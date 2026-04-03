import glob
import os
import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Point, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon

from .metric import AbstractMetric
from .utils import buffer_shapely_polygon


class MoveAlongWall_Metric(AbstractMetric):
    def __init__(self, config, pWall: float = 0.2) -> None:
        super().__init__(config)
        self.metricName = "STAY_ALONG_WALL"

        boundary = config.get("Boundary")
        if boundary is None:
            raise ValueError("STAY_ALONG_WALL requires config['Boundary'].")

        if isinstance(boundary, Polygon):
            self.boundary_region = boundary
        else:
            self.boundary_region = Polygon(boundary)

        cfg_pwall = config.get("stay_along_wall_pWall", config.get("pWall", pWall))
        self.pWall = float(cfg_pwall)

        cfg_wall_px = config.get("stay_along_wall_distance_px", None)
        self.wall_distance_px = float(cfg_wall_px) if cfg_wall_px is not None else None

        interior_geom: BaseGeometry = buffer_shapely_polygon(
            self.boundary_region,
            self.pWall,
            distance_px=self.wall_distance_px,
        )
        self.interior_polygon = self._coerce_polygon(interior_geom)

        self.scores_by_id: List[float] = []
        self.map = config.get("Map Image", None)

    @staticmethod
    def _coerce_polygon(geom: BaseGeometry) -> Polygon:
        if geom is None or getattr(geom, "is_empty", True):
            return Polygon()

        if isinstance(geom, Polygon):
            return geom

        if isinstance(geom, MultiPolygon):
            return max(list(geom.geoms), key=lambda g: g.area)

        coerced = geom.buffer(0)
        if isinstance(coerced, Polygon):
            return coerced
        if isinstance(coerced, MultiPolygon):
            return max(list(coerced.geoms), key=lambda g: g.area)

        return Polygon()

    @staticmethod
    def _first_valid_index(traj: List[Optional[Tuple[float, float]]]) -> Optional[int]:
        for i, value in enumerate(traj):
            if value is not None:
                return i
        return None

    @staticmethod
    def _last_valid_index(traj: List[Optional[Tuple[float, float]]]) -> Optional[int]:
        for i in range(len(traj) - 1, -1, -1):
            if traj[i] is not None:
                return i
        return None

    @staticmethod
    def _pick_latest(folder: str, pattern: str) -> Optional[str]:
        matches = glob.glob(os.path.join(folder, pattern))
        if not matches:
            return None
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return matches[0]

    @staticmethod
    def _load_inroom_ids(folder: str) -> set:
        tracker_path = MoveAlongWall_Metric._pick_latest(folder, "*_TrackerOutput.json")
        if tracker_path is None:
            return set()

        try:
            with open(tracker_path, "r") as f:
                tracker_output = json.load(f)
        except Exception:
            return set()

        inroom_ids = set()
        for frame_entry in tracker_output:
            for obj in frame_entry.get("objects", []):
                tid = obj.get("id")
                if tid is None:
                    continue
                if obj.get("identity_role") == "inroom" or obj.get("is_inroom", False):
                    inroom_ids.add(int(tid))
        return inroom_ids

    @staticmethod
    def _load_position_cache(folder: str) -> List[Dict[str, Any]]:
        cache_path = MoveAlongWall_Metric._pick_latest(folder, "*_PositionCache.txt")
        if cache_path is None:
            raise FileNotFoundError(f"No PositionCache found in {folder}")

        df = pd.read_csv(cache_path)
        cols = {c.strip().lower(): c for c in df.columns}
        frame_col = cols.get("frame")
        id_col = cols.get("id")
        x_col = cols.get("mapx")
        y_col = cols.get("mapy")
        if frame_col is None or id_col is None or x_col is None or y_col is None:
            raise ValueError(f"Unexpected PositionCache format: {cache_path}")

        df = df[[frame_col, id_col, x_col, y_col]].dropna()
        if df.empty:
            return []

        df[frame_col] = df[frame_col].astype(int)
        df[id_col] = df[id_col].astype(int)
        df[x_col] = df[x_col].astype(float)
        df[y_col] = df[y_col].astype(float)

        inroom_ids = MoveAlongWall_Metric._load_inroom_ids(folder)
        df = df[~df[id_col].isin(list(inroom_ids))].copy()
        if df.empty:
            return []

        max_frame = int(df[frame_col].max())
        tracks: Dict[int, List[Optional[Tuple[float, float]]]] = {}

        for tid, g in df.groupby(id_col):
            tid_i = int(tid)
            traj: List[Optional[Tuple[float, float]]] = [None] * max_frame
            for _, row in g.iterrows():
                fidx = int(row[frame_col])
                if 1 <= fidx <= max_frame:
                    traj[fidx - 1] = (float(row[x_col]), float(row[y_col]))
            tracks[tid_i] = traj

        return [{"id": int(tid), "traj": traj} for tid, traj in tracks.items()]

    @staticmethod
    def _load_capture_frames(folder: str) -> Dict[int, int]:
        cache_path = MoveAlongWall_Metric._pick_latest(folder, "*_PodCache.txt")
        if cache_path is None:
            return {}

        try:
            df = pd.read_csv(cache_path)
        except Exception:
            return {}

        if df is None or df.empty:
            return {}

        cols = {c.strip().lower(): c for c in df.columns}
        assigned_col = cols.get("assigned_id")
        capture_frame_col = cols.get("capture_frame")
        if assigned_col is None or capture_frame_col is None:
            return {}

        capture_map: Dict[int, int] = {}
        for _, row in df[[assigned_col, capture_frame_col]].dropna().iterrows():
            try:
                capture_map[int(row[assigned_col])] = int(row[capture_frame_col])
            except Exception:
                continue
        return capture_map

    @staticmethod
    def _entry_map(track_dicts: List[Dict[str, Any]]) -> Dict[int, int]:
        starts = []
        for track in track_dicts:
            tid = int(track["id"])
            first = MoveAlongWall_Metric._first_valid_index(track["traj"])
            if first is not None:
                starts.append((tid, int(first)))
        starts.sort(key=lambda x: x[1])
        return {tid: i + 1 for i, (tid, _) in enumerate(starts)}

    def _near_wall_stats(
        self,
        traj: List[Optional[Tuple[float, float]]],
        entry_frame: int,
        end_frame: int,
        fps: float,
    ) -> Tuple[float, float]:
        safe_count = 0
        unsafe_count = 0
        seen = 0

        for frame_idx in range(entry_frame, end_frame + 1):
            point_xy = traj[frame_idx - 1]
            if point_xy is None:
                continue

            seen += 1
            x, y = float(point_xy[0]), float(point_xy[1])

            if self.interior_polygon.contains(Point(x, y)):
                unsafe_count += 1
            else:
                safe_count += 1

        score = float(safe_count) / float(seen) if seen > 0 else 0.0
        unsafe_time_sec = float(unsafe_count) / float(fps) if seen > 0 else 0.0
        return score, unsafe_time_sec

    def process(self, ctx):
        self.scores_by_id = []

        capture_map: Dict[int, int] = {}
        for _, info in (ctx.pod_capture or {}).items():
            assigned_id = info.get("assigned_id")
            capture_frame = info.get("capture_frame")
            if assigned_id is not None and capture_frame is not None:
                try:
                    capture_map[int(assigned_id)] = int(capture_frame)
                except Exception:
                    continue

        inroom_ids = set(getattr(ctx, "inroom_ids", []) or [])

        for track_id, positions in ctx.tracks_by_id.items():
            track_id = int(track_id)
            if track_id in inroom_ids:
                continue

            first = self._first_valid_index(positions)
            last = self._last_valid_index(positions)
            if first is None or last is None:
                continue

            entry_frame = first + 1
            end_frame = min(capture_map.get(track_id, last + 1), last + 1)
            end_frame = max(entry_frame, end_frame)

            score, _ = self._near_wall_stats(
                positions,
                entry_frame=entry_frame,
                end_frame=end_frame,
                fps=float(self.config.get("frame_rate", 30) or 30),
            )
            self.scores_by_id.append(score)

    def getFinalScore(self) -> float:
        if not self.scores_by_id:
            return 0.0
        return round(float(np.mean(self.scores_by_id)), 2)

    def expertCompare(
        self,
        session_folder: str,
        expert_folder: str,
        map_image=None,
        _config: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ) -> Dict[str, Any]:
        fps = float(self.config.get("frame_rate", 30) or 30)
        fps = fps if fps > 0 else 30.0

        def _coerce_map_image(img_or_path):
            if img_or_path is None:
                return None
            if isinstance(img_or_path, str):
                return cv2.imread(img_or_path) if os.path.exists(img_or_path) else None
            return img_or_path

        if map_image is None:
            map_image = self.map
        map_image = _coerce_map_image(map_image)

        expert_img_path = os.path.join(session_folder, "STAY_ALONG_WALL_Expert.jpg")
        trainee_img_path = os.path.join(session_folder, "STAY_ALONG_WALL_Trainee.jpg")
        txt_path = os.path.join(session_folder, "STAY_ALONG_WALL_Comparison.txt")
        os.makedirs(session_folder, exist_ok=True)

        if map_image is None:
            err_text = "There was an error while processing this comparison. Missing map image."
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(err_text)
            return {
                "Name": "STAY_ALONG_WALL",
                "Type": "SideBySide",
                "ExpertImageLocation": expert_img_path,
                "TraineeImageLocation": trainee_img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        try:
            expert_tracks = self._load_position_cache(expert_folder)
            trainee_tracks = self._load_position_cache(session_folder)
        except Exception:
            err_text = "There was an error while processing this comparison. Missing or invalid PositionCache."
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(err_text)
            return {
                "Name": "STAY_ALONG_WALL",
                "Type": "SideBySide",
                "ExpertImageLocation": expert_img_path,
                "TraineeImageLocation": trainee_img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        if len(expert_tracks) == 0 or len(trainee_tracks) == 0:
            err_text = "There was an error while processing this comparison. No valid tracks found."
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(err_text)
            return {
                "Name": "STAY_ALONG_WALL",
                "Type": "SideBySide",
                "ExpertImageLocation": expert_img_path,
                "TraineeImageLocation": trainee_img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        expert_capture = self._load_capture_frames(expert_folder)
        trainee_capture = self._load_capture_frames(session_folder)

        expert_entry = self._entry_map(expert_tracks)
        trainee_entry = self._entry_map(trainee_tracks)

        def _prep_infos(
            track_dicts: List[Dict[str, Any]],
            capture_map: Dict[int, int],
            entry_map: Dict[int, int],
        ) -> List[Dict[str, Any]]:
            infos = []
            for track in track_dicts:
                tid = int(track["id"])
                traj = track["traj"]

                first = self._first_valid_index(traj)
                last = self._last_valid_index(traj)
                if first is None or last is None:
                    continue

                entry_frame = first + 1
                last_seen = last + 1
                reached_pod = tid in capture_map
                end_frame = min(int(capture_map.get(tid, last_seen)), last_seen)
                end_frame = max(entry_frame, end_frame)

                score, unsafe_time_sec = self._near_wall_stats(
                    traj,
                    entry_frame=entry_frame,
                    end_frame=end_frame,
                    fps=fps,
                )
                infos.append(
                    {
                        "track_id": tid,
                        "entry_number": int(entry_map.get(tid, 0)) or None,
                        "traj": traj,
                        "entry_frame": entry_frame,
                        "end_frame": end_frame,
                        "score": float(score),
                        "unsafe_time_sec": float(unsafe_time_sec),
                        "reached_pod": bool(reached_pod),
                    }
                )

            infos.sort(key=lambda d: int(d["entry_number"] or 10**9))
            return infos

        expert_infos = _prep_infos(expert_tracks, expert_capture, expert_entry)
        trainee_infos = _prep_infos(trainee_tracks, trainee_capture, trainee_entry)

        self.__generateExpertCompareGraphic(
            output_folder=session_folder,
            map_view=map_image,
            boundary=self.boundary_region,
            interior=self.interior_polygon,
            infos=expert_infos,
            out_name="STAY_ALONG_WALL_Expert.jpg",
            title="Expert",
        )
        self.__generateExpertCompareGraphic(
            output_folder=session_folder,
            map_view=map_image,
            boundary=self.boundary_region,
            interior=self.interior_polygon,
            infos=trainee_infos,
            out_name="STAY_ALONG_WALL_Trainee.jpg",
            title="Trainee",
        )

        expert_scores = [float(info.get("score", 0.0)) for info in expert_infos]
        trainee_scores = [float(info.get("score", 0.0)) for info in trainee_infos]
        expert_unsafe = [float(info.get("unsafe_time_sec", 0.0)) for info in expert_infos]
        trainee_unsafe = [float(info.get("unsafe_time_sec", 0.0)) for info in trainee_infos]

        expert_final = float(np.mean(expert_scores)) if expert_scores else 0.0
        trainee_final = float(np.mean(trainee_scores)) if trainee_scores else 0.0
        delta_final = trainee_final - expert_final

        expert_unsafe_avg = float(np.mean(expert_unsafe)) if expert_unsafe else 0.0
        trainee_unsafe_avg = float(np.mean(trainee_unsafe)) if trainee_unsafe else 0.0
        delta_unsafe = trainee_unsafe_avg - expert_unsafe_avg

        if abs(delta_unsafe) < 1e-6:
            unsafe_part = (
                f"On average, the trainee spent about {trainee_unsafe_avg:.2f}s outside the safe wall band, "
                f"which matches the expert."
            )
        elif delta_unsafe < 0:
            unsafe_part = (
                f"On average, the trainee spent about {abs(delta_unsafe):.2f}s less outside the safe wall band "
                f"than the expert (T {trainee_unsafe_avg:.2f}s vs E {expert_unsafe_avg:.2f}s)."
            )
        else:
            unsafe_part = (
                f"On average, the trainee spent about {abs(delta_unsafe):.2f}s more outside the safe wall band "
                f"than the expert (T {trainee_unsafe_avg:.2f}s vs E {expert_unsafe_avg:.2f}s)."
            )

        if abs(delta_final) < 0.01:
            score_part = (
                f"Overall near-wall score looks similar (Trainee {trainee_final:.2f} vs Expert {expert_final:.2f}, "
                f"Δ {delta_final:+.2f})."
            )
        elif delta_final > 0:
            score_part = (
                f"Overall near-wall score looks better than the expert (Trainee {trainee_final:.2f} vs Expert {expert_final:.2f}, "
                f"Δ {delta_final:+.2f})."
            )
        else:
            score_part = (
                f"Overall near-wall score looks worse than the expert (Trainee {trainee_final:.2f} vs Expert {expert_final:.2f}, "
                f"Δ {delta_final:+.2f})."
            )

        max_n = max(len(expert_infos), len(trainee_infos))
        rows: List[str] = []

        header = (
            "Entry #, Trainee ID, Trainee Score, Trainee Unsafe Time (s), Expert ID, Expert Score, "
            "Expert Unsafe Time (s), Unsafe Time Δ (T−E), Score Δ (T−E), Performance\n"
        )

        eps_score = 0.01
        eps_time = 0.10

        for i in range(max_n):
            expert_info = expert_infos[i] if i < len(expert_infos) else None
            trainee_info = trainee_infos[i] if i < len(trainee_infos) else None

            expert_id = expert_info.get("track_id") if expert_info is not None else None
            trainee_id = trainee_info.get("track_id") if trainee_info is not None else None
            expert_score = expert_info.get("score") if expert_info is not None else None
            trainee_score = trainee_info.get("score") if trainee_info is not None else None
            expert_unsafe_time = expert_info.get("unsafe_time_sec") if expert_info is not None else None
            trainee_unsafe_time = trainee_info.get("unsafe_time_sec") if trainee_info is not None else None

            if expert_score is not None and trainee_score is not None:
                score_diff = float(trainee_score) - float(expert_score)
                score_diff_str = f"{score_diff:+.2f}"
            else:
                score_diff = None
                score_diff_str = "N/A"

            if expert_unsafe_time is not None and trainee_unsafe_time is not None:
                time_diff = float(trainee_unsafe_time) - float(expert_unsafe_time)
                time_diff_str = f"{time_diff:+.2f}s"
            else:
                time_diff = None
                time_diff_str = "N/A"

            if score_diff is None:
                performance = "N/A"
            elif abs(score_diff) <= eps_score:
                if time_diff is None or abs(time_diff) <= eps_time:
                    performance = "SIMILAR"
                elif time_diff < -eps_time:
                    performance = "BETTER"
                else:
                    performance = "WORSE"
            elif score_diff > eps_score:
                performance = "BETTER"
            else:
                performance = "WORSE"

            rows.append(
                f"{i+1}, "
                f"{trainee_id if trainee_id is not None else 'N/A'}, "
                f"{('N/A' if trainee_score is None else f'{float(trainee_score):.2f}')}, "
                f"{('N/A' if trainee_unsafe_time is None else f'{float(trainee_unsafe_time):.2f}')}, "
                f"{expert_id if expert_id is not None else 'N/A'}, "
                f"{('N/A' if expert_score is None else f'{float(expert_score):.2f}')}, "
                f"{('N/A' if expert_unsafe_time is None else f'{float(expert_unsafe_time):.2f}')}, "
                f"{time_diff_str}, {score_diff_str}, {performance}"
            )

        header_line = header.strip()
        details_csv = header_line + "\n" + "\n".join(rows)
        text = unsafe_part + "\n" + score_part + "\n\n" + details_csv

        def _broken_table(headers: List[str], data_rows: List[List[str]]) -> str:
            if not data_rows:
                return "(no rows)"

            widths = []
            for j, header_cell in enumerate(headers):
                max_cell = max([len(header_cell)] + [len(row[j]) for row in data_rows])
                widths.append(max_cell)

            sep = " | "

            def _fmt(values: List[str]) -> str:
                return sep.join(values[i].ljust(widths[i]) for i in range(len(headers)))

            broken_line = sep.join("-" * w for w in widths)
            out = [_fmt(headers), broken_line]
            out.extend(_fmt(row) for row in data_rows)
            return "\n".join(out)

        pretty_headers = [h.strip() for h in header_line.split(",")]
        pretty_rows: List[List[str]] = []
        for row_line in rows:
            parts = [p.strip() for p in row_line.split(",")]
            while len(parts) < len(pretty_headers):
                parts.append("N/A")
            pretty_rows.append(parts[: len(pretty_headers)])

        details_pretty = _broken_table(pretty_headers, pretty_rows)
        saved_text = unsafe_part + "\n" + score_part + "\n\n" + details_pretty + "\n"

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(saved_text)

        return {
            "Name": "STAY_ALONG_WALL",
            "Type": "SideBySide",
            "ExpertImageLocation": expert_img_path,
            "TraineeImageLocation": trainee_img_path,
            "TxtLocation": txt_path,
            "Text": text,
        }

    @staticmethod
    def __generateExpertCompareGraphic(
        output_folder: str,
        map_view,
        boundary: Polygon,
        interior: Polygon,
        infos: List[Dict[str, Any]],
        out_name: str,
        title: str,
    ) -> None:
        os.makedirs(output_folder, exist_ok=True)

        img = map_view.copy()
        h, w = img.shape[:2]
        original_w = w

        predefined_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128),
        ]

        def _color_for_id(track_id: int):
            return predefined_colors[int(track_id) % len(predefined_colors)]

        overlay = img.copy()

        def _iter_polys(geom):
            if geom is None or getattr(geom, "is_empty", True):
                return []
            if getattr(geom, "geom_type", None) == "Polygon":
                return [geom]
            if getattr(geom, "geom_type", None) == "MultiPolygon":
                return list(geom.geoms)
            return []

        def _poly_pts(poly: Polygon) -> Optional[np.ndarray]:
            try:
                return np.asarray(list(poly.exterior.coords), dtype=np.int32)
            except Exception:
                return None

        for poly in _iter_polys(boundary):
            pts = _poly_pts(poly)
            if pts is not None:
                cv2.fillPoly(overlay, [pts], (60, 180, 60))

        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in _iter_polys(interior):
            pts = _poly_pts(poly)
            if pts is not None:
                cv2.fillPoly(mask, [pts], 255)

        overlay[mask == 255] = img[mask == 255]
        img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)

        try:
            for poly in _iter_polys(boundary):
                coords = np.asarray(list(poly.exterior.coords), dtype=np.int32)
                cv2.polylines(img, [coords], isClosed=True, color=(255, 255, 255), thickness=2)
        except Exception:
            pass

        try:
            for poly in _iter_polys(interior):
                coords = np.asarray(list(poly.exterior.coords), dtype=np.int32)
                cv2.polylines(img, [coords], isClosed=True, color=(255, 255, 255), thickness=1)
        except Exception:
            pass

        safe_segments: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int, int]]] = []
        unsafe_segments: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int, int]]] = []
        safe_alpha = 0.35

        for info in infos:
            tid = int(info.get("track_id"))
            traj: List[Optional[Tuple[float, float]]] = info.get("traj")
            entry_frame = int(info.get("entry_frame"))
            end_frame = int(info.get("end_frame"))

            color = _color_for_id(tid)
            pts: List[Optional[Tuple[int, int]]] = []
            inside_flags: List[Optional[bool]] = []

            for frame_idx in range(entry_frame, end_frame + 1):
                pt = traj[frame_idx - 1] if 0 <= (frame_idx - 1) < len(traj) else None
                if pt is None:
                    pts.append(None)
                    inside_flags.append(None)
                    continue

                x, y = float(pt[0]), float(pt[1])
                pts.append((int(round(x)), int(round(y))))
                inside_flags.append(bool(interior.contains(Point(x, y))))

            prev_pt = None
            prev_inside = None
            for pt, inside in zip(pts, inside_flags):
                if pt is None or inside is None:
                    prev_pt = None
                    prev_inside = None
                    continue

                if prev_pt is not None and prev_inside is not None:
                    is_unsafe = bool(inside) or bool(prev_inside)
                    if is_unsafe:
                        unsafe_segments.append((prev_pt, pt, color))
                    else:
                        safe_segments.append((prev_pt, pt, color))

                prev_pt = pt
                prev_inside = inside

        if safe_segments:
            safe_overlay = img.copy()
            for p1, p2, color in safe_segments:
                cv2.line(safe_overlay, p1, p2, color, 2, cv2.LINE_AA)
            img = cv2.addWeighted(safe_overlay, safe_alpha, img, 1.0 - safe_alpha, 0)

        for p1, p2, color in unsafe_segments:
            cv2.line(img, p1, p2, (255, 255, 255), 6, cv2.LINE_AA)
            cv2.line(img, p1, p2, color, 3, cv2.LINE_AA)

        for info in infos:
            if not info.get("reached_pod", False):
                continue

            traj = info.get("traj")
            end_frame = int(info.get("end_frame"))
            color = _color_for_id(int(info.get("track_id")))

            star_pt = None
            for frame_idx in range(end_frame, 0, -1):
                if 0 <= (frame_idx - 1) < len(traj) and traj[frame_idx - 1] is not None:
                    x, y = float(traj[frame_idx - 1][0]), float(traj[frame_idx - 1][1])
                    star_pt = (int(round(x)), int(round(y)))
                    break

            if star_pt is None:
                continue

            cv2.drawMarker(img, star_pt, (0, 0, 0), markerType=cv2.MARKER_STAR, markerSize=16, thickness=2, line_type=cv2.LINE_AA)
            cv2.drawMarker(img, star_pt, color, markerType=cv2.MARKER_STAR, markerSize=16, thickness=1, line_type=cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        pad = 12
        sw = 14
        gap = 10
        line_h = 22

        items = []
        for info in infos:
            entry_number = info.get("entry_number")
            track_id = info.get("track_id")
            if entry_number is None or track_id is None:
                continue
            items.append((int(entry_number), int(track_id)))
        items.sort(key=lambda x: x[0])

        lines = [title, "Safe band: green", "Star = POD reached"] + [f"Entrant #{entry}" for entry, _ in items]
        max_w = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_w = max(max_w, tw)

        panel_w = pad * 2 + sw + gap + max_w
        panel_h = pad * 2 + line_h * max(1, len(lines))

        extra_right = panel_w + pad * 2
        bg = tuple(int(x) for x in img[0, 0].tolist())
        canvas = np.full((h, original_w + extra_right, 3), bg, dtype=np.uint8)
        canvas[:, :original_w] = img
        img = canvas

        x0 = original_w + pad
        y0 = pad

        cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
        cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (220, 220, 220), 2)

        y = y0 + pad + 16
        cv2.putText(img, title, (x0 + pad, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        y = y0 + pad + line_h
        cv2.rectangle(img, (x0 + pad, y + 4), (x0 + pad + sw, y + 4 + sw), (60, 180, 60), -1)
        cv2.putText(img, "Safe band: green", (x0 + pad + sw + gap, y + 16), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h

        star_cx = x0 + pad + sw // 2
        star_cy = y + 12
        cv2.drawMarker(img, (star_cx, star_cy), (0, 0, 0), markerType=cv2.MARKER_STAR, markerSize=12, thickness=2, line_type=cv2.LINE_AA)
        cv2.drawMarker(img, (star_cx, star_cy), (255, 255, 255), markerType=cv2.MARKER_STAR, markerSize=12, thickness=1, line_type=cv2.LINE_AA)
        cv2.putText(img, "Star = POD reached", (x0 + pad + sw + gap, y + 16), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h

        for entry_number, track_id in items:
            color = _color_for_id(track_id)
            cv2.rectangle(img, (x0 + pad, y + 4), (x0 + pad + sw, y + 4 + sw), color, -1)
            cv2.rectangle(img, (x0 + pad, y + 4), (x0 + pad + sw, y + 4 + sw), (255, 255, 255), 1)
            cv2.putText(img, f"Entrant #{entry_number}", (x0 + pad + sw + gap, y + 16), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y += line_h

        cv2.imwrite(os.path.join(output_folder, out_name), img)