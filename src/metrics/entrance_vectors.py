import os
import math
from typing import Optional, List, Dict, Tuple, Any

import cv2
import numpy as np
import pandas as pd

from .metric import AbstractMetric
from ._shared import (
    DoorAxes,
    classify_entry_side,
    door_for_entry,
    first_entry_frame,
    fit_entry_velocity,
    load_door_axes,
    load_entry_polygon_points,
    pick_latest,
    select_entry_tracks,
    team_size,
)
from ..utils.run_metadata import resolve_fps_from_metadata


# ---------------------------------------------------------------------------
# Developer constant for the entrance-vectors direction estimator.
# Not exposed in per-map JSON. The metric no longer applies a magnitude
# floor or a projection cutoff: every entrant with non-zero motion gets
# committed to whichever path axis (p̂_A or p̂_B) their motion is more
# aligned with. Gates were removed because a single UNKNOWN entrant
# silently inverted the alternation interpretation for the whole rest of
# the team.
# ---------------------------------------------------------------------------
ENTRY_VECTOR_WINDOW_SEC = 2.0          # window over which direction is fit


class EntranceVectors_Metric(AbstractMetric):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.metricName = "ENTRANCE_VECTORS"
        self.num_tracks = team_size(config)
        self.map = config.get("Map Image", None)
        self.tracks: Dict[int, List[Any]] = {}

        # Door axes are derived from the room boundary + the entry polygons
        # file. Both are already standard per-map config; no new keys.
        self._doors: List[DoorAxes] = self._build_doors(config)

    @staticmethod
    def _build_doors(config) -> List[DoorAxes]:
        boundary = config.get("Boundary") if isinstance(config, dict) else None
        if boundary is None:
            return []
        if hasattr(boundary, "exterior"):
            try:
                boundary_pts = list(boundary.exterior.coords)
            except Exception:
                boundary_pts = []
        else:
            boundary_pts = list(boundary)
        door_polys = load_entry_polygon_points(config.get("entry_polys_path"))
        if not boundary_pts or not door_polys:
            return []
        return load_door_axes(boundary_pts, door_polys)

    def process(self, ctx):
        inroom_ids = set(getattr(ctx, "inroom_ids", []) or [])
        self.tracks = {
            int(track_id): traj
            for track_id, traj in ctx.tracks_by_id.items()
            if int(track_id) not in inroom_ids
        }

    def getFinalScore(self) -> float:
        frame_rate = float(self.config.get("frame_rate", 30.0) or 30.0)

        selected = select_entry_tracks(self.tracks, num_tracks=self.num_tracks)
        if len(selected) < 2:
            return -1

        # Per-entrant: classify side using the door the entrant actually used,
        # the LSQ velocity over the entry window, and projection onto the
        # door's two reference path axes (p̂_A, p̂_B).
        signs_by_door: Dict[int, List[int]] = {}
        for _, trk in selected:
            info = EntranceVectors_Metric._entrance_info(
                trk,
                doors=self._doors,
                frame_rate=frame_rate,
                window_sec=ENTRY_VECTOR_WINDOW_SEC,
            )
            sign = int(info.get("sign", 0))
            if sign == 0:
                continue
            door_id = int(info.get("door_index", -1))
            signs_by_door.setdefault(door_id, []).append(sign)

        # Alternation is only meaningful among entrants who used the same door.
        # In single-door maps (the current case), there's exactly one bucket;
        # in multi-door scenarios we sum alternations and pairs across doors.
        alts = 0
        pairs = 0
        for signs in signs_by_door.values():
            for i in range(1, len(signs)):
                pairs += 1
                if signs[i] != signs[i - 1]:
                    alts += 1

        if pairs == 0:
            return -1
        return round(alts / pairs, 2)

    @staticmethod
    def _first_valid_index(trk: List[Any]) -> Optional[int]:
        for i, v in enumerate(trk):
            if v is not None:
                return i
        return None

    @staticmethod
    def _entrance_info(
        trk: List[Any],
        *,
        doors: List[DoorAxes],
        frame_rate: float,
        window_sec: float,
    ) -> Dict[str, Any]:
        """Compute the per-entrant entry record using door axes + LSQ velocity.

        The output schema is preserved (start_frame, start_xy, end_xy, dx, dy,
        sign, side) so downstream code (graphics, analysis JSON) can consume
        it unchanged. ``z_cross`` is no longer part of the model and is
        omitted.
        """
        empty: Dict[str, Any] = {
            "start_frame": None,
            "start_xy": None,
            "end_xy": None,
            "dx": None,
            "dy": None,
            "sign": 0,
            "side": "UNKNOWN",
            "door_index": -1,
        }
        if trk is None:
            return empty

        start_idx = first_entry_frame(trk, doors)
        if start_idx is None:
            return empty

        entry_pt = trk[start_idx]
        if entry_pt is None:
            for k in range(start_idx, len(trk)):
                if trk[k] is not None:
                    start_idx = k
                    entry_pt = trk[k]
                    break
        if entry_pt is None:
            return empty

        door = door_for_entry(entry_pt, doors)
        door_index = -1
        if door is not None:
            try:
                door_index = doors.index(door)
            except ValueError:
                door_index = -1

        v = fit_entry_velocity(trk, start_idx, frame_rate, window_sec)
        if v is None:
            end_xy = None
            dx = dy = None
            sign = 0
            side = "UNKNOWN"
        else:
            m_vec = v * float(window_sec)
            sign, side, _, _ = classify_entry_side(m_vec, door)
            dx, dy = float(m_vec[0]), float(m_vec[1])
            end_xy = (float(entry_pt[0]) + dx, float(entry_pt[1]) + dy)

        return {
            "start_frame": int(start_idx) + 1,
            "start_xy": (float(entry_pt[0]), float(entry_pt[1])),
            "end_xy": end_xy,
            "dx": dx,
            "dy": dy,
            "sign": int(sign),
            "side": side,
            "door_index": int(door_index),
        }

    def expertCompare(self, session_folder: str, expert_folder: str, map_image=None, pod=None):
        _pick_latest = pick_latest

        def _load_position_cache(folder: str) -> List[Dict[str, Any]]:
            cache_path = _pick_latest(folder, "*_PositionCache.txt")
            if cache_path is None:
                raise FileNotFoundError(f"No PositionCache found in {folder}")

            df = pd.read_csv(cache_path)
            cols = {c.lower(): c for c in df.columns}
            frame_col = cols.get("frame")
            id_col = cols.get("id")
            x_col = cols.get("mapx")
            y_col = cols.get("mapy")
            if frame_col is None or id_col is None or x_col is None or y_col is None:
                raise ValueError(f"Unexpected PositionCache format: {cache_path}")

            df = df[[frame_col, id_col, x_col, y_col]].dropna()
            df[frame_col] = df[frame_col].astype(int)
            df[id_col] = df[id_col].astype(int)
            df[x_col] = df[x_col].astype(float)
            df[y_col] = df[y_col].astype(float)

            tracker_path = _pick_latest(folder, "*_TrackerOutput.json")
            inroom_ids = set()
            if tracker_path is not None:
                try:
                    import json
                    with open(tracker_path, "r") as f:
                        tracker_output = json.load(f)
                    for frame_entry in tracker_output:
                        for obj in frame_entry.get("objects", []):
                            tid = obj.get("id")
                            if tid is None:
                                continue
                            if obj.get("identity_role") == "inroom" or obj.get("is_inroom", False):
                                inroom_ids.add(int(tid))
                except Exception:
                    inroom_ids = set()

            max_frame = int(df[frame_col].max()) if len(df) else 0
            tracks = {}
            for tid, g in df.groupby(id_col):
                tid = int(tid)
                if tid in inroom_ids:
                    continue

                traj = [None] * max_frame
                for _, row in g.iterrows():
                    fidx = int(row[frame_col])
                    if 1 <= fidx <= max_frame:
                        traj[fidx - 1] = (float(row[x_col]), float(row[y_col]))
                tracks[tid] = traj

            return [{"id": tid, "traj": traj} for tid, traj in tracks.items()]

        def _per_track_info(
            track: Dict[str, Any],
            *,
            frame_rate: float,
            window_sec: float,
        ) -> Dict[str, Any]:
            info = EntranceVectors_Metric._entrance_info(
                track["traj"],
                doors=self._doors,
                frame_rate=frame_rate,
                window_sec=window_sec,
            )
            info["track_id"] = int(track["id"])
            return info

        if map_image is None:
            map_image = self.map
            if isinstance(map_image, str):
                map_image = cv2.imread(map_image) if os.path.exists(map_image) else None

        os.makedirs(session_folder, exist_ok=True)
        expert_img_path = os.path.join(session_folder, "ENTRANCE_VECTORS_Reference.png")
        trainee_img_path = os.path.join(session_folder, "ENTRANCE_VECTORS_Trainee.png")
        txt_path = os.path.join(session_folder, "ENTRANCE_VECTORS_Comparison.txt")

        if map_image is None:
            err_text = "There was an error while processing this comparison. Missing map image."
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(err_text)
            return {
                "Name": "ENTRANCE_VECTORS",
                "Type": "SideBySide",
                "ReferenceImageLocation": expert_img_path,
                "TraineeImageLocation": trainee_img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        # Read each run's own recorded fps so the entry-velocity fit window
        # (window_sec × fps frames) is correct per side when the two runs were
        # recorded at different frame rates. Falls back to config fps.
        config_fps = float(self.config.get("frame_rate", 30.0) or 30.0)
        trainee_fps = resolve_fps_from_metadata(session_folder, fallback=config_fps) or config_fps
        reference_fps = resolve_fps_from_metadata(expert_folder, fallback=config_fps) or config_fps
        window_sec = ENTRY_VECTOR_WINDOW_SEC

        try:
            expert_tracks = _load_position_cache(expert_folder)
            trainee_tracks = _load_position_cache(session_folder)
        except Exception:
            err_text = "There was an error while processing this comparison. Missing or invalid PositionCache."
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(err_text)
            return {
                "Name": "ENTRANCE_VECTORS",
                "Type": "SideBySide",
                "ReferenceImageLocation": expert_img_path,
                "TraineeImageLocation": trainee_img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        def _select_for_scoring(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            by_tid = {int(t["id"]): t for t in tracks}
            ordered = select_entry_tracks(
                {tid: rec["traj"] for tid, rec in by_tid.items()},
                num_tracks=self.num_tracks,
            )
            return [by_tid[tid] for tid, _ in ordered]

        expert_tracks = _select_for_scoring(expert_tracks)
        trainee_tracks = _select_for_scoring(trainee_tracks)

        if len(expert_tracks) == 0 or len(trainee_tracks) == 0:
            err_text = "There was an error while processing this comparison. No valid tracks found."
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(err_text)
            return {
                "Name": "ENTRANCE_VECTORS",
                "Type": "SideBySide",
                "ReferenceImageLocation": expert_img_path,
                "TraineeImageLocation": trainee_img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        expert_infos = [
            _per_track_info(t, frame_rate=reference_fps, window_sec=window_sec)
            for t in expert_tracks
        ]
        trainee_infos = [
            _per_track_info(t, frame_rate=trainee_fps, window_sec=window_sec)
            for t in trainee_tracks
        ]

        def _alternation_score_and_by_entry(signs: List[int]) -> Tuple[float, List[str]]:
            idxs = [i for i, s in enumerate(signs) if s != 0]
            by_entry = ["N/A"] * len(signs)
            if len(idxs) >= 1:
                by_entry[idxs[0]] = "N/A"

            alt = 0
            for j in range(1, len(idxs)):
                prev_i = idxs[j - 1]
                cur_i = idxs[j]
                is_alt = np.sign(signs[cur_i]) != np.sign(signs[prev_i])
                if is_alt:
                    alt += 1
                    by_entry[cur_i] = "1"
                else:
                    by_entry[cur_i] = "0"

            denom = max(1, len(idxs) - 1)
            return float(alt) / float(denom), by_entry

        expert_signs = [int(info.get("sign", 0)) for info in expert_infos]
        trainee_signs = [int(info.get("sign", 0)) for info in trainee_infos]
        expert_alt_score, expert_alt_by_entry = _alternation_score_and_by_entry(expert_signs)
        trainee_alt_score, trainee_alt_by_entry = _alternation_score_and_by_entry(trainee_signs)

        EntranceVectors_Metric.__generateExpertCompareGraphic(
            output_folder=session_folder,
            expert_infos=expert_infos,
            trainee_infos=trainee_infos,
            map_view=map_image,
        )

        max_n = max(len(expert_infos), len(trainee_infos))
        valid = 0
        match = 0
        per_entry_lines = []

        for i in range(max_n):
            einfo = expert_infos[i] if i < len(expert_infos) else None
            tinfo = trainee_infos[i] if i < len(trainee_infos) else None

            expert_id = einfo.get("track_id") if einfo is not None else None
            trainee_id = tinfo.get("track_id") if tinfo is not None else None

            e_sign = einfo.get("sign") if einfo is not None else 0
            t_sign = tinfo.get("sign") if tinfo is not None else 0

            if einfo is not None and tinfo is not None and e_sign != 0 and t_sign != 0:
                is_match = e_sign == t_sign
                valid += 1
                match += 1 if is_match else 0
                match_str = "YES" if is_match else "NO"
            else:
                match_str = "N/A"

            t_alt = trainee_alt_by_entry[i] if i < len(trainee_alt_by_entry) else "N/A"
            e_alt = expert_alt_by_entry[i] if i < len(expert_alt_by_entry) else "N/A"

            def _alt_num(v: str) -> Optional[int]:
                return int(v) if v in ("0", "1") else None

            t_num = _alt_num(t_alt)
            e_num = _alt_num(e_alt)

            if t_num is not None and e_num is not None:
                pair_diff = t_num - e_num
                pair_diff_str = f"{pair_diff:+d}"
                if pair_diff > 0:
                    pair_cmp = "BETTER"
                elif pair_diff < 0:
                    pair_cmp = "WORSE"
                else:
                    pair_cmp = "SIMILAR"
            else:
                pair_diff_str = "N/A"
                pair_cmp = "N/A"

            per_entry_lines.append(
                f"{i+1}, {trainee_id if trainee_id is not None else 'N/A'}, "
                f"{expert_id if expert_id is not None else 'N/A'}, {match_str}, "
                f"{pair_diff_str}, {pair_cmp}"
            )

        alt_summary = (
            f"Alternation score: trainee {trainee_alt_score * 100:.1f}% vs reference {expert_alt_score * 100:.1f}%."
        )

        if valid > 0:
            sign_match_pct = (match / valid) * 100.0
            summary_line = (
                f"The trainee matches the reference about {sign_match_pct:.1f}% of the time on entrance side. "
                f"{alt_summary}"
            )
        else:
            summary_line = (
                "Couldn't compute a match percent (not enough paired entries with a clear entrance side). "
                f"{alt_summary}"
            )

        rows = []
        for line in per_entry_lines:
            parts = [p.strip() for p in line.split(",")]
            while len(parts) < 6:
                parts.append("N/A")
            entry, trainee_id, reference_id, match_str, pair_diff_str, pair_cmp = parts[:6]
            rows.append(
                {
                    "Entry": entry,
                    "Trainee ID": trainee_id,
                    "Reference ID": reference_id,
                    "Match": match_str,
                    "PairScoreΔ (T−R)": pair_diff_str,
                    "Trainee vs Reference": pair_cmp,
                }
            )

        df_table = pd.DataFrame(rows)

        details_header = "Entry, Trainee ID, Reference ID, Match, PairScoreΔ (T−R), Trainee vs Reference\n"
        details_csv = details_header + "\n".join(per_entry_lines)
        text = summary_line + "\n\n" + details_csv

        def _dotted_table(df: pd.DataFrame) -> str:
            if df is None or df.empty:
                return "(no rows)"

            cols = [str(c) for c in df.columns]
            data_rows = [["" if v is None else str(v) for v in row] for row in df.to_numpy().tolist()]

            widths = []
            for j, c in enumerate(cols):
                max_cell = max([len(c)] + [len(r[j]) for r in data_rows])
                widths.append(max_cell)

            sep = " | "

            def _fmt_row(values: List[str]) -> str:
                return sep.join(v.ljust(widths[i]) for i, v in enumerate(values))

            line_parts = ["-" * w for w in widths]
            broken_line = sep.join(line_parts)

            out = [_fmt_row(cols), broken_line]
            out.extend(_fmt_row(r) for r in data_rows)
            return "\n".join(out)

        table_pretty = _dotted_table(df_table)
        saved_text = summary_line + "\n\n" + table_pretty + "\n"

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(saved_text)

        return {
            "Name": "ENTRANCE_VECTORS",
            "Type": "SideBySide",
            "ReferenceImageLocation": expert_img_path,
            "TraineeImageLocation": trainee_img_path,
            "TxtLocation": txt_path,
            "Text": text,
        }

    @staticmethod
    def __generateExpertCompareGraphic(output_folder, expert_infos, trainee_infos, map_view):
        expert_view = map_view.copy()
        trainee_view = map_view.copy()

        predefined_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128),
        ]

        # Text rendered via the shared TrueType helper so the legend reads
        # as crisp typed text instead of OpenCV's wavy stroke fonts.
        from .utils import draw_text, text_size

        def _draw_legend(img, infos, title="Legend"):
            h0, w0 = img.shape[:2]
            diag = math.hypot(w0, h0)
            # Scale text size with the canvas — small maps get smaller text.
            title_size = max(15, min(22, int(round(diag / 90.0))))
            body_size = max(13, min(18, int(round(diag / 110.0))))

            line_h = int(round(title_size * 1.55))
            swatch = int(round(body_size * 1.05))
            gap = int(round(body_size * 0.65))
            pad = int(round(body_size * 0.9))

            # Width: widest of the title (in title size) and the body lines.
            max_text_w = text_size(title, size_px=title_size)[0]
            for i in range(1, len(infos) + 1):
                w, _ = text_size(f"Entrant #{i}", size_px=body_size)
                max_text_w = max(max_text_w, w)

            panel_w = pad * 2 + swatch + gap + max_text_w
            panel_h = pad * 2 + line_h * (1 + len(infos))

            extra_right = panel_w + pad * 2
            extra_bottom = max(0, panel_h + pad * 2 - h0)

            bg = tuple(int(x) for x in img[0, 0].tolist())
            canvas = np.full((h0 + extra_bottom, w0 + extra_right, 3), bg, dtype=np.uint8)
            canvas[:h0, :w0] = img
            img = canvas

            x0 = w0 + pad
            y0 = pad

            overlay = img.copy()
            panel_bg = (28, 28, 28)
            border = (220, 220, 220)
            border_thick = max(1, int(round(diag / 900.0)))
            cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), panel_bg, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.82, img, 0.18, 0, dst=img)
            cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), border, border_thick, cv2.LINE_AA)

            # Title row (TrueType, larger size).
            draw_text(
                img, title, (x0 + pad, y0 + pad),
                size_px=title_size, fill_bgr=(255, 255, 255),
            )

            # Per-entrant rows: colour swatch + label.
            y = y0 + pad + line_h
            for entry_num, info in enumerate(infos, start=1):
                tid = info.get("track_id")
                tid = int(tid) if tid is not None else entry_num
                color = predefined_colors[tid % len(predefined_colors)]

                sx1 = x0 + pad
                sy1 = y + int(round(body_size * 0.25))
                sx2 = sx1 + swatch
                sy2 = sy1 + swatch

                cv2.rectangle(img, (sx1, sy1), (sx2, sy2), color, -1, cv2.LINE_AA)
                cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (255, 255, 255), border_thick, cv2.LINE_AA)

                draw_text(
                    img, f"Entrant #{entry_num}",
                    (sx2 + gap, y),
                    size_px=body_size, fill_bgr=(255, 255, 255),
                )
                y += line_h

            return img

        def _draw_arrow(img, start_xy, end_xy, color, idx, total):
            if start_xy is None or end_xy is None:
                return

            sx, sy = float(start_xy[0]), float(start_xy[1])
            ex0, ey0 = float(end_xy[0]), float(end_xy[1])

            v = np.array([ex0 - sx, ey0 - sy], dtype=float)
            n = float(np.linalg.norm(v))
            if n == 0:
                return
            u = v / n

            h, w = img.shape[:2]
            diag = math.hypot(w, h)
            arrow_len = max(70.0, 0.14 * diag)
            ex = sx + u[0] * arrow_len
            ey = sy + u[1] * arrow_len

            perp = np.array([-u[1], u[0]], dtype=float)
            spread = idx - (total + 1) / 2.0
            offset = perp * (14.0 * spread)

            p1 = (int(round(sx + offset[0])), int(round(sy + offset[1])))
            p2 = (int(round(ex + offset[0])), int(round(ey + offset[1])))

            thick = max(2, int(round(diag / 520.0)))
            cv2.arrowedLine(img, p1, p2, (0, 0, 0), thick + 3, cv2.LINE_AA, tipLength=0.28)
            cv2.arrowedLine(img, p1, p2, color, thick, cv2.LINE_AA, tipLength=0.28)
            cv2.circle(img, p1, max(4, thick + 1), (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, p1, max(3, thick), color, -1, cv2.LINE_AA)

        total_e = len(expert_infos)
        for entry_num, info in enumerate(expert_infos, start=1):
            tid = info.get("track_id")
            tid = int(tid) if tid is not None else entry_num
            color = predefined_colors[tid % len(predefined_colors)]
            _draw_arrow(expert_view, info.get("start_xy"), info.get("end_xy"), color, idx=entry_num, total=total_e)

        total_t = len(trainee_infos)
        for entry_num, info in enumerate(trainee_infos, start=1):
            tid = info.get("track_id")
            tid = int(tid) if tid is not None else entry_num
            color = predefined_colors[tid % len(predefined_colors)]
            _draw_arrow(trainee_view, info.get("start_xy"), info.get("end_xy"), color, idx=entry_num, total=total_t)

        expert_view = _draw_legend(expert_view, expert_infos, title="Reference")
        trainee_view = _draw_legend(trainee_view, trainee_infos, title="Trainee")

        cv2.imwrite(os.path.join(output_folder, "ENTRANCE_VECTORS_Trainee.png"), trainee_view)
        cv2.imwrite(os.path.join(output_folder, "ENTRANCE_VECTORS_Reference.png"), expert_view)