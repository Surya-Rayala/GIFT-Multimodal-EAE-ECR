import glob
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metric import AbstractMetric


class POD_Metric(AbstractMetric):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.metricName = "IDENTIFY_AND_HOLD_DESIGNATED_AREA"
        self.pod = config.get("POD", [])
        self.num_tracks = len(self.pod)
        self.tracks: Dict[int, List[Optional[Tuple[float, float]]]] = {}
        self.pod_assignment: Dict[int, Optional[int]] = {}
        self._final_score: float = 0.0

    def process(self, ctx):
        inroom_ids = set(getattr(ctx, "inroom_ids", []) or [])
        self.tracks = {
            int(track_id): traj
            for track_id, traj in ctx.tracks_by_id.items()
            if int(track_id) not in inroom_ids
        }

    @staticmethod
    def _first_non_null(trk: List[Optional[Tuple[float, float]]]) -> Optional[int]:
        for i, value in enumerate(trk):
            if value is not None:
                return i
        return None

    @staticmethod
    def _traj_len(trk: List[Optional[Tuple[float, float]]]) -> int:
        return int(sum(1 for value in trk if value is not None))

    def _pod_points_list(self) -> List[np.ndarray]:
        raw = self.pod.tolist() if isinstance(self.pod, np.ndarray) else list(self.pod)
        out: List[np.ndarray] = []
        for point in raw:
            try:
                out.append(np.asarray(point, dtype=float).reshape(2))
            except Exception:
                continue
        return out

    def getFinalScore(self) -> float:
        if not self.tracks:
            self._final_score = 0.0
            return 0.0

        pod_points = self._pod_points_list()
        if not pod_points:
            self._final_score = 0.0
            return 0.0

        track_items = list(self.tracks.items())
        track_items.sort(key=lambda item: self._traj_len(item[1]), reverse=True)
        track_items = track_items[: self.num_tracks]

        if len(track_items) < self.num_tracks:
            self._final_score = 0.0
            return 0.0

        track_items.sort(
            key=lambda item: (
                self._first_non_null(item[1])
                if self._first_non_null(item[1]) is not None
                else 10**9
            )
        )

        self.pod_assignment = self._assign_pods(track_items, pod_points)

        denom = 5000.0
        scores: List[float] = []

        for track_id, traj in track_items:
            pod_idx = self.pod_assignment.get(int(track_id))
            if pod_idx is None or pod_idx < 0 or pod_idx >= len(pod_points):
                continue

            pod_xy = pod_points[int(pod_idx)]
            pts = np.array([pt for pt in traj if pt is not None], dtype=float)
            if pts.size == 0:
                continue

            dists = np.linalg.norm(pts - pod_xy.reshape(1, 2), axis=1)
            per_frame_score = np.exp(-(dists * dists) / denom)
            scores.append(float(np.mean(per_frame_score)))

        self._final_score = float(np.mean(scores)) if scores else 0.0
        return round(self._final_score, 2)

    def _assign_pods(
        self,
        track_items: List[Tuple[int, List[Optional[Tuple[float, float]]]]],
        pod_points: List[np.ndarray],
    ) -> Dict[int, Optional[int]]:
        first_trk = track_items[0][1]
        first_pts = [pt for pt in first_trk if pt is not None][:30]

        if not first_pts:
            doorway_x = float(pod_points[0][0])
            doorway_y = float(pod_points[0][1])
            entrance_sign = 1
        elif len(first_pts) < 2:
            doorway_x = float(first_pts[0][0])
            doorway_y = float(first_pts[0][1])
            pod_mean_x = float(np.mean([pt[0] for pt in pod_points]))
            entrance_sign = 1 if doorway_x >= pod_mean_x else -1
        else:
            doorway_x = float(first_pts[0][0])
            doorway_y = float(first_pts[0][1])
            x_coords = [float(pt[0]) for pt in first_pts]
            deltas = np.diff(x_coords)
            entrance_sign = 1 if float(np.mean(deltas)) >= 0 else -1

        pods_plus = [(idx, pt) for idx, pt in enumerate(pod_points) if float(pt[0]) > doorway_x]
        pods_minus = [(idx, pt) for idx, pt in enumerate(pod_points) if float(pt[0]) <= doorway_x]

        pods_plus.sort(key=lambda item: abs(float(item[1][1]) - doorway_y), reverse=True)
        pods_minus.sort(key=lambda item: abs(float(item[1][1]) - doorway_y), reverse=True)

        out: Dict[int, Optional[int]] = {}
        plus_idx = 0
        minus_idx = 0
        current_side = int(entrance_sign)

        for track_id, _traj in track_items:
            assigned: Optional[int] = None

            if current_side == 1 and plus_idx < len(pods_plus):
                assigned = int(pods_plus[plus_idx][0])
                plus_idx += 1
            elif current_side == -1 and minus_idx < len(pods_minus):
                assigned = int(pods_minus[minus_idx][0])
                minus_idx += 1
            else:
                if plus_idx < len(pods_plus):
                    assigned = int(pods_plus[plus_idx][0])
                    plus_idx += 1
                elif minus_idx < len(pods_minus):
                    assigned = int(pods_minus[minus_idx][0])
                    minus_idx += 1

            out[int(track_id)] = assigned
            current_side *= -1

        return out

    def expertCompare(
        self,
        session_folder: str,
        expert_folder: str,
        _map_image=None,
        config: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ) -> Dict[str, Any]:
        def _pick_latest(folder: str, pattern: str) -> Optional[str]:
            matches = glob.glob(os.path.join(folder, pattern))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        def _load_inroom_ids(folder: str) -> Set[int]:
            import json

            tracker_path = _pick_latest(folder, "*_TrackerOutput.json")
            if tracker_path is None:
                return set()

            try:
                with open(tracker_path, "r") as f:
                    tracker_output = json.load(f)
            except Exception:
                return set()

            inroom_ids: Set[int] = set()
            for frame_entry in tracker_output:
                for obj in frame_entry.get("objects", []):
                    tid = obj.get("id")
                    if tid is None:
                        continue
                    if obj.get("identity_role") == "inroom" or obj.get("is_inroom", False):
                        inroom_ids.add(int(tid))
            return inroom_ids

        def _load_position_tracks(
            folder: str,
        ) -> Tuple[Dict[int, List[Optional[Tuple[float, float]]]], float]:
            path = _pick_latest(folder, "*_PositionCache.txt")
            if path is None:
                return {}, 30.0

            try:
                df = pd.read_csv(path)
            except Exception:
                return {}, 30.0

            if df is None or df.empty:
                return {}, 30.0

            cols = {c.strip().lower(): c for c in df.columns}
            frame_col = cols.get("frame")
            id_col = cols.get("id")
            x_col = cols.get("mapx")
            y_col = cols.get("mapy")
            if frame_col is None or id_col is None or x_col is None or y_col is None:
                return {}, 30.0

            df = df[[frame_col, id_col, x_col, y_col]].dropna()
            if df.empty:
                return {}, 30.0

            df[frame_col] = df[frame_col].astype(int)
            df[id_col] = df[id_col].astype(int)
            df[x_col] = df[x_col].astype(float)
            df[y_col] = df[y_col].astype(float)

            fps_local = float((config or {}).get("frame_rate", self.config.get("frame_rate", 30.0)) or 30.0)
            if fps_local <= 0:
                fps_local = 30.0

            inroom_ids_local = _load_inroom_ids(folder)
            df = df[~df[id_col].isin(list(inroom_ids_local))].copy()
            if df.empty:
                return {}, fps_local

            max_frame = int(df[frame_col].max())
            tracks: Dict[int, List[Optional[Tuple[float, float]]]] = {}

            for tid, group in df.groupby(id_col):
                tid_int = int(tid)
                traj: List[Optional[Tuple[float, float]]] = [None] * max_frame
                for _, row in group.iterrows():
                    frame_idx = int(row[frame_col])
                    if 1 <= frame_idx <= max_frame:
                        traj[frame_idx - 1] = (float(row[x_col]), float(row[y_col]))
                tracks[tid_int] = traj

            return tracks, fps_local

        def _select_top_tracks(
            tracks: Dict[int, List[Optional[Tuple[float, float]]]],
            n_local: int,
        ) -> List[Tuple[int, List[Optional[Tuple[float, float]]]]]:
            items = list(tracks.items())
            items.sort(key=lambda item: self._traj_len(item[1]), reverse=True)
            items = items[:n_local]
            items.sort(
                key=lambda item: (
                    self._first_non_null(item[1])
                    if self._first_non_null(item[1]) is not None
                    else 10**9
                )
            )
            return items

        def _entry_map(tracks: Dict[int, List[Optional[Tuple[float, float]]]]) -> Dict[int, int]:
            starts = []
            for tid, traj in tracks.items():
                first = self._first_non_null(traj)
                if first is not None:
                    starts.append((int(tid), int(first)))
            starts.sort(key=lambda item: item[1])
            return {tid: i + 1 for i, (tid, _first) in enumerate(starts)}

        def _hold_stats(
            track_items: List[Tuple[int, List[Optional[Tuple[float, float]]]]],
            pod_points_local: List[np.ndarray],
            assignment: Dict[int, Optional[int]],
            fps_local: float,
        ) -> Tuple[float, Dict[int, Dict[str, float]], List[float], List[float]]:
            denom = 5000.0

            firsts = [
                first
                for _tid, traj in track_items
                for first in [self._first_non_null(traj)]
                if first is not None
            ]
            first_entry = int(min(firsts)) if firsts else 0

            per_pod: Dict[int, Dict[str, float]] = {}
            for tid, _traj in track_items:
                pod_idx = assignment.get(int(tid))
                if pod_idx is None:
                    continue
                per_pod[int(pod_idx)] = {
                    "track_id": float(int(tid)),
                    "score": 0.0,
                }

            max_len = max((len(traj) for _tid, traj in track_items), default=0)
            curve_t: List[float] = []
            curve_s: List[float] = []

            cumulative_sum = 0.0
            cumulative_n = 0

            for frame_idx in range(first_entry, max_len):
                frame_scores: List[float] = []
                for tid, traj in track_items:
                    pt = traj[frame_idx] if frame_idx < len(traj) else None
                    if pt is None:
                        continue

                    pod_idx = assignment.get(int(tid))
                    if pod_idx is None or pod_idx < 0 or pod_idx >= len(pod_points_local):
                        continue

                    dist = float(
                        np.linalg.norm(
                            np.asarray(pt, dtype=float) - pod_points_local[int(pod_idx)].reshape(2)
                        )
                    )
                    score = float(np.exp(-(dist * dist) / denom))
                    frame_scores.append(score)

                if frame_scores:
                    frame_mean = float(np.mean(frame_scores))
                    cumulative_sum += frame_mean
                    cumulative_n += 1
                    curve_t.append((frame_idx - first_entry) / max(fps_local, 1.0))
                    curve_s.append(cumulative_sum / cumulative_n)

            total_scores: List[float] = []
            track_map = dict(track_items)
            for pod_idx, stat in per_pod.items():
                tid = int(stat["track_id"])
                traj = track_map.get(tid)
                if traj is None:
                    continue

                pod_xy = pod_points_local[int(pod_idx)]
                pts = np.array([pt for pt in traj if pt is not None], dtype=float)
                if pts.size == 0:
                    continue

                dists = np.linalg.norm(pts - pod_xy.reshape(1, 2), axis=1)
                stat["score"] = float(np.mean(np.exp(-(dists * dists) / denom)))
                total_scores.append(stat["score"])

            overall = float(np.mean(total_scores)) if total_scores else 0.0
            return overall, per_pod, curve_t, curve_s

        def _pretty_table(headers: List[str], data_rows: List[List[str]]) -> str:
            if not headers:
                return ""

            normalized_rows: List[List[str]] = []
            for row in data_rows or []:
                rr = list(row)
                if len(rr) < len(headers):
                    rr = rr + ["N/A"] * (len(headers) - len(rr))
                elif len(rr) > len(headers):
                    rr = rr[: len(headers)]
                normalized_rows.append(["" if value is None else str(value) for value in rr])

            widths: List[int] = []
            for j, header in enumerate(headers):
                max_len = len(str(header))
                for row in normalized_rows:
                    if j < len(row):
                        max_len = max(max_len, len(str(row[j])))
                widths.append(max_len)

            sep = " | "
            header_line = sep.join(str(header).ljust(widths[i]) for i, header in enumerate(headers))
            dash_line = sep.join("-" * widths[i] for i in range(len(headers)))

            lines = [header_line, dash_line]
            for row in normalized_rows:
                lines.append(sep.join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))
            return "\n".join(lines)

        pod_points = self._pod_points_list()
        os.makedirs(session_folder, exist_ok=True)

        img_path = os.path.join(session_folder, "IDENTIFY_AND_HOLD_DESIGNATED_AREA_Comparison.jpg")
        txt_path = os.path.join(session_folder, "IDENTIFY_AND_HOLD_DESIGNATED_AREA_Comparison.txt")

        if not pod_points:
            err_text = "There was an error while processing this comparison. Missing POD coordinates."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "IDENTIFY_AND_HOLD_DESIGNATED_AREA",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        expert_tracks, expert_fps = _load_position_tracks(expert_folder)
        trainee_tracks, trainee_fps = _load_position_tracks(session_folder)

        if not expert_tracks and not trainee_tracks:
            err_text = "There was an error while processing this comparison. Missing or invalid PositionCache."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "IDENTIFY_AND_HOLD_DESIGNATED_AREA",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        pod_count = int(len(pod_points))
        expert_items = _select_top_tracks(expert_tracks, pod_count)
        trainee_items = _select_top_tracks(trainee_tracks, pod_count)

        expert_assignment = self._assign_pods(expert_items, pod_points) if expert_items else {}
        trainee_assignment = self._assign_pods(trainee_items, pod_points) if trainee_items else {}

        expert_score, expert_per_pod, expert_t, expert_curve = _hold_stats(
            expert_items,
            pod_points,
            expert_assignment,
            expert_fps,
        )
        trainee_score, trainee_per_pod, trainee_t, trainee_curve = _hold_stats(
            trainee_items,
            pod_points,
            trainee_assignment,
            trainee_fps,
        )

        expert_entry = _entry_map(expert_tracks)
        trainee_entry = _entry_map(trainee_tracks)

        rows: List[Dict[str, Any]] = []
        for pod_idx in range(pod_count):
            expert_stat = expert_per_pod.get(pod_idx)
            trainee_stat = trainee_per_pod.get(pod_idx)

            expert_tid = None if expert_stat is None else int(expert_stat.get("track_id", -1))
            trainee_tid = None if trainee_stat is None else int(trainee_stat.get("track_id", -1))

            rows.append(
                {
                    "pod_idx": int(pod_idx),
                    "expert_entry_number": "" if expert_tid is None else expert_entry.get(expert_tid, ""),
                    "expert_id": "" if expert_tid is None else expert_tid,
                    "expert_score": "" if expert_stat is None else float(expert_stat.get("score", 0.0)),
                    "trainee_entry_number": "" if trainee_tid is None else trainee_entry.get(trainee_tid, ""),
                    "trainee_id": "" if trainee_tid is None else trainee_tid,
                    "trainee_score": "" if trainee_stat is None else float(trainee_stat.get("score", 0.0)),
                }
            )

        try:
            df_plot = pd.DataFrame(rows)

            fig, (ax_ts, ax_bar) = plt.subplots(
                nrows=1,
                ncols=2,
                figsize=(14.5, 5.6),
                constrained_layout=True,
            )

            if expert_t:
                ax_ts.plot(expert_t, expert_curve, label="Expert", linewidth=2.5)
            if trainee_t:
                ax_ts.plot(trainee_t, trainee_curve, label="Trainee", linewidth=2.5)

            ax_ts.set_xlabel("Seconds since first team entry")
            ax_ts.set_ylabel("Cumulative POD hold score")
            ax_ts.set_title("Holding assigned PODs over time")
            ax_ts.set_ylim(0.0, 1.05)
            ax_ts.grid(True, axis="y", linestyle="--", alpha=0.35)

            if df_plot.empty:
                ax_bar.set_axis_off()
            else:
                df_plot = df_plot.copy().sort_values("pod_idx")
                x = df_plot["pod_idx"].astype(int).tolist()
                idx = np.arange(len(x), dtype=float)
                width = 0.36

                expert_scores = pd.to_numeric(df_plot["expert_score"], errors="coerce").fillna(0).to_numpy(dtype=float)
                trainee_scores = pd.to_numeric(df_plot["trainee_score"], errors="coerce").fillna(0).to_numpy(dtype=float)

                ax_bar.bar(idx - width / 2.0, expert_scores, width, label="Expert", alpha=0.9)
                ax_bar.bar(idx + width / 2.0, trainee_scores, width, label="Trainee", alpha=0.9)

                for i in range(len(x)):
                    ax_bar.text(
                        idx[i] - width / 2.0,
                        expert_scores[i] + 0.02,
                        f"~{expert_scores[i]:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
                    ax_bar.text(
                        idx[i] + width / 2.0,
                        trainee_scores[i] + 0.02,
                        f"~{trainee_scores[i]:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

                ax_bar.set_xticks(idx)
                ax_bar.set_xticklabels([f"P{int(i)}" for i in x])
                ax_bar.set_xlabel("POD")
                ax_bar.set_ylabel("Per-POD holding score")
                ax_bar.set_title("Per-POD holding score")
                ax_bar.set_ylim(0.0, 1.05)
                ax_bar.grid(True, axis="y", linestyle="--", alpha=0.35)

            h1, l1 = ax_ts.get_legend_handles_labels()
            h2, l2 = ax_bar.get_legend_handles_labels()
            seen = set()
            handles = []
            labels = []
            for handle, label in list(zip(h1, l1)) + list(zip(h2, l2)):
                if not label or label in seen:
                    continue
                seen.add(label)
                handles.append(handle)
                labels.append(label)

            if handles:
                fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.9)
                fig.subplots_adjust(right=0.82)

            plt.savefig(img_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
            plt.close(fig)
        except Exception:
            pass

        delta_final = float(trainee_score - expert_score)

        if trainee_score == expert_score:
            score_part = (
                f"Overall POD holding looks similar to the expert "
                f"(Trainee {trainee_score:.2f} vs Expert {expert_score:.2f}, Δ {delta_final:+.2f})."
            )
        elif trainee_score > expert_score:
            score_part = (
                f"Overall POD holding looks better than the expert "
                f"(Trainee {trainee_score:.2f} vs Expert {expert_score:.2f}, Δ {delta_final:+.2f})."
            )
        else:
            score_part = (
                f"Overall POD holding looks worse than the expert "
                f"(Trainee {trainee_score:.2f} vs Expert {expert_score:.2f}, Δ {delta_final:+.2f})."
            )

        diffs: List[float] = []
        for row in rows:
            expert_sc = row.get("expert_score")
            trainee_sc = row.get("trainee_score")
            if expert_sc in (None, "") or trainee_sc in (None, ""):
                continue
            diffs.append(float(trainee_sc) - float(expert_sc))

        avg_delta_score = float(np.mean(diffs)) if diffs else None

        if avg_delta_score is None:
            hold_part = "On average, per-POD holding scores could not be compared because paired POD scores were missing."
        elif avg_delta_score == 0:
            hold_part = "On average, the trainee's per-POD holding scores were similar to the expert."
        elif avg_delta_score > 0:
            hold_part = (
                f"On average, the trainee's per-POD holding scores were about {abs(avg_delta_score):.2f} higher than the expert."
            )
        else:
            hold_part = (
                f"On average, the trainee's per-POD holding scores were about {abs(avg_delta_score):.2f} lower than the expert."
            )

        lines = [
            "POD, Expert Entrant#, Expert ID, Expert ~Score, Trainee Entrant#, Trainee ID, Trainee ~Score, Score Δ (T−E), Performance",
        ]

        for row in rows:
            pod_idx = row.get("pod_idx")
            expert_ent = row.get("expert_entry_number")
            expert_id = row.get("expert_id")
            trainee_ent = row.get("trainee_entry_number")
            trainee_id = row.get("trainee_id")
            expert_sc = row.get("expert_score")
            trainee_sc = row.get("trainee_score")

            expert_sc_f = None if expert_sc in (None, "") else float(expert_sc)
            trainee_sc_f = None if trainee_sc in (None, "") else float(trainee_sc)

            delta_sc = None
            if expert_sc_f is not None and trainee_sc_f is not None:
                delta_sc = float(trainee_sc_f - expert_sc_f)

            if delta_sc is None:
                perf = "N/A"
            elif delta_sc == 0:
                perf = "SIMILAR"
            elif delta_sc > 0:
                perf = "BETTER"
            else:
                perf = "WORSE"

            lines.append(
                f"P{int(pod_idx) if pod_idx is not None else 'N/A'}, "
                f"{expert_ent if expert_ent != '' else 'N/A'}, {expert_id if expert_id != '' else 'N/A'}, "
                f"~{(0.0 if expert_sc_f is None else expert_sc_f):.2f}, "
                f"{trainee_ent if trainee_ent != '' else 'N/A'}, {trainee_id if trainee_id != '' else 'N/A'}, "
                f"~{(0.0 if trainee_sc_f is None else trainee_sc_f):.2f}, "
                f"{('N/A' if delta_sc is None else f'{delta_sc:+.2f}')}, "
                f"{perf}"
            )

        details_csv = "\n".join(lines)
        text = score_part + "\n" + hold_part + "\n\n" + details_csv

        try:
            header_cells = [cell.strip() for cell in lines[0].split(",")]
            data_cells = [[cell.strip() for cell in line.split(",")] for line in lines[1:]]
            details_pretty = _pretty_table(header_cells, data_cells)

            saved_text = score_part + "\n" + hold_part + "\n\n" + details_pretty + "\n"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(saved_text)
        except Exception:
            pass

        return {
            "Name": "IDENTIFY_AND_HOLD_DESIGNATED_AREA",
            "Type": "Single",
            "ImgLocation": img_path,
            "TxtLocation": txt_path,
            "Text": text,
        }