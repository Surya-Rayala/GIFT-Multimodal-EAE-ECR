import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metric import AbstractMetric
from .utils import arg_first_non_null


class TotalEntryTime_Metric(AbstractMetric):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.metricName = "TOTAL_TIME_OF_ENTRY"
        self.num_tracks = len(config.get("POD", []))
        self.entry_starts: List[int] = []
        self.frame_rate = float(config.get("frame_rate", 30.0) or 30.0)
        self.entry_time_threshold_sec = float(config.get("entry_time_threshold_sec", 2.0))

    def process(self, ctx):
        inroom_ids = set(getattr(ctx, "inroom_ids", []) or [])

        tracks = {
            int(tid): traj
            for tid, traj in ctx.tracks_by_id.items()
            if int(tid) not in inroom_ids
        }

        longest = sorted(
            tracks.values(),
            key=lambda traj: sum(1 for pt in traj if pt is not None),
            reverse=True,
        )[: self.num_tracks]

        sorted_tracks = sorted(
            longest,
            key=lambda traj: arg_first_non_null(traj) if arg_first_non_null(traj) is not None else 10**9,
        )
        self.entry_starts = [
            int(arg_first_non_null(traj))
            for traj in sorted_tracks
            if arg_first_non_null(traj) is not None
        ]

    def getFinalScore(self) -> float:
        if len(self.entry_starts) < 2:
            return 0.0

        delta_frames = int(self.entry_starts[-1]) - int(self.entry_starts[0])
        delta_secs = float(delta_frames) / self.frame_rate

        if delta_secs <= self.entry_time_threshold_sec:
            return 1.0

        overrun = delta_secs - self.entry_time_threshold_sec
        return round(self._exp_penalty(overrun, self.entry_time_threshold_sec), 2)

    @staticmethod
    def _exp_penalty(overrun: float, limit: float) -> float:
        if limit <= 0:
            return 0.0
        if overrun >= limit:
            return 0.0
        return float(np.exp(-overrun / (limit - overrun)))

    @staticmethod
    def expertCompare(session_folder: str, expert_folder: str, map_image=None, config=None):
        def _pick_latest(folder: str, pattern: str) -> Optional[str]:
            matches = glob.glob(os.path.join(folder, pattern))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        def _load_inroom_ids(folder: str) -> set:
            tracker_path = _pick_latest(folder, "*_TrackerOutput.json")
            if tracker_path is None:
                return set()

            try:
                import json
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

        def _load_position_cache(folder: str) -> pd.DataFrame:
            path = _pick_latest(folder, "*_PositionCache.txt")
            if path is None:
                return pd.DataFrame(columns=["frame", "id"])

            try:
                df = pd.read_csv(path)
            except Exception:
                return pd.DataFrame(columns=["frame", "id"])

            if df is None or df.empty:
                return pd.DataFrame(columns=["frame", "id"])

            cols = {c.lower(): c for c in df.columns}
            frame_col = cols.get("frame")
            id_col = cols.get("id") or cols.get("track_id") or cols.get("track")
            if frame_col is None or id_col is None:
                return pd.DataFrame(columns=["frame", "id"])

            out = pd.DataFrame({
                "frame": pd.to_numeric(df[frame_col], errors="coerce"),
                "id": pd.to_numeric(df[id_col], errors="coerce"),
            }).dropna(subset=["frame", "id"]).copy()

            if out.empty:
                return pd.DataFrame(columns=["frame", "id"])

            out["frame"] = out["frame"].astype(int)
            out["id"] = out["id"].astype(int)

            inroom_ids = _load_inroom_ids(folder)
            out = out[~out["id"].isin(list(inroom_ids))].copy()
            return out

        def _select_entry_rows(pos: pd.DataFrame, n_tracks: int) -> List[Dict[str, int]]:
            if pos is None or pos.empty or n_tracks <= 0:
                return []

            counts = pos.groupby("id").size().sort_values(ascending=False)
            starts = pos.groupby("id")["frame"].min()

            top_ids = counts.index.tolist()[:n_tracks]
            if not top_ids:
                return []

            sub = pd.DataFrame({
                "id": [int(tid) for tid in top_ids],
                "samples": [int(counts.loc[tid]) for tid in top_ids],
                "start_frame": [int(starts.loc[tid]) for tid in top_ids],
            }).sort_values("start_frame", ascending=True)

            rows: List[Dict[str, int]] = []
            for i, row in enumerate(sub.itertuples(index=False), start=1):
                rows.append({
                    "entry_number": int(i),
                    "id": int(row.id),
                    "start_frame": int(row.start_frame),
                    "samples": int(row.samples),
                })
            return rows

        def _score(delta_secs: Optional[float], threshold: float) -> Optional[float]:
            if delta_secs is None:
                return None
            if delta_secs <= threshold:
                return 1.0
            overrun = delta_secs - threshold
            if overrun >= threshold:
                return 0.0
            return round(TotalEntryTime_Metric._exp_penalty(overrun, threshold), 2)

        os.makedirs(session_folder, exist_ok=True)
        img_path = os.path.join(session_folder, "TOTAL_TIME_OF_ENTRY_Comparison.jpg")
        txt_path = os.path.join(session_folder, "TOTAL_TIME_OF_ENTRY_Comparison.txt")

        fps = 30.0
        threshold = 2.0
        n_tracks_cfg: Optional[int] = None

        if isinstance(config, dict):
            fps = float(config.get("frame_rate", fps) or fps)
            threshold = float(config.get("entry_time_threshold_sec", threshold) or threshold)
            if config.get("POD") is not None:
                try:
                    n_tracks_cfg = int(len(config.get("POD")))
                except Exception:
                    n_tracks_cfg = None

        pos_expert = _load_position_cache(expert_folder)
        pos_trainee = _load_position_cache(session_folder)

        if (pos_expert is None or pos_expert.empty) and (pos_trainee is None or pos_trainee.empty):
            err_text = "There was an error while processing this comparison. Missing or invalid PositionCache."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "TOTAL_TIME_OF_ENTRY",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        if n_tracks_cfg is not None and n_tracks_cfg > 0:
            n_tracks = n_tracks_cfg
        else:
            n_expert = int(pos_expert["id"].nunique()) if pos_expert is not None and not pos_expert.empty else 0
            n_trainee = int(pos_trainee["id"].nunique()) if pos_trainee is not None and not pos_trainee.empty else 0
            n_tracks = min(n_expert, n_trainee)

        if n_tracks <= 0:
            n_tracks = max(
                int(pos_expert["id"].nunique()) if pos_expert is not None and not pos_expert.empty else 0,
                int(pos_trainee["id"].nunique()) if pos_trainee is not None and not pos_trainee.empty else 0,
            )

        expert_entries = _select_entry_rows(pos_expert, n_tracks)
        trainee_entries = _select_entry_rows(pos_trainee, n_tracks)

        expert_frames = [row["start_frame"] for row in expert_entries]
        trainee_frames = [row["start_frame"] for row in trainee_entries]

        def _span_seconds(frames: List[int]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[float]]:
            if len(frames) < 2:
                return None, None, None, None
            first_frame = int(frames[0])
            last_frame = int(frames[-1])
            delta_frames = last_frame - first_frame
            delta_secs = float(delta_frames / fps) if fps > 0 else None
            return first_frame, last_frame, delta_frames, delta_secs

        _, _, _, expert_delta_secs = _span_seconds(expert_frames)
        _, _, _, trainee_delta_secs = _span_seconds(trainee_frames)

        expert_score = _score(expert_delta_secs, threshold)
        trainee_score = _score(trainee_delta_secs, threshold)

        delta_time = None
        if expert_delta_secs is not None and trainee_delta_secs is not None:
            delta_time = float(trainee_delta_secs - expert_delta_secs)

        try:
            fig, ax = plt.subplots(figsize=(11.5, 3.6), constrained_layout=True)

            def _to_rel_seconds(frames: List[int]) -> List[float]:
                if not frames:
                    return []
                first = frames[0]
                return [float((frame - first) / fps) for frame in frames]

            expert_t = _to_rel_seconds(expert_frames)
            trainee_t = _to_rel_seconds(trainee_frames)

            if expert_t:
                ax.scatter(expert_t, [1.0] * len(expert_t), label="Expert")
                ax.hlines(1.0, min(expert_t), max(expert_t), linewidth=2)
                ax.text(max(expert_t) + 0.05, 1.0, f"{(expert_delta_secs or 0.0):.2f}s", va="center")

            if trainee_t:
                ax.scatter(trainee_t, [0.0] * len(trainee_t), label="Trainee")
                ax.hlines(0.0, min(trainee_t), max(trainee_t), linewidth=2)
                ax.text(max(trainee_t) + 0.05, 0.0, f"{(trainee_delta_secs or 0.0):.2f}s", va="center")

            ax.axvline(threshold, linestyle="--", linewidth=1.5, label=f"{threshold:.2f}s threshold")
            ax.set_yticks([0.0, 1.0])
            ax.set_yticklabels(["Trainee", "Expert"])
            ax.set_xlabel("Seconds since first team entry")
            ax.set_title("Total time of entry: Expert vs Trainee")
            ax.grid(True, axis="x", linestyle="--", alpha=0.4)
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, framealpha=0.9)
            fig.subplots_adjust(right=0.78)

            plt.savefig(img_path, dpi=150)
            plt.close(fig)
        except Exception:
            pass

        expert_span_str = "N/A" if expert_delta_secs is None else f"{expert_delta_secs:.2f}s"
        trainee_span_str = "N/A" if trainee_delta_secs is None else f"{trainee_delta_secs:.2f}s"
        expert_score_str = "N/A" if expert_score is None else f"{expert_score:.2f}"
        trainee_score_str = "N/A" if trainee_score is None else f"{trainee_score:.2f}"

        if delta_time is None:
            time_part = "I couldn't compare total entry time because one side did not have enough valid entrants."
        elif abs(delta_time) <= 0.05:
            time_part = "Overall entry timing looks about the same as the expert."
        elif delta_time < 0:
            time_part = f"Overall, the trainee team got everyone in about {abs(delta_time):.2f}s faster than the expert."
        else:
            time_part = f"Overall, the trainee team was about {abs(delta_time):.2f}s slower than the expert to get everyone in."

        scores_part = (
            f"Spans and scores → Trainee: {trainee_span_str} (score {trainee_score_str}), "
            f"Expert: {expert_span_str} (score {expert_score_str})."
        )

        thresholds_part = f"Threshold: full score when team entry span is at or below {threshold:.2f}s."

        text = time_part + "\n" + scores_part + "\n" + thresholds_part

        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

        return {
            "Name": "TOTAL_TIME_OF_ENTRY",
            "Type": "Single",
            "ImgLocation": img_path,
            "TxtLocation": txt_path,
            "Text": text,
        }