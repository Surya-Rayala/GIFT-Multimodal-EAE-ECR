import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metric import AbstractMetric
from ._shared import (
    exponential_time_penalty,
    load_inroom_ids,
    pick_latest,
    select_entry_tracks,
    team_size,
)
from .utils import arg_first_non_null
from ..utils.run_metadata import resolve_fps_from_metadata

# Each side of the comparison gets a distinct color, applied consistently to
# both the markers and the connecting line so the chart is readable at a glance.
_TRAINEE_COLOR = "#fb923c"   # tab orange (this run)
_REFERENCE_COLOR = "#3b82f6" # tab blue (the reference run)


class TotalEntryTime_Metric(AbstractMetric):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.metricName = "TOTAL_TIME_OF_ENTRY"
        self.num_tracks = team_size(config)
        self.entry_starts: List[int] = []
        self.frame_rate = float(config.get("frame_rate", 30.0) or 30.0)
        self.hesitation_threshold = float(config.get("HESITATION_THRESHOLD", 1.0))
        self.hesitation_threshold_second = float(
            config.get("HESITATION_THRESHOLD_SECOND", self.hesitation_threshold * 2.0)
        )

    def process(self, ctx):
        selected = select_entry_tracks(
            getattr(ctx, "tracks_by_id", {}) or {},
            inroom_ids=getattr(ctx, "inroom_ids", []) or [],
            num_tracks=self.num_tracks,
        )
        self.entry_starts = [
            int(arg_first_non_null(traj))
            for _, traj in selected
            if arg_first_non_null(traj) is not None
        ]

    def getFinalScore(self) -> float:
        if len(self.entry_starts) < 2:
            return 0.0

        delta_frames = int(self.entry_starts[-1]) - int(self.entry_starts[0])
        delta_secs = float(delta_frames) / self.frame_rate
        allowed_secs = self._allowed_total_entry_duration(len(self.entry_starts))

        if delta_secs <= allowed_secs:
            return 1.0

        overrun = delta_secs - allowed_secs
        return round(self._exp_penalty(overrun, allowed_secs), 2)

    def _allowed_pair_gap_seconds(self, pair_number: int) -> float:
        return self.hesitation_threshold_second if int(pair_number) == 2 else self.hesitation_threshold

    def _allowed_total_entry_duration(self, entry_count: int) -> float:
        if int(entry_count) < 2:
            return 0.0
        return float(sum(self._allowed_pair_gap_seconds(i) for i in range(1, int(entry_count))))

    @staticmethod
    def _exp_penalty(overrun: float, limit: float) -> float:
        return exponential_time_penalty(overrun, limit)

    def expertCompare(self, session_folder: str, expert_folder: str, map_image=None, config=None):
        # The engine's primary call passes session/expert/map_image only (no
        # config), so fall back to the config this metric was built with —
        # otherwise ``team_size`` is lost and the chart plots every non-inroom
        # track (e.g. 5 dots) instead of the capped team_size entrants the
        # live score uses. Matches the instance-method contract of the other
        # three entry metrics.
        if config is None:
            config = getattr(self, "config", None)
        _pick_latest = pick_latest
        _load_inroom_ids = load_inroom_ids

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

        def _allowed_pair_gap_seconds(pair_number: int, base_threshold: float, second_threshold: float) -> float:
            return second_threshold if int(pair_number) == 2 else base_threshold

        def _allowed_total_entry_duration(entry_count: int, base_threshold: float, second_threshold: float) -> float:
            if int(entry_count) < 2:
                return 0.0
            return float(
                sum(
                    _allowed_pair_gap_seconds(i, base_threshold, second_threshold)
                    for i in range(1, int(entry_count))
                )
            )

        def _score(delta_secs: Optional[float], limit: float) -> Optional[float]:
            if delta_secs is None:
                return None
            if delta_secs <= limit:
                return 1.0
            overrun = delta_secs - limit
            if overrun >= limit:
                return 0.0
            return round(TotalEntryTime_Metric._exp_penalty(overrun, limit), 2)

        os.makedirs(session_folder, exist_ok=True)
        img_path = os.path.join(session_folder, "TOTAL_TIME_OF_ENTRY_Comparison.png")
        txt_path = os.path.join(session_folder, "TOTAL_TIME_OF_ENTRY_Comparison.txt")

        fallback_fps = 30.0
        base_threshold = 1.0
        second_threshold = 2.0
        n_tracks_cfg: Optional[int] = None

        if isinstance(config, dict):
            fallback_fps = float(config.get("frame_rate", fallback_fps) or fallback_fps)
            base_threshold = float(config.get("HESITATION_THRESHOLD", base_threshold) or base_threshold)
            second_threshold = float(
                config.get("HESITATION_THRESHOLD_SECOND", base_threshold * 2.0) or (base_threshold * 2.0)
            )
            try:
                n_tracks_cfg = team_size(config)
                if n_tracks_cfg <= 0:
                    n_tracks_cfg = None
            except Exception:
                n_tracks_cfg = None

        # Each run's own video fps lives in its RunMetadata sidecar. We read
        # them separately so the trainee + reference may have been recorded at
        # different frame rates without distorting the x-axis conversion.
        trainee_fps = resolve_fps_from_metadata(session_folder, fallback=fallback_fps) or fallback_fps
        reference_fps = resolve_fps_from_metadata(expert_folder, fallback=fallback_fps) or fallback_fps

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

        def _span_seconds(frames: List[int], side_fps: float) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[float]]:
            if len(frames) < 2:
                return None, None, None, None
            first_frame = int(frames[0])
            last_frame = int(frames[-1])
            delta_frames = last_frame - first_frame
            delta_secs = float(delta_frames / side_fps) if side_fps > 0 else None
            return first_frame, last_frame, delta_frames, delta_secs

        _, _, _, reference_delta_secs = _span_seconds(expert_frames, reference_fps)
        _, _, _, trainee_delta_secs = _span_seconds(trainee_frames, trainee_fps)
        reference_limit_secs = _allowed_total_entry_duration(len(expert_frames), base_threshold, second_threshold)
        trainee_limit_secs = _allowed_total_entry_duration(len(trainee_frames), base_threshold, second_threshold)

        reference_score = _score(reference_delta_secs, reference_limit_secs)
        trainee_score = _score(trainee_delta_secs, trainee_limit_secs)

        delta_time = None
        if reference_delta_secs is not None and trainee_delta_secs is not None:
            delta_time = float(trainee_delta_secs - reference_delta_secs)

        try:
            fig, ax = plt.subplots(figsize=(11.5, 3.6), constrained_layout=True)

            def _to_rel_seconds(frames: List[int], side_fps: float) -> List[float]:
                if not frames:
                    return []
                first = frames[0]
                return [float((frame - first) / side_fps) for frame in frames]

            reference_t = _to_rel_seconds(expert_frames, reference_fps)
            trainee_t = _to_rel_seconds(trainee_frames, trainee_fps)

            if reference_t:
                ax.scatter(reference_t, [1.0] * len(reference_t),
                           color=_REFERENCE_COLOR, s=80, zorder=3, label="Reference")
                ax.hlines(1.0, min(reference_t), max(reference_t),
                          colors=_REFERENCE_COLOR, linewidth=2.5, zorder=2)
                ax.text(max(reference_t) + 0.05, 1.0,
                        f"{(reference_delta_secs or 0.0):.2f}s", va="center")

            if trainee_t:
                ax.scatter(trainee_t, [0.0] * len(trainee_t),
                           color=_TRAINEE_COLOR, s=80, zorder=3, label="Trainee")
                ax.hlines(0.0, min(trainee_t), max(trainee_t),
                          colors=_TRAINEE_COLOR, linewidth=2.5, zorder=2)
                ax.text(max(trainee_t) + 0.05, 0.0,
                        f"{(trainee_delta_secs or 0.0):.2f}s", va="center")

            # Allowed-span reference line. Each side's score uses a limit based
            # on its OWN entrant count (reference_limit_secs / trainee_limit_secs),
            # so draw per-side ticks when they differ. When both sides have the
            # same entrant count (the common case) the limits match and we draw
            # one full-height line — identical to the prior behaviour.
            if (
                reference_limit_secs == trainee_limit_secs
                and reference_limit_secs and reference_limit_secs > 0
            ):
                ax.axvline(
                    reference_limit_secs,
                    linestyle="--",
                    color="#ef4444",
                    linewidth=1.8,
                    label=f"{reference_limit_secs:.2f}s allowed span",
                )
            else:
                def _draw_side_limit(limit, y, color):
                    if limit and limit > 0:
                        ax.plot([limit, limit], [y - 0.18, y + 0.18],
                                linestyle="--", color=color, linewidth=1.8, zorder=4)
                        ax.annotate(f"allowed {limit:.2f}s", xy=(limit, y + 0.2),
                                    ha="center", va="bottom", fontsize=8, color=color)

                _draw_side_limit(reference_limit_secs, 1.0, _REFERENCE_COLOR)
                _draw_side_limit(trainee_limit_secs, 0.0, _TRAINEE_COLOR)
            ax.set_yticks([0.0, 1.0])
            ax.set_yticklabels(["Trainee", "Reference"])
            ax.set_xlabel("Seconds since first team entry")
            ax.set_title("Total time of entry: Trainee vs Reference")
            ax.grid(True, axis="x", linestyle="--", alpha=0.4)
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, framealpha=0.9)

            plt.savefig(img_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass

        reference_span_str = "N/A" if reference_delta_secs is None else f"{reference_delta_secs:.2f}s"
        trainee_span_str = "N/A" if trainee_delta_secs is None else f"{trainee_delta_secs:.2f}s"
        reference_score_str = "N/A" if reference_score is None else f"{reference_score:.2f}"
        trainee_score_str = "N/A" if trainee_score is None else f"{trainee_score:.2f}"

        if delta_time is None:
            time_part = "I couldn't compare total entry time because one side did not have enough valid entrants."
        elif abs(delta_time) <= 0.05:
            time_part = "Overall entry timing looks about the same as the reference."
        elif delta_time < 0:
            time_part = f"Overall, the trainee team got everyone in about {abs(delta_time):.2f}s faster than the reference."
        else:
            time_part = f"Overall, the trainee team was about {abs(delta_time):.2f}s slower than the reference to get everyone in."

        scores_part = (
            f"Spans and scores → Trainee: {trainee_span_str} (score {trainee_score_str}), "
            f"Reference: {reference_span_str} (score {reference_score_str})."
        )

        if reference_limit_secs == trainee_limit_secs:
            thresholds_part = (
                f"Allowed team entry span: {reference_limit_secs:.2f}s based on {len(trainee_frames)} entrants, "
                f"using {base_threshold:.2f}s per gap and {second_threshold:.2f}s for entrants 2->3."
            )
        else:
            thresholds_part = (
                f"Allowed team entry span -> Trainee: {trainee_limit_secs:.2f}s ({len(trainee_frames)} entrants), "
                f"Reference: {reference_limit_secs:.2f}s ({len(expert_frames)} entrants). "
                f"Uses {base_threshold:.2f}s per gap and {second_threshold:.2f}s for entrants 2->3."
            )

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
