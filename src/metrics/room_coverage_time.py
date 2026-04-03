import glob
import io
import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metric import AbstractMetric


class TotalRoomCoverageTime_Metric(AbstractMetric):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.metricName = "TOTAL_FLOOR_COVERAGE_TIME"
        self.threshold = float(config.get("coverage_time_threshold", 3))
        self.time_to_full = None

    def process(self, ctx):
        room_cov = getattr(ctx, "room_coverage", None)
        if room_cov is None:
            self.time_to_full = None
        else:
            self.time_to_full = room_cov.get("time_to_full")

    def getFinalScore(self) -> float:
        if self.time_to_full is None:
            return 0.0
        if self.time_to_full <= self.threshold:
            return 1.0

        overrun = self.time_to_full - self.threshold
        return round(self._exp_penalty(overrun, self.threshold), 2)

    def _exp_penalty(self, overrun: float, limit: float) -> float:
        if overrun >= limit:
            return 0.0
        return float(np.exp(-(overrun) / (limit - overrun)))

    @staticmethod
    def expertCompare(session_folder: str, expert_folder: str, map_image=None, config=None, **kwargs):
        def _pick_latest(folder: str, pattern: str) -> Optional[str]:
            matches = glob.glob(os.path.join(folder, pattern))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        def _parse_room_coverage_cache(path: str) -> Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
            with open(path, "r") as f:
                lines = [ln.rstrip("\n") for ln in f.readlines()]

            blank_idx = None
            for i, ln in enumerate(lines):
                if ln.strip() == "":
                    blank_idx = i
                    break

            table_lines = lines if blank_idx is None else lines[:blank_idx]
            summary_lines = [] if blank_idx is None else lines[blank_idx + 1 :]

            df = pd.DataFrame(columns=["frame", "coverage_fraction"])
            if len(table_lines) >= 2:
                try:
                    df = pd.read_csv(io.StringIO("\n".join(table_lines)))
                except Exception:
                    df = pd.DataFrame(columns=["frame", "coverage_fraction"])

            if not df.empty:
                cols = {c.lower(): c for c in df.columns}
                fcol = cols.get("frame")
                ccol = cols.get("coverage_fraction") or cols.get("coverage") or cols.get("fraction")

                if fcol is None or ccol is None:
                    df = pd.DataFrame(columns=["frame", "coverage_fraction"])
                else:
                    out = pd.DataFrame(
                        {
                            "frame": pd.to_numeric(df[fcol], errors="coerce"),
                            "coverage_fraction": pd.to_numeric(df[ccol], errors="coerce"),
                        }
                    ).dropna(subset=["frame"]).copy()
                    out["frame"] = out["frame"].astype(int)
                    out = out.sort_values("frame")
                    out["coverage_fraction"] = out["coverage_fraction"].clip(lower=0.0, upper=1.0)
                    df = out

            summary: Dict[str, Optional[float]] = {
                "first_non_entry_frame": None,
                "time_to_full_seconds": None,
                "final_fraction": None,
            }

            for ln in summary_lines:
                if not ln.strip():
                    continue
                parts = [p.strip() for p in ln.split(",", 1)]
                if len(parts) != 2:
                    continue

                key, value = parts
                if key == "first_non_enemy_frame":
                    key = "first_non_entry_frame"

                if key not in summary:
                    continue

                if value == "":
                    summary[key] = None
                    continue

                try:
                    if key == "first_non_entry_frame":
                        summary[key] = float(int(float(value)))
                    else:
                        summary[key] = float(value)
                except Exception:
                    summary[key] = None

            if summary.get("final_fraction") is None and not df.empty:
                summary["final_fraction"] = float(df["coverage_fraction"].iloc[-1])

            return df, summary

        def _frame_rate_from_config() -> float:
            fps = 30.0
            if config is not None:
                try:
                    fps = float(config.get("frame_rate") or config.get("fps") or fps)
                except Exception:
                    pass
            try:
                fps = float(kwargs.get("frame_rate") or kwargs.get("fps") or fps)
            except Exception:
                pass
            return fps if fps > 0 else 30.0

        def _load_tracker_output_roles(folder: str) -> Tuple[Dict[int, str], Dict[int, str]]:
            path = _pick_latest(folder, "*_TrackerOutput.json")
            if path is None:
                return {}, {}

            try:
                import json

                with open(path, "r") as f:
                    data = json.load(f)
            except Exception:
                return {}, {}

            identity_role_by_id: Dict[int, str] = {}
            birth_location_by_id: Dict[int, str] = {}

            if not isinstance(data, list):
                return {}, {}

            for frame_entry in data:
                for obj in frame_entry.get("objects", []) or []:
                    tid = obj.get("id")
                    if tid is None:
                        continue
                    try:
                        tid = int(tid)
                    except Exception:
                        continue

                    role = obj.get("identity_role")
                    birth_location = obj.get("birth_location")

                    if role is not None:
                        identity_role_by_id[tid] = str(role)
                    if birth_location is not None:
                        birth_location_by_id[tid] = str(birth_location)

            return identity_role_by_id, birth_location_by_id

        def _load_position_cache_first_entry(folder: str, inroom_id_start: int) -> Optional[int]:
            path = _pick_latest(folder, "*_PositionCache.txt")
            if path is None:
                return None

            try:
                df = pd.read_csv(path)
            except Exception:
                return None

            if df is None or df.empty:
                return None

            cols = {c.lower(): c for c in df.columns}
            f_col = cols.get("frame")
            id_col = cols.get("id") or cols.get("track_id") or cols.get("track")
            if f_col is None or id_col is None:
                return None

            out = pd.DataFrame(
                {
                    "frame": pd.to_numeric(df[f_col], errors="coerce"),
                    "id": pd.to_numeric(df[id_col], errors="coerce"),
                }
            ).dropna(subset=["frame", "id"]).copy()

            if out.empty:
                return None

            out["frame"] = out["frame"].astype(int)
            out["id"] = out["id"].astype(int)

            identity_role_by_id, birth_location_by_id = _load_tracker_output_roles(folder)

            def _is_entry_track(tid: int) -> bool:
                role = identity_role_by_id.get(tid)
                birth_location = birth_location_by_id.get(tid)
                if role == "entry" or birth_location == "entry":
                    return True
                if role == "inroom" or birth_location == "inroom":
                    return False
                return tid < inroom_id_start

            out = out[out["id"].map(_is_entry_track)].copy()
            if out.empty:
                return None

            return int(out["frame"].min())

        def _start_frame_team_entry(folder: str, df: pd.DataFrame, inroom_id_start: int) -> int:
            entry = _load_position_cache_first_entry(folder, inroom_id_start)
            if entry is not None:
                return int(entry)
            if df is not None and not df.empty:
                return int(df["frame"].min())
            return 1

        def _post_entry_df(df: pd.DataFrame, start_frame: int, fps: float) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame(columns=["frame", "coverage_fraction", "time_sec"])

            d = df[df["frame"] >= int(start_frame)].copy()
            if d.empty:
                return pd.DataFrame(columns=["frame", "coverage_fraction", "time_sec"])

            d["time_sec"] = (d["frame"] - int(start_frame)) / float(fps)
            d = d.sort_values("time_sec")
            return d[["frame", "coverage_fraction", "time_sec"]]

        def _score_time_to_full(t: Optional[float], limit: float) -> float:
            if t is None or (isinstance(t, float) and np.isnan(t)):
                return 0.0
            tt = float(t)
            if tt <= limit:
                return 1.0
            overrun = tt - limit
            if overrun >= limit:
                return 0.0
            return float(np.exp(-(overrun) / (limit - overrun)))

        def _generate_plot(out_path: str, ex_post: pd.DataFrame, tr_post: pd.DataFrame, threshold_sec: Optional[float] = None) -> None:
            if (ex_post is None or ex_post.empty) and (tr_post is None or tr_post.empty):
                return

            fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)

            if ex_post is not None and not ex_post.empty:
                ax.plot(ex_post["time_sec"], ex_post["coverage_fraction"], label="Expert", color="tab:blue")
            if tr_post is not None and not tr_post.empty:
                ax.plot(tr_post["time_sec"], tr_post["coverage_fraction"], label="Trainee", color="tab:orange")

            eps = 1e-6
            ex_full = None
            tr_full = None

            if ex_post is not None and not ex_post.empty:
                hit = ex_post[ex_post["coverage_fraction"] >= (1.0 - eps)]
                if not hit.empty:
                    ex_full = float(hit["time_sec"].iloc[0])

            if tr_post is not None and not tr_post.empty:
                hit = tr_post[tr_post["coverage_fraction"] >= (1.0 - eps)]
                if not hit.empty:
                    tr_full = float(hit["time_sec"].iloc[0])

            if ex_full is not None and tr_full is not None:
                ax.set_xlim(0.0, max(ex_full, tr_full) + 0.25)

            if threshold_sec is not None and np.isfinite(float(threshold_sec)):
                ax.axvline(
                    float(threshold_sec),
                    linestyle="--",
                    linewidth=1.2,
                    color="black",
                    alpha=0.8,
                    label=f"Threshold ({float(threshold_sec):.2f}s)",
                )

            ax.set_xlabel("Seconds since first entry track appears")
            ax.set_ylabel("Coverage fraction")
            ax.set_title("Room visual coverage over time (expert vs trainee)")
            ax.set_ylim(0, 1.05)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            ax.legend(loc="lower right")

            plt.savefig(out_path, dpi=150)
            plt.close(fig)

        os.makedirs(session_folder, exist_ok=True)
        img_path = os.path.join(session_folder, "TOTAL_FLOOR_COVERAGE_TIME_Comparison.jpg")
        txt_path = os.path.join(session_folder, "TOTAL_FLOOR_COVERAGE_TIME_Comparison.txt")

        expert_path = _pick_latest(expert_folder, "*_RoomCoverageCache.txt")
        trainee_path = _pick_latest(session_folder, "*_RoomCoverageCache.txt")

        if expert_path is None or trainee_path is None:
            err_text = "There was an error while processing this comparison. Missing RoomCoverageCache in expert and/or trainee folder."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "TOTAL_FLOOR_COVERAGE_TIME",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        try:
            ex_df, ex_sum = _parse_room_coverage_cache(expert_path)
            tr_df, tr_sum = _parse_room_coverage_cache(trainee_path)
        except Exception:
            err_text = "There was an error while processing this comparison. RoomCoverageCache could not be parsed."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "TOTAL_FLOOR_COVERAGE_TIME",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        if ex_df.empty and tr_df.empty:
            err_text = "There was an error while processing this comparison. No per-frame coverage data found."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "TOTAL_FLOOR_COVERAGE_TIME",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        fps = _frame_rate_from_config()

        threshold = 3.0
        if config is not None:
            try:
                threshold = float(config.get("coverage_time_threshold", threshold))
            except Exception:
                pass

        inroom_id_start = 99
        if config is not None:
            try:
                inroom_id_start = int(config.get("inroom_id_start", inroom_id_start))
            except Exception:
                pass
        try:
            inroom_id_start = int(kwargs.get("inroom_id_start") or inroom_id_start)
        except Exception:
            pass

        ex_start = _start_frame_team_entry(expert_folder, ex_df, inroom_id_start)
        tr_start = _start_frame_team_entry(session_folder, tr_df, inroom_id_start)

        ex_post = _post_entry_df(ex_df, ex_start, fps)
        tr_post = _post_entry_df(tr_df, tr_start, fps)

        expert_score = _score_time_to_full(ex_sum.get("time_to_full_seconds"), threshold)
        trainee_score = _score_time_to_full(tr_sum.get("time_to_full_seconds"), threshold)

        try:
            _generate_plot(img_path, ex_post, tr_post, threshold_sec=threshold)
        except Exception:
            pass

        ex_time_full = ex_sum.get("time_to_full_seconds")
        tr_time_full = tr_sum.get("time_to_full_seconds")

        if ex_time_full is None and tr_time_full is None:
            time_part = "Neither run reached full coverage."
        elif ex_time_full is None and tr_time_full is not None:
            time_part = f"Trainee reached full coverage in {float(tr_time_full):.2f}s; expert did not reach full coverage."
        elif ex_time_full is not None and tr_time_full is None:
            time_part = f"Expert reached full coverage in {float(ex_time_full):.2f}s; trainee did not reach full coverage."
        else:
            dt = float(tr_time_full) - float(ex_time_full)
            if abs(dt) <= 0.05:
                time_part = "Trainee and expert were about the same on time to full coverage."
            elif dt < 0:
                time_part = f"Trainee reached full coverage about {abs(dt):.2f}s faster than the expert."
            else:
                time_part = f"Trainee reached full coverage about {abs(dt):.2f}s slower than the expert."

        ds = float(trainee_score - expert_score)
        if abs(ds) <= 0.01:
            score_part = f"Scores were basically the same (Trainee {trainee_score:.2f}, Expert {expert_score:.2f})."
        elif ds > 0:
            score_part = f"On score, the trainee came in higher (Trainee {trainee_score:.2f} vs Expert {expert_score:.2f}, +{abs(ds):.2f})."
        else:
            score_part = f"On score, the trainee came in lower (Trainee {trainee_score:.2f} vs Expert {expert_score:.2f}, -{abs(ds):.2f})."

        thresholds_part = f"Thresholds: full score when time_to_full ≤ {float(threshold):.2f}s."

        text = "\n".join([time_part, score_part, thresholds_part])

        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

        return {
            "Name": "TOTAL_FLOOR_COVERAGE_TIME",
            "Type": "Single",
            "ImgLocation": img_path,
            "TxtLocation": txt_path,
            "Text": text,
        }