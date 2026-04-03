import glob
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metric import AbstractMetric


class RoomCoverage_Metric(AbstractMetric):
    """Final room visual coverage fraction (0–1) from context.room_coverage."""

    def __init__(self, config):
        super().__init__(config)
        self.metricName = "FLOOR_COVERAGE"
        self._final_score = 0.0

    def process(self, context):
        coverage_data = getattr(context, "room_coverage", None)
        if coverage_data is None:
            self._final_score = 0.0
            return

        self._final_score = float(coverage_data.get("final_fraction", 0.0) or 0.0)

    def getFinalScore(self):
        return float(self._final_score)

    @staticmethod
    def expertCompare(session_folder: str, expert_folder: str, map_image=None, config=None):
        def _pick_latest(folder: str, pattern: str) -> Optional[str]:
            matches = glob.glob(os.path.join(folder, pattern))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        def _load_room_coverage_cache(folder: str) -> Tuple[pd.DataFrame, dict]:
            path = _pick_latest(folder, "*_RoomCoverageCache.txt")
            if path is None:
                raise FileNotFoundError(f"No RoomCoverageCache found in {folder}")

            with open(path, "r") as f:
                lines = [ln.strip() for ln in f.readlines()]

            sep_idx = None
            for i, ln in enumerate(lines):
                if ln == "":
                    sep_idx = i
                    break

            curve_lines = lines
            summary_lines: List[str] = []
            if sep_idx is not None:
                curve_lines = lines[:sep_idx]
                summary_lines = lines[sep_idx + 1 :]

            curve_df = pd.DataFrame(columns=["frame", "coverage_fraction"])
            if len(curve_lines) >= 2:
                rows = []
                for ln in curve_lines[1:]:
                    if not ln:
                        continue
                    parts = [p.strip() for p in ln.split(",")]
                    if len(parts) < 2:
                        continue
                    rows.append({"frame": parts[0], "coverage_fraction": parts[1]})

                if rows:
                    curve_df = pd.DataFrame(rows)
                    curve_df["frame"] = pd.to_numeric(curve_df["frame"], errors="coerce")
                    curve_df["coverage_fraction"] = pd.to_numeric(curve_df["coverage_fraction"], errors="coerce")
                    curve_df = curve_df.dropna(subset=["frame", "coverage_fraction"]).copy()
                    curve_df["frame"] = curve_df["frame"].astype(int)
                    curve_df["coverage_fraction"] = curve_df["coverage_fraction"].astype(float)
                    curve_df = curve_df.sort_values("frame")

            summary = {
                "first_non_entry_frame": None,
                "time_to_full_seconds": None,
                "final_fraction": None,
            }

            for ln in summary_lines:
                if not ln:
                    continue
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) < 2:
                    continue

                key = parts[0]
                raw_val = parts[1]

                if raw_val == "":
                    val = None
                else:
                    try:
                        val = float(raw_val)
                    except Exception:
                        val = None

                if key == "first_non_enemy_frame":
                    key = "first_non_entry_frame"

                if key in summary:
                    summary[key] = val

            return curve_df, summary

        os.makedirs(session_folder, exist_ok=True)

        fig_path = os.path.join(session_folder, "FLOOR_COVERAGE_Comparison.jpg")
        txt_path = os.path.join(session_folder, "FLOOR_COVERAGE_Comparison.txt")

        try:
            expert_curve, expert_summary = _load_room_coverage_cache(expert_folder)
            trainee_curve, trainee_summary = _load_room_coverage_cache(session_folder)
        except Exception:
            err_text = "There was an error while processing this comparison. Missing or invalid RoomCoverageCache."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "FLOOR_COVERAGE",
                "Type": "Single",
                "ImgLocation": fig_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        expert_final = expert_summary.get("final_fraction")
        if expert_final is None:
            expert_final = float(expert_curve["coverage_fraction"].iloc[-1]) if not expert_curve.empty else 0.0

        trainee_final = trainee_summary.get("final_fraction")
        if trainee_final is None:
            trainee_final = float(trainee_curve["coverage_fraction"].iloc[-1]) if not trainee_curve.empty else 0.0

        try:
            fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)

            x = np.arange(2, dtype=float)
            expert_color = "tab:blue"
            trainee_color = "tab:orange"
            full_color = "tab:green"

            bars_ex = ax.bar(x[0], float(expert_final), width=0.6, color=expert_color, label="Expert")
            bars_tr = ax.bar(x[1], float(trainee_final), width=0.6, color=trainee_color, label="Trainee")

            ax.axhline(1.0, linestyle="--", linewidth=2, color=full_color, label="Full coverage (1.0)")

            ax.set_xticks(x)
            ax.set_xticklabels(["Expert", "Trainee"])
            ax.set_ylabel("Final coverage fraction")
            ax.set_title("Final room visual coverage: Expert vs Trainee")
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)

            for rect in list(bars_ex) + list(bars_tr):
                h = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    h + 0.02,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)

            plt.savefig(fig_path, dpi=150)
            plt.close(fig)
        except Exception:
            pass

        delta = float(trainee_final) - float(expert_final)
        if abs(delta) <= 0.02:
            comp = "about the same as"
        elif delta > 0:
            comp = "higher than"
        else:
            comp = "lower than"

        text = (
            f"Final floor coverage (score = final coverage fraction, 0–1): "
            f"Trainee {float(trainee_final):.3f} vs Expert {float(expert_final):.3f} "
            f"(Δ T−E = {delta:+.3f}). The trainee is {comp} the expert."
        )

        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

        return {
            "Name": "FLOOR_COVERAGE",
            "Type": "Single",
            "ImgLocation": fig_path,
            "TxtLocation": txt_path,
            "Text": text,
        }