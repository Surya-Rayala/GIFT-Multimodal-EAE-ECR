import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metric import AbstractMetric
from src.helper_functions import compute_threat_clearance


class ThreatClearance_Metric(AbstractMetric):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.metricName = "THREAT_CLEARANCE"
        self.visual_angle: float = float(config.get("visual_angle_degrees", 20.0))
        self.score_value: float = 0.0
        self.clearance_map: Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]] = {}
        self.total_inroom: int = 0
        self.cleared_inroom: int = 0

    def process(self, ctx) -> None:
        inroom_ids = set(getattr(ctx, "inroom_ids", []) or [])

        if hasattr(ctx, "threat_clearance") and ctx.threat_clearance:
            self.clearance_map = ctx.threat_clearance
        else:
            self.clearance_map = compute_threat_clearance(
                ctx.tracker_output,
                ctx.keypoint_details,
                ctx.gaze_info,
                inroom_ids=list(inroom_ids),
                visual_angle_deg=self.visual_angle,
            )

        if not inroom_ids:
            inferred_inroom_ids = {
                int(track_id)
                for track_id, value in self.clearance_map.items()
                if value is not None
            }
            inroom_ids = inferred_inroom_ids

        self.total_inroom = len(inroom_ids)
        self.cleared_inroom = sum(
            1
            for track_id in inroom_ids
            if track_id in self.clearance_map and self.clearance_map[track_id][0] is not None
        )

        self.score_value = self.cleared_inroom / self.total_inroom if self.total_inroom else 1.0

    def getFinalScore(self) -> float:
        return float(self.score_value)

    @staticmethod
    def expertCompare(
        session_folder: str,
        expert_folder: str,
        _map_image=None,
        config: Optional[dict] = None,
    ):
        def _pick_latest(folder: str, pattern: str) -> Optional[str]:
            matches = glob.glob(os.path.join(folder, pattern))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        def _inroom_id_start_from_config(cfg: Optional[dict]) -> int:
            if cfg and isinstance(cfg, dict):
                try:
                    return int(cfg.get("inroom_id_start", 99))
                except Exception:
                    return 99
            return 99

        def _frame_rate_from_config(cfg: Optional[dict]) -> float:
            if cfg and isinstance(cfg, dict):
                try:
                    return float(cfg.get("frame_rate", 30.0))
                except Exception:
                    return 30.0
            return 30.0

        def _load_position_cache_min(folder: str) -> pd.DataFrame:
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
            f_col = cols.get("frame")
            id_col = cols.get("id") or cols.get("track_id") or cols.get("track")
            if f_col is None or id_col is None:
                return pd.DataFrame(columns=["frame", "id"])

            out = pd.DataFrame(
                {
                    "frame": pd.to_numeric(df[f_col], errors="coerce"),
                    "id": pd.to_numeric(df[id_col], errors="coerce"),
                }
            ).dropna(subset=["frame", "id"])

            if out.empty:
                return pd.DataFrame(columns=["frame", "id"])

            out["frame"] = out["frame"].astype(int)
            out["id"] = out["id"].astype(int)
            return out

        def _first_entry_frame(folder: str, *, inroom_id_start: int) -> Optional[int]:
            pos = _load_position_cache_min(folder)
            if pos.empty:
                return None

            entry_tracks = pos[pos["id"] < inroom_id_start]
            if entry_tracks.empty:
                return None
            return int(entry_tracks["frame"].min())

        def _load_threat_clearance_cache(folder: str) -> pd.DataFrame:
            path = _pick_latest(folder, "*_ThreatClearanceCache.txt")
            if path is None:
                raise FileNotFoundError(f"No ThreatClearanceCache found in {folder}")

            df = pd.read_csv(path)
            if df is None or df.empty:
                return pd.DataFrame(
                    columns=["inroom_id", "immediate_frame", "contact_end_frame", "clearing_friend"]
                )

            cols = {c.lower(): c for c in df.columns}
            i_col = cols.get("inroom_id") or cols.get("enemy_id") or cols.get("id")
            s_col = cols.get("immediate_frame") or cols.get("start_frame") or cols.get("start")
            end_col = cols.get("contact_end_frame") or cols.get("end_frame") or cols.get("end")
            f_col = cols.get("clearing_friend") or cols.get("clearing_friend_id") or cols.get("friend")

            if i_col is None or s_col is None or end_col is None:
                raise ValueError(f"Unexpected ThreatClearanceCache format: {path}")

            out = pd.DataFrame(
                {
                    "inroom_id": pd.to_numeric(df[i_col], errors="coerce"),
                    "immediate_frame": pd.to_numeric(df[s_col], errors="coerce"),
                    "contact_end_frame": pd.to_numeric(df[end_col], errors="coerce"),
                    "clearing_friend": pd.to_numeric(df[f_col], errors="coerce") if f_col is not None else np.nan,
                }
            ).dropna(subset=["inroom_id"])

            out["inroom_id"] = out["inroom_id"].astype(int)

            for c in ["immediate_frame", "contact_end_frame", "clearing_friend"]:
                out[c] = pd.to_numeric(out[c], errors="coerce")
                out.loc[out[c] == -1, c] = np.nan

            return out

        def _trend(delta: Optional[float], *, eps: float = 0.25) -> str:
            if delta is None or (isinstance(delta, float) and np.isnan(delta)):
                return "N/A"
            d = float(delta)
            if abs(d) <= eps:
                return "SIMILAR"
            return "FASTER" if d < 0 else "SLOWER"

        def _pretty_table(headers: List[str], data_rows: List[List[str]]) -> str:
            if not headers:
                return ""

            norm_rows: List[List[str]] = []
            for r in data_rows or []:
                rr = list(r)
                if len(rr) < len(headers):
                    rr = rr + ["N/A"] * (len(headers) - len(rr))
                elif len(rr) > len(headers):
                    rr = rr[: len(headers)]
                norm_rows.append(["" if v is None else str(v) for v in rr])

            widths: List[int] = []
            for j, h in enumerate(headers):
                max_len = len(str(h))
                for r in norm_rows:
                    max_len = max(max_len, len(str(r[j])))
                widths.append(max_len)

            sep = " | "
            header_line = sep.join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
            dash_line = sep.join("-" * widths[i] for i in range(len(headers)))

            out_lines = [header_line, dash_line]
            for r in norm_rows:
                out_lines.append(sep.join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))
            return "\n".join(out_lines)

        os.makedirs(session_folder, exist_ok=True)
        img_path = os.path.join(session_folder, "THREAT_CLEARANCE_Comparison.jpg")
        txt_path = os.path.join(session_folder, "THREAT_CLEARANCE_Comparison.txt")

        inroom_id_start = _inroom_id_start_from_config(config)
        fps = _frame_rate_from_config(config)

        try:
            df_ex = _load_threat_clearance_cache(expert_folder)
            df_tr = _load_threat_clearance_cache(session_folder)
        except Exception:
            err_text = (
                "There was an error while processing this comparison. "
                "Missing or invalid ThreatClearanceCache."
            )
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "THREAT_CLEARANCE",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        ex_t0 = _first_entry_frame(expert_folder, inroom_id_start=inroom_id_start)
        tr_t0 = _first_entry_frame(session_folder, inroom_id_start=inroom_id_start)

        ex_by_inroom = df_ex.set_index("inroom_id") if not df_ex.empty else pd.DataFrame().set_index(pd.Index([]))
        tr_by_inroom = df_tr.set_index("inroom_id") if not df_tr.empty else pd.DataFrame().set_index(pd.Index([]))

        inroom_ids = (
            sorted(set(df_ex["inroom_id"].tolist()) | set(df_tr["inroom_id"].tolist()))
            if (not df_ex.empty or not df_tr.empty)
            else []
        )

        rows: List[Dict] = []
        ex_cleared = 0
        tr_cleared = 0
        both_cleared_deltas: List[float] = []

        for inroom_id in inroom_ids:
            ex_end = None
            if inroom_id in ex_by_inroom.index:
                r = ex_by_inroom.loc[inroom_id]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                ex_end = r.get("contact_end_frame")

            tr_end = None
            if inroom_id in tr_by_inroom.index:
                r = tr_by_inroom.loc[inroom_id]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                tr_end = r.get("contact_end_frame")

            ex_is_cleared = ex_end is not None and not (isinstance(ex_end, float) and np.isnan(ex_end))
            tr_is_cleared = tr_end is not None and not (isinstance(tr_end, float) and np.isnan(tr_end))

            ex_cleared += 1 if ex_is_cleared else 0
            tr_cleared += 1 if tr_is_cleared else 0

            ex_sec = None
            tr_sec = None

            if ex_is_cleared and ex_t0 is not None:
                ex_sec = (float(ex_end) - float(ex_t0)) / fps
                if ex_sec < 0:
                    ex_sec = 0.0

            if tr_is_cleared and tr_t0 is not None:
                tr_sec = (float(tr_end) - float(tr_t0)) / fps
                if tr_sec < 0:
                    tr_sec = 0.0

            delta = None
            if ex_sec is not None and tr_sec is not None:
                delta = float(tr_sec) - float(ex_sec)
                both_cleared_deltas.append(delta)

            rows.append(
                {
                    "inroom_id": int(inroom_id),
                    "expert_cleared": "YES" if ex_is_cleared else "NO",
                    "trainee_cleared": "YES" if tr_is_cleared else "NO",
                    "expert_clear_end_sec_from_team0": "" if ex_sec is None else round(float(ex_sec), 3),
                    "trainee_clear_end_sec_from_team0": "" if tr_sec is None else round(float(tr_sec), 3),
                    "delta_clear_end_sec_trainee_minus_expert": "" if delta is None else round(float(delta), 3),
                }
            )

        out_df = pd.DataFrame(rows)

        n_inroom = len(inroom_ids)
        ex_rate = (ex_cleared / n_inroom) if n_inroom > 0 else 1.0
        tr_rate = (tr_cleared / n_inroom) if n_inroom > 0 else 1.0

        diff_pp = (tr_rate - ex_rate) * 100.0
        if abs(diff_pp) <= 5.0:
            rate_part = "Overall clearance rate looks about the same as the expert."
        elif diff_pp > 0:
            rate_part = f"Overall, the trainee cleared more in-room threats than the expert (+{abs(diff_pp):.0f}pp)."
        else:
            rate_part = f"Overall, the trainee cleared fewer in-room threats than the expert (-{abs(diff_pp):.0f}pp)."

        parts: List[str] = []
        parts.append(
            f"Clearance rate: Trainee {tr_cleared}/{n_inroom} ({tr_rate*100:.1f}%) vs "
            f"Expert {ex_cleared}/{n_inroom} ({ex_rate*100:.1f}%)."
        )
        parts.append(rate_part)

        if both_cleared_deltas:
            avg_dt = float(np.mean(both_cleared_deltas))
            if abs(avg_dt) <= 0.25:
                parts.append(
                    f"For in-room tracks cleared by both teams, completion time was similar on average "
                    f"(Δ T−E = {avg_dt:+.2f}s)."
                )
            elif avg_dt < 0:
                parts.append(
                    f"For in-room tracks cleared by both teams, the trainee finished clearance about "
                    f"{abs(avg_dt):.2f}s faster on average."
                )
            else:
                parts.append(
                    f"For in-room tracks cleared by both teams, the trainee finished clearance about "
                    f"{abs(avg_dt):.2f}s slower on average."
                )
        else:
            parts.append(
                "I couldn't compute a shared clearance-time average "
                "(no in-room tracks cleared by both teams with valid entry alignment)."
            )

        lines = [
            "InRoom ID, Expert Cleared, Expert Time(s), Trainee Cleared, Trainee Time(s), Clearance Speed, Performance",
        ]

        for r in rows:
            inroom_id = r.get("inroom_id")
            ex_c = r.get("expert_cleared")
            tr_c = r.get("trainee_cleared")
            ex_t = r.get("expert_clear_end_sec_from_team0")
            tr_t = r.get("trainee_clear_end_sec_from_team0")
            dt = r.get("delta_clear_end_sec_trainee_minus_expert")

            ex_t_str = "N/A" if ex_t in (None, "") else f"{float(ex_t):.2f}"
            tr_t_str = "N/A" if tr_t in (None, "") else f"{float(tr_t):.2f}"
            speed = _trend(dt)

            ex_yes = ex_c == "YES"
            tr_yes = tr_c == "YES"

            if ex_yes == tr_yes:
                perf = "SIMILAR"
            elif tr_yes and not ex_yes:
                perf = "BETTER"
            else:
                perf = "WORSE"

            if not (ex_yes and tr_yes):
                speed = "N/A"

            lines.append(
                f"{int(inroom_id) if inroom_id is not None else 'N/A'}, "
                f"{ex_c}, {ex_t_str}, {tr_c}, {tr_t_str}, {speed}, {perf}"
            )

        details_csv = "\n".join(lines)
        text = "\n".join(parts) + "\n\n" + details_csv

        try:
            header_cells = [c.strip() for c in lines[0].split(",")]
            data_cells = [[c.strip() for c in ln.split(",")] for ln in lines[1:]]
            details_pretty = _pretty_table(header_cells, data_cells)

            saved_text = "\n".join(parts) + "\n\n" + details_pretty + "\n"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(saved_text)
        except Exception:
            pass

        try:
            ThreatClearance_Metric.__generateExpertCompareGraphic(
                output_path=img_path,
                df=out_df,
            )
        except Exception:
            pass

        return {
            "Name": "THREAT_CLEARANCE",
            "Type": "Single",
            "ImgLocation": img_path,
            "TxtLocation": txt_path,
            "Text": text,
        }

    @staticmethod
    def __generateExpertCompareGraphic(output_path: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return

        plot_df = df.copy()
        plot_df["inroom_id"] = pd.to_numeric(plot_df["inroom_id"], errors="coerce")
        plot_df = plot_df.dropna(subset=["inroom_id"])
        if plot_df.empty:
            return

        plot_df["inroom_id"] = plot_df["inroom_id"].astype(int)
        inroom_ids = plot_df["inroom_id"].tolist()
        idx = np.arange(len(inroom_ids), dtype=float)
        width = 0.36

        ex_cleared = plot_df["expert_cleared"].astype(str).str.upper().eq("YES").to_numpy(dtype=bool)
        tr_cleared = plot_df["trainee_cleared"].astype(str).str.upper().eq("YES").to_numpy(dtype=bool)

        dt = pd.to_numeric(
            plot_df["delta_clear_end_sec_trainee_minus_expert"],
            errors="coerce",
        ).to_numpy(dtype=float)

        dt_valid = np.where(ex_cleared & tr_cleared, dt, np.nan)
        dt_plot = np.nan_to_num(dt_valid, nan=0.0)

        bar_colors = []
        for v in dt_valid:
            if not np.isfinite(v):
                bar_colors.append("lightgray")
            elif v < 0:
                bar_colors.append("tab:green")
            elif v > 0:
                bar_colors.append("tab:red")
            else:
                bar_colors.append("lightgray")

        fig, (ax0, ax1) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(12.0, 6.8),
            constrained_layout=True,
        )

        ax0.bar(idx - width / 2.0, ex_cleared.astype(float), width, label="Expert", color="tab:blue")
        ax0.bar(idx + width / 2.0, tr_cleared.astype(float), width, label="Trainee", color="tab:orange")
        ax0.set_ylim(0.0, 1.15)
        ax0.set_ylabel("Cleared (1=yes, 0=no)")
        ax0.set_title("Threat clearance status by in-room track")
        ax0.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax0.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, framealpha=0.9)

        ax1.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
        ax1.bar(idx, dt_plot, width=0.6, color=bar_colors, alpha=0.85)
        ax1.set_ylabel("Δ time (T−E) seconds")
        ax1.set_title("Clearance completion time difference (only where both cleared)")
        ax1.grid(True, axis="y", linestyle="--", alpha=0.35)

        for i in range(len(inroom_ids)):
            if ex_cleared[i] and not tr_cleared[i]:
                ax1.text(idx[i], 0.02, "ONLY EXPERT", ha="center", va="bottom", fontsize=8, rotation=90)
            elif tr_cleared[i] and not ex_cleared[i]:
                ax1.text(idx[i], 0.02, "ONLY TRAINEE", ha="center", va="bottom", fontsize=8, rotation=90)
            elif (not ex_cleared[i]) and (not tr_cleared[i]):
                ax1.text(idx[i], 0.02, "NONE", ha="center", va="bottom", fontsize=8, rotation=90)
            else:
                if np.isfinite(dt_valid[i]):
                    ax1.text(
                        idx[i],
                        dt_plot[i],
                        f"{dt_valid[i]:+.2f}",
                        ha="center",
                        va="bottom" if dt_plot[i] >= 0 else "top",
                        fontsize=8,
                    )

        ax1.set_xticks(idx)
        ax1.set_xticklabels([str(e) for e in inroom_ids])
        ax1.set_xlabel("InRoom ID")
        ax0.set_xticks(idx)
        ax0.set_xticklabels([str(e) for e in inroom_ids])

        ymax = max(1.0, float(np.nanmax(np.abs(dt_valid))) if np.any(np.isfinite(dt_valid)) else 1.0)
        ax1.set_ylim(-ymax * 1.25, ymax * 1.25)

        fig.subplots_adjust(right=0.78)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)