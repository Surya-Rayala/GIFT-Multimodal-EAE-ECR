import glob
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metric import AbstractMetric


class CapturePodTime_Metric(AbstractMetric):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.metricName = "POD_CAPTURE_TIME"
        self.time_limits: List[float] = config.get("pod_time_limits", [1.0, 3.0, 1.5, 2.0])
        self._scores_by_soldier: Optional[Dict[int, float]] = None

    def _exp_penalty(self, overrun: float, limit: float) -> float:
        if limit <= 0:
            return 0.0
        if overrun >= limit:
            return 0.0
        return float(np.exp(-(overrun) / (limit - overrun)))

    def _score_single_pod(self, capture_time: Optional[float], limit: float) -> float:
        if capture_time is None:
            return 0.0

        capture_time = float(capture_time)
        limit = float(limit)

        if capture_time <= limit:
            return 1.0

        return self._exp_penalty(capture_time - limit, limit)

    def process(self, ctx):
        pod_capture = getattr(ctx, "pod_capture", {}) or {}
        if not pod_capture:
            self._scores_by_soldier = {}
            return

        max_idx = max(pod_capture.keys()) if pod_capture else -1
        limits = list(self.time_limits) if self.time_limits else []
        if max_idx >= 0:
            if not limits:
                limits = [0.0] * (max_idx + 1)
            elif len(limits) <= max_idx:
                limits.extend([limits[-1]] * (max_idx + 1 - len(limits)))

        per_person: Dict[int, List[float]] = {}
        for pod_idx, info in pod_capture.items():
            soldier_id = info.get("assigned_id")
            if soldier_id is None:
                continue

            capture_time = info.get("capture_time_sec")
            if capture_time is not None:
                try:
                    capture_time = float(capture_time)
                except Exception:
                    capture_time = None

            limit = float(limits[pod_idx]) if pod_idx < len(limits) else float(limits[-1])
            score = self._score_single_pod(capture_time, limit)
            per_person.setdefault(int(soldier_id), []).append(score)

        self._scores_by_soldier = {
            sid: round(float(np.mean(scores)), 2)
            for sid, scores in per_person.items()
        }

    def getFinalScore(self) -> float:
        if not self._scores_by_soldier:
            return 0.0
        overall = float(np.mean(list(self._scores_by_soldier.values())))
        return round(overall, 2)

    @staticmethod
    def expertCompare(session_folder: str, expert_folder: str, map_image=None, config=None):
        def _pick_latest(folder: str, pattern: str) -> Optional[str]:
            matches = glob.glob(os.path.join(folder, pattern))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        def _load_pod_cache(folder: str) -> pd.DataFrame:
            path = _pick_latest(folder, "*_PodCache.txt")
            if path is None:
                raise FileNotFoundError(f"No PodCache found in {folder}")

            df = pd.read_csv(path)
            if df is None or df.empty:
                return pd.DataFrame(columns=["pod_idx", "assigned_id", "capture_time_sec", "capture_frame"])

            cols = {c.lower(): c for c in df.columns}
            pod_col = cols.get("pod_idx") or cols.get("pod") or cols.get("podindex")
            aid_col = cols.get("assigned_id") or cols.get("assigned") or cols.get("track_id")
            time_col = cols.get("capture_time_sec") or cols.get("capture_time") or cols.get("time")
            frame_col = cols.get("capture_frame") or cols.get("frame")

            if pod_col is None or aid_col is None or time_col is None:
                raise ValueError(f"Unexpected PodCache format: {path}")

            out = pd.DataFrame({
                "pod_idx": df[pod_col],
                "assigned_id": df[aid_col],
                "capture_time_sec": df[time_col],
                "capture_frame": df[frame_col] if frame_col is not None else np.nan,
            })

            out["pod_idx"] = pd.to_numeric(out["pod_idx"], errors="coerce").astype("Int64")
            out["assigned_id"] = pd.to_numeric(out["assigned_id"], errors="coerce").astype("Int64")
            out["capture_time_sec"] = pd.to_numeric(out["capture_time_sec"], errors="coerce")
            out["capture_frame"] = pd.to_numeric(out["capture_frame"], errors="coerce")

            out = out.dropna(subset=["pod_idx"]).copy()
            out["pod_idx"] = out["pod_idx"].astype(int)
            return out

        def _load_tracker_roles(folder: str) -> Dict[int, str]:
            path = _pick_latest(folder, "*_TrackerOutput.json")
            if path is None:
                return {}

            try:
                import json
                with open(path, "r") as f:
                    data = json.load(f)
            except Exception:
                return {}

            role_by_id: Dict[int, str] = {}
            if not isinstance(data, list):
                return role_by_id

            for frame_entry in data:
                for obj in frame_entry.get("objects", []) or []:
                    tid = obj.get("id")
                    role = obj.get("identity_role")
                    if tid is None or role is None:
                        continue
                    try:
                        role_by_id[int(tid)] = str(role)
                    except Exception:
                        continue

            return role_by_id

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
            return out

        def _entry_map_from_position_cache(folder: str, inroom_id_start: int) -> Dict[int, int]:
            pos = _load_position_cache(folder)
            if pos.empty:
                return {}

            role_by_id = _load_tracker_roles(folder)

            def _is_entry_track(track_id: int) -> bool:
                role = role_by_id.get(track_id)
                if role == "entry":
                    return True
                if role == "inroom":
                    return False
                return int(track_id) < int(inroom_id_start)

            pos = pos[pos["id"].map(_is_entry_track)].copy()
            if pos.empty:
                return {}

            starts = pos.groupby("id")["frame"].min().sort_values()
            return {int(tid): int(i + 1) for i, tid in enumerate(starts.index.tolist())}

        def _exp_penalty(overrun: float, limit: float) -> float:
            if limit <= 0:
                return 0.0
            if overrun >= limit:
                return 0.0
            return float(np.exp(-(overrun) / (limit - overrun)))

        def _score_single_pod(capture_time: Optional[float], limit: float) -> float:
            if capture_time is None or pd.isna(capture_time):
                return 0.0
            capture_time = float(capture_time)
            limit = float(limit)
            if capture_time <= limit:
                return 1.0
            return _exp_penalty(capture_time - limit, limit)

        def _ensure_limits(max_pod_idx: int) -> List[float]:
            limits = None
            if isinstance(config, dict):
                limits = config.get("pod_time_limits")
            if not limits:
                limits = [1.0, 3.0, 1.5, 2.0]
            limits = list(limits)

            if max_pod_idx < 0:
                return limits
            if not limits:
                return [0.0] * (max_pod_idx + 1)
            if len(limits) <= max_pod_idx:
                limits.extend([limits[-1]] * (max_pod_idx + 1 - len(limits)))
            return limits

        def _inroom_id_start_from_config(cfg) -> int:
            if not isinstance(cfg, dict):
                return 99
            try:
                return int(cfg.get("inroom_id_start", 99))
            except Exception:
                return 99

        os.makedirs(session_folder, exist_ok=True)
        img_path = os.path.join(session_folder, "POD_CAPTURE_TIME_Comparison.jpg")
        txt_path = os.path.join(session_folder, "POD_CAPTURE_TIME_Comparison.txt")

        try:
            df_expert = _load_pod_cache(expert_folder)
            df_trainee = _load_pod_cache(session_folder)
            inroom_id_start = _inroom_id_start_from_config(config)
            expert_entry_map = _entry_map_from_position_cache(expert_folder, inroom_id_start)
            trainee_entry_map = _entry_map_from_position_cache(session_folder, inroom_id_start)
        except Exception:
            err_text = "There was an error while processing this comparison. Missing or invalid PodCache."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "POD_CAPTURE_TIME",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        if df_expert.empty and df_trainee.empty:
            err_text = "There was an error while processing this comparison. No POD entries found."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "POD_CAPTURE_TIME",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        pod_ids = sorted(set(df_expert["pod_idx"].tolist()) | set(df_trainee["pod_idx"].tolist()))
        max_pod_idx = max(pod_ids) if pod_ids else -1
        limits = _ensure_limits(max_pod_idx)

        ex_by_pod = df_expert.set_index("pod_idx") if not df_expert.empty else pd.DataFrame().set_index(pd.Index([]))
        tr_by_pod = df_trainee.set_index("pod_idx") if not df_trainee.empty else pd.DataFrame().set_index(pd.Index([]))

        rows: List[Dict[str, Any]] = []

        for pod_idx in pod_ids:
            limit = float(limits[pod_idx]) if pod_idx < len(limits) else float(limits[-1])

            ex_assigned = None
            ex_time = None
            ex_frame = None
            if pod_idx in ex_by_pod.index:
                ex_row = ex_by_pod.loc[pod_idx]
                if isinstance(ex_row, pd.DataFrame):
                    ex_row = ex_row.iloc[0]
                ex_assigned = ex_row.get("assigned_id")
                ex_time = ex_row.get("capture_time_sec")
                ex_frame = ex_row.get("capture_frame")

            tr_assigned = None
            tr_time = None
            tr_frame = None
            if pod_idx in tr_by_pod.index:
                tr_row = tr_by_pod.loc[pod_idx]
                if isinstance(tr_row, pd.DataFrame):
                    tr_row = tr_row.iloc[0]
                tr_assigned = tr_row.get("assigned_id")
                tr_time = tr_row.get("capture_time_sec")
                tr_frame = tr_row.get("capture_frame")

            ex_score = _score_single_pod(ex_time, limit)
            tr_score = _score_single_pod(tr_time, limit)

            dt = None
            if ex_time is not None and tr_time is not None and not pd.isna(ex_time) and not pd.isna(tr_time):
                dt = float(tr_time) - float(ex_time)

            if dt is None:
                trend = "N/A"
            elif dt < 0:
                trend = "FASTER"
            elif dt > 0:
                trend = "SLOWER"
            else:
                trend = "MATCH"

            rows.append({
                "pod_idx": int(pod_idx),
                "time_limit_sec": round(limit, 3),
                "expert_entry_number": "" if ex_assigned is None or pd.isna(ex_assigned) else expert_entry_map.get(int(ex_assigned), ""),
                "expert_id": "" if ex_assigned is None or pd.isna(ex_assigned) else int(ex_assigned),
                "expert_capture_time_sec": "" if ex_time is None or pd.isna(ex_time) else round(float(ex_time), 3),
                "expert_capture_frame": "" if ex_frame is None or pd.isna(ex_frame) else int(ex_frame),
                "expert_pod_score": round(float(ex_score), 3),
                "trainee_entry_number": "" if tr_assigned is None or pd.isna(tr_assigned) else trainee_entry_map.get(int(tr_assigned), ""),
                "trainee_id": "" if tr_assigned is None or pd.isna(tr_assigned) else int(tr_assigned),
                "trainee_capture_time_sec": "" if tr_time is None or pd.isna(tr_time) else round(float(tr_time), 3),
                "trainee_capture_frame": "" if tr_frame is None or pd.isna(tr_frame) else int(tr_frame),
                "trainee_pod_score": round(float(tr_score), 3),
                "delta_time_sec_trainee_minus_expert": "" if dt is None else round(float(dt), 3),
                "trainee_vs_expert_time": trend,
                "delta_score_trainee_minus_expert": round(float(tr_score - ex_score), 3),
            })

        pods_df = pd.DataFrame(rows)

        try:
            CapturePodTime_Metric.__generateExpertCompareGraphic(
                output_path=img_path,
                pods_df=pods_df,
            )
        except Exception:
            pass

        dt_series = pd.to_numeric(pods_df.get("delta_time_sec_trainee_minus_expert"), errors="coerce").dropna()
        avg_dt = float(dt_series.mean()) if len(dt_series) > 0 else None
        n_dt = int(len(dt_series))

        ds_series = pd.to_numeric(pods_df.get("delta_score_trainee_minus_expert"), errors="coerce").dropna()
        avg_ds = float(ds_series.mean()) if len(ds_series) > 0 else None

        ex_times = pd.to_numeric(pods_df.get("expert_capture_time_sec"), errors="coerce")
        tr_times = pd.to_numeric(pods_df.get("trainee_capture_time_sec"), errors="coerce")

        ex_has = ex_times.notna()
        tr_has = tr_times.notna()

        total_pods = int(len(pod_ids))
        ex_time_count = int(ex_has.sum())
        tr_time_count = int(tr_has.sum())

        missing_expert = int((~ex_has & tr_has).sum())
        missing_trainee = int((ex_has & ~tr_has).sum())
        missing_both = int((~ex_has & ~tr_has).sum())
        excluded = missing_expert + missing_trainee + missing_both

        def _excluded_breakdown() -> str:
            parts = []
            if missing_expert:
                parts.append(f"{missing_expert} missing expert")
            if missing_trainee:
                parts.append(f"{missing_trainee} missing trainee")
            if missing_both:
                parts.append(f"{missing_both} missing both")
            return "; ".join(parts) if parts else "0"

        def _excluded_pods_str() -> str:
            excluded_mask = ~(ex_has & tr_has)
            if not excluded_mask.any():
                return ""
            pod_list = pods_df.loc[excluded_mask, "pod_idx"].tolist()
            return ", ".join(f"P{int(p)}" for p in pod_list if p is not None and not pd.isna(p))

        def _exclusion_clause() -> str:
            if excluded <= 0:
                return ""
            pods = _excluded_pods_str()
            pods_part = f" ({pods})" if pods else ""
            return f"; excluded {excluded} POD(s){pods_part} due to missing capture time(s) ({_excluded_breakdown()})"

        if avg_dt is None:
            time_part = (
                "Average capture-time gap: N/A. "
                "This metric only averages time differences for PODs where both expert and trainee have a capture time. "
                f"Here, expert has times for {ex_time_count}/{total_pods} PODs and trainee has times for {tr_time_count}/{total_pods} PODs, "
                "so there are no PODs with times on both sides to compare."
            )
        elif avg_dt < 0:
            time_part = (
                f"Average capture-time gap (T − E): {avg_dt:.2f}s (trainee faster). "
                f"Computed over {n_dt}/{total_pods} PODs where both expert and trainee recorded a capture time"
                f"{_exclusion_clause()}."
            )
        elif avg_dt > 0:
            time_part = (
                f"Average capture-time gap (T − E): {avg_dt:.2f}s (trainee slower). "
                f"Computed over {n_dt}/{total_pods} PODs where both expert and trainee recorded a capture time"
                f"{_exclusion_clause()}."
            )
        else:
            time_part = (
                "Average capture-time gap (T − E): 0.00s (match). "
                f"Computed over {n_dt}/{total_pods} PODs where both expert and trainee recorded a capture time"
                f"{_exclusion_clause()}."
            )

        if avg_ds is None:
            score_part = "I couldn't compute an average score difference."
        elif avg_ds > 0:
            score_part = f"On score, the trainee came in about {abs(avg_ds):.3f} higher than the expert on average."
        elif avg_ds < 0:
            score_part = f"On score, the trainee came in about {abs(avg_ds):.3f} lower than the expert on average."
        else:
            score_part = "On score, the trainee matched the expert on average."

        thresholds_part = (
            "Time limits used (sec): " +
            ", ".join(f"P{int(pid)}={float(limits[pid]):.2f}s" for pid in pod_ids)
            if pod_ids else
            "Time limits used (sec): N/A"
        )

        lines = [
            "POD, Expert Entrant#, Expert ID, Trainee Entrant#, Trainee ID, Time Δ (T−E), Score Δ (T−E), Performance"
        ]

        for row in rows:
            pod_idx = row.get("pod_idx")
            e_ent = row.get("expert_entry_number")
            e_id = row.get("expert_id")
            t_ent = row.get("trainee_entry_number")
            t_id = row.get("trainee_id")
            dt = row.get("delta_time_sec_trainee_minus_expert")
            ds = row.get("delta_score_trainee_minus_expert")

            dt_str = "N/A" if dt in (None, "") else f"{float(dt):+.2f}s"
            ds_str = "N/A" if ds in (None, "") else f"{float(ds):+.3f}"

            if ds in (None, ""):
                perf = "N/A"
            else:
                ds_float = float(ds)
                if ds_float > 0:
                    perf = "BETTER"
                elif ds_float < 0:
                    perf = "WORSE"
                else:
                    perf = "SIMILAR"

            lines.append(
                f"P{int(pod_idx) if pod_idx is not None else 'N/A'}, "
                f"{e_ent if e_ent != '' else 'N/A'}, {e_id if e_id != '' else 'N/A'}, "
                f"{t_ent if t_ent != '' else 'N/A'}, {t_id if t_id != '' else 'N/A'}, "
                f"{dt_str}, {ds_str}, {perf}"
            )

        details_csv = "\n".join(lines)
        text = time_part + " " + score_part + "\n" + thresholds_part + "\n" + details_csv

        def _broken_table(headers: List[str], data_rows: List[List[str]]) -> str:
            if not data_rows:
                return "(no rows)"

            widths = [max(len(h), max(len(r[i]) for r in data_rows)) for i, h in enumerate(headers)]
            sep = " | "

            def _fmt(values: List[str]) -> str:
                return sep.join(values[i].ljust(widths[i]) for i in range(len(headers)))

            out = [_fmt(headers), sep.join("-" * w for w in widths)]
            out.extend(_fmt(r) for r in data_rows)
            return "\n".join(out)

        pretty_headers = [h.strip() for h in lines[0].split(",")] if lines else []
        pretty_rows: List[List[str]] = []
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(",")]
            while len(parts) < len(pretty_headers):
                parts.append("N/A")
            pretty_rows.append(parts[: len(pretty_headers)])

        details_pretty = _broken_table(pretty_headers, pretty_rows) if pretty_headers else "(no rows)"
        saved_text = time_part + " " + score_part + "\n" + thresholds_part + "\n\n" + details_pretty + "\n"

        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(saved_text)
        except Exception:
            pass

        return {
            "Name": "POD_CAPTURE_TIME",
            "Type": "Single",
            "ImgLocation": img_path,
            "TxtLocation": txt_path,
            "Text": text,
        }

    @staticmethod
    def __generateExpertCompareGraphic(output_path: str, pods_df: pd.DataFrame) -> None:
        if pods_df is None or pods_df.empty:
            return

        df = pods_df.copy()
        df["pod_idx"] = pd.to_numeric(df["pod_idx"], errors="coerce")
        df = df.dropna(subset=["pod_idx"]).copy()
        if df.empty:
            return

        x = df["pod_idx"].astype(int).tolist()

        def _to_float_array(col: str) -> np.ndarray:
            return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        ex_time = _to_float_array("expert_capture_time_sec")
        tr_time = _to_float_array("trainee_capture_time_sec")
        limits = _to_float_array("time_limit_sec")
        limits2 = limits * 2.0

        ex_plot = np.nan_to_num(ex_time, nan=0.0)
        tr_plot = np.nan_to_num(tr_time, nan=0.0)

        width = 0.35
        idx = np.arange(len(x), dtype=float)

        fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)

        ax.bar(idx - width / 2.0, ex_plot, width, label="Expert")
        ax.bar(idx + width / 2.0, tr_plot, width, label="Trainee")

        ax.scatter(idx, limits, marker="_", s=800, linewidths=3, label="Time limit")

        valid_limits2 = ~np.isnan(limits2)
        if np.any(valid_limits2):
            ax.scatter(
                idx[valid_limits2],
                limits2[valid_limits2],
                marker="_",
                s=600,
                linewidths=2,
                label="2× time limit (score = 0 from here)",
            )

        ax.set_xticks(idx)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_xlabel("POD index")
        ax.set_ylabel("Capture time (sec)")
        ax.set_title("POD capture time: Expert vs Trainee (0 = not captured)")
        ax.legend(loc="upper right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        ymax = max(
            float(np.nanmax(ex_plot)) if len(ex_plot) else 0.0,
            float(np.nanmax(tr_plot)) if len(tr_plot) else 0.0,
            float(np.nanmax(limits)) if len(limits) and np.any(~np.isnan(limits)) else 0.0,
            float(np.nanmax(limits2)) if len(limits2) and np.any(~np.isnan(limits2)) else 0.0,
            1.0,
        )
        ax.set_ylim(0.0, ymax * 1.15)

        plt.savefig(output_path, dpi=150)
        plt.close(fig)