import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import AbstractMetric


def _gaze_triangle(origin, direction, half_angle_deg, length=10000.0):
    d = np.asarray(direction, dtype=np.float32)
    n = np.linalg.norm(d)
    if n == 0.0:
        return np.zeros((3, 2), dtype=np.float32)

    d /= n
    ang = np.deg2rad(float(half_angle_deg))
    cos_a, sin_a = float(np.cos(ang)), float(np.sin(ang))

    rot_left = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    rot_right = np.array([[cos_a, sin_a], [-sin_a, cos_a]], dtype=np.float32)

    o = np.asarray(origin, dtype=np.float32)
    left_vec = rot_left @ d * length
    right_vec = rot_right @ d * length
    return np.stack([o, o + left_vec, o + right_vec], axis=0)


def _triangle_box_intersect(triangle, box):
    tri = np.asarray(triangle, dtype=np.float32).reshape(-1, 1, 2)
    x1, y1, x2, y2 = map(float, box)
    rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).reshape(-1, 1, 2)
    inter_area, _ = cv2.intersectConvexConvex(tri, rect)
    return inter_area > 0.0


class ThreatCoverage_Metric(AbstractMetric):
    """Fraction of counted frames where at least one active in-room threat is covered by gaze."""

    def __init__(self, config):
        super().__init__(config)
        self.metricName = "THREAT_COVERAGE"
        self.visual_angle_deg = float(config.get("visual_angle_degrees", 20.0))
        self._final_score = 0.0

    def process(self, ctx):
        inroom_ids = set(getattr(ctx, "inroom_ids", []) or [])
        if not inroom_ids:
            self._final_score = 1.0
            return

        all_frame_count = len(ctx.all_frames)
        all_ids = set(ctx.tracks_by_id.keys())
        entry_ids = sorted(tid for tid in all_ids if tid not in inroom_ids)

        present_frames = 0
        covered_frames = 0
        half_angle = self.visual_angle_deg / 2.0

        threat_clearance = getattr(ctx, "threat_clearance", {}) or {}

        for frame_idx in range(1, all_frame_count + 1):
            any_entry_present = any((frame_idx, tid) in ctx.bbox_details for tid in entry_ids)
            if not any_entry_present:
                continue

            active_inroom_ids = []
            for inroom_id in inroom_ids:
                if (frame_idx, inroom_id) not in ctx.bbox_details:
                    continue

                clearance_tuple = threat_clearance.get(inroom_id)
                cleared = False
                if clearance_tuple is not None:
                    _, end_frame, _ = clearance_tuple
                    if end_frame is not None and frame_idx >= end_frame:
                        cleared = True

                if not cleared:
                    active_inroom_ids.append(inroom_id)

            if not active_inroom_ids:
                continue

            present_frames += 1
            looked_at_any = False

            for entry_id in entry_ids:
                gaze = ctx.gaze_info.get((frame_idx, entry_id))
                if gaze is None:
                    continue

                ox, oy, dx, dy = gaze
                direction = np.array([dx, dy], dtype=np.float32)
                if float(np.linalg.norm(direction)) == 0.0:
                    looked_at_any = True
                    break

                tri = _gaze_triangle(np.array([ox, oy], dtype=np.float32), direction, half_angle)

                for inroom_id in active_inroom_ids:
                    bbox = ctx.bbox_details[(frame_idx, inroom_id)]
                    if _triangle_box_intersect(tri, bbox):
                        looked_at_any = True
                        break

                if looked_at_any:
                    break

            if looked_at_any:
                covered_frames += 1

        self._final_score = covered_frames / present_frames if present_frames > 0 else 1.0

    def getFinalScore(self) -> float:
        return round(self._final_score, 2)

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

        def _load_tracker_output(folder: str) -> List[Dict]:
            path = _pick_latest(folder, "*_TrackerOutput.json")
            if path is None:
                raise FileNotFoundError(f"No TrackerOutput found in {folder}")
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Unexpected TrackerOutput format: {path}")
            return data

        def _load_gaze_cache(folder: str) -> Dict[Tuple[int, int], Tuple[float, float, float, float]]:
            path = _pick_latest(folder, "*_GazeCache.txt")
            if path is None:
                return {}

            df = pd.read_csv(path)
            if df is None or df.empty:
                return {}

            cols = {c.lower(): c for c in df.columns}
            f_col = cols.get("frame")
            id_col = cols.get("id")
            ox_col = cols.get("ox")
            oy_col = cols.get("oy")
            dx_col = cols.get("dx")
            dy_col = cols.get("dy")
            if None in (f_col, id_col, ox_col, oy_col, dx_col, dy_col):
                return {}

            out: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
            for _, r in df.iterrows():
                try:
                    fr = int(r[f_col])
                    tid = int(r[id_col])
                    out[(fr, tid)] = (
                        float(r[ox_col]),
                        float(r[oy_col]),
                        float(r[dx_col]),
                        float(r[dy_col]),
                    )
                except Exception:
                    continue
            return out

        def _load_clearance_cache(folder: str) -> Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]]:
            path = _pick_latest(folder, "*_ThreatClearanceCache.txt")
            if path is None:
                return {}

            df = pd.read_csv(path)
            if df is None or df.empty:
                return {}

            cols = {c.lower(): c for c in df.columns}
            id_col = cols.get("inroom_id") or cols.get("enemy_id") or cols.get("id")
            s_col = cols.get("immediate_frame") or cols.get("start_frame") or cols.get("start")
            e_col = cols.get("contact_end_frame") or cols.get("end_frame") or cols.get("end")
            f_col = cols.get("clearing_friend") or cols.get("friend") or cols.get("clearing")

            if id_col is None:
                return {}

            out: Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]] = {}

            def _to_int(x) -> Optional[int]:
                try:
                    if pd.isna(x):
                        return None
                    xi = int(x)
                    return None if xi < 0 else xi
                except Exception:
                    return None

            for _, r in df.iterrows():
                try:
                    inroom_id = int(r[id_col])
                except Exception:
                    continue

                start = _to_int(r[s_col]) if s_col is not None else None
                end = _to_int(r[e_col]) if e_col is not None else None
                fid = _to_int(r[f_col]) if f_col is not None else None
                out[inroom_id] = (start, end, fid)

            return out

        def _frame_rate_from_config(cfg: Optional[dict]) -> float:
            if cfg and isinstance(cfg, dict):
                try:
                    return float(cfg.get("frame_rate", 30.0))
                except Exception:
                    return 30.0
            return 30.0

        def _visual_angle_from_config(cfg: Optional[dict]) -> float:
            if cfg and isinstance(cfg, dict):
                try:
                    return float(cfg.get("visual_angle_degrees", 20.0))
                except Exception:
                    return 20.0
            return 20.0

        def _inroom_id_start_from_config(cfg: Optional[dict]) -> int:
            if cfg and isinstance(cfg, dict):
                try:
                    return int(cfg.get("inroom_id_start", 99))
                except Exception:
                    return 99
            return 99

        def _extract_frame_maps(tracker_output: List[Dict]) -> Tuple[
            Dict[Tuple[int, int], Tuple[float, float, float, float]],
            Dict[int, set],
            Dict[int, set],
            Dict[int, set],
        ]:
            bbox_map: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
            ids_per_frame: Dict[int, set] = {}
            entry_ids_per_frame: Dict[int, set] = {}
            inroom_ids_per_frame: Dict[int, set] = {}

            for entry in tracker_output:
                fr = int(entry.get("frame", 0))
                objs = entry.get("objects", []) or []
                ids_per_frame.setdefault(fr, set())
                entry_ids_per_frame.setdefault(fr, set())
                inroom_ids_per_frame.setdefault(fr, set())

                for obj in objs:
                    tid = obj.get("id")
                    bb = obj.get("bbox")
                    if tid is None or bb is None:
                        continue

                    try:
                        tid = int(tid)
                        x1, y1, x2, y2 = bb
                    except Exception:
                        continue

                    bbox_map[(fr, tid)] = (float(x1), float(y1), float(x2), float(y2))
                    ids_per_frame[fr].add(tid)

                    role = obj.get("identity_role")
                    is_entry = bool(obj.get("is_entry", False))
                    is_inroom = bool(obj.get("is_inroom", False))
                    birth_location = obj.get("birth_location")

                    if role == "entry" or is_entry or birth_location == "entry":
                        entry_ids_per_frame[fr].add(tid)
                    if role == "inroom" or is_inroom or birth_location == "inroom":
                        inroom_ids_per_frame[fr].add(tid)

            return bbox_map, ids_per_frame, entry_ids_per_frame, inroom_ids_per_frame

        def _infer_semantic_ids(
            tracker_output: List[Dict],
            clearance_map: Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]],
            inroom_id_start: int,
        ) -> Tuple[List[int], List[int]]:
            all_ids = set()
            entry_ids = set()
            inroom_ids = set()

            for frame_entry in tracker_output:
                for obj in frame_entry.get("objects", []) or []:
                    tid = obj.get("id")
                    if tid is None:
                        continue
                    try:
                        tid = int(tid)
                    except Exception:
                        continue

                    all_ids.add(tid)
                    role = obj.get("identity_role")
                    is_entry = bool(obj.get("is_entry", False))
                    is_inroom = bool(obj.get("is_inroom", False))
                    birth_location = obj.get("birth_location")

                    if role == "entry" or is_entry or birth_location == "entry":
                        entry_ids.add(tid)
                    if role == "inroom" or is_inroom or birth_location == "inroom":
                        inroom_ids.add(tid)

            if not inroom_ids and clearance_map:
                inroom_ids = set(int(k) for k in clearance_map.keys())

            if not inroom_ids:
                inroom_ids = {tid for tid in all_ids if tid >= inroom_id_start}

            if not entry_ids:
                entry_ids = {tid for tid in all_ids if tid not in inroom_ids}

            return sorted(entry_ids), sorted(inroom_ids)

        def _first_entry_frame(
            tracker_output: List[Dict],
            entry_ids: List[int],
        ) -> Optional[int]:
            entry_id_set = set(entry_ids)
            min_fr = None

            for frame_entry in tracker_output:
                fr = int(frame_entry.get("frame", 0))
                for obj in frame_entry.get("objects", []) or []:
                    tid = obj.get("id")
                    if tid is None:
                        continue
                    try:
                        tid = int(tid)
                    except Exception:
                        continue

                    if tid in entry_id_set:
                        if min_fr is None or fr < min_fr:
                            min_fr = fr
            return min_fr

        def _evaluate_folder(
            folder: str,
            *,
            fps: float,
            visual_angle_deg: float,
            inroom_id_start: int,
        ) -> Dict[str, object]:
            tracker_output = _load_tracker_output(folder)
            gaze_info = _load_gaze_cache(folder)
            clearance = _load_clearance_cache(folder)

            bbox_map, ids_per_frame, _, _ = _extract_frame_maps(tracker_output)
            entry_ids, inroom_ids = _infer_semantic_ids(tracker_output, clearance, inroom_id_start)

            entry_id_set = set(entry_ids)
            inroom_id_set = set(inroom_ids)

            first_entry = _first_entry_frame(tracker_output, entry_ids)
            if first_entry is None:
                first_entry = 1

            max_frame = max(ids_per_frame.keys()) if ids_per_frame else 0
            half_angle = visual_angle_deg / 2.0

            per_inroom_present: Dict[int, int] = {tid: 0 for tid in inroom_ids}
            per_inroom_covered: Dict[int, int] = {tid: 0 for tid in inroom_ids}

            present_frames = 0
            covered_frames = 0
            curve_t: List[float] = []
            curve_cov: List[float] = []

            def _inroom_uncleared(inroom_id: int, frame_idx: int) -> bool:
                tup = clearance.get(inroom_id)
                if tup is None:
                    return True
                _, end, _ = tup
                if end is None:
                    return True
                return frame_idx < int(end)

            for frame_idx in range(int(first_entry), int(max_frame) + 1):
                active_entry_ids = [tid for tid in entry_id_set if (frame_idx, tid) in bbox_map]
                if not active_entry_ids:
                    continue

                active_inroom_ids = [
                    tid
                    for tid in inroom_id_set
                    if (frame_idx, tid) in bbox_map and _inroom_uncleared(tid, frame_idx)
                ]
                if not active_inroom_ids:
                    continue

                present_frames += 1

                for inroom_id in active_inroom_ids:
                    per_inroom_present[inroom_id] += 1

                looked_any = False
                looked_by_inroom: Dict[int, bool] = {tid: False for tid in active_inroom_ids}

                for entry_id in active_entry_ids:
                    g = gaze_info.get((frame_idx, entry_id))
                    if g is None:
                        continue

                    ox, oy, dx, dy = g
                    direction = np.array([dx, dy], dtype=np.float32)

                    if float(np.linalg.norm(direction)) == 0.0:
                        looked_any = True
                        for inroom_id in active_inroom_ids:
                            looked_by_inroom[inroom_id] = True
                        break

                    tri = _gaze_triangle(np.array([ox, oy], dtype=np.float32), direction, half_angle)

                    for inroom_id in active_inroom_ids:
                        if looked_by_inroom[inroom_id]:
                            continue
                        bbox = bbox_map[(frame_idx, inroom_id)]
                        if _triangle_box_intersect(tri, bbox):
                            looked_by_inroom[inroom_id] = True
                            looked_any = True

                    if looked_any and all(looked_by_inroom.values()):
                        break

                if looked_any:
                    covered_frames += 1

                for inroom_id, ok in looked_by_inroom.items():
                    if ok:
                        per_inroom_covered[inroom_id] += 1

                curve_t.append((frame_idx - first_entry) / fps)
                curve_cov.append(float(covered_frames / max(1, present_frames)))

            final_cov = float(covered_frames / present_frames) if present_frames > 0 else 0.0

            return {
                "first_entry_frame": int(first_entry),
                "fps": float(fps),
                "visual_angle_deg": float(visual_angle_deg),
                "entry_ids": sorted(entry_ids),
                "inroom_ids": sorted(inroom_ids),
                "final_coverage": float(final_cov),
                "present_frames": int(present_frames),
                "covered_frames": int(covered_frames),
                "per_inroom_present": per_inroom_present,
                "per_inroom_covered": per_inroom_covered,
                "clearance": clearance,
                "curve_t": curve_t,
                "curve_cov": curve_cov,
            }

        def _generate_plot(
            *,
            out_path: str,
            expert_res: Dict[str, object],
            trainee_res: Dict[str, object],
            per_inroom_df: pd.DataFrame,
        ) -> None:
            ex_t = expert_res.get("curve_t", [])
            ex_c = expert_res.get("curve_cov", [])
            tr_t = trainee_res.get("curve_t", [])
            tr_c = trainee_res.get("curve_cov", [])

            if (not ex_t and not tr_t) and (per_inroom_df is None or per_inroom_df.empty):
                return

            fig, (ax_ts, ax_bar) = plt.subplots(
                nrows=1,
                ncols=2,
                figsize=(14.5, 5.6),
                constrained_layout=True,
            )

            if ex_t:
                ax_ts.plot(ex_t, ex_c, label="Expert", color="tab:blue", linewidth=2.5)
            if tr_t:
                ax_ts.plot(tr_t, tr_c, label="Trainee", color="tab:orange", linewidth=2.5)

            ex_clear = expert_res.get("clearance") or {}
            tr_clear = trainee_res.get("clearance") or {}
            ex_first = float(expert_res.get("first_entry_frame", 1))
            tr_first = float(trainee_res.get("first_entry_frame", 1))
            ex_fps = float(expert_res.get("fps", 30.0))
            tr_fps = float(trainee_res.get("fps", 30.0))

            def _mark_clear_times(clearance_map, first_frame, fps_local, color):
                if not clearance_map or fps_local <= 0:
                    return
                for inroom_id, tup in clearance_map.items():
                    try:
                        inroom_id_i = int(inroom_id)
                    except Exception:
                        continue
                    if tup is None or len(tup) < 2 or tup[1] is None:
                        continue
                    try:
                        t_sec = (float(int(tup[1])) - float(first_frame)) / float(fps_local)
                    except Exception:
                        continue
                    if not np.isfinite(t_sec):
                        continue

                    ax_ts.axvline(float(t_sec), linestyle=":", color=color, alpha=0.45, linewidth=1.2)
                    ax_ts.text(
                        float(t_sec),
                        1.045,
                        f"InRoom ID: {inroom_id_i}",
                        rotation=90,
                        color=color,
                        ha="center",
                        va="top",
                        fontsize=7,
                        alpha=0.9,
                        clip_on=True,
                    )

            _mark_clear_times(ex_clear, ex_first, ex_fps, "tab:blue")
            _mark_clear_times(tr_clear, tr_first, tr_fps, "tab:orange")

            ax_ts.set_xlabel("Seconds since first team entry")
            ax_ts.set_ylabel("Cumulative threat coverage")
            ax_ts.set_title("Threat coverage until threats are cleared")
            ax_ts.set_ylim(0.0, 1.05)
            ax_ts.grid(True, axis="y", linestyle="--", alpha=0.35)

            if per_inroom_df is None or per_inroom_df.empty:
                ax_bar.set_axis_off()
            else:
                dfp = per_inroom_df.copy()
                dfp["inroom_id"] = pd.to_numeric(dfp["inroom_id"], errors="coerce")
                dfp = dfp.dropna(subset=["inroom_id"]).sort_values("inroom_id")

                if dfp.empty:
                    ax_bar.set_axis_off()
                else:
                    inroom_ids = [int(x) for x in dfp["inroom_id"].tolist()]
                    idx = np.arange(len(inroom_ids), dtype=float)
                    width = 0.36

                    ex_fps = float(expert_res.get("fps", 30.0))
                    tr_fps = float(trainee_res.get("fps", 30.0))

                    ex_p = pd.to_numeric(dfp["expert_inroom_present_frames"], errors="coerce").fillna(0).to_numpy(dtype=float)
                    ex_cov = pd.to_numeric(dfp["expert_inroom_covered_frames"], errors="coerce").fillna(0).to_numpy(dtype=float)
                    tr_p = pd.to_numeric(dfp["trainee_inroom_present_frames"], errors="coerce").fillna(0).to_numpy(dtype=float)
                    tr_cov = pd.to_numeric(dfp["trainee_inroom_covered_frames"], errors="coerce").fillna(0).to_numpy(dtype=float)

                    ex_unseen_sec = np.maximum(0.0, (ex_p - ex_cov) / max(ex_fps, 1.0))
                    tr_unseen_sec = np.maximum(0.0, (tr_p - tr_cov) / max(tr_fps, 1.0))

                    ex_score = np.where(ex_p > 0, ex_cov / ex_p, 0.0)
                    tr_score = np.where(tr_p > 0, tr_cov / tr_p, 0.0)

                    ax_bar.bar(idx - width / 2.0, ex_unseen_sec, width, label="Expert", color="tab:blue", alpha=0.9)
                    ax_bar.bar(idx + width / 2.0, tr_unseen_sec, width, label="Trainee", color="tab:orange", alpha=0.9)

                    for i in range(len(inroom_ids)):
                        ax_bar.text(
                            idx[i] - width / 2.0,
                            ex_unseen_sec[i] + 0.02,
                            f"~{float(ex_score[i]):.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )
                        ax_bar.text(
                            idx[i] + width / 2.0,
                            tr_unseen_sec[i] + 0.02,
                            f"~{float(tr_score[i]):.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

                    ax_bar.set_xticks(idx)
                    ax_bar.set_xticklabels([str(e) for e in inroom_ids])
                    ax_bar.set_xlabel("InRoom ID")
                    ax_bar.set_ylabel("Unseen time (seconds)")
                    ax_bar.set_title("Per-inroom unseen time while uncleared (labels show ~coverage score)")
                    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.35)

            h1, l1 = ax_ts.get_legend_handles_labels()
            h2, l2 = ax_bar.get_legend_handles_labels()
            seen = set()
            handles = []
            labels = []
            for h, lab in list(zip(h1, l1)) + list(zip(h2, l2)):
                if not lab or lab in seen:
                    continue
                seen.add(lab)
                handles.append(h)
                labels.append(lab)

            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    framealpha=0.9,
                )
                fig.subplots_adjust(right=0.82)

            plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
            plt.close(fig)

        os.makedirs(session_folder, exist_ok=True)
        img_path = os.path.join(session_folder, "THREAT_COVERAGE_Comparison.jpg")
        txt_path = os.path.join(session_folder, "THREAT_COVERAGE_Comparison.txt")

        fps = _frame_rate_from_config(config)
        visual_angle_deg = _visual_angle_from_config(config)
        inroom_id_start = _inroom_id_start_from_config(config)

        try:
            expert_res = _evaluate_folder(
                expert_folder,
                fps=fps,
                visual_angle_deg=visual_angle_deg,
                inroom_id_start=inroom_id_start,
            )
            trainee_res = _evaluate_folder(
                session_folder,
                fps=fps,
                visual_angle_deg=visual_angle_deg,
                inroom_id_start=inroom_id_start,
            )
        except Exception:
            err_text = (
                "There was an error while processing this comparison. "
                "Missing TrackerOutput, GazeCache, or ThreatClearanceCache."
            )
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "THREAT_COVERAGE",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        ex_inroom = set((expert_res.get("per_inroom_present") or {}).keys())
        tr_inroom = set((trainee_res.get("per_inroom_present") or {}).keys())
        inroom_ids = sorted(list(ex_inroom | tr_inroom))

        rows: List[Dict[str, object]] = []
        for inroom_id in inroom_ids:
            ex_p = int((expert_res.get("per_inroom_present") or {}).get(inroom_id, 0))
            ex_c = int((expert_res.get("per_inroom_covered") or {}).get(inroom_id, 0))
            tr_p = int((trainee_res.get("per_inroom_present") or {}).get(inroom_id, 0))
            tr_c = int((trainee_res.get("per_inroom_covered") or {}).get(inroom_id, 0))
            rows.append(
                {
                    "inroom_id": int(inroom_id),
                    "expert_inroom_present_frames": ex_p,
                    "expert_inroom_covered_frames": ex_c,
                    "trainee_inroom_present_frames": tr_p,
                    "trainee_inroom_covered_frames": tr_c,
                }
            )

        per_inroom_df = pd.DataFrame(rows) if rows else pd.DataFrame([])

        try:
            _generate_plot(
                out_path=img_path,
                expert_res=expert_res,
                trainee_res=trainee_res,
                per_inroom_df=per_inroom_df,
            )
        except Exception:
            pass

        ex_final = float(expert_res.get("final_coverage", 0.0))
        tr_final = float(trainee_res.get("final_coverage", 0.0))
        delta_final = float(tr_final - ex_final)

        if abs(delta_final) <= 0.02:
            score_part = (
                f"Overall threat coverage looks about the same as the expert "
                f"(Trainee {tr_final:.2f} vs Expert {ex_final:.2f}, Δ {delta_final:+.2f})."
            )
        elif delta_final > 0:
            score_part = (
                f"Overall threat coverage looks better than the expert "
                f"(Trainee {tr_final:.2f} vs Expert {ex_final:.2f}, Δ {delta_final:+.2f})."
            )
        else:
            score_part = (
                f"Overall threat coverage looks worse than the expert "
                f"(Trainee {tr_final:.2f} vs Expert {ex_final:.2f}, Δ {delta_final:+.2f})."
            )

        def _avg_uncovered_sec(res: Dict[str, object]) -> Optional[float]:
            present = res.get("per_inroom_present") or {}
            covered = res.get("per_inroom_covered") or {}
            try:
                fps_local = float(res.get("fps", fps))
            except Exception:
                fps_local = fps

            if fps_local <= 0:
                return None

            vals: List[float] = []
            for inroom_id, p in present.items():
                try:
                    p_int = int(p)
                except Exception:
                    continue
                if p_int <= 0:
                    continue
                try:
                    c_int = int(covered.get(inroom_id, 0))
                except Exception:
                    c_int = 0
                vals.append(max(0.0, (p_int - c_int) / fps_local))
            return float(np.mean(vals)) if vals else None

        ex_avg_u = _avg_uncovered_sec(expert_res)
        tr_avg_u = _avg_uncovered_sec(trainee_res)

        if ex_avg_u is None or tr_avg_u is None:
            uncovered_part = (
                "On average, uncovered time per in-room threat before clearance couldn't be computed "
                "(missing per-inroom counts)."
            )
        else:
            du = float(tr_avg_u - ex_avg_u)
            if abs(du) <= 0.10:
                uncovered_part = (
                    "On average, the trainee had about the same uncovered time per in-room threat "
                    "before clearance as the expert."
                )
            elif du < 0:
                uncovered_part = (
                    f"On average, the trainee had about {abs(du):.2f}s less uncovered time per in-room threat "
                    f"before clearance than the expert (better coverage)."
                )
            else:
                uncovered_part = (
                    f"On average, the trainee had about {abs(du):.2f}s more uncovered time per in-room threat "
                    f"before clearance than the expert."
                )

        text = score_part + "\n" + uncovered_part

        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

        return {
            "Name": "THREAT_COVERAGE",
            "Type": "Single",
            "ImgLocation": img_path,
            "TxtLocation": txt_path,
            "Text": text,
        }