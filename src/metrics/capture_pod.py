import glob
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .metric import AbstractMetric


class IdentifyAndCapturePods_Metric(AbstractMetric):
    def __init__(self, config):
        super().__init__(config)
        self.metricName = "IDENTIFY_AND_CAPTURE_POD"
        self.score = 0.0
        self.pod = config.get("POD", None)
        self.map = config.get("Map Image", None)

    def process(self, context):
        pod_capture = context.pod_capture or {}
        total_pods = len(pod_capture)
        occupied = sum(
            1
            for _, info in pod_capture.items()
            if info.get("assigned_id") is not None and info.get("capture_frame") is not None
        )
        self.score = occupied / total_pods if total_pods > 0 else 1.0

    def getFinalScore(self):
        return round(float(self.score), 2)

    def expertCompare(
        self,
        session_folder: str,
        expert_folder: str,
        map_image=None,
        pod=None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        def _pick_latest(folder: str, pattern: str) -> Optional[str]:
            matches = glob.glob(os.path.join(folder, pattern))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        def _load_pod_cache(folder: str) -> pd.DataFrame:
            cache_path = _pick_latest(folder, "*_PodCache.txt")
            if cache_path is None:
                raise FileNotFoundError(f"No PodCache found in {folder}")

            df = pd.read_csv(cache_path)
            cols = {c.strip().lower(): c for c in df.columns}
            required = ["pod_idx", "assigned_id", "capture_time_sec", "capture_frame"]
            missing = [c for c in required if c not in cols]
            if missing:
                raise ValueError(f"Unexpected PodCache format: {cache_path} (missing {missing})")

            df = df[[cols["pod_idx"], cols["assigned_id"], cols["capture_time_sec"], cols["capture_frame"]]].copy()
            df.columns = ["pod_idx", "assigned_id", "capture_time_sec", "capture_frame"]

            df["pod_idx"] = pd.to_numeric(df["pod_idx"], errors="coerce").astype("Int64")
            df["assigned_id"] = pd.to_numeric(df["assigned_id"], errors="coerce").astype("Int64")
            df["capture_time_sec"] = pd.to_numeric(df["capture_time_sec"], errors="coerce")
            df["capture_frame"] = pd.to_numeric(df["capture_frame"], errors="coerce").astype("Int64")

            df = df.dropna(subset=["pod_idx"]).copy()
            df["pod_idx"] = df["pod_idx"].astype(int)
            df["captured"] = (~df["assigned_id"].isna()) & (~df["capture_frame"].isna())
            df["assigned"] = ~df["assigned_id"].isna()
            return df.sort_values("pod_idx").reset_index(drop=True)

        def _load_tracker_roles(folder: str) -> Tuple[Dict[int, str], Dict[int, str]]:
            tracker_path = _pick_latest(folder, "*_TrackerOutput.json")
            if tracker_path is None:
                return {}, {}

            try:
                import json

                with open(tracker_path, "r") as f:
                    data = json.load(f)
            except Exception:
                return {}, {}

            role_by_id: Dict[int, str] = {}
            birth_location_by_id: Dict[int, str] = {}

            if not isinstance(data, list):
                return role_by_id, birth_location_by_id

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
                        role_by_id[tid] = str(role)
                    if birth_location is not None:
                        birth_location_by_id[tid] = str(birth_location)

            return role_by_id, birth_location_by_id

        def _load_entry_map(folder: str, inroom_id_start: int) -> Dict[int, int]:
            pos_path = _pick_latest(folder, "*_PositionCache.txt")
            if pos_path is None:
                return {}

            try:
                dfp = pd.read_csv(pos_path)
            except Exception:
                return {}

            cols = {c.strip().lower(): c for c in dfp.columns}
            if not {"frame", "id"}.issubset(set(cols.keys())):
                return {}

            frame_col = cols["frame"]
            id_col = cols["id"]

            dfp = dfp[[frame_col, id_col]].dropna().copy()
            dfp[frame_col] = pd.to_numeric(dfp[frame_col], errors="coerce")
            dfp[id_col] = pd.to_numeric(dfp[id_col], errors="coerce")
            dfp = dfp.dropna(subset=[frame_col, id_col]).copy()
            if dfp.empty:
                return {}

            dfp[frame_col] = dfp[frame_col].astype(int)
            dfp[id_col] = dfp[id_col].astype(int)

            role_by_id, birth_location_by_id = _load_tracker_roles(folder)

            def _is_entry_track(track_id: int) -> bool:
                role = role_by_id.get(track_id)
                birth_location = birth_location_by_id.get(track_id)

                if role == "entry" or birth_location == "entry":
                    return True
                if role == "inroom" or birth_location == "inroom":
                    return False
                return int(track_id) < int(inroom_id_start)

            dfp = dfp[dfp[id_col].map(_is_entry_track)].copy()
            if dfp.empty:
                return {}

            first_frame_by_id = dfp.groupby(id_col)[frame_col].min().sort_values()
            return {int(tid): int(i) for i, tid in enumerate(first_frame_by_id.index.tolist(), start=1)}

        def _summary(df: pd.DataFrame) -> Tuple[int, int, float]:
            total = int(len(df))
            captured = int(df["captured"].sum()) if total > 0 else 0
            score = (captured / total) if total > 0 else 1.0
            return total, captured, float(score)

        def _coerce_map_image(img_or_path):
            if img_or_path is None:
                return None
            if isinstance(img_or_path, str):
                return cv2.imread(img_or_path) if os.path.exists(img_or_path) else None
            return img_or_path

        def _inroom_id_start_from_config(cfg: Optional[Dict[str, Any]]) -> int:
            if not isinstance(cfg, dict):
                return 99
            try:
                return int(cfg.get("inroom_id_start", 99))
            except Exception:
                return 99

        if map_image is None:
            map_image = getattr(self, "map", None)
        map_image = _coerce_map_image(map_image)

        if pod is None:
            pod = getattr(self, "pod", None)

        predefined_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 0, 0),
            (0, 128, 0),
            (0, 0, 128),
            (128, 128, 0),
            (128, 0, 128),
            (0, 128, 128),
        ]

        expert_img_path = os.path.join(session_folder, "IDENTIFY_AND_CAPTURE_POD_Expert.jpg")
        trainee_img_path = os.path.join(session_folder, "IDENTIFY_AND_CAPTURE_POD_Trainee.jpg")
        txt_path = os.path.join(session_folder, "IDENTIFY_AND_CAPTURE_POD_Comparison.txt")

        if map_image is None or pod is None:
            os.makedirs(session_folder, exist_ok=True)
            err_text = "There was an error while processing this comparison. Missing map image or POD coordinates."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "IDENTIFY_AND_CAPTURE_POD",
                "Type": "SideBySide",
                "ExpertImageLocation": expert_img_path,
                "TraineeImageLocation": trainee_img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        try:
            pods_list = pod.tolist() if isinstance(pod, np.ndarray) else list(pod)
        except Exception:
            pods_list = []

        pods_xy = []
        for p in pods_list:
            try:
                pods_xy.append((float(p[0]), float(p[1])))
            except Exception:
                continue

        inroom_id_start = _inroom_id_start_from_config(config if config is not None else self.config)

        try:
            expert_df = _load_pod_cache(expert_folder)
            trainee_df = _load_pod_cache(session_folder)
            expert_entry_map = _load_entry_map(expert_folder, inroom_id_start)
            trainee_entry_map = _load_entry_map(session_folder, inroom_id_start)
        except Exception:
            os.makedirs(session_folder, exist_ok=True)
            err_text = "There was an error while processing this comparison. Missing or invalid PodCache in one or both folders."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "IDENTIFY_AND_CAPTURE_POD",
                "Type": "Text",
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        merged = pd.merge(
            expert_df,
            trainee_df,
            on="pod_idx",
            how="outer",
            suffixes=("_expert", "_trainee"),
        ).sort_values("pod_idx").reset_index(drop=True)

        e_total, e_captured, e_score = _summary(expert_df)
        t_total, t_captured, t_score = _summary(trainee_df)

        expert_entry_num = (
            merged["assigned_id_expert"].apply(lambda x: expert_entry_map.get(int(x)) if pd.notna(x) else pd.NA)
            if "assigned_id_expert" in merged.columns
            else pd.Series([pd.NA] * len(merged))
        )
        trainee_entry_num = (
            merged["assigned_id_trainee"].apply(lambda x: trainee_entry_map.get(int(x)) if pd.notna(x) else pd.NA)
            if "assigned_id_trainee" in merged.columns
            else pd.Series([pd.NA] * len(merged))
        )

        entry_match = (
            expert_entry_num.notna()
            & trainee_entry_num.notna()
            & (expert_entry_num.astype("Int64") == trainee_entry_num.astype("Int64"))
        )
        entry_match_count = int(entry_match.sum())
        total_pods_compared = int(len(merged))

        expert_assigned_id = merged["assigned_id_expert"] if "assigned_id_expert" in merged.columns else pd.Series([pd.NA] * len(merged))
        trainee_assigned_id = merged["assigned_id_trainee"] if "assigned_id_trainee" in merged.columns else pd.Series([pd.NA] * len(merged))

        os.makedirs(session_folder, exist_ok=True)

        def _status_map(df_side: pd.DataFrame, entry_map: Dict[int, int]) -> Dict[int, Dict[str, Any]]:
            out: Dict[int, Dict[str, Any]] = {}
            for _, row in df_side.iterrows():
                pod_idx = int(row["pod_idx"])
                assigned_id = row.get("assigned_id")
                captured = bool(row.get("captured", False))

                tid = None
                entrant_number = None
                if pd.notna(assigned_id):
                    try:
                        tid = int(assigned_id)
                        entrant_number = entry_map.get(tid)
                    except Exception:
                        tid = None
                        entrant_number = None

                out[pod_idx] = {
                    "entry": entrant_number,
                    "id": tid,
                    "captured": captured,
                }
            return out

        expert_status = _status_map(expert_df, expert_entry_map)
        trainee_status = _status_map(trainee_df, trainee_entry_map)

        self.__generateExpertCompareGraphic(
            output_folder=session_folder,
            map_view=map_image,
            pods_xy=pods_xy,
            status=expert_status,
            out_name="IDENTIFY_AND_CAPTURE_POD_Expert.jpg",
            title="Expert",
            predefined_colors=predefined_colors,
        )
        self.__generateExpertCompareGraphic(
            output_folder=session_folder,
            map_view=map_image,
            pods_xy=pods_xy,
            status=trainee_status,
            out_name="IDENTIFY_AND_CAPTURE_POD_Trainee.jpg",
            title="Trainee",
            predefined_colors=predefined_colors,
        )

        e_pct = e_score * 100.0
        t_pct = t_score * 100.0
        diff_pp = t_pct - e_pct

        if abs(diff_pp) <= 1.0:
            perf = "about the same as"
        elif diff_pp > 1.0:
            perf = "better than"
        else:
            perf = "worse than"

        summary = (
            f"Entry assignment matched on {entry_match_count}/{total_pods_compared} PODs. "
            f"Overall, the trainee looks {perf} the expert on POD capture "
            f"(Trainee {t_pct:.1f}%, Expert {e_pct:.1f}%)."
        )

        lines = [
            "POD, Expert Entrant#, Expert ID, Trainee Entrant#, Trainee ID, Entrant# Match, Trainee Captured, Performance",
        ]

        for i, row in merged.iterrows():
            pod_idx = row.get("pod_idx")

            e_ent = expert_entry_num.iloc[i] if i < len(expert_entry_num) else pd.NA
            t_ent = trainee_entry_num.iloc[i] if i < len(trainee_entry_num) else pd.NA

            e_id = expert_assigned_id.iloc[i] if i < len(expert_assigned_id) else pd.NA
            t_id = trainee_assigned_id.iloc[i] if i < len(trainee_assigned_id) else pd.NA

            ent_match = bool(entry_match.iloc[i]) if i < len(entry_match) else False

            e_cap = bool(row.get("captured_expert")) if pd.notna(row.get("captured_expert")) else False
            t_cap = bool(row.get("captured_trainee")) if pd.notna(row.get("captured_trainee")) else False

            if e_cap == t_cap:
                pod_perf = "SIMILAR"
            elif t_cap and not e_cap:
                pod_perf = "BETTER"
            else:
                pod_perf = "WORSE"

            lines.append(
                f"P{int(pod_idx) if pd.notna(pod_idx) else 'N/A'}, "
                f"{int(e_ent) if pd.notna(e_ent) else 'N/A'}, {int(e_id) if pd.notna(e_id) else 'N/A'}, "
                f"{int(t_ent) if pd.notna(t_ent) else 'N/A'}, {int(t_id) if pd.notna(t_id) else 'N/A'}, "
                f"{'YES' if ent_match else 'NO'}, {'YES' if t_cap else 'NO'}, {pod_perf}"
            )

        details_csv = "\n".join(lines)
        text = summary + "\n" + details_csv

        def _broken_table(headers: List[str], data_rows: List[List[str]]) -> str:
            if not data_rows:
                return "(no rows)"

            widths = []
            for j, h in enumerate(headers):
                max_cell = max([len(h)] + [len(r[j]) for r in data_rows])
                widths.append(max_cell)

            sep = " | "

            def _fmt(values: List[str]) -> str:
                return sep.join(values[i].ljust(widths[i]) for i in range(len(headers)))

            broken_line = sep.join("-" * w for w in widths)
            out = [_fmt(headers), broken_line]
            out.extend(_fmt(r) for r in data_rows)
            return "\n".join(out)

        pretty_headers = [h.strip() for h in lines[0].split(",")]
        pretty_rows: List[List[str]] = []
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(",")]
            while len(parts) < len(pretty_headers):
                parts.append("N/A")
            pretty_rows.append(parts[: len(pretty_headers)])

        details_pretty = _broken_table(pretty_headers, pretty_rows)
        saved_text = summary + "\n\n" + details_pretty + "\n"

        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(saved_text)
        except Exception:
            pass

        return {
            "Name": "IDENTIFY_AND_CAPTURE_POD",
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
        pods_xy: List[Tuple[float, float]],
        status: Dict[int, Dict[str, Any]],
        out_name: str,
        title: str,
        predefined_colors: List[Tuple[int, int, int]],
    ) -> None:
        os.makedirs(output_folder, exist_ok=True)
        img = map_view.copy()
        h, w = img.shape[:2]
        w0 = w

        diag = math.hypot(w, h)
        radius = int(max(10, min(18, 0.018 * diag)))

        entrant_items = sorted(
            {
                (v.get("entry"), v.get("id"))
                for v in status.values()
                if v.get("entry") is not None and v.get("id") is not None
            },
            key=lambda t: int(t[0]),
        )

        def _color_for_id(track_id: int):
            return predefined_colors[int(track_id) % len(predefined_colors)]

        for pod_idx, (px, py) in enumerate(pods_xy):
            pod_status = status.get(int(pod_idx), {"entry": None, "id": None, "captured": False})
            entrant_number = pod_status.get("entry")
            track_id = pod_status.get("id")
            captured = bool(pod_status.get("captured", False))

            cx, cy = int(round(px)), int(round(py))

            if entrant_number is None or track_id is None:
                ring = (180, 180, 180)
                cv2.circle(img, (cx, cy), radius, ring, 3)
                cv2.putText(img, "?", (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ring, 2)
            else:
                color = _color_for_id(int(track_id))
                if captured:
                    cv2.circle(img, (cx, cy), radius, color, -1)
                    cv2.circle(img, (cx, cy), radius, (255, 255, 255), 2)
                else:
                    cv2.circle(img, (cx, cy), radius, color, 3)
                    cv2.line(img, (cx - radius + 3, cy - radius + 3), (cx + radius - 3, cy + radius - 3), color, 3)
                    cv2.line(img, (cx - radius + 3, cy + radius - 3), (cx + radius - 3, cy - radius + 3), color, 3)

            label = f"P{pod_idx}"
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
            lx, ly = cx - tw // 2, cy - radius - 6
            cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        pad = 12
        swatch = 14
        gap = 10
        line_h = 22

        lines = [title] + [f"Entrant #{entrant}" for entrant, _ in entrant_items]
        max_w = 0
        for text in lines:
            (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
            max_w = max(max_w, tw)

        panel_w = pad * 2 + swatch + gap + max_w
        panel_h = pad * 2 + line_h * max(1, len(lines))

        extra_right = panel_w + pad * 2
        extra_bottom = max(0, panel_h + pad * 2 - h)

        bg = tuple(int(x) for x in img[0, 0].tolist())
        canvas = np.full((h + extra_bottom, w0 + extra_right, 3), bg, dtype=np.uint8)
        canvas[:h, :w0] = img
        img = canvas
        h, w = img.shape[:2]

        x0 = w0 + pad
        y0 = pad

        cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
        cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (220, 220, 220), 2)

        cv2.putText(img, title, (x0 + pad, y0 + pad + 16), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        y = y0 + pad + line_h
        for entrant, track_id in entrant_items:
            color = _color_for_id(int(track_id))
            sx1, sy1 = x0 + pad, y + 4
            sx2, sy2 = sx1 + swatch, sy1 + swatch
            cv2.rectangle(img, (sx1, sy1), (sx2, sy2), color, -1)
            cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (255, 255, 255), 1)
            text = f"Entrant #{entrant}"
            cv2.putText(img, text, (sx2 + gap, y + 16), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y += line_h

        cv2.imwrite(os.path.join(output_folder, out_name), img)