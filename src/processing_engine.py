import json
import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from tqdm import tqdm

from libs.Track import OCSORT
from libs.Track.processing.utils import load_entry_polygons, load_pixel_mapper
from src.helper_functions import (
    annotate_camera_tracking_with_clearance,
    annotate_camera_video,
    annotate_camera_with_gaze_triangle,
    annotate_clearance_video,
    annotate_map_pod_video,
    annotate_map_pod_with_paths_video,
    annotate_map_video,
    annotate_map_with_gaze,
    compute_gaze_vector,
    compute_room_coverage,
    compute_threat_clearance,
    initialize_keypoint_indices,
    run_pod_analysis,
    save_gaze_cache,
    save_metrics_cache,
    save_position_cache,
    save_room_coverage_cache,
    save_threat_clearance_cache,
)

from .metrics import *
from .metrics.context import MetricContext
from .utils.config import load_config, load_vmeta
from .utils.transcode import transcode
from .utils.video import count_video_frames, get_video_framerate
from .utils.vmeta import generate_vmeta

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger("mmengine").setLevel(logging.ERROR)
logging.getLogger("mmdet").setLevel(logging.ERROR)
logging.getLogger("libs.Track.trackers.basetracker").setLevel(logging.ERROR)


VIS_VMETA_SUFFIX: Dict[str, str] = {
    "annotate_camera_video": "_Tracking_Overlays",
    "annotate_camera_with_gaze_triangle": "_Gaze_Triangles",
    "annotate_clearance_video": "_Clearance_Callouts",
    "annotate_camera_tracking_with_clearance": "_Tracking_WithClearance",
    "annotate_map_video": "_Tracking_Map",
    "annotate_map_pod_video": "_Tracking_PodAreas",
    "annotate_map_pod_with_paths_video": "_Tracking_PodAreasWithTrails",
    "annotate_map_with_gaze": "_Gaze_MapCleared",
}

DEFAULT_ENABLED_VISUALIZATIONS: List[str] = [
    "annotate_camera_with_gaze_triangle",
    "annotate_map_with_gaze",
    "annotate_map_pod_with_paths_video",
    "annotate_camera_tracking_with_clearance",
]


class ProcessingEngine:
    def __init__(self, force_transcode: bool = False, config=None):
        self.force_transcode = force_transcode
        self.metrics = []
        self.processed_vmetas = set()
        self.playback_time = 0
        self.metric_names = set()
        self.output_directory = ""
        self.video_basename = ""
        self.writeXML = True
        self.config = config if config is not None else None
        logging.debug("Ready to receive requests.")

    def _initialize_components(self, vmeta_path: Optional[str] = None):
        if vmeta_path is None:
            return

        point_mapping_path = os.path.join(os.path.dirname(vmeta_path), self.config["point_mapping_path"])
        self.mapper = load_pixel_mapper(
            point_mapping_path,
            ransac_reproj_threshold=float(self.config.get("homography_ransac_thresh", 3.0)),
            confidence=float(self.config.get("homography_confidence", 0.999)),
            max_iters=int(self.config.get("homography_max_iters", 2000)),
        )

        entry_polys_path = os.path.join(os.path.dirname(vmeta_path), self.config["entry_polys_path"])
        self.entry_polys = load_entry_polygons(entry_polys_path)

        self.boundary = self.config.get("Boundary", None)
        if self.boundary is not None and not hasattr(self.boundary, "contains"):
            try:
                self.boundary = Polygon(self.boundary)
            except Exception:
                self.boundary = None

        initialize_keypoint_indices(self.config)

        self.inferencer = MMPoseInferencer(
            pose2d=self.config["pose2d_config"],
            pose2d_weights=self.config["pose2d_weights"],
            device=self.config.get("device", "cpu"),
            det_model=self.config["det_model"],
            det_weights=self.config["det_weights"],
            det_cat_ids=self.config.get("det_cat_ids", (0,)),
        )

        self.tracker = OCSORT(
            per_class=False,
            det_thresh=0.3,
            max_age=1000,
            min_hits=1,
            asso_threshold=0.6,
            delta_t=5,
            asso_func="diou",
            inertia=0.2,
            use_byte=True,
            pixel_mapper=self.mapper,
            limit_entry=True,
            entry_polys=self.entry_polys,
            entry_window_time=float("inf"),
            boundary=self.boundary,
            boundary_pad_pct=self.config.get("boundary_pad_pct", 0.05),
            track_enemy=self.config.get("track_enemy", True),
            entry_conf_threshold=0.3,
        )

        self.box_conf_threshold = self.config.get("box_conf_threshold", 0.5)
        self.pose_conf_threshold = self.config.get("pose_conf_threshold", 0.5)
        self.device = self.config.get("device", "cpu")
        self.map_img = self.config.get("Map Image", None)
        self.keypoint_indices = self.config.get("keypoint_indices", None)

    def mt_initialize(self, messages, vmeta_paths, output_path):
        logging.debug(f"Received initialization message with {len(vmeta_paths)} vmeta files.")

        try:
            for vmeta_path in vmeta_paths:
                if vmeta_path not in self.processed_vmetas:
                    logging.debug(f"Starting processing of {vmeta_path}...")
                    self.processed_vmetas.add(vmeta_path)
                    config = self.__load_config(vmeta_path, output_path)

                    self.config = config
                    self._initialize_components(vmeta_path)
                    self.metrics += self.__assess()
                else:
                    logging.debug(f"Vmeta file has previously been processed. Skipping: {vmeta_path}")

            logging.debug("Finished Processing. Ready for metric queries.")
            return "ready"
        except Exception as e:
            logging.error("Unable to process the received vmeta files. Printing stack trace...")
            logging.error(e, exc_info=True)
            return "error"

    def __parse_messages(self, messages):
        all_keys = set()
        fire_messages = []
        entity_state_messages = []

        for msg in messages:
            msg_dict = json.loads(msg)
            all_keys.update(msg_dict.keys())

            if "triggerSqueeze" in msg_dict:
                fire_messages.append(msg_dict)
            elif "appearance" in msg_dict:
                entity_state_messages.append(msg_dict)

        self.msg_keys = all_keys

    def __load_config(self, vmeta_path, output_path):
        config_path, video_path, start_time = load_vmeta(vmeta_path)
        config = load_config(config_path)
        config["start_time"] = start_time
        config["video_path"] = video_path
        config["output_path"] = output_path
        if self.force_transcode:
            video_path = transcode(video_path)
        fps = get_video_framerate(video_path)
        if fps is None:
            fps = 30
        config["frame_rate"] = fps
        config["frame_time"] = int((1 / fps) * 1000)
        return config

    def _is_in_entry_region(self, point):
        if not getattr(self, "entry_polys", None):
            return False
        for entry_poly in self.entry_polys:
            if entry_poly.contains(point):
                return True
        return False

    def _resolve_inroom_ids(self, tracker_output, config):
        """Resolve confirmed in-room track IDs from tracker metadata."""
        inroom_ids = set()
        inroom_id_start = int(config.get("inroom_id_start", getattr(self.tracker, "INROOM_FINAL_ID_START", 99)))

        for frame_entry in tracker_output:
            for obj in frame_entry["objects"]:
                tid = obj.get("id")
                if tid is None:
                    continue

                role = obj.get("identity_role")
                is_inroom = bool(obj.get("is_inroom", False))
                birth_location = obj.get("birth_location")

                if role == "inroom" or is_inroom or birth_location == "inroom":
                    inroom_ids.add(int(tid))
                elif role is None and birth_location is None and int(tid) >= inroom_id_start:
                    inroom_ids.add(int(tid))

        return sorted(inroom_ids)

    def _resolve_entry_ids(self, tracker_output):
        """Resolve confirmed entry track IDs from tracker metadata."""
        entry_ids = set()
        for frame_entry in tracker_output:
            for obj in frame_entry["objects"]:
                tid = obj.get("id")
                if tid is None:
                    continue

                role = obj.get("identity_role")
                is_entry = bool(obj.get("is_entry", False))
                birth_location = obj.get("birth_location")

                if role == "entry" or is_entry or birth_location == "entry":
                    entry_ids.add(int(tid))
        return sorted(entry_ids)

    def __assess(self):
        config = self.config

        enabled = config.get("enabled_visualizations", None)
        if not isinstance(enabled, list) or not enabled:
            enabled = list(DEFAULT_ENABLED_VISUALIZATIONS)
        enabled = [name for name in enabled if name in VIS_VMETA_SUFFIX]
        self.enabled_visualizations = enabled

        visual_angle_deg = float(config.get("visual_angle_degrees", 20.0))
        half_visual_angle_deg = visual_angle_deg / 2.0

        min_threat_interaction_time_sec = float(config.get("min_threat_interaction_time_sec", 1.0))
        threat_interaction_frames = int(min_threat_interaction_time_sec * config.get("frame_rate", 30.0))

        metrics: List[AbstractMetric] = [
            IdentifyAndCapturePods_Metric(config),
            CapturePodTime_Metric(config),
            MoveAlongWall_Metric(config),
            EntranceVectors_Metric(config),
            EntranceHesitation_Metric(config),
            ThreatClearance_Metric(config),
            TeammateCoverage_Metric(config),
            ThreatCoverage_Metric(config),
            RoomCoverage_Metric(config),
            TotalRoomCoverageTime_Metric(config),
        ]

        vs = cv2.VideoCapture(config["video_path"])
        frame_total = count_video_frames(vs)

        self.output_directory = os.path.abspath(config["output_path"])
        self.video_basename = os.path.splitext(os.path.basename(config["video_path"]))[0]

        cv2.imwrite(os.path.join(self.output_directory, "EmptyMap.jpg"), config["Map Image"])
        logging.debug(f"Saved map cache to: {os.path.join(self.output_directory, 'EmptyMap.jpg')}")

        tracker_output: List[dict] = []
        all_map_points: List[list] = []
        gaze_info: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
        all_frames: List[list] = []
        processed_frames = 0
        tracks_by_id: Dict[int, List[Optional[Tuple[float, float]]]] = {}

        for frame_num in tqdm(range(1, frame_total + 1), desc="Processing frames", unit="frame"):
            ret, frame = vs.read()
            if not ret or frame is None:
                break

            processed_frames += 1

            infer_results = self.inferencer(
                frame,
                return_vis=False,
                bbox_thr=self.box_conf_threshold,
                kpt_thr=self.pose_conf_threshold,
                pose_based_nms=True,
            )
            res = next(infer_results)
            instances = res["predictions"][0]

            dets = []
            matched_keypoints = []

            for inst in instances:
                bbox = inst["bbox"][0]
                x1, y1, x2, y2 = bbox
                frame_h, frame_w = frame.shape[:2]
                if x1 <= 0 and y1 <= 0 and x2 >= frame_w and y2 >= frame_h:
                    continue

                bbox_score = inst.get("bbox_score", 0.0)
                dets.append([x1, y1, x2, y2, bbox_score, 0])

                kps = np.array(inst["keypoints"])
                kp_scores = inst.get("keypoint_scores", None)
                if kp_scores is not None:
                    scores = np.array(kp_scores)
                else:
                    scores = np.ones((kps.shape[0],))
                kps3 = np.concatenate([kps, scores[:, None]], axis=1)
                matched_keypoints.append(kps3)

            dets = np.array(dets)
            if dets.size == 0:
                dets = np.empty((0, 6))

            if matched_keypoints:
                matched_keypoints = np.array(matched_keypoints)
            else:
                matched_keypoints = np.empty((0, 26, 3))

            trackers = self.tracker.update(
                dets,
                frame,
                keypoints=matched_keypoints,
                keypoint_confidence_threshold=self.pose_conf_threshold,
                keypoint_indices=self.keypoint_indices,
            )

            frame_objects = []
            for trk in trackers:
                trk_id = trk.get("track_id")
                x1 = int(trk.get("top_left_x", 0))
                y1 = int(trk.get("top_left_y", 0))
                w = int(trk.get("width", 0))
                h = int(trk.get("height", 0))
                bbox = [x1, y1, x1 + w, y1 + h]
                kps = trk.get("keypoints", None)

                bbox_confidence = trk.get("confidence", None)
                track_class = trk.get("class", None)
                ankle_based_point = trk.get("ankle_based_point", None)
                current_map_pos = trk.get("current_map_pos", None)
                map_velocity = trk.get("map_velocity", None)

                if ankle_based_point is not None:
                    ankle_based_point = np.asarray(ankle_based_point, dtype=float).reshape(-1)
                    ankle_based_point = ankle_based_point.tolist() if ankle_based_point.size >= 2 else None

                if current_map_pos is not None:
                    current_map_pos = np.asarray(current_map_pos, dtype=float).reshape(-1)
                    current_map_pos = current_map_pos.tolist() if current_map_pos.size >= 2 else None

                if map_velocity is not None:
                    map_velocity = np.asarray(map_velocity, dtype=float).reshape(-1)
                    map_velocity = map_velocity.tolist() if map_velocity.size >= 2 else None

                if kps is not None:
                    keypoints = kps[:, :2].tolist()
                    keypoint_scores = kps[:, 2].tolist()
                else:
                    keypoints, keypoint_scores = [], []

                if kps is not None and kps.shape[0] == 26:
                    gvec = compute_gaze_vector(kps)
                    if gvec is not None:
                        origin, direction = gvec
                        gaze_info[(frame_num, trk_id)] = (
                            float(origin[0]),
                            float(origin[1]),
                            float(direction[0]),
                            float(direction[1]),
                        )

                frame_objects.append(
                    {
                        "id": trk_id,
                        "bbox": bbox,
                        "confidence": float(bbox_confidence) if bbox_confidence is not None else None,
                        "class": int(track_class) if track_class is not None else None,
                        "ankle_based_point": ankle_based_point,
                        "current_map_pos": current_map_pos,
                        "map_velocity": map_velocity,
                        "keypoints": keypoints,
                        "keypoint_scores": keypoint_scores,
                        "identity_role": trk.get("identity_role", None),
                        "birth_location": trk.get("birth_location", None),
                        "is_inroom": bool(trk.get("is_inroom", False)),
                        "is_entry": bool(trk.get("is_entry", False)),
                    }
                )

            tracker_output.append({"frame": frame_num, "objects": frame_objects})

            map_points = []
            for trk in trackers:
                trk_id = trk.get("track_id")
                current_map_pos = trk.get("current_map_pos")
                if current_map_pos is not None:
                    cur = np.asarray(current_map_pos, dtype=float).reshape(-1)
                    if cur.size < 2 or not np.isfinite(cur[:2]).all():
                        continue

                    mapX, mapY = float(cur[0]), float(cur[1])
                    if self.boundary is not None:
                        point = Point(mapX, mapY)
                        if self.boundary.contains(point) or self._is_in_entry_region(point):
                            map_points.append([trk_id, mapX, mapY])
                        else:
                            nearest_point = nearest_points(self.boundary, point)[0]
                            map_points.append([trk_id, nearest_point.x, nearest_point.y])

            for tid in tracks_by_id.keys():
                tracks_by_id[tid].append(None)

            for tid, mx, my in map_points:
                if tid not in tracks_by_id:
                    tracks_by_id[tid] = [None] * (processed_frames - 1)
                    tracks_by_id[tid].append((float(mx), float(my)))
                else:
                    tracks_by_id[tid][-1] = (float(mx), float(my))

            all_frames.append(map_points)
            for idx, mx, my in map_points:
                all_map_points.append([frame_num, idx, mx, my])

        inroom_ids = self._resolve_inroom_ids(tracker_output, config)
        entry_ids = self._resolve_entry_ids(tracker_output)
        logging.debug(f"Resolved entry IDs from tracker metadata: {entry_ids}")
        logging.debug(f"Resolved in-room IDs from tracker metadata: {inroom_ids}")

        fall_frames = None

        save_position_cache(all_map_points, self.output_directory, self.video_basename)
        save_gaze_cache(gaze_info, self.output_directory, self.video_basename)

        if "annotate_camera_video" in self.enabled_visualizations:
            annotate_camera_video(
                tracker_output=tracker_output,
                frame_rate=config["frame_rate"],
                output_directory=self.output_directory,
                video_basename=self.video_basename,
                inroom_ids=inroom_ids,
                gaze_conf_threshold=self.pose_conf_threshold,
                video_path=config["video_path"],
            )

        if "annotate_camera_with_gaze_triangle" in self.enabled_visualizations:
            annotate_camera_with_gaze_triangle(
                tracker_output=tracker_output,
                gaze_info=gaze_info,
                frame_rate=config["frame_rate"],
                output_directory=self.output_directory,
                video_basename=self.video_basename,
                inroom_ids=inroom_ids,
                half_angle_deg=half_visual_angle_deg,
                alpha=0.2,
                show_inroom_gaze=False,
                video_path=config["video_path"],
            )

        if "annotate_map_video" in self.enabled_visualizations:
            annotate_map_video(
                config["Map Image"],
                all_map_points,
                config["frame_rate"],
                self.output_directory,
                self.video_basename,
                total_frames=processed_frames,
                inroom_ids=inroom_ids,
            )

        coverage_data = None
        if self.boundary is not None:
            if "annotate_map_with_gaze" in self.enabled_visualizations:
                annotate_map_with_gaze(
                    map_image=config["Map Image"],
                    pixel_mapper=self.mapper,
                    gaze_info=gaze_info,
                    room_boundary_coords=list(self.boundary.exterior.coords),
                    frame_rate=config["frame_rate"],
                    output_directory=self.output_directory,
                    video_basename=self.video_basename,
                    inroom_ids=inroom_ids,
                    show_inroom_gaze=False,
                    half_angle_deg=half_visual_angle_deg,
                    alpha=0.1,
                    total_frames=processed_frames,
                    accumulated_clear=True,
                )

            coverage_data = compute_room_coverage(
                map_image=config["Map Image"],
                pixel_mapper=self.mapper,
                gaze_info=gaze_info,
                room_boundary_coords=list(self.boundary.exterior.coords),
                frame_rate=config["frame_rate"],
                total_frames=processed_frames,
                inroom_ids=inroom_ids,
                half_angle_deg=half_visual_angle_deg,
            )
            save_room_coverage_cache(coverage_data, self.output_directory, self.video_basename)

        tracker_json_path = os.path.join(self.output_directory, f"{self.video_basename}_TrackerOutput.json")
        with open(tracker_json_path, "w") as f:
            json.dump(tracker_output, f, indent=4)
        logging.debug(f"Saved TrackerOutput to: {tracker_json_path}")
        logging.debug(
            f"Saved PositionCache to: {os.path.join(self.output_directory, f'{self.video_basename}_PositionCache.txt')}"
        )

        pods_cfg = config.get("POD", [])
        pod_groups = config.get("pod_groups", None)
        working_radius = config.get("pod_working_radius", 40.0)
        capture_threshold = config.get("pod_capture_threshold_sec", 0.1)

        assignment, dynamic_work_areas, pod_capture_data = run_pod_analysis(
            tracks_by_id=tracks_by_id,
            tracker_output=tracker_output,
            pods_cfg=pods_cfg,
            pod_groups=pod_groups,
            pixel_mapper=self.mapper,
            boundary=self.boundary,
            inroom_ids=inroom_ids,
            working_radius=working_radius,
            frame_rate=config["frame_rate"],
            capture_threshold_sec=capture_threshold,
            save_cache=True,
            output_directory=self.output_directory,
            video_basename=self.video_basename,
        )

        if "annotate_map_pod_video" in self.enabled_visualizations:
            annotate_map_pod_video(
                config["Map Image"],
                all_map_points=all_map_points,
                assignment=assignment,
                dynamic_work_areas=dynamic_work_areas,
                pod_capture_data=pod_capture_data,
                frame_rate=config["frame_rate"],
                output_directory=self.output_directory,
                video_basename=self.video_basename,
                total_frames=processed_frames,
                inroom_ids=inroom_ids,
            )

        if "annotate_map_pod_with_paths_video" in self.enabled_visualizations:
            annotate_map_pod_with_paths_video(
                config["Map Image"],
                all_map_points=all_map_points,
                assignment=assignment,
                dynamic_work_areas=dynamic_work_areas,
                pod_capture_data=pod_capture_data,
                frame_rate=config["frame_rate"],
                output_directory=self.output_directory,
                video_basename=self.video_basename,
                total_frames=processed_frames,
                inroom_ids=inroom_ids,
            )

        bbox_details = {}
        keypoint_details = {}
        for entry in tracker_output:
            frame_idx = entry["frame"]
            for obj in entry["objects"]:
                tid = obj["id"]
                bbox_details[(frame_idx, tid)] = tuple(obj["bbox"])
                keypoint_details[(frame_idx, tid)] = (obj["keypoints"], obj["keypoint_scores"])

        context = MetricContext(
            tracker_output=tracker_output,
            all_frames=all_frames,
            tracks_by_id=tracks_by_id,
            gaze_info=gaze_info,
            bbox_details=bbox_details,
            keypoint_details=keypoint_details,
            fall_frames=fall_frames,
            map_points=all_map_points,
            entry_ids=entry_ids,
            inroom_ids=inroom_ids,
            room_coverage=coverage_data,
            pod_capture=pod_capture_data,
        )

        clearance_map = compute_threat_clearance(
            tracker_output,
            keypoint_details,
            gaze_info,
            inroom_ids=inroom_ids,
            visual_angle_deg=visual_angle_deg,
            intersection_frames=threat_interaction_frames,
            wrist_frames=max(1, int(threat_interaction_frames * 0.1)),
            gaze_frames=max(1, int(threat_interaction_frames * 0.5)),
        )

        if "annotate_clearance_video" in self.enabled_visualizations:
            annotate_clearance_video(
                tracker_output=tracker_output,
                clearance_map=clearance_map,
                frame_rate=config["frame_rate"],
                output_directory=self.output_directory,
                video_basename=self.video_basename,
                inroom_ids=inroom_ids,
                video_path=config["video_path"],
            )

        if "annotate_camera_tracking_with_clearance" in self.enabled_visualizations:
            annotate_camera_tracking_with_clearance(
                tracker_output=tracker_output,
                clearance_map=clearance_map,
                frame_rate=config["frame_rate"],
                output_directory=self.output_directory,
                video_basename=self.video_basename,
                inroom_ids=inroom_ids,
                gaze_conf_threshold=self.pose_conf_threshold,
                show_clearing_id=True,
                video_path=config["video_path"],
            )

        save_threat_clearance_cache(clearance_map, self.output_directory, self.video_basename)
        context.threat_clearance = clearance_map

        metric_scores = []
        for m in metrics:
            m.process(context)
            score = m.getFinalScore()

            assessment = "below"
            if isinstance(score, (int, float)):
                if score > 0.9:
                    assessment = "above"
                elif score > 0.5:
                    assessment = "at"

            metric_entry = {
                "metric_name": m.metricName,
                "score": score,
                "assessment": assessment,
                "timestamp": config["start_time"],
            }
            metric_scores.append(metric_entry)

        save_metrics_cache(metric_scores, self.output_directory, self.video_basename)
        return metric_scores

    def get_assessment(self, timestamp, metric_name):
        timestamp = int(timestamp)
        if self.writeXML:
            for viz_name in getattr(self, "enabled_visualizations", []) or []:
                suffix = VIS_VMETA_SUFFIX.get(viz_name)
                if not suffix:
                    continue
                generate_vmeta(self.output_directory, self.video_basename, suffix)
            self.writeXML = False

        desired_metrics = []
        for metric in self.metrics:
            if metric["metric_name"] == metric_name:
                desired_metrics.append(metric)

        self.playback_time = max(timestamp, self.playback_time)
        if len(desired_metrics) > 0:
            logging.debug(f"Returning {len(desired_metrics)} results for query {metric_name} at time {timestamp}")
        else:
            logging.debug(f"Metric query for time {timestamp} received but no metrics match.")
        return json.dumps(desired_metrics, indent=4)

    def compare_expert(self, metric_name, session_folder, expert_folder, vmeta_path=None):
        """Compare a trainee (session_folder) against an expert (expert_folder) for a metric."""
        logging.debug("")
        logging.debug(f"Starting Expert Comparison for metric {metric_name}.")

        if not session_folder or not os.path.isdir(session_folder):
            logging.debug("Error: Supplied session folder is missing or not a directory: %s", session_folder)
            return json.dumps(
                {
                    "Name": metric_name,
                    "Type": "SideBySide",
                    "ImgLocation": os.path.join(session_folder or "", "error_image.jpg"),
                    "Text": "There was an error while processing this comparison. The supplied session folder is invalid.",
                },
                indent=4,
            )

        if not expert_folder or not os.path.isdir(expert_folder):
            logging.debug("Error: Supplied expert folder is missing or not a directory: %s", expert_folder)
            return json.dumps(
                {
                    "Name": metric_name,
                    "Type": "SideBySide",
                    "ImgLocation": os.path.join(session_folder, "error_image.jpg"),
                    "Text": "There was an error while processing this comparison. The supplied expert folder is invalid.",
                },
                indent=4,
            )

        config = None
        if vmeta_path:
            try:
                config_path, _, _ = load_vmeta(vmeta_path)
                config = load_config(config_path)
            except Exception:
                logging.error("Error: Unable to load config from vmeta_path: %s", vmeta_path, exc_info=True)
                config = None

        map_path_session = os.path.join(session_folder, "EmptyMap.jpg")
        map_path_expert = os.path.join(expert_folder, "EmptyMap.jpg")
        map_path = map_path_session if os.path.exists(map_path_session) else (
            map_path_expert if os.path.exists(map_path_expert) else None
        )
        map_image = cv2.imread(map_path) if map_path is not None else None

        def _call_metric(metric_cls):
            if not hasattr(metric_cls, "expertCompare"):
                raise AttributeError(f"{metric_cls} has no expertCompare")

            metric_obj = None
            try:
                metric_obj = metric_cls(config or {})
            except TypeError:
                metric_obj = None

            target = metric_obj if metric_obj is not None else metric_cls

            try:
                return target.expertCompare(
                    session_folder=session_folder,
                    expert_folder=expert_folder,
                    map_image=map_image,
                    config=config,
                )
            except TypeError:
                try:
                    return target.expertCompare(
                        session_folder=session_folder,
                        expert_folder=expert_folder,
                        map_image=map_image,
                    )
                except TypeError:
                    try:
                        return target.expertCompare(session_folder, expert_folder, map_image, config)
                    except TypeError:
                        return target.expertCompare(session_folder, expert_folder, map_image)

        try:
            metric_map = {
                "ENTRANCE_HESITATION": EntranceHesitation_Metric,
                "ENTRANCE_VECTORS": EntranceVectors_Metric,
                "STAY_ALONG_WALL": MoveAlongWall_Metric,
                "IDENTIFY_AND_HOLD_DESIGNATED_AREA": POD_Metric,
                "TOTAL_TIME_OF_ENTRY": TotalEntryTime_Metric,
                "IDENTIFY_AND_CAPTURE_POD": IdentifyAndCapturePods_Metric,
                "POD_CAPTURE_TIME": CapturePodTime_Metric,
                "THREAT_CLEARANCE": ThreatClearance_Metric,
                "TEAMMATE_COVERAGE": TeammateCoverage_Metric,
                "THREAT_COVERAGE": ThreatCoverage_Metric,
                "FLOOR_COVERAGE": RoomCoverage_Metric,
                "TOTAL_FLOOR_COVERAGE_TIME": TotalRoomCoverageTime_Metric,
            }

            metric_cls = metric_map.get(metric_name)
            if metric_cls is None:
                logging.debug("Metric %s not implemented in compare_expert mapping.", metric_name)
                out = {
                    "Name": metric_name,
                    "Type": "SideBySide",
                    "ImgLocation": os.path.join(session_folder, "error_image.jpg"),
                    "Text": "There was an error while processing this comparison. Most likely, this metric is not yet implemented.",
                }
            else:
                logging.debug("Started computing %s", metric_name)
                out = _call_metric(metric_cls)

        except Exception:
            logging.error("Error while running expert comparison for %s", metric_name, exc_info=True)
            out = {
                "Name": metric_name,
                "Type": "SideBySide",
                "ImgLocation": os.path.join(session_folder, "error_image.jpg"),
                "Text": "There was an error while processing this comparison. Most likely, this metric is not yet implemented.",
            }

        logging.debug(f"Completed analysis of metric {metric_name}.")
        return json.dumps(out, indent=4)

    def make_path(self, raw_path):
        paths = []
        max_len = int(np.max(raw_path[:, 0]))
        for trk_idx in np.unique(raw_path[:, 1]):
            idx_path = raw_path[raw_path[:, 1] == trk_idx]
            idx_path_corrected = []
            for frame_num in range(1, max_len + 1):
                raw_frame = np.squeeze(idx_path[idx_path[:, 0] == frame_num])
                if raw_frame is not None and len(raw_frame) > 0:
                    idx_path_corrected.append((raw_frame[2], raw_frame[3]))
                else:
                    idx_path_corrected.append(None)
            paths.append(idx_path_corrected)
        return paths