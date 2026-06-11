"""Drop-in replacement for ``mmpose.apis.MMPoseInferencer``.

Accepts the same constructor + call signatures used by
``src/processing_engine.py`` and yields the same ``{"predictions": [[<inst>...]]}``
schema. Internally delegates detection + pose to a backend chosen by
``runtime.autoselect``.
"""
from __future__ import annotations

from typing import Any, Iterator, Sequence

import numpy as np

from libs.giftpose.postprocess.pose_nms import nearby_joints_nms
from libs.giftpose.runtime.autoselect import select_backend


# Legacy-config-string -> short identifier mapping. Both the new
# tags and the old vendored mmpose paths resolve here.
_DET_TAGS = {
    "rtmdet-m-person-640",
    "libs/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py",
}
_POSE_TAGS = {
    "rtmpose-x-halpe26-384x288",
    "libs/mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py",
}


def _validate_tag(tag: str, valid: set[str], kind: str) -> None:
    if tag not in valid:
        raise ValueError(
            f"Unsupported {kind} model identifier {tag!r}. "
            f"Supported values: {sorted(valid)}"
        )


class MMPoseInferencer:
    """Compatible-shape drop-in for the mmpose inferencer.

    The constructor accepts the exact kwargs already used by
    ``src/processing_engine.py``. ``pose2d`` and ``det_model`` are treated as
    opaque architecture tags; only specific identifiers are accepted (see
    ``_POSE_TAGS`` / ``_DET_TAGS`` above).
    """

    def __init__(
        self,
        pose2d: str,
        pose2d_weights: str,
        det_model: str,
        det_weights: str,
        device: str = "cpu",
        det_cat_ids: Sequence[int] = (0,),
        prefer_backend: str | None = None,
        flip_test: bool | None = None,
        compile_for_inference: bool | None = None,
        det_score_thr: float = 0.3,
        det_iou_threshold: float = 0.6,
        det_max_per_img: int = 100,
    ) -> None:
        _validate_tag(pose2d, _POSE_TAGS, "pose2d")
        _validate_tag(det_model, _DET_TAGS, "det_model")
        # det_cat_ids: ignored — the GIFT detector is single-class person.
        self._det_cat_ids = tuple(det_cat_ids)

        # ``det_score_thr`` is the minimum detection confidence the detector
        # bothers to emit (its NMS floor). It defaults to 0.3 to match the
        # downstream tracker — OCSORT runs with ``det_thresh`` / ``entry_conf_
        # threshold`` = 0.3 and discards anything weaker, so producing (and
        # pose-estimating) boxes below 0.3 is wasted work. Callers wire this to
        # ``box_conf_threshold`` so there is a single meaningful detection floor.
        self.backend = select_backend(
            det_weights=det_weights,
            pose_weights=pose2d_weights,
            device=device,
            prefer=prefer_backend,
            compile_for_inference=bool(compile_for_inference),
            det_score_thr=det_score_thr,
            det_iou_threshold=det_iou_threshold,
            det_max_per_img=det_max_per_img,
        )

        # ``flip_test`` doubles the pose batch (original + horizontally
        # flipped crops, averaged) for ~0.5-1 px better keypoint accuracy
        # at ~2x pose-forward cost. Disabling halves pose time per frame —
        # the single biggest perf knob in this pipeline.
        if flip_test is not None:
            self.backend.flip_test = bool(flip_test)

    def __call__(
        self,
        image: np.ndarray,
        return_vis: bool = False,
        bbox_thr: float = 0.0,
        kpt_thr: float = 0.0,  # noqa: ARG002 — visualization-only in mmpose; ignored here
        pose_based_nms: bool = False,
        **_: Any,
    ) -> Iterator[dict]:
        """Yield exactly one result dict per call (the legacy mmpose
        inferencer is a generator; the engine consumes it via ``next()``).
        """
        instances = self._infer_one(image, bbox_thr, pose_based_nms)
        yield {"predictions": [instances]}

    def _infer_one(
        self,
        image: np.ndarray,
        bbox_thr: float,
        pose_based_nms: bool,
    ) -> list[dict]:
        det = self.backend.predict_detector(image)

        if bbox_thr > 0 and det.scores.size > 0:
            keep = det.scores > bbox_thr
            det.boxes_xyxy = det.boxes_xyxy[keep]
            det.scores = det.scores[keep]

        # Drop degenerate (sub-pixel) boxes. A zero-area box yields a singular
        # top-down warp (apply_inverse_warps_batched -> "Singular matrix") and a
        # NaN IoU in the tracker; such detections are spurious anyway.
        if det.boxes_xyxy.shape[0] > 0:
            wh = det.boxes_xyxy[:, 2:] - det.boxes_xyxy[:, :2]
            keep = (wh[:, 0] > 1.0) & (wh[:, 1] > 1.0)
            if not keep.all():
                det.boxes_xyxy = det.boxes_xyxy[keep]
                det.scores = det.scores[keep]

        if det.boxes_xyxy.shape[0] == 0:
            return []

        kpts, kp_scores = self.backend.predict_pose(image, det.boxes_xyxy)

        if pose_based_nms and len(det.scores) > 1:
            num_keypoints = kpts.shape[1]
            kpts_db = [
                {"keypoints": kpts[i], "score": float(det.scores[i])}
                for i in range(len(det.scores))
            ]
            keep_idx = nearby_joints_nms(
                kpts_db,
                num_nearby_joints_thr=num_keypoints // 3,
            )
            keep_idx = np.asarray(keep_idx, dtype=np.int64)
            det.boxes_xyxy = det.boxes_xyxy[keep_idx]
            det.scores = det.scores[keep_idx]
            kpts = kpts[keep_idx]
            kp_scores = kp_scores[keep_idx]

        results: list[dict] = []
        for i in range(det.boxes_xyxy.shape[0]):
            x1, y1, x2, y2 = det.boxes_xyxy[i].tolist()
            results.append(
                {
                    "bbox": [[x1, y1, x2, y2]],
                    "bbox_score": float(det.scores[i]),
                    "keypoints": kpts[i],
                    "keypoint_scores": kp_scores[i],
                }
            )
        return results
