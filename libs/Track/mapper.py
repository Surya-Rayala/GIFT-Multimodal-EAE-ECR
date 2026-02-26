# Mapper.py
import numpy as np
import cv2
from typing import Iterable, Optional, Union, Tuple

ArrayLike = Union[np.ndarray, Iterable[float], Iterable[Iterable[float]]]

class PixelMapper:
    """
    Planar homography mapper between image pixel coordinates and a target plane coordinate system.

    Expects correspondences:
      pixel_array: Nx2 (x,y) in image pixels
      target_array: Nx2 (X,Y) in map/world plane units
    """
    def __init__(
        self,
        pixel_array: ArrayLike,
        target_array: ArrayLike,
        *,
        method: str = "RANSAC",                 # "RANSAC", "LMEDS", "DIRECT"
        ransac_reproj_threshold: float = 3.0,
        confidence: float = 0.999,
        max_iters: int = 2000,
        refine: bool = True,                    # re-fit using inliers only
        min_inliers: int = 4,
    ):
        src = np.asarray(pixel_array, dtype=np.float64).reshape(-1, 2)
        dst = np.asarray(target_array, dtype=np.float64).reshape(-1, 2)

        if src.shape != dst.shape:
            raise ValueError(f"pixel_array and target_array must match shape, got {src.shape} vs {dst.shape}")
        if src.shape[0] < 4:
            raise ValueError(f"Need at least 4 correspondences, got {src.shape[0]}")

        # Convert to OpenCV format Nx1x2
        src_cv = src.reshape(-1, 1, 2)
        dst_cv = dst.reshape(-1, 1, 2)

        method_upper = method.upper()
        if method_upper == "RANSAC":
            H, mask = cv2.findHomography(
                src_cv, dst_cv,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_reproj_threshold,
                confidence=confidence,
                maxIters=max_iters,
            )
        elif method_upper == "LMEDS":
            H, mask = cv2.findHomography(src_cv, dst_cv, method=cv2.LMEDS)
        elif method_upper == "DIRECT":
            H, mask = cv2.findHomography(src_cv, dst_cv, method=0)
        else:
            raise ValueError(f"Unknown method={method}. Use RANSAC/LMEDS/DIRECT")

        if H is None or not np.isfinite(H).all():
            raise ValueError("Homography estimation failed (H is None or non-finite). Check calibration points.")

        mask = mask.astype(bool).reshape(-1) if mask is not None else np.ones((src.shape[0],), dtype=bool)

        # Optional refinement: recompute with inliers only using a direct method
        if refine:
            inliers = int(mask.sum())
            if inliers < min_inliers:
                raise ValueError(f"Too few inliers for a stable homography: {inliers}/{len(mask)}")
            src_in = src_cv[mask]
            dst_in = dst_cv[mask]
            H_ref, _ = cv2.findHomography(src_in, dst_in, method=0)
            if H_ref is not None and np.isfinite(H_ref).all():
                H = H_ref

        self.H = H.astype(np.float64)
        self.inlier_mask = mask
        self.H_inv = np.linalg.inv(self.H)

    # ---------- core transforms ----------
    def pixel_to_map(self, points: ArrayLike) -> np.ndarray:
        """
        Accepts:
          - (x,y)
          - Nx2 array
        Returns:
          - (X,Y) for single input
          - Nx2 for batch input
        """
        pts = np.asarray(points, dtype=np.float64)
        pts = pts.reshape(-1, 2)

        if not np.isfinite(pts).all():
            # Return NaNs for bad inputs rather than crashing
            out = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
            return out[0] if out.shape[0] == 1 else out

        pts_cv = pts.reshape(-1, 1, 2).astype(np.float64)
        trans = cv2.perspectiveTransform(pts_cv, self.H)
        trans = trans.reshape(-1, 2)

        return trans[0] if trans.shape[0] == 1 else trans

    def map_to_pixel(self, points: ArrayLike) -> np.ndarray:
        """Inverse mapping (useful for debugging/sanity checks)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(pts).all():
            out = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
            return out[0] if out.shape[0] == 1 else out

        pts_cv = pts.reshape(-1, 1, 2).astype(np.float64)
        trans = cv2.perspectiveTransform(pts_cv, self.H_inv)
        trans = trans.reshape(-1, 2)

        return trans[0] if trans.shape[0] == 1 else trans

    # ---------- detection helpers ----------
    def detection_to_map(self, xywh: ArrayLike, weights=None) -> np.ndarray:
        """
        xywh expected: (x_center, y_center, w, h) in pixel units.
        Uses bottom-center point.
        """
        b = np.asarray(xywh, dtype=np.float64).reshape(-1)
        if b.size != 4:
            raise ValueError(f"Expected xywh with 4 values, got shape={np.asarray(xywh).shape}")

        x, y, w, h = b
        bottom_center = np.array([x, y + h / 2.0], dtype=np.float64)
        return self.pixel_to_map(bottom_center)

    # ---------- calibration QA ----------
    def reprojection_errors(self, pixel_array: ArrayLike, target_array: ArrayLike) -> np.ndarray:
        """Per-point Euclidean reprojection error in target/map space."""
        src = np.asarray(pixel_array, dtype=np.float64).reshape(-1, 2)
        dst = np.asarray(target_array, dtype=np.float64).reshape(-1, 2)
        pred = self.pixel_to_map(src)
        return np.linalg.norm(pred - dst, axis=1)