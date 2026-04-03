from typing import Iterable, Optional

import numpy as np
from shapely.geometry import Point, Polygon


def calculate_wall_polygon(bounds: np.ndarray, p_wall: float = 0.2) -> np.ndarray:
    """
    Build an inner wall polygon from 4 boundary corner points.

    Expected point order:
        top-left, top-right, bottom-right, bottom-left
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (4, 2):
        raise ValueError("bounds must have shape (4, 2)")

    dist_top = abs(bounds[0, 0] - bounds[1, 0])
    dist_bottom = abs(bounds[2, 0] - bounds[3, 0])
    dist_left = abs(bounds[0, 1] - bounds[3, 1])
    dist_right = abs(bounds[1, 1] - bounds[2, 1])

    tl = (
        int(bounds[0, 0] + p_wall * dist_top),
        int(bounds[0, 1] + p_wall * dist_left),
    )
    tr = (
        int(bounds[1, 0] - p_wall * dist_top),
        int(bounds[1, 1] + p_wall * dist_right),
    )
    br = (
        int(bounds[2, 0] - p_wall * dist_bottom),
        int(bounds[2, 1] - p_wall * dist_right),
    )
    bl = (
        int(bounds[3, 0] + p_wall * dist_bottom),
        int(bounds[3, 1] - p_wall * dist_left),
    )

    return np.array([tl, tr, br, bl], dtype=int)


def buffer_shapely_polygon(
    poly: Polygon,
    factor: float = 0.2,
    swell: bool = False,
    *,
    distance_px: Optional[float] = None,
):
    """
    Resize a shapely polygon using buffer().

    Preferred mode:
        distance_px -> explicit distance in polygon coordinate units.

    Backward-compatible mode:
        factor -> distance derived from the polygon bounding box.
    """
    if poly is None or poly.is_empty:
        return poly

    if distance_px is not None:
        distance = abs(float(distance_px))
    else:
        xs = list(poly.exterior.coords.xy[0])
        ys = list(poly.exterior.coords.xy[1])

        x_center = 0.5 * (min(xs) + max(xs))
        y_center = 0.5 * (min(ys) + max(ys))

        min_corner = Point(min(xs), min(ys))
        center = Point(x_center, y_center)
        distance = center.distance(min_corner) * float(factor)

    return poly.buffer(distance if swell else -distance)


def non_null_len(iterable: Iterable) -> int:
    return sum(1 for item in iterable if item is not None)


def arg_first_non_null(iterable: Iterable) -> int:
    for i, item in enumerate(iterable):
        if item is not None:
            return i
    return 10**9


def len_comparator(item1, item2) -> int:
    return non_null_len(item1) - non_null_len(item2)


def arg_first_comparator(item1, item2) -> int:
    return arg_first_non_null(item1) - arg_first_non_null(item2)


def get_odd(v: int) -> int:
    return v - 1 if v % 2 == 0 else v