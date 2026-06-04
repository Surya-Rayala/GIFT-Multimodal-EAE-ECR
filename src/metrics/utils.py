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


# ----------------------------------------------------------------------
# Text rendering for compare-image legends.
#
# OpenCV's ``cv2.putText`` uses the Hershey stroke fonts — small vector
# strokes rasterised at every pixel. Even with ``cv2.LINE_AA`` they read as
# wavy / hand-drawn next to real typed text. These helpers route legend
# text through Pillow's TrueType renderer instead, which produces crisp
# kerned glyphs identical to system text. The cost is one BGR ↔ PIL Image
# round-trip per call; for the small handful of legend strings per artifact
# it's negligible.
# ----------------------------------------------------------------------


import os as _os

import cv2 as _cv2  # type: ignore[import-untyped]


def _candidate_font_paths() -> list[str]:
    """Search order for a TrueType font: macOS, Linux, Windows, then the
    DejaVu Sans copy that matplotlib bundles (always available since
    matplotlib is a project dependency)."""
    paths = [
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    # matplotlib's bundled DejaVu Sans — last-resort but always present.
    try:
        import matplotlib

        paths.append(
            _os.path.join(
                _os.path.dirname(matplotlib.__file__),
                "mpl-data", "fonts", "ttf", "DejaVuSans.ttf",
            )
        )
    except Exception:
        pass
    return paths


_FONT_CACHE: dict = {}


def get_legend_font(size_px: int):
    """Return a PIL ImageFont sized in pixels. Cached per size."""
    from PIL import ImageFont

    if size_px in _FONT_CACHE:
        return _FONT_CACHE[size_px]
    for path in _candidate_font_paths():
        if _os.path.isfile(path):
            try:
                font = ImageFont.truetype(path, size_px)
                _FONT_CACHE[size_px] = font
                return font
            except Exception:
                continue
    # Absolute fallback — PIL's stylised default. Should never trigger
    # because matplotlib's font is always available.
    font = ImageFont.load_default()
    _FONT_CACHE[size_px] = font
    return font


def draw_text(
    img_bgr,
    text: str,
    xy: tuple[int, int],
    *,
    size_px: int = 14,
    fill_bgr: tuple[int, int, int] = (255, 255, 255),
    halo_bgr: Optional[tuple[int, int, int]] = None,
) -> None:
    """Render ``text`` onto a cv2 BGR image at ``xy`` using a TrueType font.

    Mutates ``img_bgr`` in place. ``halo_bgr`` if given draws a 1-pixel
    outline behind the text (useful when text sits on a busy background).
    Coordinate convention matches ``cv2.putText`` baseline-ish positioning:
    PIL's text() draws from the top-left of the glyph box, so callers that
    were placing text with cv2 may want to shift ``y`` up by ~size_px.
    """
    from PIL import Image, ImageDraw
    import numpy as _np

    pil = Image.fromarray(_cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = get_legend_font(size_px)

    fill_rgb = (int(fill_bgr[2]), int(fill_bgr[1]), int(fill_bgr[0]))
    if halo_bgr is not None:
        halo_rgb = (int(halo_bgr[2]), int(halo_bgr[1]), int(halo_bgr[0]))
        for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            draw.text((xy[0] + dx, xy[1] + dy), text, font=font, fill=halo_rgb)
    draw.text(xy, text, font=font, fill=fill_rgb)

    img_bgr[:] = _cv2.cvtColor(_np.array(pil), _cv2.COLOR_RGB2BGR)


def text_size(text: str, *, size_px: int = 14) -> tuple[int, int]:
    """Width × height of a text string at the given size, in pixels.

    Used by callers that need to size legend panels around the text.
    Returns ``(w, h)`` based on PIL's ``getbbox``.
    """
    font = get_legend_font(size_px)
    # PIL's textbbox returns (l, t, r, b); width = r-l, height = b-t.
    bbox = font.getbbox(text)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])