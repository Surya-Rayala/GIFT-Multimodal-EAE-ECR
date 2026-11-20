from xml.etree import ElementTree
import os
import json
import numpy as np
from shapely.geometry import Polygon
import cv2

from .media_source import file_url_to_path, is_stream_url, vmeta_stem
from .paths import resolve_all_config_paths


def load_vmeta(path):
    """Parse a ``.vmeta.xml`` → ``(cfg_file, video, start_time, role, title)``.

    ``video`` is either a file path (resolved against the vmeta's folder) or,
    when ``technical/location`` holds a stream URL (``udp://…``, ``rtsp://…``
    — see :func:`src.utils.media_source.is_stream_url`), the URL itself,
    untouched: a URL must never be joined onto the course folder.
    """
    ns = "{http://ltsc.ieee.org/xsd/LOM}"
    tree = ElementTree.parse(path)
    root = tree.getroot()

    # Find video file name at technical/location
    technical = root.find("{}technical".format(ns))
    if technical is None:
        raise SyntaxError("Missing technical field in vmeta file.")
    location = technical.find("{}location".format(ns))
    if location is None:
        raise SyntaxError("Missing location attribute in technical field of vmeta file.")
    video_file = (location.text or "").strip()
    if not video_file:
        raise SyntaxError("Empty location in technical field of vmeta file.")
    is_stream = is_stream_url(video_file)
    if not is_stream:
        video_file = file_url_to_path(video_file)

    # Find room cfg file and start time
    general = root.find("{}general".format(ns))
    if general is None:
        raise SyntaxError("Missing general field in vmeta file.")
    cfg_file = None
    start_time = None
    role = None
    title = None
    for identifier in general.findall("{}identifier".format(ns)):
        catalog = identifier.find("{}catalog".format(ns))
        entry = identifier.find("{}entry".format(ns))
        if catalog is not None and entry is not None:
            if catalog.text == "start_time":
                start_time = entry.text
            elif catalog.text == "space_metadata_file":
                cfg_file = entry.text
            elif catalog.text == "role":
                role = (entry.text or "").strip().lower()
            elif catalog.text == "title":
                title = (entry.text or "").strip()
    if start_time is None:
        raise SyntaxError("Unable to find start_time in vmeta file.")
    if cfg_file is None:
        raise SyntaxError("Unable to find space_metadata_file in vmeta file.")

    if role is None:
        role = "trainee"
    elif role not in ("expert", "trainee"):
        raise SyntaxError(
            f"Invalid role {role!r} in vmeta file. Must be 'expert' or 'trainee'."
        )

    if not title:
        # Outputs are named from the title. Without one, a file keeps its
        # video stem; a stream URL has no meaningful stem, so use the vmeta
        # file name instead (minus GIFT's ``stream-`` copy prefix).
        title = (vmeta_stem(path) if is_stream
                 else os.path.splitext(os.path.basename(video_file))[0])

    # Format full paths based on vmeta path
    dir_name = os.path.dirname(path)
    cfg_file = os.path.join(dir_name, cfg_file)
    if not is_stream:
        video_file = os.path.join(dir_name, video_file)
    return cfg_file, video_file, int(start_time), role, title

def load_config(path):
    with open(path) as config_file:
        config = json.load(config_file)

    # Resolve every known path key (config-anchored + project-anchored) to
    # absolute, normalized form. Single source of truth lives in
    # ``src/utils/paths.py``. Downstream consumers can use the resolved
    # values directly with no further path arithmetic.
    resolve_all_config_paths(config, path)

    boundary = np.array(config["Boundary"]).astype("int")
    pods = np.array(config["POD"]).astype("int")
    map_image = cv2.imread(config["MapPath"]) if config.get("MapPath") else None

    config["Boundary"] = Polygon(boundary)
    config["POD"] = pods
    config["Map Image"] = map_image

    return config