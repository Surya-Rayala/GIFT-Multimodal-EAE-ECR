"""Media-source classification and output naming.

A ``.vmeta.xml`` ``technical/location`` is either a video file path (relative
to the vmeta's folder, or absolute) or a live-stream URL. GIFT's streaming
integration rewrites the location of a copied ``stream-*.vmeta.xml`` to e.g.
``udp://127.0.0.1:1234?pkt_size=1316``; live GoPro feeds use other hosts,
ports or protocols, so detection keys on the *shape* of a URL (a scheme
followed by ``:``), never on one fixed address.

Output artifacts are named from the vmeta ``title`` (see
:func:`safe_basename`) — never from the location, which for a stream is a URL
and would produce names like ``1234?pkt_size=1316_Analysis.json``.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Optional
from urllib.parse import unquote, urlsplit, urlunsplit

# Protocols ffmpeg can ingest live. A location with one of these schemes is a
# stream even when written without ``//`` (e.g. ``udp:127.0.0.1:1234``); any
# other ``scheme://...`` is also treated as a stream URL and handed to ffmpeg
# as-is. ``file:`` is the one URL that names a local file.
STREAM_SCHEMES = frozenset({
    "udp", "rtp", "srtp", "rtsp", "rtsps", "rtmp", "rtmps", "rtmpt", "rtmpe",
    "srt", "rist", "tcp", "tls", "sctp", "http", "https", "hls", "mms",
    "mmsh", "mmst", "zmq", "ftp", "sftp",
})

_SCHEME_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9+.\-]*):")


def _scheme_of(location: str) -> Optional[str]:
    m = _SCHEME_RE.match(location or "")
    if not m:
        return None
    scheme = m.group(1).lower()
    # A one-letter "scheme" is a Windows drive letter (C:\videos\a.mp4).
    if len(scheme) < 2:
        return None
    return scheme


def is_stream_url(location: Optional[str]) -> bool:
    """True when ``location`` names a live/network stream rather than a file.

    Matches any known streaming protocol (``udp:``, ``rtsp:``, ``srt:``,
    ``http(s):`` …) and, more generally, any ``scheme://`` URL other than
    ``file://``. Windows drive paths (``C:\\…``) and plain relative/absolute
    paths are files.
    """
    if not isinstance(location, str):
        return False
    scheme = _scheme_of(location)
    if scheme is None or scheme == "file":
        return False
    rest = location.strip()[len(scheme) + 1:]
    return scheme in STREAM_SCHEMES or rest.startswith("//")


def file_url_to_path(location: str) -> str:
    """``file:///abs/path`` → ``/abs/path`` (other strings pass through)."""
    if _scheme_of(location) != "file":
        return location
    parts = urlsplit(location.strip())
    path = unquote(parts.path or "")
    if os.name == "nt" and re.match(r"^/[A-Za-z]:", path):
        path = path[1:]
    if parts.netloc and parts.netloc not in ("", "localhost"):
        path = f"//{parts.netloc}{path}"  # UNC share
    return path


def redact_url(url: Optional[str]) -> Optional[str]:
    """Strip credentials (``user:pass@``) from a URL for logs and artifacts."""
    if not isinstance(url, str) or "@" not in url:
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.netloc or "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, f"***@{host}", parts.path, parts.query, parts.fragment))


# Device names Windows refuses as file stems (case-insensitive).
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
# Path separators, Windows-reserved punctuation, control characters, and ``%``
# (OpenCV's writer treats ``%0Nd`` in a filename as an image-sequence pattern).
_UNSAFE_CHARS_RE = re.compile(r'[<>:"/\\|?*%\x00-\x1f\x7f]+')


def safe_basename(text: Optional[str], *, fallback: str = "untitled", max_len: int = 120) -> str:
    """Filesystem-safe artifact stem derived from a human title.

    Keeps the title readable (letters, digits, unicode, ``-``, ``.``, ``_``
    survive) while removing everything that is unsafe on Windows, macOS or
    Linux: path separators and reserved punctuation become ``_``, whitespace
    runs become ``_``, leading/trailing dots and underscores are trimmed, and
    Windows device names (``CON``, ``NUL`` …) are prefixed. Never returns an
    empty string.
    """
    s = unicodedata.normalize("NFKC", str(text or "")).strip()
    s = _UNSAFE_CHARS_RE.sub("_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_{2,}", "_", s)
    s = s.strip(" ._")
    if not s:
        s = fallback
    if s.split(".")[0].upper() in _WINDOWS_RESERVED:
        s = f"_{s}"
    s = s[:max_len].rstrip(" ._")
    return s or fallback


def vmeta_stem(vmeta_path: str) -> str:
    """``.../stream-run3.vmeta.xml`` → ``run3``.

    Fallback output name when a stream vmeta has no title: the vmeta file name
    without its ``.vmeta.xml`` suffix and without the ``stream-`` prefix GIFT
    adds to the copies it generates for streaming.
    """
    name = os.path.basename(vmeta_path or "")
    for suffix in (".vmeta.xml", ".xml"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name.lower().startswith("stream-") and len(name) > len("stream-"):
        name = name[len("stream-"):]
    return name
