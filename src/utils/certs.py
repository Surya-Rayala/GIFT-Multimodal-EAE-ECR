"""Verify HTTPS with certifi's CA bundle instead of the OS certificate store.

The Windows store can hold a malformed entry that breaks every urllib /
torch.hub download with ``ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]``; certifi is
the bundle requests/HuggingFace already use, so it's the consistent choice.
"""
from __future__ import annotations

_installed = False


def install_certifi_https() -> None:
    """Route this process's HTTPS through certifi. Idempotent; no-op if certifi
    can't be imported (urllib then keeps its default context)."""
    global _installed
    if _installed:
        return
    try:
        import ssl
        import certifi

        ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(
            *a, cafile=certifi.where(), **k
        )
        _installed = True
    except Exception:
        pass
