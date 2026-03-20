"""Helpers for importing OpenCV lazily.

This project supports headless environments where importing ``cv2`` may fail
because GUI system libraries are absent.  Modules that only need OpenCV during
specific runtime paths should use :func:`get_cv2` instead of importing ``cv2``
at module import time.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional


def get_cv2(required: bool = True) -> Optional[ModuleType]:
    """Return the imported ``cv2`` module.

    Parameters
    ----------
    required:
        When ``True`` (default), re-raise the import error.  When ``False``,
        return ``None`` instead so callers can keep non-OpenCV code paths
        importable in headless test environments.
    """
    try:
        return importlib.import_module("cv2")
    except ImportError:
        if required:
            raise
        return None
