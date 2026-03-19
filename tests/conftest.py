"""Shared pytest fixtures for robo-vision tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def black_frame():
    """200×200 all-black BGR frame."""
    return np.zeros((200, 200, 3), dtype=np.uint8)


@pytest.fixture
def bright_spot_frame():
    """200×200 black frame with a single bright white circle at (100, 100)."""
    import cv2

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(frame, (100, 100), 6, (255, 255, 255), -1)
    return frame


@pytest.fixture
def two_spots_frame():
    """200×200 black frame with two bright spots at (50, 50) and (150, 150)."""
    import cv2

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(frame, (50, 50), 6, (255, 255, 255), -1)
    cv2.circle(frame, (150, 150), 6, (255, 255, 255), -1)
    return frame
