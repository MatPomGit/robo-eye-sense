"""Abstract base class for all detector implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .results import Detection


class BaseDetector(ABC):
    """Abstract base class for all detector implementations.

    Every concrete detector (AprilTag, QR-code, laser-spot, or any future
    detector) must inherit from this class and implement :meth:`detect` and
    :meth:`get_name`.

    This base class ensures a uniform interface, making it straightforward
    to add new detector types to the pipeline without modifying the
    orchestrator.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on *frame* and return found objects.

        Parameters
        ----------
        frame:
            Input image as a NumPy array.  The expected format depends on
            the concrete detector (e.g. grayscale for AprilTags, BGR for
            QR-codes and laser spots).

        Returns
        -------
        List[Detection]
            Zero or more detections found in the frame.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return a short human-readable name for this detector.

        Returns
        -------
        str
            E.g. ``"AprilTag"``, ``"QRCode"``, ``"LaserSpot"``.
        """
        ...
