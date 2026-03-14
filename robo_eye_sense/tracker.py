"""Centroid-based multi-object tracker.

Labeled objects (AprilTags, QR codes) are matched by their
``(detection_type, identifier)`` pair so that a physical marker always
keeps the same track ID even after a brief occlusion.

Unlabeled objects (laser spots) are matched by nearest-centroid with a
configurable maximum pixel distance.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from .results import Detection


class CentroidTracker:
    """Assigns persistent track IDs to detected objects across video frames.

    Parameters
    ----------
    max_disappeared:
        Number of consecutive frames a track may be absent before it is
        removed from the active set.
    max_distance:
        Maximum pixel distance allowed when centroid-matching unlabeled
        objects (e.g. laser spots).
    """

    def __init__(self, max_disappeared: int = 10, max_distance: int = 50) -> None:
        self._next_id: int = 0
        # (DetectionType, identifier) -> track_id  (for labeled objects)
        self._labeled_tracks: Dict[Tuple, int] = {}
        # track_id -> last-known centre  (for unlabeled objects)
        self._unlabeled_tracks: Dict[int, Tuple[int, int]] = {}
        # track_id -> consecutive frames without a match
        self._disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Assign ``track_id`` to every detection in *detections* in-place.

        Parameters
        ----------
        detections:
            List of :class:`~robo_eye_sense.results.Detection` objects from
            the current frame.  The ``track_id`` attribute of each is set
            before returning.

        Returns
        -------
        List[Detection]
            The same list with ``track_id`` values populated.
        """
        labeled = [d for d in detections if d.identifier is not None]
        unlabeled = [d for d in detections if d.identifier is None]
        self._update_labeled(labeled)
        self._update_unlabeled(unlabeled)
        return detections

    @property
    def active_track_count(self) -> int:
        """Total number of currently active tracks (labeled + unlabeled)."""
        return len(self._labeled_tracks) + len(self._unlabeled_tracks)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _new_id(self) -> int:
        track_id = self._next_id
        self._next_id += 1
        return track_id

    def _update_labeled(self, detections: List[Detection]) -> None:
        seen_keys: Set[Tuple] = set()
        for d in detections:
            key = (d.detection_type, d.identifier)
            seen_keys.add(key)
            if key in self._labeled_tracks:
                d.track_id = self._labeled_tracks[key]
            else:
                d.track_id = self._new_id()
                self._labeled_tracks[key] = d.track_id
            self._disappeared[d.track_id] = 0  # reset counter on each match

        # Increment disappeared counters for absent labeled tracks
        for key, track_id in list(self._labeled_tracks.items()):
            if key not in seen_keys:
                self._disappeared[track_id] = self._disappeared.get(track_id, 0) + 1
                if self._disappeared[track_id] > self.max_disappeared:
                    del self._labeled_tracks[key]
                    self._disappeared.pop(track_id, None)

    def _update_unlabeled(self, detections: List[Detection]) -> None:
        if not detections:
            # No detections this frame – age all existing unlabeled tracks
            for track_id in list(self._unlabeled_tracks.keys()):
                self._disappeared[track_id] = (
                    self._disappeared.get(track_id, 0) + 1
                )
                if self._disappeared[track_id] > self.max_disappeared:
                    del self._unlabeled_tracks[track_id]
                    self._disappeared.pop(track_id, None)
            return

        if not self._unlabeled_tracks:
            # No existing tracks – register every detection as a new track
            for d in detections:
                d.track_id = self._new_id()
                self._unlabeled_tracks[d.track_id] = d.center
                self._disappeared[d.track_id] = 0
            return

        track_ids = list(self._unlabeled_tracks.keys())
        track_centers = [self._unlabeled_tracks[tid] for tid in track_ids]
        input_centers = [d.center for d in detections]

        # Build Euclidean distance matrix  (tracks × detections)
        D = np.array(
            [
                [np.hypot(tc[0] - ic[0], tc[1] - ic[1]) for ic in input_centers]
                for tc in track_centers
            ],
            dtype=float,
        )

        used_rows: Set[int] = set()
        used_cols: Set[int] = set()

        # Greedy assignment: iterate pairs from nearest to farthest
        for row, col in sorted(
            ((r, c) for r in range(D.shape[0]) for c in range(D.shape[1])),
            key=lambda rc: D[rc[0], rc[1]],
        ):
            if D[row, col] > self.max_distance:
                break
            if row in used_rows or col in used_cols:
                continue
            track_id = track_ids[row]
            detections[col].track_id = track_id
            self._unlabeled_tracks[track_id] = detections[col].center
            self._disappeared[track_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Age unmatched existing tracks
        for row in set(range(len(track_ids))) - used_rows:
            track_id = track_ids[row]
            self._disappeared[track_id] = (
                self._disappeared.get(track_id, 0) + 1
            )
            if self._disappeared[track_id] > self.max_disappeared:
                del self._unlabeled_tracks[track_id]
                self._disappeared.pop(track_id, None)

        # Register new tracks for unmatched detections
        for col in set(range(len(detections))) - used_cols:
            track_id = self._new_id()
            detections[col].track_id = track_id
            self._unlabeled_tracks[track_id] = detections[col].center
            self._disappeared[track_id] = 0
