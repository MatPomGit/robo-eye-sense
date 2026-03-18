"""Centroid-based multi-object tracker.

Labeled objects (AprilTags, QR codes) are matched by their
``(detection_type, identifier)`` pair so that a physical marker always
keeps the same track ID even after a brief occlusion.

Unlabeled objects (laser spots) are matched by nearest-centroid with a
configurable maximum pixel distance.  When *use_kalman* is enabled every
unlabeled track maintains a 4-state Kalman filter
(position + velocity: ``[x, y, vx, vy]``) so that the tracker can
predict where a fast-moving or temporarily blurred object will reappear
and match it even if the detection centroid has shifted noticeably.
This Kalman mode is automatically activated in ROBUST pipeline mode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .results import Detection


def _make_kalman(center: Tuple[int, int]) -> Any:
    """Return a new Kalman filter initialised at *center*.

    Requires OpenCV (cv2) and is called only when Kalman tracking is enabled.

    State vector:  ``[x, y, vx, vy]``
    Measurement:   ``[x, y]``
    Model:         constant-velocity (velocity is propagated unchanged).
    """
    import cv2

    kf = cv2.KalmanFilter(4, 2)

    # H – measurement matrix: observe x and y only
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
    )

    # F – transition matrix: constant-velocity model
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], dtype=np.float32
    )

    # Q – process noise: modest acceleration uncertainty
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05

    # R – measurement noise: moderate trust in (potentially blurry) centres
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0

    # P – initial posterior error covariance
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    # Initialise state at the given centre with zero velocity
    kf.statePost = np.array(
        [[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32
    )
    return kf


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
    use_kalman:
        When ``True`` each unlabeled track maintains a Kalman filter that
        predicts the object's next position using a constant-velocity model.
        Matching is performed against the *predicted* position rather than
        the last observed centroid, which makes the tracker robust to
        motion blur and temporary detection failures.
    """

    def __init__(
        self,
        max_disappeared: int = 10,
        max_distance: int = 50,
        use_kalman: bool = False,
    ) -> None:
        self._next_id: int = 0
        # (DetectionType, identifier) -> track_id  (for labeled objects)
        self._labeled_tracks: Dict[Tuple, int] = {}
        # track_id -> last-known center  (for unlabeled objects)
        self._unlabeled_tracks: Dict[int, Tuple[int, int]] = {}
        # track_id -> consecutive frames without a match
        self._disappeared: Dict[int, int] = {}
        # track_id -> KalmanFilter  (populated when use_kalman is True)
        self._kalman_filters: Dict[int, Any] = {}

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self._use_kalman = use_kalman

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def use_kalman(self) -> bool:
        """Whether Kalman-filter prediction is active for unlabeled tracks."""
        return self._use_kalman

    @use_kalman.setter
    def use_kalman(self, value: bool) -> None:
        if value and not self._use_kalman:
            # Create Kalman filters for all currently active tracks
            for tid, center in self._unlabeled_tracks.items():
                if tid not in self._kalman_filters:
                    self._kalman_filters[tid] = _make_kalman(center)
        elif not value:
            self._kalman_filters.clear()
        self._use_kalman = value

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

    def _register_unlabeled(self, d: Detection) -> int:
        """Register *d* as a new unlabeled track and return its track ID."""
        track_id = self._new_id()
        d.track_id = track_id
        self._unlabeled_tracks[track_id] = d.center
        self._disappeared[track_id] = 0
        if self._use_kalman:
            self._kalman_filters[track_id] = _make_kalman(d.center)
        return track_id

    def _update_unlabeled(self, detections: List[Detection]) -> None:
        if self._use_kalman:
            self._update_unlabeled_kalman(detections)
        else:
            self._update_unlabeled_centroid(detections)

    # ---- original centroid matching ----------------------------------

    def _update_unlabeled_centroid(self, detections: List[Detection]) -> None:
        if not detections:
            for track_id in list(self._unlabeled_tracks.keys()):
                self._disappeared[track_id] = (
                    self._disappeared.get(track_id, 0) + 1
                )
                if self._disappeared[track_id] > self.max_disappeared:
                    del self._unlabeled_tracks[track_id]
                    self._disappeared.pop(track_id, None)
            return

        if not self._unlabeled_tracks:
            for d in detections:
                self._register_unlabeled(d)
            return

        track_ids = list(self._unlabeled_tracks.keys())
        track_centers = [self._unlabeled_tracks[tid] for tid in track_ids]
        input_centers = [d.center for d in detections]

        D = np.array(
            [
                [np.hypot(tc[0] - ic[0], tc[1] - ic[1]) for ic in input_centers]
                for tc in track_centers
            ],
            dtype=float,
        )

        used_rows: Set[int] = set()
        used_cols: Set[int] = set()

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

        for row in set(range(len(track_ids))) - used_rows:
            track_id = track_ids[row]
            self._disappeared[track_id] = (
                self._disappeared.get(track_id, 0) + 1
            )
            if self._disappeared[track_id] > self.max_disappeared:
                del self._unlabeled_tracks[track_id]
                self._disappeared.pop(track_id, None)

        for col in set(range(len(detections))) - used_cols:
            self._register_unlabeled(detections[col])

    # ---- Kalman-filter matching --------------------------------------

    def _update_unlabeled_kalman(self, detections: List[Detection]) -> None:
        """Kalman-filter variant of the unlabeled update step.

        For each existing track the filter predicts the object's next
        position.  Matching is performed against predicted positions (not
        last observed centres), which tolerates larger inter-frame
        displacements caused by fast motion or motion blur.
        """
        # Step 1 – predict next position for every existing track
        predicted: Dict[int, Tuple[int, int]] = {}
        for tid in list(self._unlabeled_tracks.keys()):
            kf = self._kalman_filters.get(tid)
            if kf is None:
                kf = _make_kalman(self._unlabeled_tracks[tid])
                self._kalman_filters[tid] = kf
            pred = kf.predict()
            predicted[tid] = (int(pred[0, 0]), int(pred[1, 0]))

        if not detections:
            # No detections this frame – age all tracks; predictions already ran
            for track_id in list(self._unlabeled_tracks.keys()):
                self._disappeared[track_id] = (
                    self._disappeared.get(track_id, 0) + 1
                )
                if self._disappeared[track_id] > self.max_disappeared:
                    del self._unlabeled_tracks[track_id]
                    self._kalman_filters.pop(track_id, None)
                    self._disappeared.pop(track_id, None)
            return

        if not self._unlabeled_tracks:
            for d in detections:
                self._register_unlabeled(d)
            return

        track_ids = list(predicted.keys())
        pred_centers = [predicted[tid] for tid in track_ids]
        input_centers = [d.center for d in detections]

        # Distance matrix: predicted positions vs measured positions
        D = np.array(
            [
                [np.hypot(pc[0] - ic[0], pc[1] - ic[1]) for ic in input_centers]
                for pc in pred_centers
            ],
            dtype=float,
        )

        used_rows: Set[int] = set()
        used_cols: Set[int] = set()

        for row, col in sorted(
            ((r, c) for r in range(D.shape[0]) for c in range(D.shape[1])),
            key=lambda rc: D[rc[0], rc[1]],
        ):
            if D[row, col] > self.max_distance:
                break
            if row in used_rows or col in used_cols:
                continue

            track_id = track_ids[row]
            measurement = np.array(
                [[detections[col].center[0]], [detections[col].center[1]]],
                dtype=np.float32,
            )
            corrected = self._kalman_filters[track_id].correct(measurement)
            # Store the Kalman-corrected position as the new track centre
            self._unlabeled_tracks[track_id] = (
                int(corrected[0, 0]),
                int(corrected[1, 0]),
            )
            detections[col].track_id = track_id
            self._disappeared[track_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Age unmatched existing tracks (prediction already done above)
        for row in set(range(len(track_ids))) - used_rows:
            track_id = track_ids[row]
            self._disappeared[track_id] = (
                self._disappeared.get(track_id, 0) + 1
            )
            if self._disappeared[track_id] > self.max_disappeared:
                del self._unlabeled_tracks[track_id]
                self._kalman_filters.pop(track_id, None)
                self._disappeared.pop(track_id, None)

        # Register new tracks for unmatched detections
        for col in set(range(len(detections))) - used_cols:
            self._register_unlabeled(detections[col])
