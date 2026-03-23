from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .offset_scenario import estimate_focal_length_px, _DEFAULT_HFOV_DEG
from .results import Detection, DetectionType


@dataclass
class AutoFollowResult:
    """Result of a single auto-follow computation.

    Attributes
    ----------
    position_vector:
        ``(dx, dy)`` pixel displacement from the current marker position
        to the frame centre.  Positive ``dx`` means the camera must move
        *right*; positive ``dy`` means it must move *down*.
        ``(0.0, 0.0)`` when no target marker is found.
    yaw:
        Estimated yaw angle (degrees) from the camera's optical axis to
        the target marker.  Positive means the marker is to the right of
        the camera centre.  ``0.0`` when no target is found.
    target_marker_id:
        Identifier of the marker currently being followed, or ``None``.
    target_found:
        ``True`` if the target marker was found in the current frame.
    visible_marker_ids:
        List of all visible AprilTag marker IDs in the current frame.
    marker_positions:
        Mapping from marker identifier to its ``(x, y)`` pixel centre.
    predicted_position:
        Predicted target position used when the marker is temporarily not
        visible. ``None`` means no prediction was available.
    tracking_state:
        ``"detected"`` for a direct observation, ``"predicted"`` when a
        short-term prediction was used, or ``"lost"`` when there is no
        valid target estimate.
    frames_since_seen:
        Number of consecutive frames since the target was last directly
        observed. ``0`` means the marker was detected in the current frame.
    compensated_yaw:
        Yaw corrected by subtracting the measured head / camera yaw angle.
        This allows upstream control to suppress periodic head sway.
    """

    position_vector: Tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0
    target_marker_id: Optional[str] = None
    target_found: bool = False
    visible_marker_ids: List[str] = field(default_factory=list)
    marker_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    predicted_position: Optional[Tuple[float, float]] = None
    tracking_state: str = "lost"
    frames_since_seen: int = 0
    compensated_yaw: float = 0.0


def compute_follow_vector(
    detections: List[Detection],
    frame_width: int = 640,
    frame_height: int = 480,
    target_marker_id: Optional[str] = None,
    hfov_deg: float = _DEFAULT_HFOV_DEG,
) -> AutoFollowResult:
    """Compute the follow vector for a given set of detections."""
    marker_positions: Dict[str, Tuple[int, int]] = {}
    for d in detections:
        if d.detection_type == DetectionType.APRIL_TAG and d.identifier is not None:
            marker_positions[d.identifier] = d.center

    visible_ids = sorted(marker_positions.keys())

    if not visible_ids:
        return AutoFollowResult(
            visible_marker_ids=visible_ids,
            marker_positions=marker_positions,
        )

    if target_marker_id is not None and target_marker_id in marker_positions:
        chosen_id = target_marker_id
    else:
        chosen_id = visible_ids[0]

    mx, my = marker_positions[chosen_id]
    cx = frame_width / 2.0
    cy = frame_height / 2.0
    dx = cx - mx
    dy = cy - my
    focal = estimate_focal_length_px(frame_width, hfov_deg)
    yaw = math.degrees(math.atan2(float(mx - cx), focal))

    return AutoFollowResult(
        position_vector=(dx, dy),
        yaw=yaw,
        target_marker_id=chosen_id,
        target_found=True,
        visible_marker_ids=visible_ids,
        marker_positions=marker_positions,
        predicted_position=(float(mx), float(my)),
        tracking_state="detected",
        frames_since_seen=0,
        compensated_yaw=yaw,
    )


class AutoFollowScenario:
    """Continuous auto-follow scenario with temporal filtering.

    Besides frame-by-frame AprilTag centering, this wrapper adds two
    practical countermeasures against head sway and motion blur:

    * exponential smoothing of the target position to suppress control
      jitter caused by oscillating head motion,
    * short-term constant-velocity prediction so that a target can be
      maintained across a few missed frames.

    If the caller can provide the current head/camera yaw angle, the
    scenario also returns a yaw value compensated for this oscillation.
    """

    def __init__(
        self,
        camera: object,
        detector: object,
        frame_width: int = 640,
        frame_height: int = 480,
        target_marker_id: Optional[str] = None,
        smoothing: float = 0.35,
        prediction_horizon_frames: int = 4,
        hfov_deg: float = _DEFAULT_HFOV_DEG,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._target_marker_id = target_marker_id
        self._smoothing = min(1.0, max(0.0, float(smoothing)))
        self._prediction_horizon_frames = max(0, int(prediction_horizon_frames))
        self._hfov_deg = float(hfov_deg)
        self._last_target_position: Optional[Tuple[float, float]] = None
        self._last_target_id: Optional[str] = target_marker_id
        self._last_velocity: Tuple[float, float] = (0.0, 0.0)
        self._frames_since_seen = 0
        self._last_timestamp: Optional[float] = None

    @property
    def target_marker_id(self) -> Optional[str]:
        return self._target_marker_id

    @target_marker_id.setter
    def target_marker_id(self, value: Optional[str]) -> None:
        self._target_marker_id = value
        self._last_target_position = None
        self._last_target_id = value
        self._last_velocity = (0.0, 0.0)
        self._frames_since_seen = 0
        self._last_timestamp = None

    def _yaw_from_x(self, x_pos: float) -> float:
        focal = estimate_focal_length_px(self.frame_width, self._hfov_deg)
        cx = self.frame_width / 2.0
        return math.degrees(math.atan2(float(x_pos - cx), focal))

    def _result_from_position(
        self,
        position: Tuple[float, float],
        base: AutoFollowResult,
        target_found: bool,
        tracking_state: str,
        camera_yaw_deg: float,
    ) -> AutoFollowResult:
        cx = self.frame_width / 2.0
        cy = self.frame_height / 2.0
        px, py = position
        yaw = self._yaw_from_x(px)
        return AutoFollowResult(
            position_vector=(cx - px, cy - py),
            yaw=yaw,
            target_marker_id=base.target_marker_id,
            target_found=target_found,
            visible_marker_ids=base.visible_marker_ids,
            marker_positions=base.marker_positions,
            predicted_position=(float(px), float(py)),
            tracking_state=tracking_state,
            frames_since_seen=self._frames_since_seen,
            compensated_yaw=yaw - camera_yaw_deg,
        )

    def compute_from_detections(
        self,
        detections: List[Detection],
        *,
        timestamp: Optional[float] = None,
        camera_yaw_deg: float = 0.0,
    ) -> AutoFollowResult:
        """Compute the follow vector from already-obtained detections.

        Parameters
        ----------
        detections:
            Detections already computed for the current frame.
        timestamp:
            Optional monotonic timestamp in seconds. When omitted,
            ``time.monotonic()`` is used.
        camera_yaw_deg:
            Measured head/camera yaw angle in degrees. The returned
            ``compensated_yaw`` subtracts this angle from the optical yaw.
        """
        now = time.monotonic() if timestamp is None else float(timestamp)
        dt = 0.0 if self._last_timestamp is None else max(1e-6, now - self._last_timestamp)
        self._last_timestamp = now

        current = compute_follow_vector(
            detections,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            target_marker_id=self._target_marker_id,
            hfov_deg=self._hfov_deg,
        )

        if current.target_found and current.predicted_position is not None:
            measured = current.predicted_position
            if self._last_target_position is None:
                filtered = measured
                velocity = (0.0, 0.0)
            else:
                prev_x, prev_y = self._last_target_position
                mx, my = measured
                alpha = self._smoothing
                filtered = (
                    (1.0 - alpha) * prev_x + alpha * mx,
                    (1.0 - alpha) * prev_y + alpha * my,
                )
                velocity = (
                    (filtered[0] - prev_x) / dt,
                    (filtered[1] - prev_y) / dt,
                )
            self._last_target_position = filtered
            self._last_target_id = current.target_marker_id
            self._last_velocity = velocity
            self._frames_since_seen = 0
            return self._result_from_position(
                filtered,
                current,
                target_found=True,
                tracking_state="detected",
                camera_yaw_deg=camera_yaw_deg,
            )

        if self._last_target_position is not None and self._frames_since_seen < self._prediction_horizon_frames:
            self._frames_since_seen += 1
            px = self._last_target_position[0] + self._last_velocity[0] * dt
            py = self._last_target_position[1] + self._last_velocity[1] * dt
            predicted = (px, py)
            self._last_target_position = predicted
            current.target_marker_id = self._target_marker_id or self._last_target_id or current.target_marker_id
            return self._result_from_position(
                predicted,
                current,
                target_found=False,
                tracking_state="predicted",
                camera_yaw_deg=camera_yaw_deg,
            )

        self._frames_since_seen = self._prediction_horizon_frames if self._last_target_position is not None else 0
        self._last_target_position = None
        self._last_target_id = self._target_marker_id
        self._last_velocity = (0.0, 0.0)
        current.compensated_yaw = current.yaw - camera_yaw_deg
        current.frames_since_seen = self._frames_since_seen
        return current

    def compute_current(self) -> AutoFollowResult:
        frame = self.camera.read()  # type: ignore[attr-defined]
        if frame is None:
            raise RuntimeError("Camera returned no frame.")
        detections = self.detector.process_frame(frame)  # type: ignore[attr-defined]
        return self.compute_from_detections(detections)
