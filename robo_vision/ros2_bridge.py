"""ROS2 bridge for publishing detections and subscribing to configuration.

This module starts a background thread with a ROS2 node when the ``rclpy``
package is detected.  When ``rclpy`` is not available (i.e. outside of a
ROS2 workspace) the bridge silently does nothing.

Published topics
----------------
``/robo_vision/detections``
    JSON-encoded list of current detections (one message per frame update).

``/robo_vision/robot_pose``
    JSON-encoded robot pose (position + orientation, SLAM mode only).

Subscribed topics
-----------------
``/robo_vision/config``
    JSON-encoded configuration overrides (e.g. quality, detector toggles).
    Changes are applied on the next frame.

Usage::

    from robo_vision.ros2_bridge import ROS2Bridge

    bridge = ROS2Bridge()
    if bridge.available:
        bridge.start()
        # In the detection loop:
        bridge.publish_detections(detections)
        bridge.publish_robot_pose(pose)
        new_config = bridge.get_pending_config()
        # On shutdown:
        bridge.stop()
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("robo_vision.ros2_bridge")


def _rclpy_available() -> bool:
    """Return ``True`` if ``rclpy`` is importable."""
    try:
        import importlib.util
        return importlib.util.find_spec("rclpy") is not None
    except (ImportError, ModuleNotFoundError):
        return False


class ROS2Bridge:
    """Bridge between robo-vision and a ROS2 network.

    When ``rclpy`` is not installed, all methods are safe no-ops.

    Parameters
    ----------
    node_name:
        Name of the ROS2 node.  Default ``"robo_vision_bridge"``.
    """

    def __init__(self, node_name: str = "robo_vision_bridge") -> None:
        self._node_name = node_name
        self._available = _rclpy_available()
        self._node: Any = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._pending_config: Optional[Dict[str, Any]] = None
        self._config_lock = threading.Lock()

        # Publishers (initialised in start())
        self._det_pub: Any = None
        self._pose_pub: Any = None

    @property
    def available(self) -> bool:
        """``True`` when ``rclpy`` is importable and the bridge can start."""
        return self._available

    @property
    def is_running(self) -> bool:
        """``True`` while the ROS2 spinning thread is active."""
        return self._running

    def start(self) -> bool:
        """Initialise rclpy, create the node, and start the spin thread.

        Returns
        -------
        bool
            ``True`` if the bridge was started successfully.
        """
        if not self._available:
            logger.debug("rclpy not available – ROS2 bridge disabled.")
            return False

        if self._running:
            return True

        try:
            import rclpy  # type: ignore[import]
            from rclpy.node import Node  # type: ignore[import]
            from std_msgs.msg import String  # type: ignore[import]

            rclpy.init()

            self._node = Node(self._node_name)

            # Publishers
            self._det_pub = self._node.create_publisher(
                String, "/robo_vision/detections", 10
            )
            self._pose_pub = self._node.create_publisher(
                String, "/robo_vision/robot_pose", 10
            )

            # Subscriber for config overrides
            self._node.create_subscription(
                String, "/robo_vision/config", self._on_config, 10
            )

            self._running = True
            self._thread = threading.Thread(
                target=self._spin, daemon=True, name="ros2-bridge"
            )
            self._thread.start()

            logger.info("ROS2 bridge started (node: %s).", self._node_name)
            return True

        except Exception as exc:
            logger.error("Failed to start ROS2 bridge: %s", exc)
            self._running = False
            return False

    def stop(self) -> None:
        """Shut down the ROS2 node and stop the spin thread."""
        if not self._running:
            return

        self._running = False
        try:
            import rclpy  # type: ignore[import]

            if self._node is not None:
                self._node.destroy_node()
                self._node = None
            rclpy.shutdown()
        except Exception as exc:
            logger.debug("ROS2 shutdown error (non-critical): %s", exc)

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        logger.info("ROS2 bridge stopped.")

    def publish_detections(self, detections: List[Any]) -> None:
        """Publish the current detections as a JSON string message.

        Parameters
        ----------
        detections:
            List of :class:`~robo_vision.results.Detection` objects.
        """
        if not self._running or self._det_pub is None:
            return

        try:
            from std_msgs.msg import String  # type: ignore[import]

            data = [
                {
                    "type": d.detection_type.value,
                    "id": d.identifier,
                    "center": list(d.center),
                    "track_id": d.track_id,
                    "confidence": d.confidence,
                }
                for d in detections
            ]
            msg = String()
            msg.data = json.dumps(data)
            self._det_pub.publish(msg)
        except Exception:
            pass  # Non-critical – don't crash the detection loop

    def publish_robot_pose(self, pose: Any) -> None:
        """Publish the robot pose as a JSON string message.

        Parameters
        ----------
        pose:
            A :class:`~robo_vision.marker_map.RobotPose3D` instance.
        """
        if not self._running or self._pose_pub is None:
            return

        try:
            from std_msgs.msg import String  # type: ignore[import]

            data = {
                "position": list(pose.position),
                "orientation": list(pose.orientation),
                "visible_markers": pose.visible_markers,
            }
            msg = String()
            msg.data = json.dumps(data)
            self._pose_pub.publish(msg)
        except Exception:
            pass

    def get_pending_config(self) -> Optional[Dict[str, Any]]:
        """Return and clear the most recent configuration override.

        Returns
        -------
        Optional[Dict[str, Any]]
            Configuration dict if a new override was received, else ``None``.
        """
        with self._config_lock:
            cfg = self._pending_config
            self._pending_config = None
            return cfg

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _spin(self) -> None:
        """Background thread target: spin the ROS2 node."""
        try:
            import rclpy  # type: ignore[import]

            while self._running and rclpy.ok():
                rclpy.spin_once(self._node, timeout_sec=0.1)
        except Exception as exc:
            logger.debug("ROS2 spin stopped: %s", exc)
        finally:
            self._running = False

    def _on_config(self, msg: Any) -> None:
        """Callback for configuration override messages."""
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict):
                with self._config_lock:
                    self._pending_config = data
                logger.debug("Received config override: %s", data)
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("Invalid config message: %s", exc)
