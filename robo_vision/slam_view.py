"""Interactive 3-D SLAM visualisation using matplotlib.

Provides an interactive matplotlib window with 3D axes that dynamically
displays the robot's pose and detected markers as a point cloud.  The
view supports interactive rotation, zoom, and pan.

Usage::

    from robo_vision.slam_view import SlamView3D

    view = SlamView3D()
    view.update(markers=marker_list, robot_pose=pose)
    # ... in a loop ...
    view.close()

This module is optional and only available when ``matplotlib`` is
installed.
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger("robo_vision.slam_view")


def _matplotlib_available() -> bool:
    """Return ``True`` if matplotlib is importable."""
    try:
        import importlib.util

        return importlib.util.find_spec("matplotlib") is not None
    except (ImportError, ModuleNotFoundError):
        return False


class SlamView3D:
    """Interactive 3-D visualisation of SLAM map and robot pose.

    When ``matplotlib`` is not installed, all methods are safe no-ops.

    Parameters
    ----------
    title:
        Window title.
    figsize:
        Figure size in inches ``(width, height)``.
    """

    def __init__(
        self,
        title: str = "SLAM 3D View",
        figsize: tuple[float, float] = (6, 5),
    ) -> None:
        self._title = title
        self._figsize = figsize
        self._available = _matplotlib_available()
        self._fig = None
        self._ax = None
        self._initialized = False

    @property
    def available(self) -> bool:
        """``True`` when matplotlib is importable."""
        return self._available

    def _init_figure(self) -> bool:
        """Create the matplotlib figure and 3D axes.

        Returns ``True`` on success.
        """
        if not self._available:
            return False

        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            self._fig = plt.figure(
                num=self._title, figsize=self._figsize
            )
            self._ax = self._fig.add_subplot(111, projection="3d")
            self._ax.set_xlabel("X (cm)")
            self._ax.set_ylabel("Y (cm)")
            self._ax.set_zlabel("Z (cm)")
            self._ax.set_title("SLAM Map")
            self._fig.tight_layout()
            self._initialized = True
            logger.info("SLAM 3D view initialized.")
            return True

        except Exception as exc:
            logger.warning("Could not initialize 3D view: %s", exc)
            self._available = False
            return False

    def update(
        self,
        markers: Optional[List] = None,
        robot_pose: Optional[object] = None,
    ) -> None:
        """Update the 3D plot with current marker and robot positions.

        Parameters
        ----------
        markers:
            List of :class:`~robo_vision.marker_map.MarkerPose3D` objects.
        robot_pose:
            A :class:`~robo_vision.marker_map.RobotPose3D` object, or
            ``None`` if the robot pose is unknown.
        """
        if not self._available:
            return

        if not self._initialized:
            if not self._init_figure():
                return

        try:
            import matplotlib.pyplot as plt

            self._ax.cla()
            self._ax.set_xlabel("X (cm)")
            self._ax.set_ylabel("Y (cm)")
            self._ax.set_zlabel("Z (cm)")
            self._ax.set_title("SLAM Map")

            # Draw markers as green triangles
            if markers:
                xs = [m.position[0] for m in markers]
                ys = [m.position[1] for m in markers]
                zs = [m.position[2] for m in markers]
                self._ax.scatter(
                    xs, ys, zs, c="green", marker="^", s=60,
                    label="Markers", depthshade=True,
                )
                for m in markers:
                    self._ax.text(
                        m.position[0], m.position[1], m.position[2],
                        f"  {m.marker_id}", fontsize=7, color="green",
                    )

            # Draw robot pose as a red dot
            if robot_pose is not None and robot_pose.visible_markers > 0:
                rx, ry, rz = robot_pose.position
                self._ax.scatter(
                    [rx], [ry], [rz], c="red", marker="o", s=100,
                    label="Robot", depthshade=True,
                )

            if markers or (robot_pose and robot_pose.visible_markers > 0):
                self._ax.legend(loc="upper left", fontsize=8)

            self._fig.canvas.draw_idle()
            plt.pause(0.001)

        except Exception as exc:
            logger.debug("3D view update error: %s", exc)

    def close(self) -> None:
        """Close the matplotlib window."""
        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._fig)
            except Exception:
                pass
            self._fig = None
            self._ax = None
            self._initialized = False
