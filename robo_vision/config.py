"""YAML configuration file support.

Loads settings from a YAML file and merges them with CLI arguments.
CLI arguments always take precedence over file-based configuration.

Example YAML configuration::

    # robo-vision configuration
    source: 0
    width: 1280
    height: 720
    quality: high
    mode: basic

    detectors:
      apriltag: true
      qr: false
      laser: true

    laser:
      threshold: 230
      threshold_max: 255
      channels: rg

    tag_size: 0.05
    map_file: maps/my_map.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("robo_vision.config")


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ImportError
        If PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML configuration file support. "
            "Install it with:  pip install pyyaml"
        )

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        logger.warning("Configuration file %s is empty or invalid.", path)
        return {}

    return data


def merge_config_with_args(
    config: Dict[str, Any],
    args: argparse.Namespace,
    cli_defaults: Dict[str, Any] | None = None,
) -> argparse.Namespace:
    """Merge YAML configuration with CLI arguments.

    CLI arguments take precedence over configuration file values.
    Only config values whose corresponding CLI argument was not explicitly
    set by the user are applied.

    Parameters
    ----------
    config:
        Configuration dictionary loaded from YAML.
    args:
        Parsed CLI arguments namespace.
    cli_defaults:
        Default values from the argument parser.  When a CLI argument
        equals its default, it is assumed the user did not set it
        explicitly and the YAML value is used instead.

    Returns
    -------
    argparse.Namespace
        Updated argument namespace with merged values.
    """
    if cli_defaults is None:
        cli_defaults = {}

    # Mapping from YAML keys to argparse attribute names
    flat_keys: dict[str, str] = {
        "source": "source",
        "width": "width",
        "height": "height",
        "quality": "quality",
        "mode": "mode",
        "headless": "headless",
        "gui": "gui",
        "record": "record",
        "map_file": "map_file",
        "tag_size": "tag_size",
        "target_distance": "target_distance",
        "follow_marker": "follow_marker",
        "follow_box": "follow_box",
        "chessboard_size": "chessboard_size",
        "cal": "cal",
    }

    for yaml_key, attr_name in flat_keys.items():
        if yaml_key in config:
            current = getattr(args, attr_name, None)
            default = cli_defaults.get(attr_name)
            if current == default:
                setattr(args, attr_name, config[yaml_key])
                logger.debug(
                    "Config: %s = %r (from YAML)", attr_name, config[yaml_key]
                )

    # Nested detector toggles
    detectors = config.get("detectors", {})
    if isinstance(detectors, dict):
        if "apriltag" in detectors:
            current = getattr(args, "no_apriltag", False)
            default = cli_defaults.get("no_apriltag", False)
            if current == default:
                args.no_apriltag = not detectors["apriltag"]
        if "qr" in detectors:
            current = getattr(args, "qr", False)
            default = cli_defaults.get("qr", False)
            if current == default:
                args.qr = detectors["qr"]
        if "laser" in detectors:
            current = getattr(args, "laser", False)
            default = cli_defaults.get("laser", False)
            if current == default:
                args.laser = detectors["laser"]

    # Nested laser parameters
    laser = config.get("laser", {})
    if isinstance(laser, dict):
        if "threshold" in laser:
            current = getattr(args, "laser_threshold", 240)
            default = cli_defaults.get("laser_threshold", 240)
            if current == default:
                args.laser_threshold = laser["threshold"]
        if "threshold_max" in laser:
            current = getattr(args, "laser_threshold_max", 255)
            default = cli_defaults.get("laser_threshold_max", 255)
            if current == default:
                args.laser_threshold_max = laser["threshold_max"]
        if "channels" in laser:
            current = getattr(args, "laser_channels", "rgb")
            default = cli_defaults.get("laser_channels", "rgb")
            if current == default:
                args.laser_channels = laser["channels"]

    return args
