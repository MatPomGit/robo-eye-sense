"""Tests for the Qt font directory fix in robo_vision."""

from __future__ import annotations

import importlib
import os
from unittest.mock import patch


class TestQtFontDirFix:
    """Verify _fix_qt_font_dir sets QT_QPA_FONTDIR correctly."""

    def test_sets_fontdir_when_system_fonts_exist(self):
        """If /usr/share/fonts exists the env var should be set."""
        env = os.environ.copy()
        env.pop("QT_QPA_FONTDIR", None)

        with patch.dict(os.environ, env, clear=True):
            from robo_vision import _fix_qt_font_dir

            _fix_qt_font_dir()

            if os.path.isdir("/usr/share/fonts"):
                assert os.environ.get("QT_QPA_FONTDIR") == "/usr/share/fonts"
            elif os.path.isdir("/usr/local/share/fonts"):
                assert os.environ.get("QT_QPA_FONTDIR") == "/usr/local/share/fonts"
            # else: neither dir exists – nothing to assert

    def test_respects_existing_envvar(self):
        """If QT_QPA_FONTDIR is already set, _fix_qt_font_dir must not override it."""
        with patch.dict(os.environ, {"QT_QPA_FONTDIR": "/custom/fonts"}):
            from robo_vision import _fix_qt_font_dir

            _fix_qt_font_dir()
            assert os.environ["QT_QPA_FONTDIR"] == "/custom/fonts"

    def test_noop_when_no_font_dirs(self):
        """When no candidate directories exist, the env var stays unset."""
        env = os.environ.copy()
        env.pop("QT_QPA_FONTDIR", None)

        with patch.dict(os.environ, env, clear=True), \
             patch("os.path.isdir", return_value=False):
            from robo_vision import _fix_qt_font_dir

            _fix_qt_font_dir()
            assert "QT_QPA_FONTDIR" not in os.environ

    def test_package_import_sets_fontdir(self):
        """Importing robo_vision should set QT_QPA_FONTDIR automatically."""
        env = os.environ.copy()
        env.pop("QT_QPA_FONTDIR", None)

        with patch.dict(os.environ, env, clear=True):
            # Force reimport to trigger _fix_qt_font_dir() call
            importlib.reload(importlib.import_module("robo_vision"))

            if os.path.isdir("/usr/share/fonts") or os.path.isdir("/usr/local/share/fonts"):
                assert "QT_QPA_FONTDIR" in os.environ
