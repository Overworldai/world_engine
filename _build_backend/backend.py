"""
In-tree build backend for world_engine.

Thin wrapper around setuptools.build_meta that dynamically adds macOS-only
build dependencies (mlx, nanobind) via get_requires_for_build_wheel.
This avoids requiring consumers to configure no-build-isolation or declare
build deps in their own pyproject.toml.
"""

import sys

from setuptools.build_meta import *  # noqa: F401,F403
from setuptools import build_meta as _orig


_MACOS_BUILD_DEPS = [
    "mlx>=0.29",
    "nanobind==2.10.2",
]


def get_requires_for_build_wheel(config_settings=None):
    deps = _orig.get_requires_for_build_wheel(config_settings)
    if sys.platform == "darwin":
        deps = list(deps) + _MACOS_BUILD_DEPS
    return deps


def get_requires_for_build_editable(config_settings=None):
    deps = _orig.get_requires_for_build_editable(config_settings)
    if sys.platform == "darwin":
        deps = list(deps) + _MACOS_BUILD_DEPS
    return deps
