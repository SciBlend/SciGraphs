"""Coloring core: colormap definitions and attribute helpers.

Pure-Python helpers used by the coloring UI. Keeping them here means the UI
classes only orchestrate, and the actual maths/data live in plain functions
that can be reused outside Blender.
"""

from . import colormaps  # pylint: disable=import-self
from . import attributes  # pylint: disable=import-self

__all__ = [
    "colormaps",
    "attributes",
]
