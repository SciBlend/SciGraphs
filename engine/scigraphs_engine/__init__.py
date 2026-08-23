# Nothing in this package may import bpy, gpu, bmesh or mathutils; tests in the
# SciGraphs repository parse this directory for those names. Everything below
# numpy is lazy, so a MeshSpec caller never pays for igraph and scipy.

import importlib as _importlib

__version__ = "0.1.1"

from .mesh import (  # noqa: F401
    DrawState, MeshGroup, MeshSpec, ShaderRef, LINES, POINTS, TRIS,
)
from .settings import Clause, Settings, PREFIX  # noqa: F401
from .source import ArraySource, GraphSource, EDGE, POINT  # noqa: F401

_LAZY = (
    "adaptive", "api", "blocks", "channels", "communities", "edge_styles",
    "filters", "gpu_compute", "lod", "mesh", "palette", "settings", "simplify",
    "source",
)

# Lets `from scigraphs_engine import Graph` resolve without importing api.
_API_NAMES = (
    "BackendUnavailable", "Camera", "ChannelResult", "ChannelSpec",
    "ChannelUnavailable", "Condition", "EngineError", "Geometry", "Graph",
    "Image", "Report", "UnknownChannel", "channel", "channel_names",
    "channel_specs", "colormap_names", "gpu_available",
)

__all__ = [
    "ArraySource", "Clause", "DrawState", "EDGE", "GraphSource", "LINES",
    "MeshGroup", "MeshSpec", "POINT", "POINTS", "PREFIX", "Settings",
    "ShaderRef", "TRIS", "__version__",
] + list(_API_NAMES) + list(_LAZY)


def __getattr__(name):
    """Import a submodule or a public API name on first access (PEP 562)."""
    if name in _LAZY:
        module = _importlib.import_module("." + name, __name__)
        globals()[name] = module
        return module
    if name in _API_NAMES:
        value = getattr(_importlib.import_module(".api", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY) | set(_API_NAMES))
