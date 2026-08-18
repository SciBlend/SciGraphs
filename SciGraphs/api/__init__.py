"""Scripting API: drive SciGraphs from Python (`from SciGraphs import api as sg`).

Submodules: `graphs`, `preview`, `render`, `context`, `thin`.
Registered state goes through `bpy.ops` (direct PropertyGroup writes can miss the registered add-on).
For Blender-free work use `scigraphs_engine`; for declarative YAML pipelines use `SciGraphs.core.repro`.
"""

_MODULES = ("context", "graphs", "preview", "render", "thin")

__all__ = list(_MODULES)


def __getattr__(name):
    """Lazy-load submodules so importing `sg.graphs` does not pull in render/context."""
    if name in _MODULES:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_MODULES))
