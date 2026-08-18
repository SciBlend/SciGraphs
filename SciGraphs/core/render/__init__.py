# Shim over the scigraphs_engine wheel (``engine/`` at repo root), aliased through
# sys.modules rather than star-imported, so state is shared and _names reachable.

import importlib as _importlib
import os as _os
import sys as _sys

_MODULES = (
    "adaptive", "blocks", "channels", "edge_styles", "filters", "gpu_compute",
    "lod", "mesh", "settings", "simplify", "source",
)


def _locate_engine():
    try:
        return _importlib.import_module("scigraphs_engine")
    except ModuleNotFoundError:
        pass

    here = _os.path.dirname(_os.path.abspath(__file__))
    addon_root = _os.path.dirname(_os.path.dirname(here))
    repo_root = _os.path.dirname(addon_root)

    checkout = _os.path.join(repo_root, "engine")
    if _os.path.isfile(_os.path.join(checkout, "scigraphs_engine",
                                     "__init__.py")):
        if checkout not in _sys.path:
            _sys.path.insert(0, checkout)
        return _importlib.import_module("scigraphs_engine")

    raise ModuleNotFoundError(
        "scigraphs_engine is not importable and no checkout was found next to "
        f"the add-on (looked for {checkout}). In Blender this means the "
        "scigraphs_engine wheel listed in blender_manifest.toml did not "
        "install -- reinstall the extension, or `pip install scigraphs-engine` "
        "into Blender's Python. From a checkout, `pip install ./engine` or run "
        "with the repository root on sys.path.")


_engine = _locate_engine()

_self = _sys.modules[__name__]
for _name in _MODULES:
    _module = _importlib.import_module("scigraphs_engine." + _name)
    _sys.modules[__name__ + "." + _name] = _module
    setattr(_self, _name, _module)

# Older engine builds may still use the British spelling.
_RENAMED = {"filters": (("neighborhood_overlap", "neighbourhood_overlap"),)}

for _name, _pairs in _RENAMED.items():
    _module = _sys.modules[__name__ + "." + _name]
    for _new, _old in _pairs:
        if not hasattr(_module, _new) and hasattr(_module, _old):
            setattr(_module, _new, getattr(_module, _old))


def _communities_from_edges(edges_int, num_nodes, algorithm):
    """Bridge to add-on analysis; imported lazily so the engine stays standalone."""
    from scigraphs_core.algorithms.analysis import communities_from_edges
    return communities_from_edges(edges_int, num_nodes, algorithm)


_sys.modules[__name__ + ".simplify"].set_community_detector(
    _communities_from_edges)

from scigraphs_engine.settings import Clause, Settings, PREFIX  # noqa: E402,F401
