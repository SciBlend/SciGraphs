# city2graph: graphs built out of urban data, including streets, buildings, GTFS
# feeds, Overture extracts and metapaths. Nothing is imported eagerly;
# `scigraphs_core/__init__.py` holds the lazy-attribute contract. `overture_api`
# belongs in the table although nothing here imports it: it has always been
# reachable as an attribute of this package, and that surface must not shrink.
# `area_resolver` is deliberately absent, because callers spell it
# `from ..core.city2graph.area_resolver import ...`, which needs no help here.

_LAZY = {
    # Blender-bound modules stayed with the add-on, as SciGraphs.core.<name>.
    "get_c2g": ".get_c2g",
    "metapaths": ".metapaths",
    "mobility": ".mobility",
    "overture_api": ".overture_api",
}

__all__ = list(_LAZY)


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target, __name__)
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
