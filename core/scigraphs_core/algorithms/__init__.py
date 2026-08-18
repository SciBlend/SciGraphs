# Graph algorithms: analysis, construction, flow, paths, spanning trees, stats,
# topology. Imported lazily, so a missing optional backend costs nothing here.

_LAZY = {
    "analysis": ".analysis",
    "graph": ".graph",
    "network_flow": ".network_flow",
    "pathfinding": ".pathfinding",
    "spanning": ".spanning",
    "statistics": ".statistics",
    "topology": ".topology",
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
