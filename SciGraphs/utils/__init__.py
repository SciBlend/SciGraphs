# Shared add-on utilities. Lazy on purpose: blender_helpers imports bpy, and an
# eager import here used to make every ``from ...utils.logger`` path
# unimportable outside Blender. logger now lives in scigraphs_core.

_LAZY = {
    "blender_helpers": ".blender_helpers",
    "dependencies": ".dependencies",
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
