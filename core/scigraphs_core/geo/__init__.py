# Geospatial: DEM download and processing, rasters, terrain, imagery textures.
# `imagery` stays out of the table: `from ..core.geo import imagery` resolves as a
# submodule either way, and listing it would grow the attribute surface.

_LAZY = {
    # Blender-bound modules stayed with the add-on, as SciGraphs.core.<name>.
    "dem_download": ".dem_download",
    "georaster": ".georaster",
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
