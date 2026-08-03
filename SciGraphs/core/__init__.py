# Core module for SciGraphs
#
# Subpackages are re-exported at the top level for backward compatibility, but
# they are imported LAZILY (PEP 562) rather than eagerly.

import importlib

_LAZY = {
    'analysis': '.algorithms.analysis',
    'animation': '.visualization.animation',
    'city2graph': '.city2graph',
    'coloring': '.coloring',
    'db_connector': '.data_io.db_connector',
    'dem_download': '.geo.dem_download',
    'dem_processor': '.geo.dem_processor',
    'edge_styles': '.mesh.edge_styles',
    'export_utils': '.data_io.export_utils',
    'geo_mesh': '.mesh.geo_mesh',
    'geometry': '.mesh.geometry',
    'georaster': '.geo.georaster',
    'geospatial': '.geo.geospatial',
    'graph': '.algorithms.graph',
    'importer': '.data_io.importer',
    'layout': '.mesh.layout',
    'mesh_utils': '.mesh.mesh_utils',
    'network_flow': '.algorithms.network_flow',
    'osmnx': '.osmnx',
    'osmnx_analysis': '.osmnx.analysis',
    'pathfinding': '.algorithms.pathfinding',
    'repro': '.repro',
    'spanning': '.algorithms.spanning',
    'sql_importer': '.data_io.sql_importer',
    'statistics': '.algorithms.statistics',
    'suitesparse_importer': '.data_io.suitesparse_importer',
    'terrain': '.geo.terrain',
    'text_overlay': '.visualization.text_overlay',
    'texture_api': '.geo.texture_api',
    'topology': '.algorithms.topology',
}

__all__ = list(_LAZY)


def __getattr__(name):
    """Import a re-exported subpackage on first access (PEP 562)."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target, __name__)
    globals()[name] = module          # subsequent lookups skip this hook
    return module


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
