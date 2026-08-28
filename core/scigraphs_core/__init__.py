"""SciGraphs core: analysis with no Blender. Graph algorithms, layouts, color,
I/O, OSMnx and city2graph; mesh and scene code stays in the add-on, which this
package must never import (``scripts/extract/test_core_standalone.py``). numpy is
required and the rest optional. Submodules load on first use, and ``_LAZY`` values
must stay plain strings: the purity test reads the table with ``literal_eval``."""

_LAZY = {
    'algorithms': '.algorithms',
    'analysis': '.algorithms.analysis',
    'animation': '.visualization.animation',
    'city2graph': '.city2graph',
    'coloring': '.coloring',
    'data_io': '.data_io',
    'db_connector': '.data_io.db_connector',
    'dem_download': '.geo.dem_download',
    'edge_styles': '.mesh.edge_styles',
    'export_utils': '.data_io.export_utils',
    'feature_tags': '.feature_tags',
    'geo': '.geo',
    'georaster': '.geo.georaster',
    'graph': '.algorithms.graph',
    'layout': '.mesh.layout',
    'logger': '.logger',
    'mesh': '.mesh',
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
    'topology': '.algorithms.topology',
    'visualization': '.visualization',
}

__all__ = list(_LAZY)

__version__ = "0.3.0"


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
