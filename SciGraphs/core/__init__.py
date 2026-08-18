# The half of SciGraphs that needs Blender. `_LAZY` names resolve to modules
# here; `_MOVED` names are back-compat aliases into the `scigraphs_core`
# distribution, and new code should import from there directly.
#
# Keep `_LAZY` values as plain relative strings: tests/purity/test_core_purity.py reads this table with `literal_eval` and follows string values, so anything else parses fine and silently drops those import edges.

import importlib

_LAZY = {
    'dem_processor': '.geo.dem_processor',
    'geo_mesh': '.mesh.geo_mesh',
    'geometry': '.mesh.geometry',
    'geospatial': '.geo.geospatial',
    'importer': '.data_io.importer',
    'terrain': '.geo.terrain',
    'text_overlay': '.visualization.text_overlay',
    'texture_api': '.geo.texture_api',
}

_MOVED = {
    'analysis': 'scigraphs_core.algorithms.analysis',
    'animation': 'scigraphs_core.visualization.animation',
    'city2graph': 'scigraphs_core.city2graph',
    'coloring': 'scigraphs_core.coloring',
    'db_connector': 'scigraphs_core.data_io.db_connector',
    'dem_download': 'scigraphs_core.geo.dem_download',
    'edge_styles': 'scigraphs_core.mesh.edge_styles',
    'export_utils': 'scigraphs_core.data_io.export_utils',
    'georaster': 'scigraphs_core.geo.georaster',
    'graph': 'scigraphs_core.algorithms.graph',
    'layout': 'scigraphs_core.mesh.layout',
    'mesh_utils': 'scigraphs_core.mesh.mesh_utils',
    'network_flow': 'scigraphs_core.algorithms.network_flow',
    'osmnx_analysis': 'scigraphs_core.osmnx.analysis',
    'pathfinding': 'scigraphs_core.algorithms.pathfinding',
    'repro': 'scigraphs_core.repro',
    'spanning': 'scigraphs_core.algorithms.spanning',
    'sql_importer': 'scigraphs_core.data_io.sql_importer',
    'statistics': 'scigraphs_core.algorithms.statistics',
    'suitesparse_importer': 'scigraphs_core.data_io.suitesparse_importer',
    'topology': 'scigraphs_core.algorithms.topology',
}

__all__ = sorted(set(_LAZY) | set(_MOVED))


def __getattr__(name):
    """Import a re-exported module on first access (PEP 562)."""
    target = _LAZY.get(name)
    if target is not None:
        module = importlib.import_module(target, __name__)
        globals()[name] = module
        return module

    target = _MOVED.get(name)
    if target is not None:
        module = importlib.import_module(target)
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY) | set(_MOVED))
