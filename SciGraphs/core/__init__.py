# Core module for SciGraphs addon
#
# Subpackages re-exported at the top level for backward compatibility.
# Consumers can use either:
#   from ...core import analysis          (re-export, always works)
#   from ...core.algorithms.analysis import X  (direct path)

from .algorithms import analysis
from .algorithms import graph
from .algorithms import network_flow
from .algorithms import pathfinding
from .algorithms import spanning
from .algorithms import statistics
from .algorithms import topology

from .mesh import edge_styles
from .mesh import geo_mesh
from .mesh import geometry
from .mesh import layout
from .mesh import mesh_utils

from .geo import dem_download
from .geo import dem_processor
from .geo import georaster
from .geo import geospatial
from .geo import terrain
from .geo import texture_api

from .data_io import db_connector
from .data_io import export_utils
from .data_io import importer
from .data_io import sql_importer
from .data_io import suitesparse_importer

from .visualization import animation
from .visualization import text_overlay

from .osmnx import analysis as osmnx_analysis

from . import city2graph
from . import osmnx
from . import repro
from . import coloring

__all__ = [
    'analysis',
    'graph',
    'network_flow',
    'pathfinding',
    'spanning',
    'statistics',
    'topology',
    'edge_styles',
    'geo_mesh',
    'geometry',
    'layout',
    'mesh_utils',
    'dem_download',
    'dem_processor',
    'georaster',
    'geospatial',
    'terrain',
    'texture_api',
    'db_connector',
    'export_utils',
    'importer',
    'sql_importer',
    'suitesparse_importer',
    'animation',
    'text_overlay',
    'osmnx_analysis',
    'city2graph',
    'osmnx',
    'repro',
    'coloring',
]
