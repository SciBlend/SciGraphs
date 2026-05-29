import bpy
from bpy.props import (
    StringProperty,
    EnumProperty,
    BoolProperty,
    IntProperty,
    FloatProperty,
    FloatVectorProperty,
    PointerProperty,
    CollectionProperty,
)

from .callbacks import get_column_items, get_db_profile_items, get_attribute_items, get_system_fonts

class CSVColumnItem(bpy.types.PropertyGroup):
    """Property group to store information about a CSV column."""
    name: StringProperty(name="Column Name")
    import_as_attribute: BoolProperty(name="Import", default=True)
    column_type: StringProperty(name="Type")

class SciGraphsProperties(bpy.types.PropertyGroup):
    """Properties to store the state of the SciGraphs addon."""
    
    # Data source selection
    data_source: EnumProperty(
        name="Data Source",
        description="Source of graph data",
        items=[
            ('FILE', "File", "Import from CSV or text file"),
            ('DATABASE', "Database", "Import from SQL database"),
            ('SUITESPARSE', "SuiteSparse", "Download from SuiteSparse Matrix Collection"),
            ('REPRO', "Repro", "Run or validate a reproducible pipeline"),
        ],
        default='FILE',
    )
    
    filepath: StringProperty(
        name="File Path",
        description="Path to the data file (CSV, TXT, GEXF)",
        subtype='FILE_PATH',
    )
    
    # SuiteSparse properties
    suitesparse_id: StringProperty(
        name="Matrix ID",
        description="SuiteSparse matrix identifier: 'Group/Name' (e.g. Grund/bayer09) or full URL",
        default="Grund/bayer09",
    )
    suitesparse_mode: EnumProperty(
        name="Graph Mode",
        description="How to interpret the sparse matrix as a graph",
        items=[
            ('BIPARTITE', "Bipartite", "Row-Column bipartite graph (preserves matrix structure, elongated layouts)"),
            ('SYMMETRIC', "Symmetric (A+A^T)", "Symmetric adjacency graph (denser, rounder layouts)"),
        ],
        default='BIPARTITE',
    )
    suitesparse_giant_only: BoolProperty(
        name="Giant Component Only",
        description="Keep only the largest connected component (recommended)",
        default=True,
    )
    suitesparse_status: StringProperty(
        name="Status",
        description="Status of last SuiteSparse download",
        default="",
    )
    
    csv_delimiter: EnumProperty(
        name="Delimiter",
        description="Column separator character for CSV/text files",
        items=[
            (',', "Comma (,)", "Use comma as delimiter"),
            (';', "Semicolon (;)", "Use semicolon as delimiter"),
            ('\t', "Tab", "Use tab as delimiter"),
            ('|', "Pipe (|)", "Use pipe as delimiter"),
            (' ', "Space", "Use space as delimiter"),
        ],
        default=',',
    )
    
    # SQL Database properties
    db_profile_index: EnumProperty(
        name="Database Connection",
        description="Select a database connection profile",
        items=get_db_profile_items,
    )
    
    sql_query: StringProperty(
        name="SQL Query",
        description="SQL query to execute (SELECT only)",
        default="SELECT source, target FROM edges",
    )
    
    sql_columns_cache: StringProperty(
        name="SQL Columns Cache",
        description="Cached column names from last SQL query (pipe-separated)",
        default="",
    )
    
    sql_row_count: IntProperty(
        name="Row Count",
        description="Number of rows returned by the last query",
        default=0,
        min=0,
    )
    
    sql_query_status: StringProperty(
        name="Query Status",
        description="Status message from the last query execution",
        default="",
    )
    
    source_column: EnumProperty(
        name="Source Column",
        description="Column containing source nodes",
        items=get_column_items,
    )
    
    target_column: EnumProperty(
        name="Target Column",
        description="Column containing target nodes",
        items=get_column_items,
    )
    
    is_directed: BoolProperty(
        name="Directed Graph",
        description="Treat edges as having a direction",
        default=False,
    )
    
    remove_self_loops: BoolProperty(
        name="Remove Self-Loops",
        description="Discard edges where source and target are the same node",
        default=True,
    )
    
    # Geospatial properties
    use_geospatial: BoolProperty(
        name="Enable Geospatial Mode",
        description="Automatically detect and use geographic coordinates",
        default=False,
    )
    
    latitude_column: EnumProperty(
        name="Latitude Column",
        description="Column containing latitude values",
        items=get_column_items,
    )
    
    longitude_column: EnumProperty(
        name="Longitude Column",
        description="Column containing longitude values",
        items=get_column_items,
    )
    
    geocode_columns: BoolProperty(
        name="Auto-Geocode Countries",
        description="Automatically convert country names to coordinates",
        default=True,
    )
    
    # Temporal data properties
    has_temporal_data: BoolProperty(
        name="Has Temporal Data",
        description="Enable temporal data filtering and aggregation",
        default=False,
    )
    
    time_column: EnumProperty(
        name="Time Column",
        description="Column containing time/date information",
        items=get_column_items,
    )
    
    time_aggregation: EnumProperty(
        name="Time Aggregation",
        description="How to aggregate temporal data",
        items=[
            ('ALL', "All Time", "Aggregate all time periods"),
            ('YEAR', "By Year", "Aggregate by year"),
            ('MONTH', "By Month", "Keep monthly granularity"),
            ('RANGE', "Custom Range", "Select specific time range"),
        ],
        default='ALL',
    )
    
    time_range_start: StringProperty(
        name="Start Period",
        description="Start time period (format: YYYY or YYYY-MM)",
        default="",
    )
    
    time_range_end: StringProperty(
        name="End Period",
        description="End time period (format: YYYY or YYYY-MM)",
        default="",
    )
    
    # Edge weight property
    weight_column: EnumProperty(
        name="Edge Weight Column",
        description="Column containing edge weights/values",
        items=get_column_items,
    )
    
    # Globe visualization properties
    show_globe: BoolProperty(
        name="Show Earth Globe",
        description="Display Earth sphere for reference",
        default=True,
    )
    
    globe_radius: FloatProperty(
        name="Globe Radius",
        description="Radius of the Earth globe",
        default=5.0,
        min=1.0,
        max=20.0,
    )
    
    globe_material: EnumProperty(
        name="Globe Material",
        description="Visual style for the Earth globe",
        items=[
            ('SIMPLE', "Simple", "Solid blue color (default)"),
            ('OCEAN', "Ocean", "Blue with subtle wave-like texture"),
            ('WIREFRAME', "Wireframe", "Transparent with grid lines"),
            ('TOPOGRAPHIC', "Topographic", "Height-based color gradient"),
            ('WORLD_MAP', "World Map", "Land vs ocean using real geographic data (slower)"),
        ],
        default='SIMPLE',
    )
    
    globe_subdivisions: IntProperty(
        name="Globe Subdivisions",
        description="UV sphere resolution (higher = smoother but slower)",
        default=64,
        min=16,
        max=512,
    )
    
    map_resolution: EnumProperty(
        name="Map Resolution",
        description="Level of detail for World Map material",
        items=[
            ('110m', "Low (110m)", "Fast - ~1 MB, basic shapes"),
            ('50m', "Medium (50m)", "Balanced - ~3 MB, good detail"),
            ('10m', "High (10m)", "Detailed - ~20 MB, very accurate"),
        ],
        default='10m',
    )
    
    map_feature_type: EnumProperty(
        name="Map Feature Type",
        description="Type of geographic data to display",
        items=[
            ('LAND', "Land Masses", "Show continents and islands (simple, fastest)"),
            ('COASTLINE', "Detailed Coastline", "High-detail coastal boundaries (medium)"),
            ('LAND_OCEAN', "Land + Ocean", "Land with ocean floor features (detailed)"),
            ('BATHYMETRY', "Bathymetry", "Ocean depth contours (very detailed)"),
            ('RIVERS_LAKES', "Rivers + Lakes", "Water bodies and rivers (detailed)"),
        ],
        default='COASTLINE',
    )
    
    globe_theme_api: EnumProperty(
        name="Globe Theme",
        description="Texture theme for the Earth globe (configure API keys in addon preferences)",
        items=[
            ('NONE', "None", "Use material style without satellite texture"),
            ('NASA_BLUE_MARBLE', "Realistic Satellite", "Satellite imagery (NASA/alternative sources)"),
            ('NASA_VIIRS', "Night Lights", "Earth at night - city lights"),
            ('NATURAL_EARTH', "Natural Earth", "Natural Earth shaded relief"),
            ('URBAN_DARK', "Urban Dark/Neon", "Dark map with illuminated urban areas (procedural)"),
            ('TOPOGRAPHIC_SHADED', "Topographic Shaded", "Elevation-based shaded relief (procedural)"),
            ('DATA_OVERLAY', "Data Overlay", "Transparent base for data visualization (procedural)"),
        ],
        default='NONE',
    )
    
    globe_texture_resolution: EnumProperty(
        name="Texture Resolution",
        description="Resolution of downloaded texture (higher = better quality, larger file)",
        items=[
            ('2K', "2K (2048x1024)", "Fast download, lower quality (~1MB)"),
            ('4K', "4K (4096x2048)", "Balanced quality (~4MB)"),
            ('8K', "8K (8192x4096)", "High quality (~15MB)"),
        ],
        default='4K',
    )
    
    globe_water_specular: FloatProperty(
        name="Water Specular",
        description="Specular reflection intensity for ocean areas",
        default=0.8,
        min=0.0,
        max=1.0,
    )
    
    globe_water_roughness: FloatProperty(
        name="Water Roughness",
        description="Surface roughness for ocean areas (lower = shinier)",
        default=0.2,
        min=0.0,
        max=1.0,
    )
    
    globe_land_roughness: FloatProperty(
        name="Land Roughness",
        description="Surface roughness for land areas (higher = more matte)",
        default=0.8,
        min=0.0,
        max=1.0,
    )
    
    globe_bump_strength: FloatProperty(
        name="Bump Strength",
        description="Intensity of surface relief/bump mapping",
        default=0.1,
        min=0.0,
        max=1.0,
    )
    
    edge_style: EnumProperty(
        name="Edge Style",
        description="How edges are rendered in geospatial mode",
        items=[
            ('STRAIGHT', "Straight Lines", "Direct 3D connections"),
            ('GREAT_CIRCLE', "Great Circle Arcs", "Curved paths along sphere surface"),
        ],
        default='GREAT_CIRCLE',
    )
    
    # CSV column selection
    available_csv_columns: CollectionProperty(type=CSVColumnItem)

    auto_layout_on_import: BoolProperty(
        name="Auto Layout on Import",
        description="Automatically apply the selected layout after creating a graph",
        default=True,
    )
    
    layout_algorithm: EnumProperty(
        name="Layout Algorithm",
        description="Algorithm to determine node positions",
        items=[
            # === 2D LAYOUTS ===
            ('GRID', "Grid (2D)", "Arrange nodes in a 2D grid (instant)"),
            ('SPRING', "Spring (2D - NetworkX)", "Force-directed 2D layout (slow)"),
            ('FORCEATLAS2', "ForceAtlas2 (2D)", "Gephi's algorithm, requires optional fa2 package (medium)"),
            ('IGRAPH_DRL_2D', "DrL (2D - igraph)", "Distributed Recursive Layout 2D, very fast for huge graphs (very fast)"),
            ('IGRAPH_DH', "Davidson-Harel (2D - igraph)", "Simulated annealing approach (medium)"),
            ('IGRAPH_GRAPHOPT', "Graphopt (2D - igraph)", "Energy-based optimization (fast)"),
            ('CIRCLE_PACKING', "Circle Packing (2D - Koebe)", "Koebe theorem: tangent circles for planar graphs (medium)"),
            
            # === 3D GEOMETRIC LAYOUTS ===
            ('RANDOM', "Random (3D)", "Distribute nodes randomly in 3D space (instant)"),
            ('SPHERE', "Sphere (3D)", "Distribute nodes on sphere surface using Fibonacci algorithm (instant)"),
            ('SPIRAL_3D', "Spiral (3D)", "Arrange nodes in upward spiral pattern (instant)"),
            ('HELIX', "Helix (3D)", "Double helix pattern like DNA structure (instant)"),
            ('CUBE', "Cube (3D)", "Distribute nodes in and on a cube (instant)"),
            
            # === 3D GRAPH-BASED LAYOUTS ===
            ('SPECTRAL_3D', "Spectral (3D)", "Use graph Laplacian eigenvectors for 3D positioning (fast)"),
            ('MDS_3D', "MDS (3D)", "Multidimensional scaling using shortest path distances (medium)"),
            ('HIERARCHICAL_3D', "Hierarchical (3D)", "Tree-like hierarchy in layers (fast)"),
            ('BIPARTITE_3D', "Bipartite (3D)", "Two parallel planes for bipartite graphs (fast)"),
            
            # === 3D FORCE-DIRECTED LAYOUTS ===
            ('YIFAN_HU', "Yifan Hu (3D)", "Scalable force-directed placement via bundled scigraphs-utils"),
            ('IGRAPH_DRL', "DrL (3D - igraph)", "Distributed Recursive Layout for huge graphs 100k+ (very fast)"),
            ('IGRAPH_FR', "Fruchterman-Reingold (3D - igraph)", "Classic force-directed in 3D (fast)"),
            ('IGRAPH_KK', "Kamada-Kawai (3D - igraph)", "Deterministic 3D layout, reproducible (medium)"),
            ('IGRAPH_LGL', "LGL (3D - igraph)", "Large Graph Layout, optimized for massive graphs (fast)"),
            ('SPRING_3D', "Spring (3D - NetworkX)", "Force-directed 3D layout (very slow)"),

            # === GRAPHVIZ LAYOUTS (scigraphs-utils) ===
            ('GRAPHVIZ_DOT', "Graphviz Dot (2D)", "Hierarchical layout via bundled scigraphs-utils"),
            ('GRAPHVIZ_NEATO', "Graphviz Neato (2D/3D)", "Spring model layout via bundled scigraphs-utils"),
            ('GRAPHVIZ_FDP', "Graphviz FDP (2D/3D)", "Force-directed placement via bundled scigraphs-utils"),
            ('GRAPHVIZ_SFDP', "Graphviz SFDP (2D/3D)", "Scalable force-directed placement via bundled scigraphs-utils"),
            ('GRAPHVIZ_TWOPI', "Graphviz Twopi (2D)", "Radial layout via bundled scigraphs-utils"),
            ('GRAPHVIZ_CIRCO', "Graphviz Circo (2D)", "Circular layout via bundled scigraphs-utils"),
            ('GRAPHVIZ_OSAGE', "Graphviz Osage (2D)", "Cluster layout via bundled scigraphs-utils"),
            ('GRAPHVIZ_PATCHWORK', "Graphviz Patchwork (2D)", "Patchwork layout via bundled scigraphs-utils"),
            
            # === DIRECTED GRAPH LAYOUTS ===
            ('SUGIYAMA', "Sugiyama/Layered (2D - Directed)", "Hierarchical DAG layout, minimizes crossings (fast)"),
            ('CIRCULAR_HIERARCHY', "Circular Hierarchy (2D - Directed)", "Concentric circles from roots (fast)"),
        ],
        default='YIFAN_HU',
    )

    iterations: IntProperty(
        name="Iterations",
        description="Number of iterations for layout simulation",
        default=50,
        min=1,
        max=1000,
        options=set(),
    )
    
    iterations_per_frame: IntProperty(
        name="Iterations per Frame",
        description="Number of iterations to execute per frame in interactive mode (higher = faster convergence, lower = smoother animation)",
        default=5,
        min=1,
        max=50,
        options=set(),
    )
    
    layout_scale: FloatProperty(
        name="Scale",
        description="Overall scale of the layout",
        default=5.0,
        min=0.1,
        max=100.0,
        options=set(),
    )
    
    # Advanced Force-Directed Parameters
    repulsion_strength: FloatProperty(
        name="Repulsion Strength",
        description="Strength of repulsive forces between nodes",
        default=1.0,
        min=0.0,
        max=10.0,
    )
    
    attraction_strength: FloatProperty(
        name="Attraction Strength",
        description="Strength of attractive forces along edges",
        default=1.0,
        min=0.0,
        max=10.0,
    )
    
    gravity_strength: FloatProperty(
        name="Gravity",
        description="Strength of gravity pulling nodes to center",
        default=0.1,
        min=0.0,
        max=5.0,
    )
    
    cooling_factor: FloatProperty(
        name="Cooling Factor",
        description="Temperature cooling rate (lower = slower cooling)",
        default=0.95,
        min=0.1,
        max=1.0,
    )
    
    initial_temperature: FloatProperty(
        name="Initial Temperature",
        description="Starting temperature for simulated annealing",
        default=1.0,
        min=0.1,
        max=10.0,
    )
    
    # Execution Control
    execution_speed: FloatProperty(
        name="Execution Speed",
        description="Time between iterations (seconds, lower = faster)",
        default=0.01,
        min=0.001,
        max=1.0,
    )
    
    auto_stop_threshold: FloatProperty(
        name="Auto-Stop Threshold",
        description="Stop when energy change is below this (0 = never stop)",
        default=0.0,
        min=0.0,
        max=1.0,
    )
    
    barnes_hut_theta: FloatProperty(
        name="Barnes-Hut Theta",
        description="Approximation parameter for force calculation (lower = more accurate)",
        default=1.2,
        min=0.0,
        max=3.0,
    )
    
    edge_distance: FloatProperty(
        name="Edge Distance",
        description="Optimal distance between connected nodes",
        default=1.0,
        min=0.1,
        max=10.0,
    )
    
    # ========================================
    # TOPOLOGICAL ANALYSIS PROPERTIES
    # ========================================
    
    topology_analysis_mode: EnumProperty(
        name="Analysis Mode",
        description="Type of topological analysis to perform",
        items=[
            ('SURFACE', "Surface Embedding", "Analyze planarity, genus, and face structure"),
        ],
        default='SURFACE',
    )
    
    topology_surface_type: EnumProperty(
        name="Surface Type",
        description="Target surface for embedding analysis",
        items=[
            ('PLANE', "Plane (Genus 0)", "Check if graph is planar"),
        ],
        default='PLANE',
    )
    
    topology_show_faces: BoolProperty(
        name="Show Faces",
        description="Visualize detected faces for planar graphs",
        default=False,
    )
    
    show_hud_overlay: BoolProperty(
        name="Show HUD Overlay",
        description="Display graph statistics overlay in viewport",
        default=True,
    )

# Inject properties defined in domain-specific modules.
# This keeps all property names on the same SciGraphsProperties class
# so existing code (`context.scene.scigraphs.osmnx_place_name`) works unchanged.

from .osmnx_scene_properties import OSMNX_SCENE_PROPERTIES
from .layout_properties import LAYOUT_PROPERTIES
from .splitter_properties import SPLITTER_PROPERTIES
from .text_overlay_properties import TEXT_OVERLAY_PROPERTIES
from .edge_style_properties import EDGE_STYLE_PROPERTIES

SciGraphsProperties.__annotations__.update(OSMNX_SCENE_PROPERTIES)
SciGraphsProperties.__annotations__.update(LAYOUT_PROPERTIES)
SciGraphsProperties.__annotations__.update(SPLITTER_PROPERTIES)
SciGraphsProperties.__annotations__.update(TEXT_OVERLAY_PROPERTIES)
SciGraphsProperties.__annotations__.update(EDGE_STYLE_PROPERTIES)

def register():
    bpy.utils.register_class(CSVColumnItem)
    bpy.utils.register_class(SciGraphsProperties)
    bpy.types.Scene.scigraphs = PointerProperty(type=SciGraphsProperties)

def unregister():
    del bpy.types.Scene.scigraphs
    bpy.utils.unregister_class(SciGraphsProperties)
    bpy.utils.unregister_class(CSVColumnItem)
