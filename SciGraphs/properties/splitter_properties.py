from .callbacks import get_attribute_items
from bpy.props import (
    StringProperty,
    EnumProperty,
    BoolProperty,
    IntProperty,
    FloatProperty,
    FloatVectorProperty,
)

SPLITTER_PROPERTIES = {
    # ========================================
    # NETWORK SPLITTER 3D
    # ========================================

    'splitter_criterion': EnumProperty(
        name="Split Criterion",
        description="Criterion to split the network into Z layers",
        items=[
            ('COMMUNITY', "Community Detection", "Detect communities using pySurprise algorithms"),
            ('ATTRIBUTE', "Node Attribute", "Use a categorical or numeric node attribute"),
            ('DEGREE', "Node Degree", "Split by degree ranges (low/medium/high connectivity)"),
            ('COMPONENT', "Connected Components", "Each component on separate layer"),
            ('CENTRALITY', "Centrality Ranges", "Split by centrality score ranges"),
            ('CUSTOM', "Custom Expression", "Use a custom Python expression"),
        ],
        default='COMMUNITY',
    ),

    'splitter_attribute': EnumProperty(
        name="Split Attribute",
        description="Node attribute to use for splitting (when criterion is ATTRIBUTE)",
        items=get_attribute_items,
    ),

    'splitter_layer_height': FloatProperty(
        name="Layer Height",
        description="Vertical distance between layers",
        default=2.0,
        min=0.1,
        max=20.0,
    ),

    'splitter_layer_order': EnumProperty(
        name="Layer Order",
        description="How to order the layers on Z axis",
        items=[
            ('SIZE_ASC', "Size Ascending", "Smallest groups at bottom"),
            ('SIZE_DESC', "Size Descending", "Largest groups at bottom"),
            ('VALUE_ASC', "Value Ascending", "Lowest values at bottom"),
            ('VALUE_DESC', "Value Descending", "Highest values at bottom"),
            ('ALPHA', "Alphabetical", "Alphabetical order for categorical"),
        ],
        default='SIZE_DESC',
    ),

    'splitter_degree_bins': IntProperty(
        name="Degree Bins",
        description="Number of bins for degree-based splitting",
        default=3,
        min=2,
        max=10,
    ),

    'splitter_centrality_bins': IntProperty(
        name="Centrality Bins",
        description="Number of bins for centrality-based splitting",
        default=3,
        min=2,
        max=10,
    ),

    'splitter_community_algorithm': EnumProperty(
        name="Community Algorithm",
        description="Algorithm for community detection (pySurprise)",
        items=[
            ('CPM', "CPM", "Clique Percolation Method (Traag et al. 2011)"),
            ('INFOMAP', "Infomap", "Map equation (Rosvall & Bergstrom 2008)"),
            ('RB', "RB", "Reichardt-Bornholdt Potts model (2006)"),
            ('RN', "RN", "Ronhovde-Nussinov resolution-free (2010)"),
            ('RNSC', "RNSC", "Restricted Neighbourhood Search Clustering (King et al. 2004)"),
            ('SCLUSTER', "SCluster", "Hierarchical clustering (Aldecoa & Marín 2010)"),
            ('UVCLUSTER', "UVCluster", "Iterative cluster analysis (Arnau et al. 2005)"),
        ],
        default='RN',
    ),

    'splitter_community_resolution': FloatProperty(
        name="Resolution",
        description="Resolution parameter for community detection (higher = more communities)",
        default=1.0,
        min=0.1,
        max=5.0,
    ),

    'splitter_preserve_xy': BoolProperty(
        name="Preserve XY Positions",
        description="Keep original X and Y positions, only modify Z",
        default=True,
    ),

    'splitter_center_layers': BoolProperty(
        name="Center Layers",
        description="Center each layer around its centroid",
        default=False,
    ),

    'splitter_scale_by_size': BoolProperty(
        name="Scale by Layer Size",
        description="Scale layer XY extent based on number of nodes",
        default=False,
    ),

    'splitter_inter_layer_edges': EnumProperty(
        name="Inter-Layer Edges",
        description="How to handle edges between layers",
        items=[
            ('STRAIGHT', "Straight", "Direct straight lines between layers"),
            ('CURVED', "Curved", "Smooth curves between layers"),
            ('HIDE', "Hide", "Hide inter-layer edges"),
        ],
        default='STRAIGHT',
    ),

    'splitter_base_z': FloatProperty(
        name="Base Z",
        description="Z coordinate for the first (bottom) layer",
        default=0.0,
        min=-100.0,
        max=100.0,
    ),

    # Visualization Options
    'show_forces': BoolProperty(
        name="Show Forces",
        description="Visualize force vectors during execution",
        default=False,
    ),

    'update_viewport': BoolProperty(
        name="Update Viewport",
        description="Update 3D view during execution (slower but visual feedback)",
        default=True,
    ),

    'node_size': FloatProperty(
        name="Node Size",
        description="Radius of node primitives",
        default=0.02,
        min=0.0,
        max=1.0,
    ),

    'node_resolution': IntProperty(
        name="Node Resolution",
        description="Resolution of generated node geometry",
        default=10,
        min=3,
        max=128,
    ),

    'node_shape': EnumProperty(
        name="Node Shape",
        description="Generated mesh shape for nodes when supported by the visualization node tree",
        items=[
            ('SPHERE', "Sphere", "UV sphere nodes"),
            ('ICOSPHERE', "Icosphere", "Icosphere nodes"),
            ('CUBE', "Cube", "Cube nodes"),
            ('CONE', "Cone", "Cone nodes"),
            ('CYLINDER', "Cylinder", "Cylinder-style nodes"),
        ],
        default='SPHERE',
    ),

    'node_scale_multiplier': FloatProperty(
        name="Node Attribute Multiplier",
        description="Multiplier for attribute-driven node scaling",
        default=1.0,
        min=0.0,
        max=100.0,
    ),

    'edge_thickness': FloatProperty(
        name="Edge Thickness",
        description="Thickness of edge tubes (0 = invisible edges)",
        default=0.0,
        min=0.0,
        max=0.5,
    ),

    'edge_resolution': IntProperty(
        name="Edge Resolution",
        description="Resolution of edge tube profiles",
        default=8,
        min=3,
        max=64,
    ),

    'edge_thickness_multiplier': FloatProperty(
        name="Edge Attribute Multiplier",
        description="Multiplier for attribute-driven edge thickness",
        default=1.0,
        min=0.0,
        max=100.0,
    ),

    'show_arrows': BoolProperty(
        name="Show Direction Arrows",
        description="Display arrows for directed graph edges when supported",
        default=False,
    ),

    'arrow_size': FloatProperty(
        name="Arrow Size",
        description="Size of direction arrows",
        default=0.15,
        min=0.001,
        max=5.0,
    ),

    'arrow_position': FloatProperty(
        name="Arrow Position",
        description="Position of direction arrows along each edge",
        default=0.7,
        min=0.0,
        max=1.0,
    ),

    'centrality_method': EnumProperty(
        name="Centrality Method",
        description="Method for calculating node centrality",
        items=[
            ('degree', "Degree", "Number of connections"),
            ('betweenness', "Betweenness", "Number of shortest paths through node"),
            ('closeness', "Closeness", "Average distance to all other nodes"),
            ('eigenvector', "Eigenvector", "Influence based on connections"),
        ],
        default='degree',
    ),

    # Directed graph specific centrality
    'directed_centrality_method': EnumProperty(
        name="Directed Centrality",
        description="Centrality metrics specific to directed graphs",
        items=[
            ('pagerank', "PageRank", "Web-style importance (Google algorithm)"),
            ('hub_score', "Hub Score", "HITS algorithm - good pointers to authorities"),
            ('authority_score', "Authority Score", "HITS algorithm - good destinations"),
            ('in_degree', "In-Degree", "Number of incoming connections (popularity)"),
            ('out_degree', "Out-Degree", "Number of outgoing connections (influence)"),
            ('katz', "Katz Centrality", "Eigenvector variant for directed graphs"),
        ],
        default='pagerank',
    ),

    # Flow animation settings
    'flow_animation_mode': EnumProperty(
        name="Animation Mode",
        description="How the flow propagates through the graph",
        items=[
            ('DISCRETE', "Discrete", "Binary activation (0 or 1), step by step propagation"),
            ('CONTINUOUS', "Continuous", "Smooth gradient activation (0 to 1), wave-like propagation"),
        ],
        default='CONTINUOUS',
    ),

    'flow_animation_loop': BoolProperty(
        name="Loop Animation",
        description="Loop the flow animation to fit the timeline range",
        default=True,
    ),

    'flow_animation_speed': IntProperty(
        name="Frames per Cycle",
        description="Number of frames for one complete flow propagation cycle",
        default=25,
        min=5,
        max=500,
    ),

    'flow_animation_smoothness': FloatProperty(
        name="Wave Width",
        description="Width of the activation gradient in continuous mode (higher = smoother)",
        default=2.0,
        min=0.5,
        max=10.0,
    ),

    # Traversal animation settings
    'traversal_algorithm': EnumProperty(
        name="Algorithm",
        description="Graph traversal algorithm to use",
        items=[
            ('BFS', "BFS", "Breadth-First Search - explores neighbors level by level"),
            ('DFS', "DFS", "Depth-First Search - explores as far as possible along each branch"),
        ],
        default='BFS',
    ),

    'traversal_start_mode': EnumProperty(
        name="Start Mode",
        description="How to select starting nodes for traversal",
        items=[
            ('AUTO', "Auto", "Automatically select node with highest degree"),
            ('MANUAL', "Manual", "Manually specify node indices"),
        ],
        default='AUTO',
    ),

    'traversal_start_nodes': StringProperty(
        name="Start Nodes",
        description="Comma-separated node indices (e.g., '0,5,10')",
        default="",
    ),

    'traversal_animation_mode': EnumProperty(
        name="Animation Mode",
        description="How the traversal propagates through the graph",
        items=[
            ('DISCRETE', "Discrete", "Binary activation (0 or 1), step by step"),
            ('CONTINUOUS', "Continuous", "Smooth gradient activation (0 to 1), wave-like"),
        ],
        default='DISCRETE',
    ),

    'traversal_animation_speed': IntProperty(
        name="Frames per Cycle",
        description="Number of frames for one complete traversal cycle",
        default=30,
        min=5,
        max=500,
    ),

    'traversal_animation_smoothness': FloatProperty(
        name="Wave Width",
        description="Width of the activation gradient in continuous mode",
        default=2.0,
        min=0.5,
        max=10.0,
    ),

    'traversal_animation_loop': BoolProperty(
        name="Loop Animation",
        description="Loop the traversal animation to fit the timeline range",
        default=True,
    ),

    'clustering_algorithm': EnumProperty(
        name="Clustering Algorithm",
        description="Algorithm for community detection (pySurprise)",
        items=[
            ('cpm', "CPM", "Clique Percolation Method (Traag et al. 2011)"),
            ('infomap', "Infomap", "Map equation (Rosvall & Bergstrom 2008)"),
            ('rb', "RB", "Reichardt-Bornholdt Potts model (2006)"),
            ('rn', "RN", "Ronhovde-Nussinov resolution-free (2010)"),
            ('rnsc', "RNSC", "Restricted Neighbourhood Search Clustering (King et al. 2004)"),
            ('scluster', "SCluster", "Hierarchical clustering (Aldecoa & Marín 2010)"),
            ('uvcluster', "UVCluster", "Iterative cluster analysis (Arnau et al. 2005)"),
        ],
        default='rn',
    ),

    'clustering_resolution': FloatProperty(
        name="Resolution",
        description="Resolution parameter (higher = more communities)",
        default=1.0,
        min=0.1,
        max=5.0,
    ),

    'clustering_seed': IntProperty(
        name="Random Seed",
        description="Seed for reproducibility (0 = random)",
        default=0,
        min=0,
    ),

    'clustering_threshold': FloatProperty(
        name="Threshold",
        description="Convergence threshold for iterative algorithms",
        default=1e-7,
        min=1e-10,
        max=1e-2,
    ),

    # Pathfinding properties
    'pathfinding_source': StringProperty(
        name="Source Node",
        description="Source node index for pathfinding",
        default="0",
    ),

    'pathfinding_target': StringProperty(
        name="Target Node",
        description="Target node index for pathfinding",
        default="1",
    ),

    'pathfinding_algorithm': EnumProperty(
        name="Algorithm",
        description="Pathfinding algorithm to use",
        items=[
            ('DIJKSTRA', "Dijkstra", "Shortest path (non-negative weights)"),
            ('ASTAR', "A*", "A* with Euclidean heuristic"),
            ('BELLMAN_FORD', "Bellman-Ford", "Handles negative weights"),
        ],
        default='DIJKSTRA',
    ),

    # Spanning tree properties
    'spanning_algorithm': EnumProperty(
        name="Algorithm",
        description="Spanning tree algorithm",
        items=[
            ('KRUSKAL', "Kruskal", "Kruskal's MST algorithm"),
            ('PRIM', "Prim", "Prim's MST algorithm"),
            ('MAXIMUM', "Maximum", "Maximum spanning tree"),
        ],
        default='KRUSKAL',
    ),

    # Network flow properties
    'flow_source': StringProperty(
        name="Source Node",
        description="Source node for flow",
        default="0",
    ),

    'flow_sink': StringProperty(
        name="Sink Node",
        description="Sink node for flow",
        default="1",
    ),

    # Rendering properties
    'rendering_preset': EnumProperty(
        name="Preset",
        description="Rendering preset",
        items=[
            ('BASIC', "Basic", "Simple diffuse material"),
            ('GLASS', "Glass", "Transparent glass-like"),
            ('METALLIC', "Metallic", "Metallic reflective"),
            ('EMISSION', "Emission", "Glowing emissive"),
            ('SCIENTIFIC', "Scientific", "Clean scientific visualization"),
        ],
        default='BASIC',
    ),

    'lighting_setup': EnumProperty(
        name="Lighting",
        description="Lighting setup",
        items=[
            ('THREE_POINT', "3-Point", "Classic 3-point lighting"),
            ('STUDIO', "Studio", "Soft studio lighting"),
            ('OUTDOOR', "Outdoor", "Outdoor sun lighting"),
        ],
        default='THREE_POINT',
    ),

    # Export properties
    'export_filepath': StringProperty(
        name="Export Path",
        description="Path for export file",
        subtype='FILE_PATH',
    ),

    'export_format': EnumProperty(
        name="Format",
        description="Export file format",
        items=[
            ('GRAPHML', "GraphML", "GraphML XML format"),
            ('GEXF', "GEXF", "Gephi GEXF format"),
            ('JSON', "JSON", "JSON format"),
            ('CSV', "CSV", "CSV edges list"),
        ],
        default='GRAPHML',
    ),

    'export_include_attributes': BoolProperty(
        name="Include Attributes",
        description="Export node attributes",
        default=True,
    ),

    # Report properties
    'report_include_powerlaw': BoolProperty(
        name="Power Law Analysis",
        description="Include power law fitting in report",
        default=True,
    ),
}
