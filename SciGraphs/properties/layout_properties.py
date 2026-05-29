from bpy.props import (
    StringProperty,
    EnumProperty,
    BoolProperty,
    IntProperty,
    FloatProperty,
)

LAYOUT_PROPERTIES = {
    # ========================================
    # ALGORITHM-SPECIFIC PARAMETERS
    # ========================================

    # --- ForceAtlas2 Parameters ---
    'fa2_scaling_ratio': FloatProperty(
        name="Scaling Ratio",
        description="Ratio de escalado de fuerzas (mayor = más separación)",
        default=2.0,
        min=1.0,
        max=5.0,
    ),

    'fa2_gravity': FloatProperty(
        name="Gravity",
        description="Gravedad hacia el centro (mayor = más compacto)",
        default=1.0,
        min=0.01,
        max=2.0,
    ),

    'fa2_strong_gravity': BoolProperty(
        name="Strong Gravity Mode",
        description="Gravedad fuerte (útil para grafos desconectados)",
        default=False,
    ),

    'fa2_lin_log_mode': BoolProperty(
        name="LinLog Mode",
        description="Modo logarítmico (mejor para grafos muy grandes)",
        default=False,
    ),

    'fa2_barnes_hut_optimize': BoolProperty(
        name="Barnes-Hut Optimization",
        description="Optimización Barnes-Hut (más rápido)",
        default=True,
    ),

    'fa2_barnes_hut_theta': FloatProperty(
        name="Barnes-Hut Theta",
        description="Precisión Barnes-Hut (menor = más preciso, más lento)",
        default=1.2,
        min=0.5,
        max=1.5,
    ),

    'fa2_jitter_tolerance': FloatProperty(
        name="Jitter Tolerance",
        description="Tolerancia de jitter (mayor = más estable)",
        default=1.0,
        min=0.1,
        max=2.0,
    ),

    'fa2_edge_weight_influence': FloatProperty(
        name="Edge Weight Influence",
        description="Influencia del peso de edges",
        default=1.0,
        min=0.0,
        max=2.0,
    ),

    # --- Fruchterman-Reingold (igraph) Parameters ---
    'igraph_fr_start_temp': FloatProperty(
        name="Start Temperature",
        description="Temperatura inicial (fracción, mayor = más movimiento inicial)",
        default=1.0,
        min=0.1,
        max=1.0,
    ),

    'igraph_fr_coolexp': FloatProperty(
        name="Cooling Exponent",
        description="Exponente de enfriamiento",
        default=1.5,
        min=0.8,
        max=1.5,
    ),

    'igraph_fr_maxdelta': FloatProperty(
        name="Max Delta",
        description="Máximo desplazamiento por iteración",
        default=0.0,  # 0 = auto
        min=0.0,
        max=10.0,
    ),

    'igraph_fr_area': FloatProperty(
        name="Area",
        description="Área de la disposición (0 = auto)",
        default=0.0,
        min=0.0,
        max=10000.0,
    ),

    'igraph_fr_repulserad': FloatProperty(
        name="Repulse Radius",
        description="Radio de repulsión (0 = auto)",
        default=0.0,
        min=0.0,
        max=10000.0,
    ),

    # --- Kamada-Kawai (igraph) Parameters ---
    'igraph_kk_maxiter': IntProperty(
        name="Max Iterations",
        description="Máximo de iteraciones (0 = 10*n)",
        default=0,
        min=0,
        max=10000,
    ),

    'igraph_kk_epsilon': FloatProperty(
        name="Epsilon",
        description="Tolerancia de convergencia (0 = auto)",
        default=0.0,
        min=0.0,
        max=1e-2,
    ),

    'igraph_kk_kkconst': FloatProperty(
        name="Spring Constant",
        description="Constante de resorte (0 = n)",
        default=0.0,
        min=0.0,
        max=1000.0,
    ),

    # --- DrL (igraph) Parameters ---
    # Global
    'igraph_drl_edge_cut': FloatProperty(
        name="Edge Cut",
        description="Edge cutting aggressiveness (0 = none, 1 = max). Cuts stressed edges in late phases",
        default=0.8, min=0.0, max=1.0, step=5,
    ),
    # Phase 1: Init
    'igraph_drl_init_iterations': IntProperty(
        name="Init Iterations", description="Iterations in init phase",
        default=0, min=0, max=5000,
    ),
    'igraph_drl_init_temperature': FloatProperty(
        name="Init Temperature", description="Start temperature in init phase",
        default=2000.0, min=0.0, max=10000.0, step=100,
    ),
    'igraph_drl_init_attraction': FloatProperty(
        name="Init Attraction", description="Attraction in init phase",
        default=10.0, min=0.0, max=100.0, step=10,
    ),
    'igraph_drl_init_damping_mult': FloatProperty(
        name="Init Damping", description="Damping multiplier in init phase",
        default=1.0, min=0.0, max=10.0, step=5,
    ),
    # Phase 2: Liquid
    'igraph_drl_liquid_iterations': IntProperty(
        name="Liquid Iterations", description="Iterations in liquid phase",
        default=200, min=0, max=5000,
    ),
    'igraph_drl_liquid_temperature': FloatProperty(
        name="Liquid Temperature", description="Start temperature in liquid phase",
        default=2000.0, min=0.0, max=10000.0, step=100,
    ),
    'igraph_drl_liquid_attraction': FloatProperty(
        name="Liquid Attraction", description="Attraction in liquid phase",
        default=10.0, min=0.0, max=100.0, step=10,
    ),
    'igraph_drl_liquid_damping_mult': FloatProperty(
        name="Liquid Damping", description="Damping multiplier in liquid phase",
        default=1.0, min=0.0, max=10.0, step=5,
    ),
    # Phase 3: Expansion
    'igraph_drl_expansion_iterations': IntProperty(
        name="Expansion Iterations", description="Iterations in expansion phase",
        default=200, min=0, max=5000,
    ),
    'igraph_drl_expansion_temperature': FloatProperty(
        name="Expansion Temperature", description="Start temperature in expansion phase",
        default=2000.0, min=0.0, max=10000.0, step=100,
    ),
    'igraph_drl_expansion_attraction': FloatProperty(
        name="Expansion Attraction", description="Attraction in expansion phase",
        default=2.0, min=0.0, max=100.0, step=10,
    ),
    'igraph_drl_expansion_damping_mult': FloatProperty(
        name="Expansion Damping", description="Damping multiplier in expansion phase",
        default=1.0, min=0.0, max=10.0, step=5,
    ),
    # Phase 4: Cooldown
    'igraph_drl_cooldown_iterations': IntProperty(
        name="Cooldown Iterations", description="Iterations in cooldown phase",
        default=200, min=0, max=5000,
    ),
    'igraph_drl_cooldown_temperature': FloatProperty(
        name="Cooldown Temperature", description="Start temperature in cooldown phase",
        default=2000.0, min=0.0, max=10000.0, step=100,
    ),
    'igraph_drl_cooldown_attraction': FloatProperty(
        name="Cooldown Attraction", description="Attraction in cooldown phase",
        default=1.0, min=0.0, max=100.0, step=10,
    ),
    'igraph_drl_cooldown_damping_mult': FloatProperty(
        name="Cooldown Damping", description="Damping multiplier in cooldown phase",
        default=0.1, min=0.0, max=10.0, step=5,
    ),
    # Phase 5: Crunch
    'igraph_drl_crunch_iterations': IntProperty(
        name="Crunch Iterations", description="Iterations in crunch phase",
        default=50, min=0, max=5000,
    ),
    'igraph_drl_crunch_temperature': FloatProperty(
        name="Crunch Temperature", description="Start temperature in crunch phase",
        default=250.0, min=0.0, max=10000.0, step=100,
    ),
    'igraph_drl_crunch_attraction': FloatProperty(
        name="Crunch Attraction", description="Attraction in crunch phase",
        default=1.0, min=0.0, max=100.0, step=10,
    ),
    'igraph_drl_crunch_damping_mult': FloatProperty(
        name="Crunch Damping", description="Damping multiplier in crunch phase",
        default=0.25, min=0.0, max=10.0, step=5,
    ),
    # Phase 6: Simmer
    'igraph_drl_simmer_iterations': IntProperty(
        name="Simmer Iterations", description="Iterations in simmer phase",
        default=100, min=0, max=5000,
    ),
    'igraph_drl_simmer_temperature': FloatProperty(
        name="Simmer Temperature", description="Start temperature in simmer phase",
        default=250.0, min=0.0, max=10000.0, step=100,
    ),
    'igraph_drl_simmer_attraction': FloatProperty(
        name="Simmer Attraction", description="Attraction in simmer phase",
        default=0.5, min=0.0, max=100.0, step=10,
    ),
    'igraph_drl_simmer_damping_mult': FloatProperty(
        name="Simmer Damping", description="Damping multiplier in simmer phase",
        default=0.0, min=0.0, max=10.0, step=5,
    ),

    # --- Yifan Hu / sfdp Parameters ---
    'sfdp_dim': EnumProperty(
        name="Dimensions",
        description="How to compute the 3D layout",
        items=[
            ('2', "2D (flat)", "2D sfdp, all nodes on z=0 plane"),
            ('2Z', "2D + Z depth", "2D sfdp XY + Z from graph structure (recommended)"),
            ('3', "3D native", "Native 3D sfdp from scigraphs-utils"),
        ],
        default='2Z',
    ),
    'sfdp_z_method': EnumProperty(
        name="Z Method",
        description="How to generate depth (Z axis) from graph structure",
        items=[
            ('SPECTRAL', "Spectral", "3rd eigenvector of graph Laplacian (best structure)"),
            ('DEGREE', "By Degree", "Z based on node degree (hubs rise up)"),
            ('BETWEENNESS', "Betweenness", "Z based on betweenness centrality"),
            ('RANDOM', "Random", "Small random Z for visual depth"),
        ],
        default='SPECTRAL',
    ),
    'sfdp_z_scale': FloatProperty(
        name="Z Scale",
        description="Scale of Z displacement relative to XY spread. 0.3 = 30% of XY range",
        default=0.3, min=0.0, max=3.0, step=5,
    ),
    'sfdp_k': FloatProperty(
        name="K (Edge Length)",
        description="Ideal edge length (spring rest length). Higher = more spread out",
        default=0.3, min=0.01, max=100.0, step=5,
    ),
    'sfdp_repulsive_force': FloatProperty(
        name="Repulsive Force",
        description="Repulsive force exponent. Higher = stronger repulsion between nodes",
        default=1.0, min=0.1, max=50.0, step=10,
    ),
    'sfdp_maxiter': IntProperty(
        name="Max Iterations",
        description="Maximum number of sfdp iterations. More = finer convergence",
        default=600, min=10, max=10000,
    ),
    'sfdp_smoothing': EnumProperty(
        name="Smoothing",
        description="Post-processing smoothing method applied to final layout",
        items=[
            ('none', "None", "No smoothing"),
            ('triangle', "Triangle", "Triangle smoothing (default, good quality)"),
            ('spring', "Spring", "Spring-based smoothing"),
            ('rng', "RNG", "Relative neighborhood graph smoothing"),
            ('power_dist', "Power Distance", "Power distance smoothing"),
        ],
        default='spring',
    ),
    'sfdp_quadtree': EnumProperty(
        name="Quadtree",
        description="Barnes-Hut quadtree scheme for force approximation",
        items=[
            ('normal', "Normal", "Standard quadtree (good quality)"),
            ('fast', "Fast", "Faster but less accurate"),
            ('none', "None", "No quadtree (exact, slow for large graphs)"),
        ],
        default='normal',
    ),
    'sfdp_levels': IntProperty(
        name="Coarsening Levels",
        description="Number of multilevel coarsening levels. More = better global structure",
        default=0, min=0, max=100,
    ),
    'sfdp_beautify': BoolProperty(
        name="Beautify Leaves",
        description="Spread leaf nodes evenly around their parent for cleaner appearance",
        default=False,
    ),
    'sfdp_overlap': EnumProperty(
        name="Overlap Removal",
        description="Method to resolve overlapping nodes after layout",
        items=[
            ('true', "Allow", "Allow overlaps (fastest)"),
            ('prism', "Prism", "Prism algorithm (good balance)"),
            ('scale', "Scale", "Scale up uniformly until no overlap"),
            ('false', "Voronoi", "Voronoi-based removal"),
        ],
        default='scale',
    ),
    'sfdp_overlap_scaling': FloatProperty(
        name="Overlap Scaling",
        description="Scaling factor for overlap removal. Negative = more compact",
        default=-4.0, min=-10.0, max=10.0, step=10,
    ),

    # --- Graphviz / scigraphs-utils Parameters ---
    'graphviz_dimension': EnumProperty(
        name="Dimensions",
        description="SciGraphs dimensionality for Graphviz layouts",
        items=[
            ('2', "2D", "Compute a 2D Graphviz layout"),
            ('2Z', "2D + Z depth", "2D Graphviz XY + Z from graph structure"),
            ('3', "3D native", "Compute a native 3D Graphviz layout when supported by the engine"),
        ],
        default='2',
    ),
    'graphviz_quiet': BoolProperty(
        name="Quiet",
        description="Suppress Graphviz diagnostic output",
        default=True,
    ),
    'graphviz_extra_graph_attrs': StringProperty(
        name="Extra Graph Attrs",
        description="Additional Graphviz graph attributes as key=value pairs separated by commas",
        default="",
    ),
    'graphviz_node_attrs': StringProperty(
        name="Node Attrs",
        description="Graphviz node attributes as key=value pairs separated by commas",
        default="",
    ),
    'graphviz_edge_attrs': StringProperty(
        name="Edge Attrs",
        description="Graphviz edge attributes as key=value pairs separated by commas",
        default="",
    ),
    'graphviz_neato_mode': EnumProperty(
        name="Mode",
        description="Neato mode attribute",
        items=[
            ('DEFAULT', "Default", "Use Graphviz default"),
            ('major', "Major", "Majorization mode"),
            ('KK', "Kamada-Kawai", "Kamada-Kawai mode"),
            ('hier', "Hierarchical", "Hierarchical mode"),
            ('ipsep', "IPSep", "IP separation mode"),
        ],
        default='DEFAULT',
    ),
    'graphviz_neato_model': EnumProperty(
        name="Model",
        description="Neato distance model",
        items=[
            ('DEFAULT', "Default", "Use Graphviz default"),
            ('shortpath', "Shortest Path", "Shortest-path distance model"),
            ('circuit', "Circuit", "Circuit resistance model"),
            ('subset', "Subset", "Subset distance model"),
            ('mds', "MDS", "Multidimensional scaling model"),
        ],
        default='DEFAULT',
    ),
    'graphviz_neato_start': StringProperty(
        name="Start",
        description="Neato start attribute, for example a seed or 'random'",
        default="",
    ),
    'graphviz_neato_maxiter': IntProperty(
        name="Max Iterations",
        description="Maximum neato iterations (0 = Graphviz default)",
        default=0, min=0, max=100000,
    ),
    'graphviz_fdp_start': StringProperty(
        name="Start",
        description="FDP start attribute, for example a seed or 'random'",
        default="",
    ),
    'graphviz_dot_directed': BoolProperty(
        name="Directed",
        description="Use a directed Graphviz graph for dot",
        default=True,
    ),
    'graphviz_dot_rankdir': EnumProperty(
        name="Rank Direction",
        description="Direction of dot ranks",
        items=[
            ('TB', "Top to Bottom", "Top to bottom"),
            ('BT', "Bottom to Top", "Bottom to top"),
            ('LR', "Left to Right", "Left to right"),
            ('RL', "Right to Left", "Right to left"),
        ],
        default='TB',
    ),
    'graphviz_dot_ranksep': FloatProperty(
        name="Rank Separation",
        description="Distance between ranks (0 = Graphviz default)",
        default=0.0, min=0.0, max=100.0,
    ),
    'graphviz_dot_nodesep': FloatProperty(
        name="Node Separation",
        description="Distance between nodes in the same rank (0 = Graphviz default)",
        default=0.0, min=0.0, max=100.0,
    ),
    'graphviz_dot_splines': EnumProperty(
        name="Splines",
        description="How dot routes edges",
        items=[
            ('false', "False", "Straight edges"),
            ('true', "True", "Spline edges"),
            ('line', "Line", "Line segments"),
            ('polyline', "Polyline", "Polyline routing"),
            ('ortho', "Orthogonal", "Orthogonal routing"),
            ('curved', "Curved", "Curved routing"),
        ],
        default='false',
    ),
    'graphviz_twopi_root': StringProperty(
        name="Root",
        description="Root node index for twopi radial layout (empty = Graphviz default)",
        default="",
    ),
    'graphviz_twopi_ranksep': FloatProperty(
        name="Rank Separation",
        description="Radial separation between ranks (0 = Graphviz default)",
        default=0.0, min=0.0, max=100.0,
    ),
    'graphviz_circo_mindist': FloatProperty(
        name="Minimum Distance",
        description="Minimum node separation for circo (0 = Graphviz default)",
        default=0.0, min=0.0, max=100.0,
    ),
    'graphviz_osage_pack': BoolProperty(
        name="Pack",
        description="Pack connected components or clusters for osage",
        default=True,
    ),
    'graphviz_osage_packmode': EnumProperty(
        name="Pack Mode",
        description="Osage packing mode",
        items=[
            ('DEFAULT', "Default", "Use Graphviz default"),
            ('node', "Node", "Pack around nodes"),
            ('clust', "Cluster", "Pack around clusters"),
            ('graph', "Graph", "Pack around graphs"),
            ('array', "Array", "Array packing"),
        ],
        default='array',
    ),

    # --- LGL (igraph) Parameters ---
    'igraph_lgl_maxiter': IntProperty(
        name="Max Iterations",
        description="Máximo de iteraciones",
        default=150,
        min=10,
        max=1000,
    ),

    'igraph_lgl_maxdelta': FloatProperty(
        name="Max Delta",
        description="Máximo desplazamiento (0 = n)",
        default=0.0,
        min=0.0,
        max=10000.0,
    ),

    'igraph_lgl_area': FloatProperty(
        name="Area",
        description="Área deseada (0 = n²)",
        default=0.0,
        min=0.0,
        max=100000.0,
    ),

    'igraph_lgl_coolexp': FloatProperty(
        name="Cool Exponent",
        description="Exponente de enfriamiento",
        default=1.5,
        min=0.8,
        max=2.0,
    ),

    'igraph_lgl_repulserad': FloatProperty(
        name="Repulse Radius",
        description="Radio de repulsión (0 = n³)",
        default=0.0,
        min=0.0,
        max=100000.0,
    ),

    'igraph_lgl_cellsize': FloatProperty(
        name="Cell Size",
        description="Tamaño de celda para acelerar (0 = sqrt(n))",
        default=0.0,
        min=0.0,
        max=1000.0,
    ),

    # --- Davidson-Harel (igraph) Parameters ---
    'igraph_dh_maxiter': IntProperty(
        name="Max Iterations",
        description="Número de iteraciones",
        default=10,
        min=10,
        max=100000,
    ),

    'igraph_dh_fineiter': IntProperty(
        name="Fine Iterations",
        description="Iteraciones de refinamiento fino",
        default=0,
        min=0,
        max=1000,
    ),

    'igraph_dh_cool_fact': FloatProperty(
        name="Cooling Factor",
        description="Factor de enfriamiento",
        default=0.95,
        min=0.9,
        max=0.99,
    ),

    'igraph_dh_weight_node_dist': FloatProperty(
        name="Weight: Node Distance",
        description="Peso de distancia entre nodos",
        default=1.0,
        min=0.0,
        max=2.0,
    ),

    'igraph_dh_weight_border': FloatProperty(
        name="Weight: Border",
        description="Peso de distancia al borde",
        default=0.0,
        min=0.0,
        max=2.0,
    ),

    'igraph_dh_weight_edge_lengths': FloatProperty(
        name="Weight: Edge Lengths",
        description="Peso de longitud de edges",
        default=1.0,
        min=0.0,
        max=2.0,
    ),

    'igraph_dh_weight_edge_crossings': FloatProperty(
        name="Weight: Edge Crossings",
        description="Peso de cruces de edges",
        default=1.0,
        min=0.0,
        max=2.0,
    ),

    'igraph_dh_weight_node_edge_dist': FloatProperty(
        name="Weight: Node-Edge Distance",
        description="Peso de distancia nodo-edge",
        default=1.0,
        min=0.0,
        max=2.0,
    ),

    # --- GraphOpt (igraph) Parameters ---
    'igraph_graphopt_niter': IntProperty(
        name="Iterations",
        description="Número de iteraciones",
        default=500,
        min=100,
        max=5000,
    ),

    'igraph_graphopt_node_charge': FloatProperty(
        name="Node Charge",
        description="Carga de nodos (menor = más agrupado)",
        default=0.001,
        min=0.0001,
        max=0.1,
    ),

    'igraph_graphopt_node_mass': FloatProperty(
        name="Node Mass",
        description="Masa de nodos",
        default=30.0,
        min=1.0,
        max=100.0,
    ),

    'igraph_graphopt_spring_length': FloatProperty(
        name="Spring Length",
        description="Longitud de resorte (0 = auto)",
        default=0.0,
        min=0.0,
        max=100.0,
    ),

    'igraph_graphopt_spring_constant': FloatProperty(
        name="Spring Constant",
        description="Constante de resorte",
        default=1.0,
        min=0.001,
        max=10.0,
    ),

    'igraph_graphopt_max_sa_movement': FloatProperty(
        name="Max SA Movement",
        description="Máximo movimiento",
        default=5.0,
        min=0.1,
        max=10.0,
    ),

    # --- Yifan Hu Parameters ---
    'yifan_hu_spring_constant': FloatProperty(
        name="Spring Constant (K)",
        description="Longitud de resorte ideal",
        default=0.3,
        min=0.01,
        max=10.0,
    ),

    'yifan_hu_step_ratio': FloatProperty(
        name="Step Ratio",
        description="Ratio de paso",
        default=1.0,
        min=0.1,
        max=5.0,
    ),

    'yifan_hu_adaptive_cooling': BoolProperty(
        name="Adaptive Cooling",
        description="Enfriamiento adaptativo",
        default=True,
    ),

    # --- Hierarchical 3D Parameters ---
    'hierarchical_layer_height': FloatProperty(
        name="Layer Height",
        description="Altura entre capas jerárquicas",
        default=1.0,
        min=0.5,
        max=5.0,
    ),

    'hierarchical_mode': EnumProperty(
        name="Direction Mode",
        description="Modo para grafos dirigidos",
        items=[
            ('out', "Out", "Desde raíz hacia fuera"),
            ('in', "In", "Hacia raíz"),
            ('all', "All", "Ignorar dirección"),
        ],
        default='out',
    ),

    # --- MDS 3D Parameters ---
    'mds_dissimilarity': EnumProperty(
        name="Dissimilarity",
        description="Métrica de disimilitud",
        items=[
            ('euclidean', "Euclidean", "Distancia euclidiana"),
            ('precomputed', "Precomputed", "Matriz precomputada"),
        ],
        default='euclidean',
    ),

    # --- Spectral Parameters ---
    'spectral_weight_attr': StringProperty(
        name="Weight Attribute",
        description="Atributo de peso de edges (vacío = sin peso)",
        default="",
    ),
}


def _without_animation(property_def):
    factory = getattr(property_def, "function", None)
    keywords = getattr(property_def, "keywords", None)
    if factory is None or keywords is None:
        return property_def

    keywords = dict(keywords)
    options = set(keywords.get("options", set()))
    options.discard('ANIMATABLE')
    keywords["options"] = options
    return factory(**keywords)


LAYOUT_PROPERTIES = {
    name: _without_animation(property_def)
    for name, property_def in LAYOUT_PROPERTIES.items()
}
