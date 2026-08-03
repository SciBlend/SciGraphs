from bpy.props import (
    StringProperty,
    EnumProperty,
    BoolProperty,
    IntProperty,
    FloatProperty,
)


def _gpu_style_refresh(self, context):
    """Repaint the GPU preview when a style parameter changes.

    The GPU render path bakes these parameters into its batch-cache
    signature, so a simple redraw is enough: the next draw notices the
    signature change and re-tessellates the edges.
    """
    try:
        from ..ui.gpu_render.state import tag_redraw
        tag_redraw()
    except Exception:  # noqa: BLE001 - never break the properties UI
        pass


EDGE_STYLE_PROPERTIES = {
    # ========================================
    # EDGE STYLE PROPERTIES
    # ========================================

    'edge_style_type': EnumProperty(
        name="Edge Style",
        description="Visual style for graph edges",
        items=[
            ('STRAIGHT', "Straight", "Direct straight lines between nodes"),
            ('CURVED', "Curved (Bezier)", "Smooth cubic Bezier curves"),
            ('QUADRATIC', "Quadratic", "Quadratic Bezier curves (simpler, faster)"),
            ('ARC', "Arc", "Circular arc segments"),
            ('BUNDLED', "Bundled (FDEB)", "Force-directed edge bundling: edges attract each other pairwise (Holten & van Wijk 2009)"),
            ('HIERARCHICAL', "Bundled (Hierarchical)", "Route edges along the cluster tree, from each endpoint up to the least common ancestor and back down (Holten 2006). Needs a clustering attribute; scales to any edge count"),
            ('FDEB', "Bundled (Force-Directed, GPU)", "The same forces as FDEB, relaxed in compute shaders against a spatial neighbourhood instead of an all-pairs matrix. No clustering attribute needed, and no 4096-edge ceiling"),
            ('SBEB', "Bundled (Skeleton, image-based)", "Attract edges toward the skeleton of their own drawn density, computed in image space (Ersoy et al. 2011). Its cost follows the raster resolution rather than the edge count"),
            ('ROUTED', "Bundled (Routed roads)", "Route edges along a grid and make shared stretches cheaper each round, so they converge onto exactly coincident roads (Lambert et al. 2010). Can be told to steer clear of crowded regions"),
            ('MINGLE', "Bundled (Agglomerative, ink)", "Merge edges into a tree of shared trunks, choosing at each step the merge that saves the most drawn ink (Gansner et al. 2011). Meeting points are explicit and exactly shared, which reads differently from the soft merging of the force-directed families"),
            ('TAPERED', "Tapered", "Variable thickness along edge (for directed graphs)"),
            ('ORTHOGONAL', "Orthogonal", "Right-angle (90°) connections"),
        ],
        default='STRAIGHT',
        update=_gpu_style_refresh,
    ),

    'edge_style_preset': EnumProperty(
        name="Preset",
        description="Pre-configured edge style settings",
        items=[
            ('CUSTOM', "Custom", "Use manual settings"),
            ('GEPHI_DEFAULT', "Gephi Default", "Gephi-style curved edges"),
            ('CYTOSCAPE_BEZIER', "Cytoscape Bezier", "Cytoscape-style Bezier curves"),
            ('SCHEMATIC', "Schematic", "Technical diagram style with orthogonal edges"),
            ('BUNDLED_DENSE', "Bundled (Dense)", "Strong bundling for very dense graphs"),
            ('FLOW_DIAGRAM', "Flow Diagram", "Tapered edges showing direction"),
            ('MINIMAL', "Minimal", "Clean straight lines"),
        ],
        default='CUSTOM',
    ),

    'edge_curvature': FloatProperty(
        name="Curvature",
        description="Intensity of edge curvature (0 = straight, 1 = maximum curve)",
        default=0.3,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_gpu_style_refresh,
    ),

    'edge_segments': IntProperty(
        name="Segments",
        description="Number of segments for curved edges (more = smoother but heavier)",
        default=8,
        min=2,
        max=32,
        update=_gpu_style_refresh,
    ),

    'edge_curve_direction': EnumProperty(
        name="Curve Direction",
        description="Direction of the curve offset",
        items=[
            ('AUTO', "Auto", "Automatic based on node positions"),
            ('CLOCKWISE', "Clockwise", "Curve bends clockwise"),
            ('COUNTER_CLOCKWISE', "Counter-Clockwise", "Curve bends counter-clockwise"),
            ('ALTERNATING', "Alternating", "Alternate direction for parallel edges"),
        ],
        default='AUTO',
        update=_gpu_style_refresh,
    ),

    'edge_parallel_offset': FloatProperty(
        name="Parallel Offset",
        description="Offset distance for parallel/multi-edges between same nodes",
        default=0.05,
        min=0.0,
        max=0.5,
        update=_gpu_style_refresh,
    ),

    'edge_auto_offset_parallel': BoolProperty(
        name="Auto-Offset Parallel",
        description="Automatically offset parallel edges to avoid overlap",
        default=True,
        update=_gpu_style_refresh,
    ),

    'edge_bundle_strength': FloatProperty(
        name="Bundle Strength",
        description="Strength of edge bundling (0 = no bundling, 1 = maximum)",
        default=0.6,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_gpu_style_refresh,
    ),

    'edge_bundle_iterations': IntProperty(
        name="Bundle Iterations",
        description="Number of iterations for edge bundling algorithm",
        default=6,
        min=1,
        max=20,
        update=_gpu_style_refresh,
    ),

    'edge_bundle_compatibility_threshold': FloatProperty(
        name="Compatibility Threshold",
        description="Minimum compatibility score for edges to be bundled together",
        default=0.6,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_gpu_style_refresh,
    ),

    # --- Hierarchical edge bundling (Holten 2006) ---------------------------

    'edge_heb_beta': FloatProperty(
        name="Bundling Strength",
        description=(
            "How closely edges follow the cluster tree. 1 hugs the hierarchy "
            "and gives the tightest bundles; 0 straightens them back to "
            "direct lines. Low values read node-to-node, high values read "
            "cluster-to-cluster"
        ),
        default=0.85,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_gpu_style_refresh,
    ),

    'edge_heb_remove_lca': BoolProperty(
        name="Skip Common Ancestor",
        description=(
            "Do not route through the meeting point of the two endpoints, so "
            "links inside one cluster stop detouring via its centre. Ignored "
            "for the shortest paths, where it would collapse every such link "
            "onto the same straight line"
        ),
        default=True,
        update=_gpu_style_refresh,
    ),

    'edge_heb_fade_long': FloatProperty(
        name="Fade Long Edges",
        description=(
            "Draw long curves more transparently than short ones, so "
            "long-range links stop painting over the local structure they "
            "fly past. 0 keeps every edge at the same opacity"
        ),
        default=0.6,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_gpu_style_refresh,
    ),

    'edge_heb_gradient': EnumProperty(
        name="Gradient",
        description=(
            "Colour along each curve. Curvature already carries the bundling "
            "and cannot also show direction, so colour does it instead"
        ),
        items=[
            ('NONE', "None", "One flat edge colour"),
            ('DIRECTION', "Direction",
             "Green at the source, red at the destination"),
            ('NODES', "Endpoint Nodes",
             "Blend the colours of the two nodes, so a strand shows which "
             "cluster it leaves and which it reaches"),
        ],
        default='NODES',
        update=_gpu_style_refresh,
    ),

    # --- Shared by every bundling mode -------------------------------------
    #
    # These act on the control polygon, whichever family produced it, so they
    # are deliberately not prefixed with one mode's name.

    'edge_bundle_turn_limit': FloatProperty(
        name="Turning Limit",
        description=(
            "Largest corner a routed edge may turn through, in degrees. Stops "
            "distant links from doubling back to join a bundle they only just "
            "reach. 180 leaves the route exactly as the algorithm produced it"
        ),
        # Plain degrees rather than subtype='ANGLE', which would make Blender
        # store and show radians and put a unit on a control whose useful range
        # the user reads in degrees.
        default=180.0,
        min=20.0,
        max=180.0,
        update=_gpu_style_refresh,
    ),

    'edge_bundle_adaptive_beta': FloatProperty(
        name="Strength Follows Length",
        description=(
            "Bundle long links harder than short ones. A link across the whole "
            "graph is what bundling is for; a link between neighbours only "
            "loses its direct reading by being routed. 0 gives every edge the "
            "same strength"
        ),
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_gpu_style_refresh,
    ),

    'edge_bundle_company': FloatProperty(
        name="Only Bundle With Company",
        description=(
            "Leave an edge straight when nothing else is going its way. "
            "Bundling a link that joins no bundle only costs: it is bent away "
            "from the line it could have been read along and joins nothing. 0 "
            "bundles every edge equally, company or not"
        ),
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_gpu_style_refresh,
    ),

    'edge_bundle_view_beta': FloatProperty(
        name="Strength Follows Zoom",
        description=(
            "Bundle hard when the graph is small on screen and let it "
            "straighten as you zoom in, where the bending only makes an edge "
            "harder to follow. Costs nothing per frame: the strength is already "
            "a shader uniform, so this changes no geometry and rebuilds nothing"
        ),
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_gpu_style_refresh,
    ),

    'edge_bundle_density_opacity': FloatProperty(
        name="Fade Crowded Routes",
        description=(
            "Draw an edge more transparently the more of the graph shares its "
            "route, so a dense bundle stops reading as one opaque mass. 0 "
            "keeps every edge at the opacity its colour asks for"
        ),
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_gpu_style_refresh,
    ),

    # --- Force-directed bundling on the GPU (FDEB) --------------------------

    'edge_fdeb_cycles': IntProperty(
        name="Cycles",
        description=(
            "Rounds of the schedule. Each one doubles the number of points a "
            "curve is free to bend at and halves how far they may move, so "
            "more cycles buy detail rather than more bending"
        ),
        default=6,
        min=1,
        max=10,
        update=_gpu_style_refresh,
    ),

    'edge_fdeb_radius': FloatProperty(
        name="Attraction Radius",
        description=(
            "How far an edge looks for company, as a fraction of the graph's "
            "diagonal. This is what replaces the all-pairs matrix: the "
            "attraction falls off with distance anyway, so cutting it off at a "
            "radius costs accuracy that is bounded and buys the edge count"
        ),
        default=0.05,
        min=0.005,
        max=0.5,
        subtype='FACTOR',
        update=_gpu_style_refresh,
    ),

    'edge_fdeb_threshold': FloatProperty(
        name="Compatibility",
        description=(
            "How compatible two edges must be before they attract each other. "
            "Separate from the setting the CPU bundling uses because it is not "
            "the same number: the visibility test multiplies into the score, so "
            "a threshold chosen against three terms is strictly stricter once "
            "there are four, and the CPU default leaves almost nothing above it"
        ),
        # Holten's own is 0.05. This sits above it because the neighbourhood
        # radius already excludes the distant pairs a low threshold would
        # otherwise admit, and below the CPU default for the reason above.
        default=0.25,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_gpu_style_refresh,
    ),

    'edge_fdeb_visibility': BoolProperty(
        name="Visibility Compatibility",
        description=(
            "Refuse to bundle edges that pass each other rather than run "
            "together, even when their angle and length agree. The CPU path "
            "skips this test for speed; on the GPU it is affordable, and it is "
            "the one that most improves the result"
        ),
        default=True,
        update=_gpu_style_refresh,
    ),

    # --- Agglomerative bundling by ink minimisation (MINGLE) ----------------

    'edge_mingle_neighbours': IntProperty(
        name="Candidates",
        description=(
            "How many nearby edges each one is offered as merge partners. More "
            "does not reliably mean tighter: measured, the ink saved is flat to "
            "slightly worse above about ten, because taking the largest saving "
            "first is not the best set of merges and one big merge can block "
            "two that were worth more together"
        ),
        default=10,
        min=1,
        max=16,
        update=_gpu_style_refresh,
    ),

    'edge_mingle_rounds': IntProperty(
        name="Rounds",
        description=(
            "How many times the merged bundles are themselves offered for "
            "merging. The recursion stops on its own when no merge saves any "
            "ink, so raising this past that point costs nothing"
        ),
        default=4,
        min=1,
        max=10,
        update=_gpu_style_refresh,
    ),

    'edge_mingle_min_gain': FloatProperty(
        name="Minimum Saving",
        description=(
            "Refuse a merge that saves less than this share of the two edges' "
            "unbundled ink. 0 takes every merge that saves anything at all"
        ),
        default=0.0,
        min=0.0,
        max=0.99,
        subtype='FACTOR',
        update=_gpu_style_refresh,
    ),

    # --- Skeleton-based bundling in image space (SBEB) ----------------------

    'edge_sbeb_resolution': IntProperty(
        name="Raster Resolution",
        description=(
            "Side of the image the bundling is computed in. This, and not the "
            "edge count, is what the cost of this mode follows"
        ),
        default=512,
        min=64,
        max=2048,
        update=_gpu_style_refresh,
    ),

    'edge_sbeb_iterations': IntProperty(
        name="Iterations",
        description="Rounds of skeletonisation and attraction",
        default=5,
        min=1,
        max=20,
        update=_gpu_style_refresh,
    ),

    'edge_sbeb_clusters': IntProperty(
        name="Edge Clusters",
        description=(
            "How many groups the edges are split into before each group is "
            "drawn and skeletonised. Too few and unrelated edges share a "
            "skeleton; too many and nothing bundles"
        ),
        default=16,
        min=1,
        max=256,
        update=_gpu_style_refresh,
    ),

    'edge_sbeb_threshold': FloatProperty(
        name="Shape Threshold",
        description=(
            "How much drawn density a pixel needs before it counts as inside "
            "the shape the skeleton is taken from"
        ),
        default=0.1,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_gpu_style_refresh,
    ),

    'edge_sbeb_attraction': FloatProperty(
        name="Attraction",
        description="How far toward its skeleton point an edge moves each round",
        default=0.6,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_gpu_style_refresh,
    ),

    'edge_sbeb_smooth': IntProperty(
        name="Smoothing Passes",
        description=(
            "Laplacian passes over each curve after attraction. Skeletons have "
            "corners and the curves inherit them; this is what takes them out"
        ),
        default=2,
        min=0,
        max=10,
        update=_gpu_style_refresh,
    ),

    'edge_sbeb_recluster': BoolProperty(
        name="Recluster Each Round",
        description=(
            "Recompute the edge groups every iteration. The only step that is "
            "not image-based, so it is also the only one whose cost grows with "
            "the edge count"
        ),
        default=False,
        update=_gpu_style_refresh,
    ),

    # --- Routed bundling over a grid ----------------------------------------

    'edge_routed_resolution': IntProperty(
        name="Grid Resolution",
        description="Cells per axis of the grid the routes are found on",
        default=128,
        min=8,
        max=512,
        update=_gpu_style_refresh,
    ),

    'edge_routed_iterations': IntProperty(
        name="Routing Rounds",
        description=(
            "How many times every edge is re-routed. Each round makes the "
            "stretches already in use cheaper, which is what pulls separate "
            "paths onto a shared road"
        ),
        default=4,
        min=1,
        max=16,
        update=_gpu_style_refresh,
    ),

    'edge_routed_reinforce': FloatProperty(
        name="Reinforcement",
        description=(
            "How much cheaper a stretch of grid becomes for having been used. "
            "0 leaves every round independent and nothing converges"
        ),
        # Stops below 1: full reinforcement makes a used cell free, and every
        # route after the first then collapses onto the first one's road
        # whatever it costs to reach it.
        default=0.5,
        min=0.0,
        max=0.99,
        subtype='FACTOR',
        update=_gpu_style_refresh,
    ),

    'edge_routed_avoid_nodes': FloatProperty(
        name="Avoid Crowded Regions",
        description=(
            "Make cells expensive in proportion to how many nodes sit in them, "
            "so routes bend around dense clusters instead of crossing them. 0 "
            "ignores where the nodes are"
        ),
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_gpu_style_refresh,
    ),

    'edge_taper_start': FloatProperty(
        name="Taper Start",
        description="Relative thickness at edge start (for tapered edges)",
        default=1.0,
        min=0.1,
        max=3.0,
        update=_gpu_style_refresh,
    ),

    'edge_taper_end': FloatProperty(
        name="Taper End",
        description="Relative thickness at edge end (for tapered edges)",
        default=0.3,
        min=0.1,
        max=3.0,
        update=_gpu_style_refresh,
    ),

    'edge_self_loop_radius': FloatProperty(
        name="Self-Loop Radius",
        description="Radius of self-loop edges (edges connecting a node to itself)",
        default=0.2,
        min=0.05,
        max=1.0,
        update=_gpu_style_refresh,
    ),

    'edge_orthogonal_style': EnumProperty(
        name="Orthogonal Style",
        description="Style of orthogonal (right-angle) edges",
        items=[
            ('HORIZONTAL_FIRST', "Horizontal First", "Go horizontal then vertical"),
            ('VERTICAL_FIRST', "Vertical First", "Go vertical then horizontal"),
            ('SHORTEST', "Shortest Path", "Choose based on shortest total length"),
            ('CENTERED', "Centered", "Meet at midpoint with two bends"),
        ],
        default='CENTERED',
        update=_gpu_style_refresh,
    ),

    # ========================================
    # NODE ATTRIBUTE IMPORT
    # ========================================

    'node_attr_filepath': StringProperty(
        name="Node Attribute File",
        description="Path to a file containing vertex-only attributes (node_name, value columns)",
        subtype='FILE_PATH',
    ),

    'node_attr_delimiter': EnumProperty(
        name="Delimiter",
        description="Column separator for the node attribute file",
        items=[
            ('\t', "Tab", "Tab-separated values"),
            (',', "Comma (,)", "Comma-separated values"),
            (';', "Semicolon (;)", "Semicolon-separated values"),
            (' ', "Space", "Space-separated values"),
        ],
        default='\t',
    ),

    'node_attr_has_header': BoolProperty(
        name="Has Header Row",
        description="Whether the first row contains column names",
        default=False,
    ),

    'edge_style_affect_existing': BoolProperty(
        name="Affect Existing Geometry",
        description="Apply style to existing edge geometry (vs only new edges)",
        default=True,
    ),

    'edge_style_preserve_osmnx': BoolProperty(
        name="Preserve OSMnx Curves",
        description="Keep original street geometry for OSMnx graphs instead of overwriting",
        default=True,
    ),
}
