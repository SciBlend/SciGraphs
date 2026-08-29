import bpy

from . import draw, filters
from ...core.render import adaptive
from .state import tag_redraw
from .toon import properties as toon_properties


def _on_enabled_update(self, context):
    if getattr(self, "scigraphs_preview_enabled", False):
        draw.enable_preview()
    else:
        draw.disable_preview()


def _on_setting_update(self, context):
    tag_redraw()


def _volume_show_get(self):
    return self.scigraphs_preview_volume_mode != 'OFF'


def _volume_show_set(self, value):
    if value:
        self.scigraphs_preview_volume_mode = (
            self.scigraphs_preview_volume_last or 'COUNT')
        return
    if self.scigraphs_preview_volume_mode != 'OFF':
        self.scigraphs_preview_volume_last = self.scigraphs_preview_volume_mode
    self.scigraphs_preview_volume_mode = 'OFF'


def _on_rebuild_update(self, context):
    draw.invalidate()


def _on_hide_mesh_update(self, context):
    draw.sync_native_display(context)
    tag_redraw()


def _on_engine_update(self, context):
    draw.apply_display_engine(context)


class SCIGRAPHS_PG_filter_slot(bpy.types.PropertyGroup):
    """A channel and a range, normalized to [0, 1] so one slider pair serves
    every channel and the stack reaches the shader unscaled."""

    channel: bpy.props.EnumProperty(
        name="Channel",
        description="What this clause filters on",
        items=filters.CHANNEL_ITEMS,
        default='DEGREE', update=_on_rebuild_update,
    )
    attr_name: bpy.props.StringProperty(
        name="Attribute",
        description=(
            "Which attribute, for the attribute channels. For Edge Spans "
            "Groups it is the grouping attribute, and empty means the "
            "coarsening one"
        ),
        default="", update=_on_rebuild_update,
    )
    range_min: bpy.props.FloatProperty(
        name="Min", description="Low end of the kept range",
        default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_on_rebuild_update,
    )
    range_max: bpy.props.FloatProperty(
        name="Max", description="High end of the kept range",
        default=1.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_on_rebuild_update,
    )
    invert: bpy.props.BoolProperty(
        name="Invert",
        description=(
            "Keep what falls outside the range instead of inside it. This is "
            "what stands in for a union: the stack is a conjunction, and the "
            "complement of one clause covers most of the cases where an OR was "
            "wanted"
        ),
        default=False, update=_on_rebuild_update,
    )
    mute: bpy.props.BoolProperty(
        name="Mute",
        description="Leave the clause configured but stop applying it",
        default=False, update=_on_rebuild_update,
    )


ANIMATABLE_ITEMS = [
    ('FORCEATLAS2', "ForceAtlas2", "Gephi's model: linear attraction, mass-weighted repulsion, adaptive speed"),
    ('SPRING', "Spring (2D)", "Fruchterman-Reingold, flat"),
    ('SPRING_3D', "Spring (3D)", "Fruchterman-Reingold in three dimensions"),
    ('IGRAPH_FR', "Fruchterman-Reingold", "The same model, entered from the igraph menu entry"),
]


def _assert_animatable_in_sync():
    """A model added to playback and not here would be silently unreachable."""
    try:
        from .playback import ITERATIVE_MODELS
    except Exception:  # noqa: BLE001 - registration must not depend on it
        return
    listed = {item[0] for item in ANIMATABLE_ITEMS}
    missing = set(ITERATIVE_MODELS) - listed
    extra = listed - set(ITERATIVE_MODELS)
    if missing or extra:
        print(f"  Animated Layout menu is out of step with the force models: "
              f"missing {sorted(missing)}, unreachable {sorted(extra)}")


LABEL_PRIORITY_ITEMS = [
    ('DISTANCE', "Camera Distance",
     "Nearest to the camera wins, which is what the code did before there was "
     "a choice"),
]
try:
    from . import filters as _filters
    LABEL_PRIORITY_ITEMS.extend(_filters.CHANNEL_ITEMS)
except Exception:  # noqa: BLE001 - the menu must exist regardless
    pass


def register_properties():
    _assert_animatable_in_sync()
    # Before the scene properties: the collection needs its type to exist.
    bpy.utils.register_class(SCIGRAPHS_PG_filter_slot)

    S = bpy.types.Scene
    S.scigraphs_filters = bpy.props.CollectionProperty(
        type=SCIGRAPHS_PG_filter_slot, name="Filter Stack",
        description="Clauses combined with AND to decide what is drawn",
    )
    S.scigraphs_filters_index = bpy.props.IntProperty(
        name="Active Filter", default=0, min=0,
    )
    S.scigraphs_display_engine = bpy.props.EnumProperty(
        name="Display Engine",
        description="How graphs are shown in the viewport",
        items=[
            ('GPU', "GPU (Fast)",
             "Draw nodes/edges directly on the GPU. Fast, interactive, ideal "
             "for exploring large graphs. Not shown in Cycles/EEVEE renders",
             'SHADING_RENDERED', 0),
            ('GEOMETRY_NODES', "CPU (Geometry Nodes)",
             "Instanced spheres/tubes via Geometry Nodes. Slower but renderable "
             "in Cycles/EEVEE with full materials",
             'GEOMETRY_NODES', 1),
        ],
        default='GPU', update=_on_engine_update,
    )
    S.scigraphs_preview_enabled = bpy.props.BoolProperty(
        name="GPU Preview",
        description="Draw the active graph as fast GPU points and lines",
        default=True, update=_on_enabled_update,
    )
    S.scigraphs_preview_animate = bpy.props.BoolProperty(
        name="Animated Layout",
        description=(
            "Keep node positions in a GPU texture instead of in the vertex "
            "buffers, so a moving layout refreshes one texture per frame "
            "instead of rebuilding every batch. Ignored where the drawn picture "
            "would not be identical (edge styles, density cloud, coarsening, "
            "spatial blocks)"
        ),
        default=False, update=_on_rebuild_update,
    )
    S.scigraphs_preview_animate_gpu = bpy.props.BoolProperty(
        name="Simulate on GPU",
        description=(
            "Run the layout simulation in compute shaders. Measured at 18x the "
            "numpy simulator on 400k nodes and 9.5x on 100k; below 20k the "
            "difference is imperceptible and the CPU path is used regardless"
        ),
        default=True,
        update=_on_setting_update,
    )
    S.scigraphs_preview_animate_algorithm = bpy.props.EnumProperty(
        name="Algorithm",
        description=(
            "Which force model the animation simulates. Only the algorithms "
            "with an iterative form appear here: the rest compute a final "
            "layout in one shot and have no intermediate states to show"
        ),
        items=ANIMATABLE_ITEMS,
        default='FORCEATLAS2',
        update=_on_setting_update,
    )
    S.scigraphs_preview_repulsion = bpy.props.EnumProperty(
        name="Repulsion",
        description=(
            "How the long-range repulsion is approximated above a few hundred "
            "nodes. Below that it is computed exactly either way"
        ),
        items=[
            ('GRID', "Uniform grid",
             "One monopole per occupied cell of a fixed 8x8x8 grid. Cheap and "
             "stale: the cell count never changes, so nodes per cell grow with "
             "the graph and the error grows with them, reaching 39% on a "
             "settled hundred-thousand-node layout"),
            ('TREE', "Barnes-Hut tree",
             "Graphviz's own quadtree, subdividing where the nodes are. Needs "
             "scigraphs-utils 0.2 or newer; without it the grid is used and "
             "nothing complains"),
        ],
        default='GRID',
        update=_on_setting_update,
    )
    S.scigraphs_preview_theta = bpy.props.FloatProperty(
        name="Theta",
        description=(
            "Barnes-Hut opening angle. A cell is used whole when it is this "
            "much narrower than its distance, so smaller is more accurate and "
            "slower. Measured against exact all-pairs, 0.6 lands within 5% in "
            "2D and 20% in 3D, and 1.2 roughly doubles both"
        ),
        default=0.6, min=0.0, max=4.0,
        update=_on_setting_update,
    )
    S.scigraphs_preview_grid_show = bpy.props.BoolProperty(
        name="Show Layout Grid",
        description=(
            "Draw the spatial structure the running simulation reads its "
            "repulsion off. It is a uniform grid, not a tree: the cell count is "
            "fixed, so the nodes per cell grow with the graph, and seeing which "
            "cell has swallowed the layout is the point"
        ),
        default=False,
        update=_on_setting_update,
    )
    S.scigraphs_preview_grid_mode = bpy.props.EnumProperty(
        name="Structure",
        description="Which of the two the overlay draws",
        items=[
            ('FAR', "Far field", "The coarse monopole grid, one box per occupied cell"),
            ('NEAR', "Near field", "The bin grid behind the neighbor list. GPU only: the numpy path asks a kd-tree instead, and a kd-tree has no boxes"),
            ('BOTH', "Both", "The coarse grid and the bins together"),
            ('TREE', "Barnes-Hut tree",
             "Graphviz's quadtree, subdividing where the nodes are. With "
             "Repulsion set to the tree these are the cells the forces came "
             "from; with it on the grid they are a preview of what the tree "
             "would do, not what ran"),
        ],
        default='FAR',
        update=_on_setting_update,
    )
    S.scigraphs_preview_tree_depth = bpy.props.IntProperty(
        name="Tree Depth",
        description=(
            "How far down the quadtree the overlay draws. Graphviz splits until "
            "a leaf holds one point, so the deepest cells are far under a pixel "
            "and the whole thing reads as a flat mesh; stopping earlier is what "
            "makes the nesting visible. Does not change the forces"
        ),
        default=6, min=1, max=12,
        update=_on_setting_update,
    )
    S.scigraphs_preview_grid_color = bpy.props.FloatVectorProperty(
        name="Grid Color", description="Color of an empty or barely used cell",
        subtype='COLOR', size=4, min=0.0, max=1.0,
        default=(0.25, 0.7, 1.0, 0.35),
        update=_on_setting_update,
    )
    S.scigraphs_preview_grid_hot = bpy.props.FloatVectorProperty(
        name="Crowded Color",
        description="Color a cell reaches when it holds the most nodes",
        subtype='COLOR', size=4, min=0.0, max=1.0,
        default=(1.0, 0.25, 0.1, 0.9),
        update=_on_setting_update,
    )
    S.scigraphs_preview_grid_by_count = bpy.props.BoolProperty(
        name="Color by Occupancy",
        description=(
            "Shade each cell between the two colors by how many nodes it holds, "
            "on a log scale. Occupancy spans four orders of magnitude on a "
            "settled graph, and a linear ramp shows one hot box and nothing else"
        ),
        default=True,
        update=_on_setting_update,
    )
    S.scigraphs_preview_grid_width = bpy.props.FloatProperty(
        name="Line Width",
        description=(
            "Cell edge thickness in pixels. Widened in the shader, so it means "
            "the same on both backends; the GPU state call this replaces is "
            "capped at 10 px by the OpenGL driver"
        ),
        default=1.5, min=0.5, max=20.0,
        update=_on_setting_update,
    )
    S.scigraphs_preview_animate_steps = bpy.props.IntProperty(
        name="Iterations / Frame",
        description=(
            "Layout iterations advanced per animation frame. This is the "
            "animation's pacing: more per frame settles the graph in fewer "
            "frames but moves further between them"
        ),
        default=1, min=1, max=50,
        update=_on_setting_update,
    )
    S.scigraphs_preview_animate_max_frames = bpy.props.IntProperty(
        name="Record Frames",
        description=(
            "How many frames of the layout to record. Within it the trajectory "
            "is kept in memory, so a scrub or a second play shows exactly what "
            "was watched. Past it the simulation keeps running but nothing is "
            "stored, so those frames neither scrub nor render the same way "
            "twice"
        ),
        default=250, min=2, max=2000,
        update=_on_setting_update,
    )
    S.scigraphs_preview_node_size = bpy.props.FloatProperty(
        name="Point Size", description="Size of node points in pixels",
        default=6.0, min=1.0, max=64.0, update=_on_setting_update,
    )
    S.scigraphs_preview_round_points = bpy.props.BoolProperty(
        name="Round Points",
        description="Draw circular points instead of squares (uses a custom shader when supported)",
        default=True, update=_on_rebuild_update,
    )
    S.scigraphs_preview_node_style = bpy.props.EnumProperty(
        name="Node Style",
        description="How nodes are drawn. Auto picks by on-screen size (LOD)",
        items=[
            ('AUTO', "Auto (LOD)", "Choose point/disk/sphere by on-screen size"),
            ('POINT', "Point", "Small fixed-size point"),
            ('DISK', "Disk", "Flat round point"),
            ('SPHERE', "Sphere Impostor", "Lit sphere impostor with real depth"),
        ],
        default='AUTO', update=_on_setting_update,
    )
    S.scigraphs_preview_edge_style = bpy.props.EnumProperty(
        name="Edge Style",
        description="How edges are drawn. Auto picks by on-screen thickness (LOD)",
        items=[
            ('AUTO', "Auto (LOD)", "Choose line/ribbon by on-screen thickness"),
            ('LINE', "Line", "Thin GPU line"),
            ('RIBBON', "Ribbon (Tube)", "Camera-facing cylinder impostor"),
            ('NONE', "Hidden", "Do not draw edges"),
        ],
        default='AUTO', update=_on_setting_update,
    )
    S.scigraphs_preview_backbone_mode = bpy.props.EnumProperty(
        name="Edge Backbone",
        description=(
            "Structural edge sparsification: draw only the important edges. "
            "The panel reports how many edges and how much total weight the "
            "backbone preserves"
        ),
        items=[
            ('ALL', "All Edges", "No sparsification"),
            ('TOPK', "Local Top-k",
             "Keep each node's k strongest edges (by the weight attribute)"),
            ('DISPARITY', "Disparity Filter",
             "Serrano-Boguna statistical backbone: keep edges that are "
             "significant for at least one endpoint (alpha threshold)"),
            ('MST', "Spanning Tree + Top-k",
             "Maximum spanning tree (guarantees connectivity) plus each "
             "node's k strongest edges"),
            ('SAMPLE', "Random Sample",
             "Deterministic uniform sample of the edges"),
        ],
        default='ALL', update=_on_rebuild_update,
    )
    S.scigraphs_preview_backbone_k = bpy.props.IntProperty(
        name="Top-k",
        description="Edges kept per node (strongest first)",
        default=3, min=1, max=64, update=_on_rebuild_update,
    )
    S.scigraphs_preview_backbone_alpha = bpy.props.FloatProperty(
        name="Alpha",
        description="Disparity filter significance level (lower = sparser)",
        default=0.05, min=0.001, max=1.0, update=_on_rebuild_update,
    )
    S.scigraphs_preview_backbone_sample = bpy.props.FloatProperty(
        name="Sample Fraction",
        description="Fraction of edges kept by the random sample",
        default=0.25, min=0.001, max=1.0, subtype='FACTOR',
        update=_on_rebuild_update,
    )
    S.scigraphs_preview_backbone_attr = bpy.props.StringProperty(
        name="Backbone Weight",
        description=(
            "EDGE-domain attribute used as edge weight by the backbone "
            "(empty = 'weight' if present, else uniform)"
        ),
        default="", update=_on_rebuild_update,
    )
    S.scigraphs_preview_density_fallback = bpy.props.BoolProperty(
        name="Density Fallback",
        description=(
            "Instead of hiding sub-pixel nodes, draw them as 1-px additive "
            "points so distant graphs read as a continuous density cloud"
        ),
        default=True, update=_on_setting_update,
    )
    S.scigraphs_preview_volume_last = bpy.props.StringProperty(
        name="Last Density Field", default='COUNT', options={'HIDDEN'},
    )
    S.scigraphs_preview_volume_show = bpy.props.BoolProperty(
        name="Density Cloud",
        description="Draw the density cloud. The field itself is chosen below",
        get=_volume_show_get, set=_volume_show_set,
    )
    S.scigraphs_preview_volume_mode = bpy.props.EnumProperty(
        name="Density Cloud",
        description=(
            "Draw the nodes as a volumetric density field instead of (or as "
            "well as) individual elements. The field is built once in world "
            "space, so its brightness is a real density and the draw cost no "
            "longer depends on how many nodes there are"
        ),
        items=[
            ('OFF', "Off", "No density cloud"),
            ('COUNT', "Node Count",
             "Density is how many nodes occupy each region"),
            ('WEIGHTED', "Attribute Mass",
             "Each node counts for its normalized attribute value, so the "
             "cloud shows where the attribute is concentrated rather than "
             "where the nodes are"),
            ('COLOR', "Node Colors",
             "Density sets the opacity and the average of the member nodes' "
             "own colors sets the hue. With nodes colored by community this "
             "gives one cloud per community"),
        ],
        default='OFF', update=_on_rebuild_update,
    )
    S.scigraphs_preview_volume_when = bpy.props.EnumProperty(
        name="Cloud Shown",
        description="When the density cloud replaces or joins the nodes",
        items=[
            ('FAR', "When Sub-pixel",
             "Only once the nodes project below a pixel, where drawing them "
             "individually stops meaning anything. Replaces the screen-space "
             "Density Fallback"),
            ('ALWAYS', "Always",
             "Draw the cloud at every distance, over the nodes"),
            ('ONLY', "Cloud Only",
             "Draw the field and nothing else: no node is drawn individually, "
             "at any distance"),
            ('HYBRID', "Crowded Regions Only",
             "Split the graph: the densest nodes are represented by the cloud "
             "and removed from the discrete batches, sparse regions keep their "
             "real nodes and edges"),
        ],
        default='FAR', update=_on_rebuild_update,
    )
    S.scigraphs_preview_volume_res = bpy.props.IntProperty(
        name="Grid Resolution",
        description=(
            "Voxels along the longest axis of the graph's bounding box. The "
            "other axes get proportionally fewer, so voxels stay cubic and the "
            "density really is per unit of volume"
        ),
        default=64, min=8, max=256, update=_on_rebuild_update,
    )
    S.scigraphs_preview_volume_smooth = bpy.props.FloatProperty(
        name="Smoothing",
        description=(
            "Kernel radius in voxels. This is what turns a histogram of nodes "
            "into a continuous density estimate; 0 shows the raw voxels"
        ),
        default=1.5, min=0.0, max=8.0, update=_on_rebuild_update,
    )
    S.scigraphs_preview_volume_threshold = bpy.props.FloatProperty(
        name="Density Threshold",
        description=(
            "Fraction of the peak density below which the cloud is fully "
            "transparent. This is the density filter: it hides the thin haze "
            "of isolated nodes and leaves the real concentrations"
        ),
        default=0.02, min=0.0, max=1.0, subtype='FACTOR',
        update=_on_rebuild_update,
    )
    S.scigraphs_preview_volume_falloff = bpy.props.FloatProperty(
        name="Falloff",
        description=(
            "Exponent of the opacity ramp above the threshold. Above 1 the "
            "cloud keeps only its dense core, below 1 it reads as a soft haze"
        ),
        default=1.0, min=0.1, max=8.0, update=_on_rebuild_update,
    )
    S.scigraphs_preview_volume_gain = bpy.props.FloatProperty(
        name="Opacity Gain",
        description=(
            "How opaque a full traverse of the densest part of the cloud is. "
            "Defined against the whole grid rather than per voxel, so changing "
            "the resolution changes the detail and not the look. Costs nothing "
            "to change: it is a draw-time uniform, so nothing is rebuilt"
        ),
        default=3.0, min=0.05, max=40.0, update=_on_setting_update,
    )
    S.scigraphs_preview_volume_levels = bpy.props.IntProperty(
        name="Isosurfaces",
        description=(
            "Draw the field as this many nested density shells instead of a "
            "cloud, at evenly spaced fractions of the peak density. A cloud "
            "shows where the density is high; shells put it on a scale you can "
            "count, and give the outer haze a definite surface. 0 draws the "
            "cloud. Costs nothing to change: it is a draw-time uniform"
        ),
        default=0, min=0, max=32, update=_on_setting_update,
    )
    S.scigraphs_preview_volume_shell = bpy.props.FloatProperty(
        name="Shell Width",
        description=(
            "Thickness of each isosurface, as a fraction of the gap between "
            "levels. Narrow shells are made proportionally denser, so this "
            "changes how sharp they are and not how bright"
        ),
        default=0.15, min=0.01, max=0.5, subtype='FACTOR',
        update=_on_setting_update,
    )
    S.scigraphs_preview_volume_slices = bpy.props.IntProperty(
        name="Slices",
        description=(
            "How many view-aligned slices integrate the field per frame. More "
            "is smoother and costs linearly more fill; too few shows as "
            "banding across the cloud"
        ),
        default=96, min=8, max=512, update=_on_setting_update,
    )
    S.scigraphs_preview_volume_hybrid_pct = bpy.props.FloatProperty(
        name="Cloud Above",
        description=(
            "In Crowded Regions Only, the percentile of local density above "
            "which a node is left to the cloud. 90 means the densest tenth of "
            "the graph becomes cloud and the rest keeps its nodes"
        ),
        default=90.0, min=0.0, max=100.0, update=_on_rebuild_update,
    )
    S.scigraphs_preview_coarsen = bpy.props.BoolProperty(
        name="Coarsen by Community",
        description=(
            "Aggregate nodes into supernodes by a community attribute and "
            "collapse inter-community edges into superedges (thickness = "
            "aggregated weight, arrows = dominant direction). Shown when "
            "communities project too small to read individually"
        ),
        default=False, update=_on_rebuild_update,
    )
    S.scigraphs_preview_coarsen_attr = bpy.props.StringProperty(
        name="Community Attribute",
        description="POINT integer attribute holding the community id",
        default="community", update=_on_rebuild_update,
    )
    S.scigraphs_preview_coarsen_size_mode = bpy.props.EnumProperty(
        name="Community Size",
        description=(
            "What makes a community bigger or smaller. The community id is "
            "an arbitrary label, so it never drives size by itself"
        ),
        items=[
            ('COUNT', "Node Count",
             "Supernode area proportional to how many nodes the community has"),
            ('ATTRIBUTE', "Attribute",
             "Supernode area proportional to a POINT scalar aggregated per "
             "community (e.g. cluster_size from Clustering)"),
            ('EXTENT', "Spatial Extent",
             "Supernode radius follows the real footprint of its members "
             "(RMS distance to the community center)"),
            ('UNIFORM', "Uniform", "All communities drawn the same size"),
        ],
        default='COUNT', update=_on_rebuild_update,
    )
    S.scigraphs_preview_coarsen_size_attr = bpy.props.StringProperty(
        name="Size Attribute",
        description=(
            "POINT scalar attribute that drives supernode size, e.g. "
            "'cluster_size' written by Clustering"
        ),
        default="cluster_size", update=_on_rebuild_update,
    )
    S.scigraphs_preview_coarsen_size_agg = bpy.props.EnumProperty(
        name="Aggregate",
        description="How the size attribute is collapsed per community",
        items=[
            ('MEAN', "Mean",
             "Average over members. Right for per-community values repeated "
             "on every node, such as cluster_size"),
            ('SUM', "Sum", "Total over members, e.g. a per-node mass"),
            ('MAX', "Max", "Largest member value"),
            ('MIN', "Min", "Smallest member value"),
        ],
        default='MEAN', update=_on_rebuild_update,
    )
    S.scigraphs_preview_coarsen_size_mult = bpy.props.FloatProperty(
        name="Max Size Multiplier",
        description=(
            "Radius of the largest community, as a multiple of the node "
            "radius. The smallest one keeps the node radius"
        ),
        default=6.0, min=1.0, max=64.0, update=_on_rebuild_update,
    )
    S.scigraphs_preview_coarsen_px = bpy.props.FloatProperty(
        name="Coarsen Below",
        description=(
            "Switch to the aggregated level when the typical community "
            "projects below this radius (reference px, 1080p image)"
        ),
        default=24.0, min=2.0, max=200.0, update=_on_setting_update,
    )
    S.scigraphs_preview_adaptive = bpy.props.BoolProperty(
        name="Adaptive Cut",
        description=(
            "Choose the level of detail per region instead of once for the "
            "whole graph: communities are split open where their members are "
            "big enough on screen to tell apart and left merged where they "
            "would only crowd each other. Needs Coarsen enabled"
        ),
        default=False, update=_on_rebuild_update,
    )
    S.scigraphs_preview_adaptive_algorithm = bpy.props.EnumProperty(
        name="Hierarchy Algorithm",
        description="Community algorithm used to build the levels above the "
                    "first one",
        items=[
            ('infomap', "Infomap", "Map equation, flow-based"),
            ('rn', "RN", "Resolution-limit-free Potts"),
            ('cpm', "CPM", "Constant Potts model"),
            ('rb', "RB", "Reichardt-Bornholdt"),
        ],
        default='infomap', update=_on_rebuild_update,
    )
    S.scigraphs_preview_adaptive_predicted = bpy.props.FloatProperty(
        name="Split Above",
        description=(
            "Split a community open only when at least this fraction of its "
            "members would be drawn clear of each other. Lower shows more "
            "detail at the cost of letting nodes overlap"
        ),
        default=0.55, min=0.05, max=1.0, update=_on_setting_update,
    )
    S.scigraphs_preview_adaptive_min_px = bpy.props.FloatProperty(
        name="Split Above Size",
        description=(
            "...and only when the members would project at least this big "
            "(reference px, 1080p image)"
        ),
        default=2.5, min=0.5, max=64.0, update=_on_setting_update,
    )
    S.scigraphs_preview_adaptive_collapse = bpy.props.FloatProperty(
        name="Merge Below",
        description=(
            "Merge a drawn element back into its parent when the measured "
            "image shows less than this fraction of it. Keep well below "
            "Split Above: the gap is what stops borderline elements from "
            "flickering between the two states"
        ),
        default=0.25, min=0.0, max=0.9, update=_on_setting_update,
    )
    S.scigraphs_preview_adaptive_merge_loss = bpy.props.FloatProperty(
        name="Fidelity Traded",
        description=(
            "How much of a community a merge may newly hide, as a fraction of "
            "its members, before the merge is refused and it is drawn out "
            "again. This is the trade: near zero draws almost every node, near "
            "one merges almost everything"
        ),
        default=adaptive.MAX_MERGE_LOSS, min=0.0, max=1.0,
        update=_on_setting_update,
    )
    S.scigraphs_preview_adaptive_freeze = bpy.props.BoolProperty(
        name="Freeze Cut",
        description=(
            "Stop adapting and keep what is currently drawn, so you can move "
            "the camera and inspect the cut chosen from the previous position "
            "without it changing as you look"
        ),
        default=False, update=_on_setting_update,
    )
    S.scigraphs_preview_render_overdraw = bpy.props.BoolProperty(
        name="Overdraw Pass",
        description=(
            "Render an 'Overdraw' AOV with the exact number of edge "
            "segments crossing each pixel: a measurable clutter heatmap"
        ),
        default=False, update=_on_setting_update,
    )
    S.scigraphs_preview_edge_radius_scale = bpy.props.FloatProperty(
        name="Edge Radius Scale",
        description=(
            "Edge tube radius as a fraction of the node radius. Keeping "
            "edges thinner than nodes preserves the visual hierarchy at "
            "every distance (nodes stay readable over the wiring)"
        ),
        default=0.45, min=0.05, max=2.0, update=_on_rebuild_update,
    )
    S.scigraphs_preview_edge_arrows = bpy.props.EnumProperty(
        name="Edge Arrows",
        description="Arrow heads at the target end of each edge",
        items=[
            ('AUTO', "Auto (Directed)",
             "Show arrows only when the graph was imported as directed"),
            ('ON', "On", "Always show arrow heads"),
            ('OFF', "Off", "Never show arrow heads"),
        ],
        default='AUTO', update=_on_rebuild_update,
    )
    S.scigraphs_preview_edge_arrow_size = bpy.props.FloatProperty(
        name="Arrow Size",
        description="Arrow head size, relative to the edge thickness",
        default=1.0, min=0.1, max=10.0, update=_on_rebuild_update,
    )
    S.scigraphs_preview_edge_size_by_attr = bpy.props.BoolProperty(
        name="Edge Width by Attribute",
        description=(
            "Scale each edge's tube thickness by an EDGE-domain scalar "
            "attribute (e.g. weight), min-max normalized"
        ),
        default=False, update=_on_rebuild_update,
    )
    S.scigraphs_preview_edge_arrow_style = bpy.props.EnumProperty(
        name="Arrow Style",
        description="How directed-edge arrowheads are drawn",
        items=[
            ('FLAT', "Flat",
             "One unlit triangle facing the camera. A mark rather than an "
             "object: it cannot be foreshortened, cannot be seen edge-on, and "
             "costs three vertices instead of eighteen"),
            ('CONE', "Cone (3D)",
             "A real lit cone with true depth. Right when the graph is a solid "
             "object in a scene; at arrowhead sizes its shading is mostly noise"),
        ],
        default='FLAT', update=_on_rebuild_update,
    )
    S.scigraphs_preview_edge_size_scale = bpy.props.EnumProperty(
        name="Width Scale",
        description="How edge magnitudes are mapped onto thickness",
        items=[
            ('LOG', "Logarithmic",
             "Spread multiplicatively. The right default for weights, which are "
             "usually heavy-tailed: a linear map puts almost every edge at the "
             "minimum width and they all look the same"),
            ('LINEAR', "Linear",
             "Min-max. Keeps ratios between values meaningful; only readable "
             "when the magnitudes are already evenly spread"),
            ('RANK', "Rank",
             "Position in the sorted order, so widths are spread evenly by "
             "construction and no distribution can collapse them. Ties keep "
             "the same width"),
        ],
        default='LOG', update=_on_rebuild_update,
    )
    S.scigraphs_preview_edge_taper = bpy.props.EnumProperty(
        name="Taper",
        description=(
            "Vary an edge's thickness along its length. The tube shader has "
            "always taken a radius per endpoint; this is what drives it"
        ),
        items=[
            ('NONE', "None", "Same thickness at both ends"),
            ('FORWARD', "To Target",
             "Thin at the source, thick at the target. Encodes direction "
             "without arrowheads, which is the only thing that still reads at "
             "a hundred thousand edges"),
            ('BACKWARD', "To Source", "Thick at the source, thin at the target"),
        ],
        default='NONE', update=_on_rebuild_update,
    )
    S.scigraphs_preview_edge_taper_amount = bpy.props.FloatProperty(
        name="Taper Amount",
        description=(
            "How strong the taper is. 0 leaves both ends equal, 1 makes the "
            "narrow end as thin as it goes"
        ),
        default=0.75, min=0.0, max=1.0, subtype='FACTOR',
        update=_on_rebuild_update,
    )
    S.scigraphs_preview_edge_attr_name = bpy.props.StringProperty(
        name="Edge Attribute",
        description="EDGE-domain scalar attribute that drives edge thickness",
        default="", update=_on_rebuild_update,
    )
    S.scigraphs_preview_edge_size_max_mult = bpy.props.FloatProperty(
        name="Edge Max Multiplier",
        description="Thickness multiplier for the highest attribute value",
        default=4.0, min=1.0, max=20.0, update=_on_rebuild_update,
    )
    S.scigraphs_preview_edge_styles_gpu = bpy.props.BoolProperty(
        name="GPU Edge Styles",
        description=(
            "Generate the Edge Styles (curved, arc, bundled, tapered, "
            "orthogonal...) live on the GPU render path, using the settings "
            "from the SciGraphs Edge Style panel. The mesh is never modified: "
            "edges are tessellated in memory at batch-build time, so style "
            "changes are instant. Baked curve points from the CPU operator "
            "are collapsed back to straight edges first"
        ),
        default=False, update=_on_rebuild_update,
    )
    S.scigraphs_preview_heb_vertex_shader = bpy.props.BoolProperty(
        name="Bundle on the GPU",
        description=(
            "Evaluate hierarchical bundling in the vertex shader instead of in "
            "numpy. Only the control polygon of each edge is uploaded, so "
            "Bundling Strength becomes a shader uniform and moving it costs a "
            "redraw instead of a rebuild. Used on graphs too large for ribbon "
            "impostors, or whenever the edge style is set to LINE; ribbons, "
            "picking and the adaptive cut need real segments and keep the "
            "numpy path. Turn off to compare the two"
        ),
        default=True, update=_on_rebuild_update,
    )
    S.scigraphs_preview_impostor_radius = bpy.props.FloatProperty(
        name="Impostor Radius",
        description="World-space radius of sphere/ribbon impostors",
        default=0.1, min=0.0001, max=1000.0, update=_on_rebuild_update,
    )
    S.scigraphs_preview_color_mode = bpy.props.EnumProperty(
        name="Color Mode",
        description="How node colors are chosen",
        items=[
            ('FLAT', "Flat", "Single flat color"),
            ('VERTEX', "Node Colors", "Active per-vertex color attribute (e.g. from Coloring)"),
            ('ATTRIBUTE', "Attribute + Colormap", "Map a scalar attribute through a colormap"),
        ],
        default='VERTEX', update=_on_rebuild_update,
    )
    S.scigraphs_preview_node_color = bpy.props.FloatVectorProperty(
        name="Node Color", description="Flat node color",
        subtype='COLOR', size=4, min=0.0, max=1.0,
        default=(0.3, 0.7, 1.0, 1.0), update=_on_rebuild_update,
    )
    S.scigraphs_preview_attr_name = bpy.props.StringProperty(
        name="Attribute",
        description="Scalar POINT attribute used for color / size / filter",
        default="", update=_on_rebuild_update,
    )
    S.scigraphs_preview_colormap = bpy.props.StringProperty(
        name="Colormap", description="Colormap name (e.g. viridis, plasma, magma)",
        default="viridis", update=_on_rebuild_update,
    )
    S.scigraphs_preview_reverse_colormap = bpy.props.BoolProperty(
        name="Reverse Colormap", default=False, update=_on_rebuild_update,
    )
    from scigraphs_core.coloring.colormaps import (
        DEFAULT_NORM_MODE, norm_mode_items_for_enum)
    S.scigraphs_preview_norm_mode = bpy.props.EnumProperty(
        name="Normalization",
        description=(
            "How attribute values are mapped onto the colormap. Degree, "
            "betweenness and PageRank are heavy-tailed: under Linear most of "
            "the graph shares the bottom few percent of the ramp"
        ),
        items=norm_mode_items_for_enum(),
        default=DEFAULT_NORM_MODE, update=_on_rebuild_update,
    )
    S.scigraphs_preview_norm_gamma = bpy.props.FloatProperty(
        name="Gamma",
        description=(
            "Bend the normalized value before the colormap reads it. Above 1 "
            "gives the low end more of the ramp, below 1 the high end"
        ),
        default=1.0, min=0.05, max=20.0, update=_on_rebuild_update,
    )
    S.scigraphs_preview_clip_low_pct = bpy.props.FloatProperty(
        name="Clip Low",
        description=(
            "Percentile taken as the bottom of the range. A single outlier "
            "otherwise sets an end of the ramp on its own"
        ),
        default=0.0, min=0.0, max=100.0, subtype='PERCENTAGE',
        update=_on_rebuild_update,
    )
    S.scigraphs_preview_clip_high_pct = bpy.props.FloatProperty(
        name="Clip High",
        description="Percentile taken as the top of the range",
        default=100.0, min=0.0, max=100.0, subtype='PERCENTAGE',
        update=_on_rebuild_update,
    )
    S.scigraphs_preview_size_by_attr = bpy.props.BoolProperty(
        name="Size by Attribute",
        description="Scale node size by the selected scalar attribute",
        default=False, update=_on_rebuild_update,
    )
    S.scigraphs_preview_size_attr_name = bpy.props.StringProperty(
        name="Size Attribute",
        description="Scalar POINT attribute driving node size. Empty follows the "
                    "Attribute above, which is what color and the filters read",
        default="", update=_on_rebuild_update,
    )
    S.scigraphs_preview_size_max_mult = bpy.props.FloatProperty(
        name="Max Size Multiplier",
        description="Size multiplier for the highest attribute value",
        # 1000 to match the radius spread a circle packing accepts. At 32 a
        # 2000-node packing was compressed tenfold and stopped looking tangent.
        default=4.0, min=1.0, max=1000.0, update=_on_rebuild_update,
    )
    S.scigraphs_preview_show_edges = bpy.props.BoolProperty(
        name="Show Edges", description="Draw graph edges",
        default=True, update=_on_setting_update,
    )
    S.scigraphs_preview_edge_color = bpy.props.FloatVectorProperty(
        name="Edge Color", description="Color (and alpha) used for edges",
        subtype='COLOR', size=4, min=0.0, max=1.0,
        default=(0.5, 0.5, 0.55, 0.35), update=_on_rebuild_update,
    )
    S.scigraphs_preview_edge_width = bpy.props.FloatProperty(
        name="Edge Width", description="Line width of edges in pixels",
        default=1.0, min=1.0, max=16.0, update=_on_setting_update,
    )
    S.scigraphs_preview_additive_edges = bpy.props.BoolProperty(
        name="Additive Edges",
        description="Use additive blending so dense edge regions glow (density look)",
        default=False, update=_on_setting_update,
    )
    S.scigraphs_preview_filter_enabled = bpy.props.BoolProperty(
        name="Filter", description="Show only nodes whose normalized attribute is in range",
        default=False, update=_on_rebuild_update,
    )
    S.scigraphs_preview_filter_min = bpy.props.FloatProperty(
        name="Filter Min", default=0.0, min=0.0, max=1.0, update=_on_rebuild_update,
    )
    S.scigraphs_preview_filter_max = bpy.props.FloatProperty(
        name="Filter Max", default=1.0, min=0.0, max=1.0, update=_on_rebuild_update,
    )
    S.scigraphs_preview_lod_enabled = bpy.props.BoolProperty(
        name="Level of Detail",
        description="Subsample points/edges when the graph exceeds Max Points",
        default=False, update=_on_rebuild_update,
    )
    S.scigraphs_preview_lod_max_points = bpy.props.IntProperty(
        name="Max Points", description="Target maximum number of drawn points",
        default=300000, min=1000, max=10000000, update=_on_rebuild_update,
    )
    S.scigraphs_preview_use_blocks = bpy.props.BoolProperty(
        name="Spatial Blocks",
        description=(
            "Partition the graph into spatial blocks and frustum-cull / budget "
            "whole blocks each frame. Recommended for very large graphs"
        ),
        default=False, update=_on_rebuild_update,
    )
    S.scigraphs_preview_block_count = bpy.props.IntProperty(
        name="Target Blocks",
        description="Approximate number of spatial blocks the graph is split into",
        default=4096, min=64, max=262144, update=_on_rebuild_update,
    )
    S.scigraphs_preview_node_budget = bpy.props.IntProperty(
        name="Node Budget",
        description="Max nodes drawn per frame in blocks mode (0 = unlimited)",
        default=1000000, min=0, max=100000000, update=_on_setting_update,
    )
    S.scigraphs_preview_render_aa = bpy.props.IntProperty(
        name="AA Samples",
        description=(
            "Supersampling factor per axis for the render engine (1 = off). "
            "The image is rendered at N x resolution and averaged down. Cost "
            "grows with N squared"
        ),
        default=2, min=1, max=8, update=_on_setting_update,
    )
    S.scigraphs_preview_render_id_pass = bpy.props.BoolProperty(
        name="Node ID Pass",
        description=(
            "Render an extra NodeID pass (24-bit encoded node index per pixel) "
            "for selection / compositing. Uses sphere impostors"
        ),
        default=False, update=_on_rebuild_update,
    )
    S.scigraphs_preview_render_labels = bpy.props.BoolProperty(
        name="Text Labels",
        description=(
            "Draw node labels directly in the render engine (viewport and "
            "F12), using the Text Labels settings from the SciGraphs panel. "
            "No compositor texture is involved; labels stay sharp over "
            "depth of field and are occlusion-tested against the depth buffer"
        ),
        default=False, update=_on_setting_update,
    )
    S.scigraphs_preview_labels_declutter = bpy.props.BoolProperty(
        name="Declutter Labels",
        description=(
            "Dynamically keep node labels from overlapping: each label tries "
            "alternative spots around its node (nearest nodes win) and is "
            "hidden when no free spot remains. Recomputed every frame, so "
            "the layout follows the camera"
        ),
        default=True, update=_on_setting_update,
    )
    S.scigraphs_preview_labels_priority = bpy.props.EnumProperty(
        name="Label Priority",
        description=(
            "Which labels win a contested spot. Distance keeps the ones nearest "
            "the camera; any other channel keeps the ones that matter, which on "
            "a large graph is rarely the same set"
        ),
        items=LABEL_PRIORITY_ITEMS, default='DISTANCE',
        update=_on_setting_update,
    )
    S.scigraphs_preview_labels_priority_attr = bpy.props.StringProperty(
        name="Priority Attribute",
        description="Which scalar attribute, when the priority channel is one",
        default="", update=_on_setting_update,
    )

    S.scigraphs_preview_legend = bpy.props.BoolProperty(
        name="Color Key",
        description=(
            "Draw the colormap's scale into the image. A figure whose colors "
            "carry a value and whose scale is missing cannot be read, and the "
            "key names the normalization too, since a log ramp with linear "
            "ticks is a lie"
        ),
        default=False, update=_on_setting_update,
    )
    S.scigraphs_preview_legend_anchor = bpy.props.EnumProperty(
        name="Key Corner",
        description="Which corner the key sits in",
        items=[('BOTTOM_RIGHT', "Bottom Right", ""),
               ('BOTTOM_LEFT', "Bottom Left", ""),
               ('TOP_RIGHT', "Top Right", ""),
               ('TOP_LEFT', "Top Left", "")],
        default='BOTTOM_RIGHT', update=_on_setting_update,
    )
    S.scigraphs_preview_legend_orient = bpy.props.EnumProperty(
        name="Key Layout",
        description="Which way the color bar lies",
        items=[('VERTICAL', "Vertical", "Tall bar, labels down the side"),
               ('HORIZONTAL', "Horizontal", "Wide bar, labels underneath. "
                                            "Costs height, which a wide figure "
                                            "has less of to spare")],
        default='VERTICAL', update=_on_setting_update,
    )
    S.scigraphs_preview_legend_x = bpy.props.FloatProperty(
        name="Key X",
        description="Shift from the chosen corner, in reference pixels at 1080p",
        default=0.0, soft_min=-960.0, soft_max=960.0,
        update=_on_setting_update,
    )
    S.scigraphs_preview_legend_y = bpy.props.FloatProperty(
        name="Key Y",
        description="Shift from the chosen corner, in reference pixels at 1080p",
        default=0.0, soft_min=-540.0, soft_max=540.0,
        update=_on_setting_update,
    )
    S.scigraphs_preview_legend_scale = bpy.props.FloatProperty(
        name="Key Size",
        description=(
            "Multiplies the whole key. It already tracks the resolution, so "
            "this is taste, or a floor for renders below 1080p where the "
            "reference sizes land too small to read"
        ),
        default=1.0, min=0.2, max=6.0, soft_min=0.5, soft_max=3.0,
        update=_on_setting_update,
    )
    S.scigraphs_preview_legend_box = bpy.props.BoolProperty(
        name="Key Backdrop",
        description=(
            "Panel and frame behind the key. Off leaves the text and bar over "
            "the image, which reads better on a plain background and worse on "
            "a busy one"
        ),
        default=True, update=_on_setting_update,
    )
    S.scigraphs_preview_edge_xray = bpy.props.FloatProperty(
        name="Occluded Edges",
        description=(
            "Redraw the hidden stretch of an edge at this opacity, so an edge "
            "diving behind a cluster stays traceable and a crossing reads as "
            "over-and-under. Drawn without depth sorting, so it reads as "
            "density rather than as correct transparency"
        ),
        default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_on_setting_update,
    )
    S.scigraphs_preview_dof_highlights = bpy.props.FloatProperty(
        name="Bokeh Highlights",
        description=(
            "Weight bright pixels more in the depth-of-field blur so "
            "out-of-focus highlights read as crisp bokeh discs "
            "(0 = plain average, physically neutral)"
        ),
        default=1.0, min=0.0, max=10.0, update=_on_setting_update,
    )
    S.scigraphs_preview_render_bg = bpy.props.FloatVectorProperty(
        name="Render Background",
        description="Clear color (RGBA) used by the SciGraphs render engine",
        subtype='COLOR', size=4, min=0.0, max=1.0,
        default=(0.05, 0.05, 0.08, 0.0), update=_on_setting_update,
    )
    S.scigraphs_preview_use_scene_lights = bpy.props.BoolProperty(
        name="Use Scene Lights",
        description=(
            "Light sphere/ribbon impostors with the scene's lamps (Sun, Point, "
            "Spot, Area). When off, a camera headlight is used"
        ),
        default=True, update=_on_setting_update,
    )
    S.scigraphs_preview_light_strength = bpy.props.FloatProperty(
        name="Light Strength",
        description="Global multiplier applied to all lights that shade the impostors",
        default=1.0, min=0.0, max=20.0, update=_on_setting_update,
    )
    S.scigraphs_preview_ambient = bpy.props.FloatProperty(
        name="Ambient",
        description=(
            "Ambient (fill) light so the shaded side of nodes is never fully "
            "black. Uses the World color as tint when available"
        ),
        default=0.25, min=0.0, max=1.0, update=_on_setting_update,
    )
    S.scigraphs_preview_rim = bpy.props.FloatProperty(
        name="Rim Light",
        description="Rim highlight strength on sphere impostors for a glossy edge",
        default=0.15, min=0.0, max=2.0, update=_on_setting_update,
    )
    S.scigraphs_preview_depth_test = bpy.props.BoolProperty(
        name="Depth Test",
        description="Occlude points/lines behind other geometry",
        default=True, update=_on_setting_update,
    )
    S.scigraphs_preview_hide_mesh = bpy.props.BoolProperty(
        name="Hide Native Mesh",
        description=(
            "Switch the graph mesh to Bounds display while the preview is on "
            "so Blender does not draw it a second time (drawn twice = slower)"
        ),
        default=True, update=_on_hide_mesh_update,
    )

    # The ``scigraphs_preview_toon_*`` names are what host._rna_properties
    # matches on. Register them here and not later: host.invalidate_properties()
    # runs just before this function, and once the memoized property set has
    # been rebuilt the toon settings vanish from snapshots until reload.
    toon_properties.register_properties()


_PROP_NAMES = (
    "scigraphs_display_engine",
    "scigraphs_preview_enabled",
    "scigraphs_preview_node_size",
    "scigraphs_preview_round_points",
    "scigraphs_preview_node_style",
    "scigraphs_preview_edge_style",
    "scigraphs_preview_edge_styles_gpu",
    "scigraphs_preview_heb_vertex_shader",
    "scigraphs_preview_backbone_mode",
    "scigraphs_preview_backbone_k",
    "scigraphs_preview_backbone_alpha",
    "scigraphs_preview_backbone_sample",
    "scigraphs_preview_backbone_attr",
    "scigraphs_preview_density_fallback",
    "scigraphs_preview_volume_mode",
    "scigraphs_preview_volume_last",
    "scigraphs_preview_volume_show",
    "scigraphs_preview_volume_when",
    "scigraphs_preview_volume_res",
    "scigraphs_preview_volume_smooth",
    "scigraphs_preview_volume_threshold",
    "scigraphs_preview_volume_falloff",
    "scigraphs_preview_volume_gain",
    "scigraphs_preview_volume_slices",
    "scigraphs_preview_volume_hybrid_pct",
    "scigraphs_preview_coarsen",
    "scigraphs_preview_coarsen_attr",
    "scigraphs_preview_coarsen_size_mode",
    "scigraphs_preview_coarsen_size_attr",
    "scigraphs_preview_coarsen_size_agg",
    "scigraphs_preview_coarsen_size_mult",
    "scigraphs_preview_coarsen_px",
    "scigraphs_preview_adaptive",
    "scigraphs_preview_adaptive_algorithm",
    "scigraphs_preview_adaptive_predicted",
    "scigraphs_preview_adaptive_min_px",
    "scigraphs_preview_adaptive_collapse",
    "scigraphs_preview_adaptive_merge_loss",
    "scigraphs_preview_adaptive_freeze",
    "scigraphs_preview_render_overdraw",
    "scigraphs_preview_edge_radius_scale",
    "scigraphs_preview_edge_arrows",
    "scigraphs_preview_edge_arrow_size",
    "scigraphs_preview_edge_size_by_attr",
    "scigraphs_preview_edge_attr_name",
    "scigraphs_preview_edge_size_max_mult",
    "scigraphs_preview_edge_size_scale",
    "scigraphs_preview_edge_arrow_style",
    "scigraphs_preview_edge_taper",
    "scigraphs_preview_edge_taper_amount",
    "scigraphs_preview_impostor_radius",
    "scigraphs_preview_color_mode",
    "scigraphs_preview_node_color",
    "scigraphs_preview_attr_name",
    "scigraphs_preview_colormap",
    "scigraphs_preview_reverse_colormap",
    "scigraphs_preview_norm_mode",
    "scigraphs_preview_norm_gamma",
    "scigraphs_preview_clip_low_pct",
    "scigraphs_preview_clip_high_pct",
    "scigraphs_preview_size_by_attr",
    "scigraphs_preview_size_attr_name",
    "scigraphs_preview_size_max_mult",
    "scigraphs_preview_show_edges",
    "scigraphs_preview_edge_color",
    "scigraphs_preview_edge_width",
    "scigraphs_preview_additive_edges",
    "scigraphs_preview_filter_enabled",
    "scigraphs_preview_filter_min",
    "scigraphs_preview_filter_max",
    "scigraphs_preview_lod_enabled",
    "scigraphs_preview_lod_max_points",
    "scigraphs_preview_use_blocks",
    "scigraphs_preview_block_count",
    "scigraphs_preview_node_budget",
    "scigraphs_preview_render_aa",
    "scigraphs_preview_render_id_pass",
    "scigraphs_preview_render_labels",
    "scigraphs_preview_labels_declutter",
    "scigraphs_preview_legend",
    "scigraphs_preview_legend_anchor",
    "scigraphs_preview_legend_box",
    "scigraphs_preview_legend_orient",
    "scigraphs_preview_legend_scale",
    "scigraphs_preview_legend_x",
    "scigraphs_preview_legend_y",
    "scigraphs_preview_edge_xray",
    "scigraphs_preview_labels_priority_attr",
    "scigraphs_preview_labels_priority",
    "scigraphs_preview_dof_highlights",
    "scigraphs_preview_render_bg",
    "scigraphs_preview_use_scene_lights",
    "scigraphs_preview_light_strength",
    "scigraphs_preview_ambient",
    "scigraphs_preview_rim",
    "scigraphs_preview_depth_test",
    "scigraphs_preview_hide_mesh",
)


def unregister_properties():
    S = bpy.types.Scene
    toon_properties.unregister_properties()
    for name in _PROP_NAMES:
        if hasattr(S, name):
            delattr(S, name)
    for name in ("scigraphs_filters", "scigraphs_filters_index"):
        if hasattr(S, name):
            delattr(S, name)
    bpy.utils.unregister_class(SCIGRAPHS_PG_filter_slot)
