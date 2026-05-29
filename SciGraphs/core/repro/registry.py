# Operator registry for reproducible SciGraphs pipelines
#
# Provides adapters and mappings between declarative pipeline specs
# and actual Blender operators.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import PipelineSchema

# Singleton registry instance
_registry: Optional["OperatorRegistry"] = None


@dataclass
class OperatorAdapter:
    """Adapter for calling a Blender operator with pipeline properties."""
    bl_idname: str
    category: str  # 'dataset', 'analysis', 'layout', 'visual', 'render', 'export', 'generic'
    description: str
    property_mapping: Dict[str, str]  # pipeline_prop -> operator_prop
    scene_props: List[str]  # scene.scigraphs properties to set before calling
    pre_call: Optional[Callable] = None  # Optional pre-processing
    post_call: Optional[Callable] = None  # Optional post-processing


class OperatorRegistry:
    """
    Registry of operator adapters for pipeline execution.

    Maps declarative pipeline specs to Blender operators with
    proper property translation and scene setup.
    """

    def __init__(self):
        self._adapters: Dict[str, OperatorAdapter] = {}
        self._by_category: Dict[str, List[str]] = {}
        self._shortcuts: Dict[str, str] = {}  # Short name -> full bl_idname
        self._initialized = False

    def register(self, adapter: OperatorAdapter) -> None:
        """Register an operator adapter."""
        self._adapters[adapter.bl_idname] = adapter
        if adapter.category not in self._by_category:
            self._by_category[adapter.category] = []
        self._by_category[adapter.category].append(adapter.bl_idname)

    def register_shortcut(self, short_name: str, bl_idname: str) -> None:
        """Register a shortcut name for an operator."""
        self._shortcuts[short_name] = bl_idname

    def get(self, bl_idname: str) -> Optional[OperatorAdapter]:
        """Get adapter by bl_idname or shortcut."""
        # Try shortcut first
        if bl_idname in self._shortcuts:
            bl_idname = self._shortcuts[bl_idname]
        return self._adapters.get(bl_idname)

    def get_by_category(self, category: str) -> List[OperatorAdapter]:
        """Get all adapters in a category."""
        ids = self._by_category.get(category, [])
        return [self._adapters[id] for id in ids]

    def list_all(self) -> List[str]:
        """List all registered operator bl_idnames."""
        return list(self._adapters.keys())

    def list_categories(self) -> List[str]:
        """List all categories."""
        return list(self._by_category.keys())

    def resolve_operator(self, spec: str) -> Tuple[str, bool]:
        """
        Resolve an operator specification to bl_idname.

        Args:
            spec: Operator spec (bl_idname, shortcut, or bpy.ops path)

        Returns:
            Tuple of (resolved_bl_idname, has_adapter)
        """
        # Try shortcut
        if spec in self._shortcuts:
            return self._shortcuts[spec], True

        # Try direct adapter lookup
        if spec in self._adapters:
            return spec, True

        # Check if it's a valid bpy.ops path (e.g., "scigraphs.apply_layout")
        if "." in spec and not spec.startswith("bpy."):
            # It's a bare bl_idname - check if operator exists
            return spec, spec in self._adapters

        return spec, False

    def initialize(self) -> None:
        """Initialize the registry with default adapters."""
        if self._initialized:
            return
        self._initialized = True

        self._register_dataset_adapters()
        self._register_analysis_adapters()
        self._register_layout_adapters()
        self._register_visual_adapters()
        self._register_topology_adapters()
        self._register_osmnx_adapters()
        self._register_export_adapters()
        self._register_render_adapters()
        self._register_shortcuts()

    def _register_dataset_adapters(self) -> None:
        """Register dataset/import adapters."""
        self.register(OperatorAdapter(
            bl_idname="scigraphs.import_osm_graph",
            category="dataset",
            description="Import OSMnx street network",
            property_mapping={
                "method": "method",
                "query": "place_query",
                "network_type": "network_type",
                "simplify": "simplify",
                "retain_all": "retain_all",
            },
            scene_props=["osmnx_method", "osmnx_place", "osmnx_network_type"],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.create_graph",
            category="dataset",
            description="Create graph from node/edge collections",
            property_mapping={},
            scene_props=["nodes_collection", "edges_collection"],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.download_suitesparse",
            category="dataset",
            description="Download SuiteSparse matrix",
            property_mapping={
                "matrix_name": "matrix_name",
            },
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.create_graph_from_sql",
            category="dataset",
            description="Create graph from SQL database",
            property_mapping={},
            scene_props=["db_type", "db_host", "db_port", "db_name", "db_user"],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.c2g_load_overture",
            category="dataset",
            description="Load City2Graph Overture data",
            property_mapping={
                "bbox": "bbox",
            },
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.c2g_load_overture_place",
            category="dataset",
            description="Load City2Graph data by place name",
            property_mapping={
                "place_name": "place_name",
            },
            scene_props=[],
        ))

    def _register_analysis_adapters(self) -> None:
        """Register analysis adapters."""
        self.register(OperatorAdapter(
            bl_idname="scigraphs.calculate_centrality",
            category="analysis",
            description="Compute centrality metrics",
            property_mapping={
                "metric": "centrality_type",
                "normalize": "normalize",
            },
            scene_props=["centrality_type"],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.calculate_clustering",
            category="analysis",
            description="Calculate clustering coefficient",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.apply_clustering",
            category="analysis",
            description="Detect and apply community clustering",
            property_mapping={
                "algorithm": "algorithm",
                "resolution": "resolution",
            },
            scene_props=["clustering_algorithm", "clustering_resolution"],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.calculate_directed_centrality",
            category="analysis",
            description="Compute directed centrality",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.detect_patterns",
            category="analysis",
            description="Detect graph patterns",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.analyze_flow",
            category="analysis",
            description="Analyze network flow",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.find_sccs",
            category="analysis",
            description="Find strongly connected components",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.calculate_global_statistics",
            category="analysis",
            description="Calculate global graph statistics",
            property_mapping={},
            scene_props=[],
        ))

    def _register_layout_adapters(self) -> None:
        """Register layout adapters."""
        self.register(OperatorAdapter(
            bl_idname="scigraphs.apply_layout",
            category="layout",
            description="Apply graph layout algorithm",
            property_mapping={
                "algorithm": "algorithm",
                "scale": "scale",
                "iterations": "iterations",
                "seed": "seed",
            },
            scene_props=["layout_algorithm", "layout_scale", "layout_iterations"],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.execute_layout_step",
            category="layout",
            description="Execute single layout iteration",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.reset_layout",
            category="layout",
            description="Reset layout to initial positions",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.network_splitter_3d",
            category="layout",
            description="Split network into 3D layers",
            property_mapping={},
            scene_props=[],
        ))

    def _register_visual_adapters(self) -> None:
        """Register visualization adapters."""
        self.register(OperatorAdapter(
            bl_idname="scigraphs.setup_visualization",
            category="visual",
            description="Setup geometry nodes visualization",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.update_appearance",
            category="visual",
            description="Update visual appearance",
            property_mapping={},
            scene_props=["node_size", "edge_thickness", "node_color", "edge_color"],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.apply_rendering_preset",
            category="visual",
            description="Apply rendering preset",
            property_mapping={
                "preset": "preset",
            },
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.setup_lighting",
            category="visual",
            description="Setup scene lighting",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.apply_edge_style",
            category="visual",
            description="Apply edge style parameters",
            property_mapping={
                "curvature": "curvature",
                "resolution": "resolution",
            },
            scene_props=["edge_curvature", "edge_resolution"],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.apply_edge_style_preset",
            category="visual",
            description="Apply edge style preset",
            property_mapping={
                "preset": "preset",
            },
            scene_props=["edge_style_preset"],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.generate_text_overlay",
            category="visual",
            description="Generate text overlay labels",
            property_mapping={},
            scene_props=[],
        ))

    def _register_topology_adapters(self) -> None:
        """Register topology adapters."""
        self.register(OperatorAdapter(
            bl_idname="scigraphs.check_planarity",
            category="topology",
            description="Check graph planarity",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.calculate_genus",
            category="topology",
            description="Calculate graph genus",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.compute_faces",
            category="topology",
            description="Compute graph faces",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.validate_crossings",
            category="topology",
            description="Validate edge crossings",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.visualize_surface",
            category="topology",
            description="Visualize topological surface",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.create_dual_graph",
            category="topology",
            description="Create dual graph",
            property_mapping={},
            scene_props=[],
        ))

    def _register_osmnx_adapters(self) -> None:
        """Register OSMnx-specific adapters."""
        self.register(OperatorAdapter(
            bl_idname="scigraphs.osmnx_centrality",
            category="osmnx",
            description="Compute OSMnx centrality",
            property_mapping={
                "centrality_type": "centrality_type",
            },
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.osmnx_shortest_path",
            category="osmnx",
            description="Compute shortest path",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.osmnx_isochrones",
            category="osmnx",
            description="Compute isochrones",
            property_mapping={
                "travel_time": "travel_time",
            },
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.osmnx_add_elevations_raster",
            category="osmnx",
            description="Add elevations from raster",
            property_mapping={
                "filepath": "filepath",
            },
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.osmnx_apply_elevation_3d",
            category="osmnx",
            description="Apply 3D elevation to network",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.osmnx_project_graph",
            category="osmnx",
            description="Project graph to UTM",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.osmnx_simplify",
            category="osmnx",
            description="Simplify graph topology",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.osmnx_consolidate",
            category="osmnx",
            description="Consolidate intersections",
            property_mapping={
                "tolerance": "tolerance",
            },
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.osmnx_basic_stats",
            category="osmnx",
            description="Compute basic network statistics",
            property_mapping={},
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.osmnx_orientation_entropy",
            category="osmnx",
            description="Compute orientation entropy",
            property_mapping={},
            scene_props=[],
        ))

    def _register_export_adapters(self) -> None:
        """Register export adapters."""
        self.register(OperatorAdapter(
            bl_idname="scigraphs.export_graph",
            category="export",
            description="Export graph to file",
            property_mapping={
                "filepath": "filepath",
                "format": "export_format",
            },
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.export_positions",
            category="export",
            description="Export node positions",
            property_mapping={
                "filepath": "filepath",
            },
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.generate_statistics_report",
            category="export",
            description="Generate statistics report",
            property_mapping={
                "filepath": "filepath",
            },
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.export_gexf",
            category="export",
            description="Export to GEXF format",
            property_mapping={
                "filepath": "filepath",
            },
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.export_graphml",
            category="export",
            description="Export to GraphML format",
            property_mapping={
                "filepath": "filepath",
            },
            scene_props=[],
        ))

        self.register(OperatorAdapter(
            bl_idname="scigraphs.osmnx_export",
            category="export",
            description="Export OSMnx graph",
            property_mapping={
                "filepath": "filepath",
                "format": "format",
            },
            scene_props=[],
        ))

    def _register_render_adapters(self) -> None:
        """Register render adapters."""
        self.register(OperatorAdapter(
            bl_idname="render.render",
            category="render",
            description="Render current view",
            property_mapping={
                "write_still": "write_still",
            },
            scene_props=[],
        ))

    def _register_shortcuts(self) -> None:
        """Register common shortcut names."""
        shortcuts = {
            # Datasets
            "import_osmnx": "scigraphs.import_osm_graph",
            "osmnx": "scigraphs.import_osm_graph",
            "create_graph": "scigraphs.create_graph",
            "suitesparse": "scigraphs.download_suitesparse",
            "sql": "scigraphs.create_graph_from_sql",
            "overture": "scigraphs.c2g_load_overture",
            "c2g_place": "scigraphs.c2g_load_overture_place",
            # Analysis
            "centrality": "scigraphs.calculate_centrality",
            "clustering": "scigraphs.apply_clustering",
            "communities": "scigraphs.apply_clustering",
            "sccs": "scigraphs.find_sccs",
            "patterns": "scigraphs.detect_patterns",
            "flow": "scigraphs.analyze_flow",
            "stats": "scigraphs.calculate_global_statistics",
            # Layout
            "layout": "scigraphs.apply_layout",
            "layout_step": "scigraphs.execute_layout_step",
            "reset_layout": "scigraphs.reset_layout",
            "splitter": "scigraphs.network_splitter_3d",
            # Visual
            "setup_vis": "scigraphs.setup_visualization",
            "appearance": "scigraphs.update_appearance",
            "preset": "scigraphs.apply_rendering_preset",
            "lighting": "scigraphs.setup_lighting",
            "edge_style": "scigraphs.apply_edge_style_preset",
            "text_overlay": "scigraphs.generate_text_overlay",
            # Topology
            "planarity": "scigraphs.check_planarity",
            "genus": "scigraphs.calculate_genus",
            "faces": "scigraphs.compute_faces",
            "crossings": "scigraphs.validate_crossings",
            "surface": "scigraphs.visualize_surface",
            "dual": "scigraphs.create_dual_graph",
            # OSMnx
            "osmnx_centrality": "scigraphs.osmnx_centrality",
            "shortest_path": "scigraphs.osmnx_shortest_path",
            "isochrones": "scigraphs.osmnx_isochrones",
            "elevation_raster": "scigraphs.osmnx_add_elevations_raster",
            "elevation_3d": "scigraphs.osmnx_apply_elevation_3d",
            "project": "scigraphs.osmnx_project_graph",
            "simplify": "scigraphs.osmnx_simplify",
            "consolidate": "scigraphs.osmnx_consolidate",
            "osmnx_stats": "scigraphs.osmnx_basic_stats",
            "orientation": "scigraphs.osmnx_orientation_entropy",
            # Export
            "export": "scigraphs.export_graph",
            "export_positions": "scigraphs.export_positions",
            "export_stats": "scigraphs.generate_statistics_report",
            "export_gexf": "scigraphs.export_gexf",
            "export_graphml": "scigraphs.export_graphml",
            "osmnx_export": "scigraphs.osmnx_export",
            # Render
            "render": "render.render",
        }
        for short, full in shortcuts.items():
            self.register_shortcut(short, full)


def get_registry() -> OperatorRegistry:
    """Get the global operator registry instance."""
    global _registry
    if _registry is None:
        _registry = OperatorRegistry()
        _registry.initialize()
    return _registry


def call_operator(
    bl_idname: str,
    props: Optional[Dict[str, Any]] = None,
    scene_props: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Call a Blender operator with given properties.

    Args:
        bl_idname: Operator bl_idname (e.g., "scigraphs.apply_layout")
        props: Properties to pass to operator
        scene_props: Scene properties to set before calling

    Returns:
        Result dictionary with 'status' and optional 'error'
    """
    try:
        import bpy
    except ImportError:
        return {"status": "error", "error": "Blender not available"}

    props = props or {}
    scene_props = scene_props or {}

    # Set scene properties if needed
    if scene_props and hasattr(bpy.context, "scene"):
        scigraphs_props = getattr(bpy.context.scene, "scigraphs", None)
        if scigraphs_props:
            for key, value in scene_props.items():
                if hasattr(scigraphs_props, key):
                    try:
                        setattr(scigraphs_props, key, value)
                    except Exception:
                        pass  # Ignore property errors

    # Get operator
    parts = bl_idname.split(".")
    if len(parts) != 2:
        return {"status": "error", "error": f"Invalid bl_idname: {bl_idname}"}

    category, name = parts
    ops_category = getattr(bpy.ops, category, None)
    if ops_category is None:
        return {"status": "error", "error": f"Operator category not found: {category}"}

    op = getattr(ops_category, name, None)
    if op is None:
        return {"status": "error", "error": f"Operator not found: {bl_idname}"}

    # Call operator
    try:
        result = op(**props)
        if result == {"FINISHED"}:
            return {"status": "success"}
        elif result == {"CANCELLED"}:
            return {"status": "cancelled"}
        else:
            return {"status": "unknown", "result": str(result)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def prepare_operator_props(
    adapter: OperatorAdapter,
    pipeline_props: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Prepare operator and scene properties from pipeline properties.

    Args:
        adapter: Operator adapter with mappings
        pipeline_props: Properties from pipeline spec

    Returns:
        Tuple of (operator_props, scene_props)
    """
    op_props = {}
    scene_props = {}

    # Map pipeline props to operator props
    for pipeline_key, op_key in adapter.property_mapping.items():
        if pipeline_key in pipeline_props:
            op_props[op_key] = pipeline_props[pipeline_key]

    # Extract scene props
    for scene_key in adapter.scene_props:
        # Map common names
        mapping = {
            "osmnx_method": "method",
            "osmnx_place": "query",
            "osmnx_network_type": "network_type",
            "layout_algorithm": "algorithm",
            "layout_scale": "scale",
            "layout_iterations": "iterations",
            "clustering_algorithm": "algorithm",
            "clustering_resolution": "resolution",
            "centrality_type": "metric",
            "node_color_attribute": "node_color",
            "node_size_attribute": "node_size",
            "colormap": "colormap",
            "edge_style_preset": "edge_style",
        }
        pipeline_key = mapping.get(scene_key, scene_key)
        if pipeline_key in pipeline_props:
            scene_props[scene_key] = pipeline_props[pipeline_key]

    return op_props, scene_props
