# SciGraphs HUD Overlay
# Draws graph statistics and info directly in the 3D viewport

import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader

_HANDLE = None
_ENABLED = False


def _get_graph_stats(obj):
    """Extract stats from a graph object."""
    if not obj or obj.type != 'MESH' or "num_nodes" not in obj:
        return None

    nodes = int(obj.get("num_nodes", 0) or 0)
    edges = int(obj.get("num_edges", 0) or 0)
    directed = bool(obj.get("is_directed", False))
    possible_edges = nodes * (nodes - 1)
    if not directed:
        possible_edges *= 0.5
    density = (edges / possible_edges) if possible_edges else 0.0
    avg_degree = (edges / nodes) if directed and nodes else (2 * edges / nodes if nodes else 0.0)

    mesh = obj.data
    attr_names = [
        attr.name for attr in mesh.attributes
        if not attr.name.startswith(".")
        and attr.name not in {'position', 'normal', 'sharp_edge', 'crease_edge', 'material_index', 'UVMap'}
    ]
    
    stats = {
        'name': obj.name,
        'nodes': nodes,
        'edges': edges,
        'directed': directed,
        'density': density,
        'avg_degree': avg_degree,
        'layout': obj.get("layout_algorithm", "unknown"),
        'has_viz': obj.modifiers.get("SciGraphs_Viz") is not None,
        'is_osmnx': obj.get("is_osmnx", False),
        'is_geospatial': obj.get("is_geospatial", False),
        'attributes': attr_names,
        'clusters': obj.get("num_clusters", None),
        'genus': obj.get("genus", None),
        'is_planar': obj.get("is_planar", None),
        'shortest_path_length': obj.get("shortest_path_length", None),
        'shortest_path_nodes': obj.get("shortest_path_nodes", None),
        'shortest_path_source': obj.get("shortest_path_source", None),
        'shortest_path_target': obj.get("shortest_path_target", None),
        'traversal': obj.get("traversal_algorithm", None),
        'traversal_count': obj.get("traversal_visited_count", None),
        'mst_edges': obj.get("mst_edges", None),
        'mst_weight': obj.get("mst_weight", None),
        'max_flow': obj.get("max_flow_value", None),
        'sccs': obj.get("num_sccs", None),
        'diameter': obj.get("stat_diameter", None),
        'avg_path': obj.get("stat_avg_path_length", None),
        'global_clustering': obj.get("stat_global_clustering", None),
        'assortativity': obj.get("stat_assortativity", None),
        'degree_min': obj.get("stat_degree_min", None),
        'degree_max': obj.get("stat_degree_max", None),
    }
    
    # Additional OSMnx stats
    if stats['is_osmnx']:
        stats['osmnx_place'] = obj.get("osmnx_query_name", obj.get("osmnx_place", "Network"))
    
    return stats


def _text_width(text, size=13):
    """Return text width in pixels for the current Blender font."""
    font_id = 0
    blf.size(font_id, size)
    return blf.dimensions(font_id, text)[0]


def _draw_text(x, y, text, size=14, color=(1.0, 1.0, 1.0, 0.9)):
    """Draw text at screen position."""
    font_id = 0
    blf.size(font_id, size)
    blf.color(font_id, *color)
    blf.position(font_id, x, y, 0)
    blf.draw(font_id, text)


def _draw_background(x, y, width, height, color=(0.1, 0.1, 0.1, 0.7)):
    """Draw a semi-transparent background rectangle."""
    vertices = [
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    ]
    indices = [(0, 1, 2), (0, 2, 3)]
    
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices=indices)
    
    gpu.state.blend_set('ALPHA')
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.blend_set('NONE')


def _draw_hud_callback():
    """Main draw callback for the HUD overlay."""
    context = bpy.context
    
    # Only draw in 3D View
    if context.area is None or context.area.type != 'VIEW_3D':
        return
    
    # Check if overlay is enabled
    props = getattr(context.scene, 'scigraphs', None)
    if props is None:
        return
    if not getattr(props, 'show_hud_overlay', True):
        return
    
    obj = context.active_object
    stats = _get_graph_stats(obj)
    
    if not stats:
        return
    
    # Build text lines
    lines = []
    lines.append(f"Graph: {stats['name']}")
    lines.append(f"Nodes: {stats['nodes']:,}  |  Edges: {stats['edges']:,}")
    lines.append(f"Avg degree: {stats['avg_degree']:.2f}  |  Density: {stats['density']:.4f}")
    
    graph_type = "Directed" if stats['directed'] else "Undirected"
    if stats['is_osmnx']:
        graph_type += " | OSMnx"
    elif stats['is_geospatial']:
        graph_type += " | Geospatial"
    lines.append(f"Type: {graph_type}")

    if stats['layout'] and stats['layout'] != "unknown":
        lines.append(f"Layout: {stats['layout']}")

    lines.append(f"Visualization: {'Geometry Nodes' if stats['has_viz'] else 'Mesh only'}")
    
    if stats['is_osmnx']:
        lines.append(f"OSMnx: {stats.get('osmnx_place', 'Network')}")

    if stats['attributes']:
        preview = ", ".join(stats['attributes'][:3])
        suffix = "..." if len(stats['attributes']) > 3 else ""
        lines.append(f"Attributes: {len(stats['attributes'])} ({preview}{suffix})")
    
    if stats['clusters'] is not None:
        lines.append(f"Clusters: {stats['clusters']}")

    if stats['shortest_path_length'] is not None:
        source = stats['shortest_path_source']
        target = stats['shortest_path_target']
        path_nodes = stats['shortest_path_nodes']
        label = f"Shortest path: {stats['shortest_path_length']:.2f}"
        if path_nodes is not None:
            label += f" ({path_nodes} nodes)"
        if source is not None and target is not None:
            label += f" {source}->{target}"
        lines.append(label)

    if stats['traversal'] is not None:
        count = stats['traversal_count'] if stats['traversal_count'] is not None else "?"
        lines.append(f"Traversal: {stats['traversal']} ({count} nodes)")

    if stats['mst_edges'] is not None:
        weight = stats['mst_weight']
        if weight is not None:
            lines.append(f"MST: {stats['mst_edges']} edges | weight {weight:.2f}")
        else:
            lines.append(f"MST: {stats['mst_edges']} edges")

    if stats['max_flow'] is not None:
        lines.append(f"Max flow: {stats['max_flow']:.2f}")

    if stats['sccs'] is not None:
        lines.append(f"Strong components: {stats['sccs']}")

    if stats['diameter'] is not None:
        if stats['avg_path'] is not None:
            lines.append(f"Diameter: {stats['diameter']}  |  Avg path: {stats['avg_path']:.3f}")
        else:
            lines.append(f"Diameter: {stats['diameter']}")

    if stats['global_clustering'] is not None:
        lines.append(f"Global clustering: {stats['global_clustering']:.3f}")

    if stats['assortativity'] is not None:
        lines.append(f"Assortativity: {stats['assortativity']:.3f}")

    if stats['degree_min'] is not None and stats['degree_max'] is not None:
        lines.append(f"Degree range: {stats['degree_min']:.0f}-{stats['degree_max']:.0f}")
    
    if stats['is_planar'] is not None:
        planar_str = "Yes" if stats['is_planar'] else "No"
        lines.append(f"Planar: {planar_str}")
        if stats['genus'] is not None:
            lines.append(f"Genus: {stats['genus']}")
    
    # Calculate background size and anchor to bottom-right.
    region = context.region
    padding = 18
    line_height = 18
    max_text_width = max(_text_width(line, 15 if i == 0 else 13) for i, line in enumerate(lines))
    max_width = min(max(300, int(max_text_width) + 24), max(320, region.width - padding * 2))
    bg_height = len(lines) * line_height + 12
    x = region.width - max_width + 8 - padding
    y = padding + 8
    
    # Draw background at bottom-right
    _draw_background(x - 8, y - 8, max_width, bg_height, (0.05, 0.05, 0.08, 0.8))
    
    # Draw lines (from bottom up)
    for i, line in enumerate(lines):
        text_y = y + (len(lines) - 1 - i) * line_height
        
        # Header in different color
        if i == 0:
            _draw_text(x, text_y, line, size=15, color=(0.4, 0.8, 1.0, 1.0))
        else:
            _draw_text(x, text_y, line, size=13, color=(0.9, 0.9, 0.9, 0.9))


def enable_hud():
    """Enable the HUD overlay."""
    global _HANDLE, _ENABLED
    if not _ENABLED:
        _HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _draw_hud_callback, (), 'WINDOW', 'POST_PIXEL'
        )
        _ENABLED = True


def disable_hud():
    """Disable the HUD overlay."""
    global _HANDLE, _ENABLED
    if _ENABLED and _HANDLE:
        bpy.types.SpaceView3D.draw_handler_remove(_HANDLE, 'WINDOW')
        _HANDLE = None
        _ENABLED = False


class SCIGRAPHS_OT_toggle_hud(bpy.types.Operator):
    """Toggle the HUD overlay on/off."""
    bl_idname = "scigraphs.toggle_hud"
    bl_label = "Toggle Graph HUD"
    bl_description = "Show/hide graph statistics overlay in viewport"

    def execute(self, context):
        global _ENABLED
        props = context.scene.scigraphs
        
        if _ENABLED:
            disable_hud()
            props.show_hud_overlay = False
            self.report({'INFO'}, "HUD overlay disabled")
        else:
            enable_hud()
            props.show_hud_overlay = True
            self.report({'INFO'}, "HUD overlay enabled")
        
        # Force redraw
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        
        return {'FINISHED'}


_CLASSES = [
    SCIGRAPHS_OT_toggle_hud,
]


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    
    # Add property for HUD toggle
    bpy.types.Scene.scigraphs_show_hud = bpy.props.BoolProperty(
        name="Show Graph HUD",
        description="Display graph statistics overlay in viewport",
        default=True,
        update=lambda self, ctx: enable_hud() if self.scigraphs_show_hud else disable_hud()
    )
    
    # Auto-enable on startup
    enable_hud()


def unregister():
    disable_hud()
    
    if hasattr(bpy.types.Scene, 'scigraphs_show_hud'):
        del bpy.types.Scene.scigraphs_show_hud
    
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
