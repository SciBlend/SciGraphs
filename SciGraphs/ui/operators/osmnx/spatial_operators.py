import bpy
from bpy.props import StringProperty
from ....core import osmnx_analysis
from .utils import (
    _get_osmnx_graph,
    _get_unprojected_graph,
    _draw_highlight_circle,
    _draw_highlight_line,
    _draw_highlight_point,
    _mark_shortest_path_attributes,
)


def _graph_has_edge_attribute(G, attr):
    """Return True iff ``attr`` is present and non-empty on every edge of G.

    Tolerates both MultiDiGraph (the OSMnx default) and DiGraph / Graph
    variants produced by ``to_digraph`` / ``to_undirected``.
    """
    if G is None or G.number_of_edges() == 0:
        return False

    if getattr(G, "is_multigraph", lambda: False)():
        edge_iter = (data for _u, _v, _k, data in G.edges(keys=True, data=True))
    else:
        edge_iter = (data for _u, _v, data in G.edges(data=True))

    for data in edge_iter:
        if data.get(attr) in (None, ""):
            return False
    return True


class SCIGRAPHS_OT_SelectNearestNode(bpy.types.Operator):
    """Interactive modal operator to select nearest node by clicking in viewport."""
    bl_idname = "scigraphs.osmnx_select_nearest_node"
    bl_label = "Select Nearest Node"
    bl_description = "Click in the viewport to find the nearest graph node (with highlight)"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        if event.type == 'MOUSEMOVE':
            self.mouse_x = event.mouse_region_x
            self.mouse_y = event.mouse_region_y
            
            self._update_highlight(context, event)
        
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._remove_draw_handler(context)
            
            result = self._find_node_at_click(context, event)
            
            if result:
                props = context.scene.scigraphs
                props.osmnx_selected_node_id = str(result['node_id'])
                
                obj = context.active_object
                obj["osmnx_last_selected_node"] = str(result['node_id'])
                if result.get('coords'):
                    obj["osmnx_last_selected_lat"] = result['coords'][0]
                    obj["osmnx_last_selected_lon"] = result['coords'][1]
                
                self.report({'INFO'}, f"Selected node: {result['node_id']}")
            else:
                self.report({'WARNING'}, "Could not find node at this position")
            
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self._remove_draw_handler(context)
            self.report({'INFO'}, "Node selection cancelled")
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        obj = context.active_object
        self.graph = _get_osmnx_graph(obj)
        
        if self.graph is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        self.query_graph = _get_unprojected_graph(obj)
        if self.query_graph is None:
            self.query_graph = self.graph
        
        self.obj = obj
        self.mouse_x = 0
        self.mouse_y = 0
        self.highlight_pos = None
        self.highlight_node_id = None
        
        self._build_vertex_lookup()
        
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback, (context,), 'WINDOW', 'POST_VIEW'
        )
        
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Move cursor to highlight nodes. Click to select (ESC to cancel)")
        return {'RUNNING_MODAL'}
    
    def _build_vertex_lookup(self):
        """Build a lookup of vertex positions for fast nearest-vertex finding."""
        import numpy as np
        
        mesh = self.obj.data
        nodes_str = self.obj.get("nodes_data", "")
        
        if not nodes_str:
            self.vertex_positions = np.array([])
            self.node_ids = []
            return
        
        self.node_ids = nodes_str.split(",")
        num_intersections = len(self.node_ids)
        
        positions = []
        for i in range(min(num_intersections, len(mesh.vertices))):
            v = mesh.vertices[i]
            positions.append([v.co.x, v.co.y, v.co.z])
        
        self.vertex_positions = np.array(positions) if positions else np.array([])
    
    def _update_highlight(self, context, event):
        """
        Update the highlighted node based on cursor position.
        
        Projects mesh vertices to screen space and finds the nearest one
        to the cursor. Works regardless of Geometry Nodes or 3D elevation.
        """
        from bpy_extras import view3d_utils
        import numpy as np
        
        region = context.region
        rv3d = context.region_data
        
        if region is None or rv3d is None:
            return
        
        if len(self.vertex_positions) == 0:
            self.highlight_pos = None
            self.highlight_node_id = None
            return
        
        cursor_2d = np.array([event.mouse_region_x, event.mouse_region_y])
        
        min_dist = float('inf')
        nearest_idx = -1
        
        for i, pos_3d in enumerate(self.vertex_positions):
            pos_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, pos_3d)
            
            if pos_2d is None:
                continue
            
            dist = np.linalg.norm(np.array([pos_2d.x, pos_2d.y]) - cursor_2d)
            
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        threshold_pixels = 50
        
        if nearest_idx >= 0 and min_dist < threshold_pixels:
            self.highlight_pos = tuple(self.vertex_positions[nearest_idx])
            self.highlight_node_id = self.node_ids[nearest_idx] if nearest_idx < len(self.node_ids) else None
        else:
            self.highlight_pos = None
            self.highlight_node_id = None
    
    def _draw_callback(self, context):
        """Draw the highlight circle."""
        if self.highlight_pos is None:
            return
        
        scale = self.obj.get("osmnx_scale", 0.001)
        radius = 30 * scale
        
        color = (1.0, 0.9, 0.0, 0.9)
        
        _draw_highlight_circle(self.highlight_pos, radius, color)
        _draw_highlight_point(self.highlight_pos, (1.0, 0.5, 0.0, 1.0), size=12.0)
    
    def _remove_draw_handler(self, context):
        """Remove the draw handler."""
        if hasattr(self, '_draw_handle') and self._draw_handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            self._draw_handle = None
    
    def _find_node_at_click(self, context, event):
        """Convert click to lat/lon and find nearest node."""
        from bpy_extras import view3d_utils
        import numpy as np
        
        region = context.region
        rv3d = context.region_data
        
        coord = (event.mouse_region_x, event.mouse_region_y)
        
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        
        depsgraph = context.evaluated_depsgraph_get()
        result, location, normal, index, obj_hit, matrix = context.scene.ray_cast(
            depsgraph, ray_origin, view_vector
        )
        
        if not result or obj_hit != self.obj:
            if abs(view_vector.z) > 0.001:
                t = -ray_origin.z / view_vector.z
                location = ray_origin + view_vector * t
            else:
                return None
        
        scale = self.obj.get("osmnx_scale", 0.001)
        
        query_graph = self.query_graph
        
        extent = osmnx_analysis.get_graph_extent(query_graph)
        if extent is None:
            return None
        
        center_lat = extent['center_lat']
        center_lon = extent['center_lon']
        
        EARTH_RADIUS = 6371000.0
        cos_lat = np.cos(np.radians(center_lat))
        meters_per_deg = np.pi / 180.0 * EARTH_RADIUS
        
        x_m = location.x / scale
        y_m = location.y / scale
        
        lat = center_lat + (y_m / meters_per_deg)
        lon = center_lon + (x_m / (meters_per_deg * cos_lat))
        
        node_id = osmnx_analysis.find_nearest_node(query_graph, lon, lat, is_projected=False)
        
        if node_id is None:
            return None
        
        coords = osmnx_analysis.get_node_coordinates(query_graph, node_id)
        
        return {
            'node_id': node_id,
            'coords': coords,
            'click_lat': lat,
            'click_lon': lon,
        }


class SCIGRAPHS_OT_SelectNearestEdge(bpy.types.Operator):
    """Interactive modal operator to select nearest edge by clicking in viewport."""
    bl_idname = "scigraphs.osmnx_select_nearest_edge"
    bl_label = "Select Nearest Edge"
    bl_description = "Click in the viewport to find the nearest graph edge (with highlight)"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        if event.type == 'MOUSEMOVE':
            self._update_highlight(context, event)
        
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._remove_draw_handler(context)
            
            result = self._find_edge_at_click(context, event)
            
            if result:
                props = context.scene.scigraphs
                props.osmnx_selected_edge_u = str(result['u'])
                props.osmnx_selected_edge_v = str(result['v'])
                
                obj = context.active_object
                obj["osmnx_last_selected_edge_u"] = result['u']
                obj["osmnx_last_selected_edge_v"] = result['v']
                
                self.report({'INFO'}, f"Selected edge: ({result['u']}, {result['v']})")
            else:
                self.report({'WARNING'}, "Could not find edge at this position")
            
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self._remove_draw_handler(context)
            self.report({'INFO'}, "Edge selection cancelled")
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        obj = context.active_object
        self.graph = _get_osmnx_graph(obj)
        
        if self.graph is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        self.query_graph = _get_unprojected_graph(obj)
        if self.query_graph is None:
            self.query_graph = self.graph
        
        self.obj = obj
        self.highlight_edges = []
        self.highlight_edge_ids = None
        
        self._build_edge_lookup()
        
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback, (context,), 'WINDOW', 'POST_VIEW'
        )
        
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Move cursor to highlight edges. Click to select (ESC to cancel)")
        return {'RUNNING_MODAL'}
    
    def _build_edge_lookup(self):
        """Build lookup structures for edge highlighting."""
        import numpy as np
        from collections import defaultdict
        
        mesh = self.obj.data
        nodes_str = self.obj.get("nodes_data", "")
        
        if not nodes_str:
            self.edge_segments = []
            self.edge_midpoints = np.array([])
            return
        
        node_ids = nodes_str.split(",")
        num_intersections = len(node_ids)
        
        mesh_adj = defaultdict(dict)
        for edge_idx, edge in enumerate(mesh.edges):
            v0, v1 = edge.vertices
            mesh_adj[v0][v1] = edge_idx
            mesh_adj[v1][v0] = edge_idx
        
        self.edge_segments = []
        midpoints = []
        
        for edge_idx, edge in enumerate(mesh.edges):
            v0, v1 = edge.vertices
            pos0 = mesh.vertices[v0].co
            pos1 = mesh.vertices[v1].co
            
            segment = {
                'start': (pos0.x, pos0.y, pos0.z),
                'end': (pos1.x, pos1.y, pos1.z),
                'v0': v0,
                'v1': v1,
                'edge_idx': edge_idx,
            }
            self.edge_segments.append(segment)
            
            mid = ((pos0.x + pos1.x) / 2, (pos0.y + pos1.y) / 2, (pos0.z + pos1.z) / 2)
            midpoints.append(mid)
        
        self.edge_midpoints = np.array(midpoints) if midpoints else np.array([])
        self.node_ids = node_ids
        self.num_intersections = num_intersections
    
    def _update_highlight(self, context, event):
        """
        Update highlighted edge based on cursor position.
        
        Projects edge midpoints to screen space and finds the nearest one.
        Works regardless of Geometry Nodes.
        """
        from bpy_extras import view3d_utils
        import numpy as np
        
        region = context.region
        rv3d = context.region_data
        
        if region is None or rv3d is None:
            return
        
        if len(self.edge_midpoints) == 0:
            self.highlight_edges = []
            return
        
        cursor_2d = np.array([event.mouse_region_x, event.mouse_region_y])
        
        min_dist = float('inf')
        nearest_idx = -1
        
        for i, mid_3d in enumerate(self.edge_midpoints):
            pos_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, mid_3d)
            
            if pos_2d is None:
                continue
            
            dist = np.linalg.norm(np.array([pos_2d.x, pos_2d.y]) - cursor_2d)
            
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        threshold_pixels = 50
        
        if nearest_idx >= 0 and min_dist < threshold_pixels:
            segment = self.edge_segments[nearest_idx]
            self.highlight_edges = [(segment['start'], segment['end'])]
            
            v0, v1 = segment['v0'], segment['v1']
            if v0 < self.num_intersections and v1 < self.num_intersections:
                self.highlight_edge_ids = (self.node_ids[v0], self.node_ids[v1])
            else:
                self.highlight_edge_ids = None
        else:
            self.highlight_edges = []
            self.highlight_edge_ids = None
    
    def _draw_callback(self, context):
        """Draw highlighted edges."""
        if not self.highlight_edges:
            return
        
        color = (0.0, 1.0, 1.0, 0.9)
        
        for start, end in self.highlight_edges:
            _draw_highlight_line(start, end, color, width=6.0)
    
    def _remove_draw_handler(self, context):
        """Remove the draw handler."""
        if hasattr(self, '_draw_handle') and self._draw_handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            self._draw_handle = None
    
    def _find_edge_at_click(self, context, event):
        """Convert click to lat/lon and find nearest edge."""
        from bpy_extras import view3d_utils
        import numpy as np
        
        region = context.region
        rv3d = context.region_data
        
        coord = (event.mouse_region_x, event.mouse_region_y)
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        
        if abs(view_vector.z) > 0.001:
            t = -ray_origin.z / view_vector.z
            location = ray_origin + view_vector * t
        else:
            return None
        
        scale = self.obj.get("osmnx_scale", 0.001)
        query_graph = self.query_graph
        
        extent = osmnx_analysis.get_graph_extent(query_graph)
        if extent is None:
            return None
        
        center_lat = extent['center_lat']
        center_lon = extent['center_lon']
        
        EARTH_RADIUS = 6371000.0
        cos_lat = np.cos(np.radians(center_lat))
        meters_per_deg = np.pi / 180.0 * EARTH_RADIUS
        
        x_m = location.x / scale
        y_m = location.y / scale
        
        lat = center_lat + (y_m / meters_per_deg)
        lon = center_lon + (x_m / (meters_per_deg * cos_lat))
        
        edge = osmnx_analysis.find_nearest_edge(query_graph, lon, lat, is_projected=False)
        
        if edge is None:
            return None
        
        u, v, key = edge
        
        return {
            'u': u,
            'v': v,
            'key': key,
        }


class SCIGRAPHS_OT_CalculateShortestPath(bpy.types.Operator):
    """Calculate shortest path between two nodes."""
    bl_idname = "scigraphs.osmnx_shortest_path"
    bl_label = "Calculate Shortest Path"
    bl_description = "Find the shortest path between source and target nodes"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or not obj.get("is_osmnx", False):
            return False
        props = context.scene.scigraphs
        return props.osmnx_shortest_path_source and props.osmnx_shortest_path_target
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        G = _get_osmnx_graph(obj)
        G_unprojected = _get_unprojected_graph(obj)
        
        if G is None and G_unprojected is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        try:
            source = int(props.osmnx_shortest_path_source)
            target = int(props.osmnx_shortest_path_target)
        except ValueError:
            self.report({'ERROR'}, "Invalid node IDs (must be integers)")
            return {'CANCELLED'}
        
        routing_graph = None
        
        if G is not None and source in G.nodes and target in G.nodes:
            routing_graph = G
        elif G_unprojected is not None and source in G_unprojected.nodes and target in G_unprojected.nodes:
            routing_graph = G_unprojected
        elif G is not None:
            routing_graph = G
        elif G_unprojected is not None:
            routing_graph = G_unprojected
        
        if routing_graph is None:
            self.report({'ERROR'}, "No valid graph for routing")
            return {'CANCELLED'}
        
        if source not in routing_graph.nodes:
            self.report({'ERROR'}, f"Source node {source} not found in graph")
            return {'CANCELLED'}
        if target not in routing_graph.nodes:
            self.report({'ERROR'}, f"Target node {target} not found in graph")
            return {'CANCELLED'}
        
        weight = props.osmnx_path_weight

        if weight == 'travel_time':
            if not _graph_has_edge_attribute(routing_graph, 'travel_time'):
                obj["osmnx_has_travel_times"] = False
                self.report(
                    {'ERROR'},
                    "Travel times missing on this graph. Add edge speeds and travel times first.",
                )
                return {'CANCELLED'}

        if weight == 'elevation_impedance' and not obj.get("osmnx_has_elevation", False):
            self.report({'ERROR'}, "Node elevations not calculated. Add elevations first.")
            return {'CANCELLED'}

        from ....core.osmnx import routing as _routing
        result = _routing.calculate_shortest_path(
            routing_graph, source, target,
            weight=weight,
            impedance_alpha=props.osmnx_impedance_alpha,
        )
        
        if result is None or 'error' in result:
            error_msg = result.get('error', 'Unknown error') if result else 'Calculation failed'
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}
        
        obj["osmnx_path_distance_m"] = result['distance_m']
        obj["osmnx_path_distance_km"] = result['distance_km']
        obj["osmnx_path_num_nodes"] = result['num_nodes']
        obj["osmnx_path_num_edges"] = result['num_edges']
        
        if 'travel_time_seconds' in result:
            obj["osmnx_path_travel_time_s"] = result['travel_time_seconds']
            obj["osmnx_path_travel_time_min"] = result['travel_time_minutes']
            msg = f"Path: {result['distance_km']:.2f} km, {result['travel_time_minutes']:.1f} min ({result['num_nodes']} nodes)"
        else:
            msg = f"Path: {result['distance_km']:.2f} km ({result['num_nodes']} nodes)"
        
        obj["osmnx_last_path"] = str(result['path'])
        
        path_nodes = result['path']
        _mark_shortest_path_attributes(obj, path_nodes)
        
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class SCIGRAPHS_OT_SelectPathSource(bpy.types.Operator):
    """Interactive selection of path source node with visual highlight."""
    bl_idname = "scigraphs.osmnx_select_path_source"
    bl_label = "Select Source Node"
    bl_description = "Click on a node to set it as path source (with highlight)"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        if event.type == 'MOUSEMOVE':
            self._update_highlight(context, event)
        
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._remove_draw_handler(context)
            
            if self.highlight_node_id is not None:
                props = context.scene.scigraphs
                props.osmnx_shortest_path_source = str(self.highlight_node_id)
                props.osmnx_selected_node_id = str(self.highlight_node_id)
                self.report({'INFO'}, f"Source set to node {self.highlight_node_id}")
            else:
                self.report({'WARNING'}, "No node under cursor")
            
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self._remove_draw_handler(context)
            self.report({'INFO'}, "Source selection cancelled")
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        obj = context.active_object
        self.obj = obj
        self.highlight_pos = None
        self.highlight_node_id = None
        
        self._build_vertex_lookup()
        
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback, (context,), 'WINDOW', 'POST_VIEW'
        )
        
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Click on a node to set as SOURCE (ESC to cancel)")
        return {'RUNNING_MODAL'}
    
    def _build_vertex_lookup(self):
        """Build vertex position lookup."""
        import numpy as np
        
        mesh = self.obj.data
        nodes_str = self.obj.get("nodes_data", "")
        
        if not nodes_str:
            self.vertex_positions = np.array([])
            self.node_ids = []
            return
        
        self.node_ids = nodes_str.split(",")
        num_intersections = len(self.node_ids)
        
        positions = []
        for i in range(min(num_intersections, len(mesh.vertices))):
            v = mesh.vertices[i]
            positions.append([v.co.x, v.co.y, v.co.z])
        
        self.vertex_positions = np.array(positions) if positions else np.array([])
    
    def _update_highlight(self, context, event):
        """Update highlighted node using screen-space projection."""
        from bpy_extras import view3d_utils
        import numpy as np
        
        region = context.region
        rv3d = context.region_data
        
        if region is None or rv3d is None:
            return
        
        if len(self.vertex_positions) == 0:
            self.highlight_pos = None
            self.highlight_node_id = None
            return
        
        cursor_2d = np.array([event.mouse_region_x, event.mouse_region_y])
        
        min_dist = float('inf')
        nearest_idx = -1
        
        for i, pos_3d in enumerate(self.vertex_positions):
            pos_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, pos_3d)
            if pos_2d is None:
                continue
            dist = np.linalg.norm(np.array([pos_2d.x, pos_2d.y]) - cursor_2d)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        threshold_pixels = 50
        
        if nearest_idx >= 0 and min_dist < threshold_pixels:
            self.highlight_pos = tuple(self.vertex_positions[nearest_idx])
            self.highlight_node_id = self.node_ids[nearest_idx] if nearest_idx < len(self.node_ids) else None
        else:
            self.highlight_pos = None
            self.highlight_node_id = None
    
    def _draw_callback(self, context):
        """Draw highlight - green for source."""
        if self.highlight_pos is None:
            return
        
        scale = self.obj.get("osmnx_scale", 0.001)
        radius = 30 * scale
        
        color = (0.2, 1.0, 0.2, 0.9)
        _draw_highlight_circle(self.highlight_pos, radius, color)
        _draw_highlight_point(self.highlight_pos, (0.0, 0.8, 0.0, 1.0), size=14.0)
    
    def _remove_draw_handler(self, context):
        if hasattr(self, '_draw_handle') and self._draw_handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            self._draw_handle = None


class SCIGRAPHS_OT_SelectPathTarget(bpy.types.Operator):
    """Interactive selection of path target node with visual highlight."""
    bl_idname = "scigraphs.osmnx_select_path_target"
    bl_label = "Select Target Node"
    bl_description = "Click on a node to set it as path target (with highlight)"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        if event.type == 'MOUSEMOVE':
            self._update_highlight(context, event)
        
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._remove_draw_handler(context)
            
            if self.highlight_node_id is not None:
                props = context.scene.scigraphs
                props.osmnx_shortest_path_target = str(self.highlight_node_id)
                props.osmnx_selected_node_id = str(self.highlight_node_id)
                self.report({'INFO'}, f"Target set to node {self.highlight_node_id}")
            else:
                self.report({'WARNING'}, "No node under cursor")
            
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self._remove_draw_handler(context)
            self.report({'INFO'}, "Target selection cancelled")
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        obj = context.active_object
        self.obj = obj
        self.highlight_pos = None
        self.highlight_node_id = None
        
        self._build_vertex_lookup()
        
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback, (context,), 'WINDOW', 'POST_VIEW'
        )
        
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Click on a node to set as TARGET (ESC to cancel)")
        return {'RUNNING_MODAL'}
    
    def _build_vertex_lookup(self):
        """Build vertex position lookup."""
        import numpy as np
        
        mesh = self.obj.data
        nodes_str = self.obj.get("nodes_data", "")
        
        if not nodes_str:
            self.vertex_positions = np.array([])
            self.node_ids = []
            return
        
        self.node_ids = nodes_str.split(",")
        num_intersections = len(self.node_ids)
        
        positions = []
        for i in range(min(num_intersections, len(mesh.vertices))):
            v = mesh.vertices[i]
            positions.append([v.co.x, v.co.y, v.co.z])
        
        self.vertex_positions = np.array(positions) if positions else np.array([])
    
    def _update_highlight(self, context, event):
        """Update highlighted node using screen-space projection."""
        from bpy_extras import view3d_utils
        import numpy as np
        
        region = context.region
        rv3d = context.region_data
        
        if region is None or rv3d is None:
            return
        
        if len(self.vertex_positions) == 0:
            self.highlight_pos = None
            self.highlight_node_id = None
            return
        
        cursor_2d = np.array([event.mouse_region_x, event.mouse_region_y])
        
        min_dist = float('inf')
        nearest_idx = -1
        
        for i, pos_3d in enumerate(self.vertex_positions):
            pos_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, pos_3d)
            if pos_2d is None:
                continue
            dist = np.linalg.norm(np.array([pos_2d.x, pos_2d.y]) - cursor_2d)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        threshold_pixels = 50
        
        if nearest_idx >= 0 and min_dist < threshold_pixels:
            self.highlight_pos = tuple(self.vertex_positions[nearest_idx])
            self.highlight_node_id = self.node_ids[nearest_idx] if nearest_idx < len(self.node_ids) else None
        else:
            self.highlight_pos = None
            self.highlight_node_id = None
    
    def _draw_callback(self, context):
        """Draw highlight - red for target."""
        if self.highlight_pos is None:
            return
        
        scale = self.obj.get("osmnx_scale", 0.001)
        radius = 30 * scale
        
        color = (1.0, 0.2, 0.2, 0.9)
        _draw_highlight_circle(self.highlight_pos, radius, color)
        _draw_highlight_point(self.highlight_pos, (0.8, 0.0, 0.0, 1.0), size=14.0)
    
    def _remove_draw_handler(self, context):
        if hasattr(self, '_draw_handle') and self._draw_handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            self._draw_handle = None


class SCIGRAPHS_OT_UseSelectedAsSource(bpy.types.Operator):
    """Use the currently selected node as path source."""
    bl_idname = "scigraphs.osmnx_use_selected_source"
    bl_label = "Use as Source"
    bl_description = "Set the selected node as the path source"
    
    @classmethod
    def poll(cls, context):
        props = context.scene.scigraphs
        return props.osmnx_selected_node_id != ""
    
    def execute(self, context):
        props = context.scene.scigraphs
        props.osmnx_shortest_path_source = props.osmnx_selected_node_id
        self.report({'INFO'}, f"Source set to node {props.osmnx_selected_node_id}")
        return {'FINISHED'}


class SCIGRAPHS_OT_UseSelectedAsTarget(bpy.types.Operator):
    """Use the currently selected node as path target."""
    bl_idname = "scigraphs.osmnx_use_selected_target"
    bl_label = "Use as Target"
    bl_description = "Set the selected node as the path target"
    
    @classmethod
    def poll(cls, context):
        props = context.scene.scigraphs
        return props.osmnx_selected_node_id != ""
    
    def execute(self, context):
        props = context.scene.scigraphs
        props.osmnx_shortest_path_target = props.osmnx_selected_node_id
        self.report({'INFO'}, f"Target set to node {props.osmnx_selected_node_id}")
        return {'FINISHED'}


class SCIGRAPHS_OT_TruncatePolygon(bpy.types.Operator):
    """Truncate graph to polygon."""
    bl_idname = "scigraphs.osmnx_truncate_polygon"
    bl_label = "Truncate to Polygon"
    bl_description = "Remove nodes outside a polygon (from selected Blender object)"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def execute(self, context):
        from ....core.osmnx import truncate
        
        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        selected_objs = [o for o in context.selected_objects if o != obj and o.type == 'MESH']
        if not selected_objs:
            self.report({'ERROR'}, "Select a mesh object to use as polygon boundary")
            return {'CANCELLED'}
        
        poly_obj = selected_objs[0]
        
        from shapely.geometry import Polygon
        mesh = poly_obj.data
        
        if len(mesh.polygons) == 0:
            self.report({'ERROR'}, "Selected object has no faces")
            return {'CANCELLED'}
        
        verts = [(v.co.x, v.co.y) for v in mesh.vertices]
        polygon = Polygon(verts)
        
        nodes_before = G.number_of_nodes()
        
        G_truncated = truncate.truncate_graph_polygon(G, polygon)
        
        if G_truncated is None:
            self.report({'ERROR'}, "Failed to truncate graph")
            return {'CANCELLED'}
        
        from ....core import importer
        if hasattr(importer, '_osmnx_graph_cache'):
            graph_id = obj.get("osmnx_graph_id", obj.name)
            importer._osmnx_graph_cache[graph_id] = G_truncated
        
        nodes_after = G_truncated.number_of_nodes()
        self.report({'INFO'}, f"Truncated: {nodes_before} -> {nodes_after} nodes")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_TruncateDistance(bpy.types.Operator):
    """Truncate graph by distance from point."""
    bl_idname = "scigraphs.osmnx_truncate_distance"
    bl_label = "Truncate by Distance"
    bl_description = "Keep only nodes within distance from a center point"
    bl_options = {'REGISTER', 'UNDO'}
    
    center_lat: bpy.props.FloatProperty(
        name="Center Latitude",
        description="Center point latitude",
        default=0.0,
        min=-90.0,
        max=90.0,
        precision=6,
    )
    
    center_lon: bpy.props.FloatProperty(
        name="Center Longitude",
        description="Center point longitude",
        default=0.0,
        min=-180.0,
        max=180.0,
        precision=6,
    )
    
    distance: bpy.props.FloatProperty(
        name="Distance (m)",
        description="Maximum distance from center",
        default=1000.0,
        min=100.0,
        max=50000.0,
    )
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def invoke(self, context, event):
        props = context.scene.scigraphs
        self.center_lat = props.osmnx_latitude
        self.center_lon = props.osmnx_longitude
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context):
        from ....core.osmnx import truncate, spatial_queries as osmnx_spatial

        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        nodes_before = G.number_of_nodes()

        # find_nearest_node expects (G, x, y); for unprojected lat/lon graphs
        # x = longitude, y = latitude.
        nearest_node = osmnx_spatial.find_nearest_node(
            G, x=self.center_lon, y=self.center_lat
        )

        if nearest_node is None:
            self.report({'ERROR'}, "Could not find nearest node to center point")
            return {'CANCELLED'}
        
        G_truncated = truncate.truncate_graph_dist(G, nearest_node, self.distance)
        
        if G_truncated is None:
            self.report({'ERROR'}, "Failed to truncate graph")
            return {'CANCELLED'}
        
        from ....core import importer
        if hasattr(importer, '_osmnx_graph_cache'):
            graph_id = obj.get("osmnx_graph_id", obj.name)
            importer._osmnx_graph_cache[graph_id] = G_truncated
        
        nodes_after = G_truncated.number_of_nodes()
        self.report({'INFO'}, f"Truncated: {nodes_before} -> {nodes_after} nodes ({self.distance/1000:.1f}km radius)")
        
        return {'FINISHED'}


class _SelectDistanceNodeBase(bpy.types.Operator):
    """Base modal eyedropper for the node-pair distance calculator.

    Subclasses must set ``_target_prop`` to either ``"osmnx_dist_node_a"``
    or ``"osmnx_dist_node_b"``.
    """

    bl_options = {'REGISTER', 'UNDO'}

    _target_prop = ""
    _label_text = ""
    _highlight_color = (0.4, 0.8, 1.0, 0.9)
    _point_color = (0.1, 0.6, 1.0, 1.0)

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)

    def modal(self, context, event):
        context.area.tag_redraw()

        if event.type == 'MOUSEMOVE':
            self._update_highlight(context, event)

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._remove_draw_handler(context)
            if self.highlight_node_id is not None:
                props = context.scene.scigraphs
                setattr(props, self._target_prop, str(self.highlight_node_id))
                self.report({'INFO'}, f"{self._label_text} = node {self.highlight_node_id}")
            else:
                self.report({'WARNING'}, "No node under cursor")
            return {'FINISHED'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self._remove_draw_handler(context)
            self.report({'INFO'}, f"{self._label_text} selection cancelled")
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, _event):
        self.obj = context.active_object
        self.highlight_pos = None
        self.highlight_node_id = None
        self._build_vertex_lookup()
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback, (context,), 'WINDOW', 'POST_VIEW'
        )
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, f"Click on a node to set as {self._label_text} (ESC to cancel)")
        return {'RUNNING_MODAL'}

    def _build_vertex_lookup(self):
        import numpy as np
        mesh = self.obj.data
        nodes_str = self.obj.get("nodes_data", "")
        if not nodes_str:
            self.vertex_positions = np.array([])
            self.node_ids = []
            return
        self.node_ids = nodes_str.split(",")
        positions = []
        for i in range(min(len(self.node_ids), len(mesh.vertices))):
            v = mesh.vertices[i]
            positions.append([v.co.x, v.co.y, v.co.z])
        self.vertex_positions = np.array(positions) if positions else np.array([])

    def _update_highlight(self, context, event):
        from bpy_extras import view3d_utils
        import numpy as np
        region = context.region
        rv3d = context.region_data
        if region is None or rv3d is None or len(self.vertex_positions) == 0:
            self.highlight_pos = None
            self.highlight_node_id = None
            return
        cursor_2d = np.array([event.mouse_region_x, event.mouse_region_y])
        min_dist = float('inf')
        nearest_idx = -1
        for i, pos_3d in enumerate(self.vertex_positions):
            pos_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, pos_3d)
            if pos_2d is None:
                continue
            dist = np.linalg.norm(np.array([pos_2d.x, pos_2d.y]) - cursor_2d)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        if nearest_idx >= 0 and min_dist < 50:
            self.highlight_pos = tuple(self.vertex_positions[nearest_idx])
            self.highlight_node_id = (
                self.node_ids[nearest_idx] if nearest_idx < len(self.node_ids) else None
            )
        else:
            self.highlight_pos = None
            self.highlight_node_id = None

    def _draw_callback(self, _context):
        if self.highlight_pos is None:
            return
        scale = self.obj.get("osmnx_scale", 0.001)
        _draw_highlight_circle(self.highlight_pos, 30 * scale, self._highlight_color)
        _draw_highlight_point(self.highlight_pos, self._point_color, size=14.0)

    def _remove_draw_handler(self, _context):
        if hasattr(self, '_draw_handle') and self._draw_handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            self._draw_handle = None


class SCIGRAPHS_OT_SelectDistanceNodeA(_SelectDistanceNodeBase):
    """Pick the first node for the node-pair distance calculator."""
    bl_idname = "scigraphs.osmnx_select_distance_node_a"
    bl_label = "Pick Node A (Distance)"
    bl_description = "Click a node to set it as the first endpoint of the distance calculator"

    _target_prop = "osmnx_dist_node_a"
    _label_text = "Node A"
    _highlight_color = (0.2, 1.0, 0.4, 0.9)
    _point_color = (0.0, 0.8, 0.2, 1.0)


class SCIGRAPHS_OT_SelectDistanceNodeB(_SelectDistanceNodeBase):
    """Pick the second node for the node-pair distance calculator."""
    bl_idname = "scigraphs.osmnx_select_distance_node_b"
    bl_label = "Pick Node B (Distance)"
    bl_description = "Click a node to set it as the second endpoint of the distance calculator"

    _target_prop = "osmnx_dist_node_b"
    _label_text = "Node B"
    _highlight_color = (1.0, 0.6, 0.1, 0.9)
    _point_color = (0.9, 0.4, 0.0, 1.0)


class SCIGRAPHS_OT_CalcNodePairDistance(bpy.types.Operator):
    """Compute several distance metrics between two graph nodes.

    Reports straight-line distance (great-circle for unprojected graphs,
    Euclidean for projected ones), shortest network distance, the resulting
    circuity, and (when ``travel_time`` is available) shortest travel time.
    Mirrors the typical OSMnx notebook pattern of comparing route length to
    crow-flies distance.
    """

    bl_idname = "scigraphs.osmnx_calc_node_pair_distance"
    bl_label = "Compute Node-Pair Distances"
    bl_description = (
        "Compare straight-line, network and travel-time distances between "
        "the two picked nodes (uses the graph's projection to choose method)"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or not obj.get("is_osmnx", False):
            return False
        props = context.scene.scigraphs
        return bool(props.osmnx_dist_node_a) and bool(props.osmnx_dist_node_b)

    def execute(self, context):
        from ....core.osmnx import distance as osmnx_distance
        from ....core.osmnx import routing as osmnx_routing

        obj = context.active_object
        props = context.scene.scigraphs

        G = _get_osmnx_graph(obj)
        G_unprojected = _get_unprojected_graph(obj)
        if G is None and G_unprojected is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}

        try:
            node_a = int(props.osmnx_dist_node_a)
            node_b = int(props.osmnx_dist_node_b)
        except ValueError:
            self.report({'ERROR'}, "Invalid node IDs")
            return {'CANCELLED'}

        if node_a == node_b:
            self.report({'WARNING'}, "Node A and Node B are the same")

        # Pick a graph that contains both nodes.
        G_active = None
        if G is not None and node_a in G.nodes and node_b in G.nodes:
            G_active = G
        elif G_unprojected is not None and node_a in G_unprojected.nodes and node_b in G_unprojected.nodes:
            G_active = G_unprojected
        if G_active is None:
            self.report({'ERROR'}, "Both nodes must belong to the same cached graph")
            return {'CANCELLED'}

        is_projected = bool(obj.get("osmnx_projected", False)) and G_active is G
        crs_label = obj.get("osmnx_crs", "EPSG:4326")

        ax = G_active.nodes[node_a].get("x")
        ay = G_active.nodes[node_a].get("y")
        bx = G_active.nodes[node_b].get("x")
        by = G_active.nodes[node_b].get("y")

        if None in (ax, ay, bx, by):
            self.report({'ERROR'}, "Node coordinates missing on the graph")
            return {'CANCELLED'}

        # Straight-line (crow-flies) distance.
        if is_projected:
            straight_m = osmnx_distance.euclidean(ay, ax, by, bx)
            straight_method = "Euclidean (projected)"
        else:
            straight_m = osmnx_distance.great_circle(ay, ax, by, bx)
            straight_method = "Great-circle (haversine)"

        if straight_m is None:
            self.report({'ERROR'}, "Failed to compute straight-line distance")
            return {'CANCELLED'}

        # Shortest network distance (length-weighted).
        net_result = osmnx_routing.calculate_shortest_path(
            G_active, node_a, node_b, weight='length'
        )
        if net_result is None or 'error' in net_result:
            err = (net_result or {}).get('error', "Network path failed")
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        network_m = float(net_result['distance_m'])
        circuity = network_m / straight_m if straight_m > 0 else float('nan')

        # Optional travel-time on the same path.
        travel_min = None
        if 'travel_time_minutes' in net_result:
            travel_min = float(net_result['travel_time_minutes'])

        # Persist the results so the panel can render them.
        obj["osmnx_pair_node_a"] = str(node_a)
        obj["osmnx_pair_node_b"] = str(node_b)
        obj["osmnx_pair_straight_m"] = float(straight_m)
        obj["osmnx_pair_straight_method"] = straight_method
        obj["osmnx_pair_network_m"] = network_m
        obj["osmnx_pair_circuity"] = float(circuity)
        obj["osmnx_pair_num_nodes"] = int(net_result['num_nodes'])
        obj["osmnx_pair_num_edges"] = int(net_result['num_edges'])
        obj["osmnx_pair_crs"] = crs_label
        if travel_min is not None:
            obj["osmnx_pair_travel_min"] = travel_min
        else:
            obj.pop("osmnx_pair_travel_min", None)

        msg = (
            f"Straight: {straight_m:.1f} m · Network: {network_m:.1f} m · "
            f"Circuity: {circuity:.3f}"
        )
        if travel_min is not None:
            msg += f" · {travel_min:.1f} min"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


classes = [
    SCIGRAPHS_OT_SelectNearestNode,
    SCIGRAPHS_OT_SelectNearestEdge,
    SCIGRAPHS_OT_CalculateShortestPath,
    SCIGRAPHS_OT_SelectPathSource,
    SCIGRAPHS_OT_SelectPathTarget,
    SCIGRAPHS_OT_UseSelectedAsSource,
    SCIGRAPHS_OT_UseSelectedAsTarget,
    SCIGRAPHS_OT_TruncatePolygon,
    SCIGRAPHS_OT_TruncateDistance,
    SCIGRAPHS_OT_SelectDistanceNodeA,
    SCIGRAPHS_OT_SelectDistanceNodeB,
    SCIGRAPHS_OT_CalcNodePairDistance,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

