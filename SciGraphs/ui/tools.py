# SciGraphs Custom Tools
# Interactive tools for graph editing in the viewport

import bpy
import numpy as np
from bpy_extras import view3d_utils
import gpu
from gpu_extras.batch import batch_for_shader
import blf
from mathutils import Vector

from ..core import geometry


_PATH_MARKER_HANDLE = None


def _get_nearest_node_index(context, event, obj):
    """Find the nearest node to mouse position."""
    if not obj or "node_positions" not in obj:
        return None, None
    
    # Get mouse coordinates
    coord = (event.mouse_region_x, event.mouse_region_y)
    
    # Get view ray
    region = context.region
    rv3d = context.region_data
    
    # Get node positions
    pos_flat = obj.get("node_positions", [])
    if not pos_flat:
        return None, None
    
    positions = np.array(pos_flat).reshape(-1, 3)
    
    # Find nearest node by projecting to screen
    min_dist = float('inf')
    nearest_idx = None
    nearest_pos = None
    
    for i, pos in enumerate(positions):
        world_pos = obj.matrix_world @ Vector(pos)
        screen_pos = view3d_utils.location_3d_to_region_2d(region, rv3d, world_pos)
        
        if screen_pos:
            dist = (screen_pos[0] - coord[0])**2 + (screen_pos[1] - coord[1])**2
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
                nearest_pos = world_pos
    
    # Threshold: 30 pixels
    if min_dist < 900:
        return nearest_idx, nearest_pos
    
    return None, None


def _path_node_world_position(obj, node_index):
    """Return a graph node world position from the stored node_positions property."""
    if node_index is None or not obj or "node_positions" not in obj:
        return None

    pos_flat = obj.get("node_positions", [])
    if not pos_flat:
        return None

    try:
        positions = np.array(pos_flat, dtype=float).reshape(-1, 3)
    except ValueError:
        return None

    if node_index < 0 or node_index >= len(positions):
        return None

    return obj.matrix_world @ Vector(positions[node_index])


def _parse_node_index(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _draw_path_selection_markers():
    """Draw temporary source/target markers while the path picker is active."""
    context = bpy.context
    if context.area is None or context.area.type != 'VIEW_3D':
        return

    obj = context.active_object
    if not obj or "num_nodes" not in obj:
        return

    props = context.scene.scigraphs
    source_pos = _path_node_world_position(obj, _parse_node_index(props.pathfinding_source))
    target_pos = _path_node_world_position(obj, _parse_node_index(props.pathfinding_target))

    points = []
    colors = []
    if source_pos is not None:
        points.append(source_pos[:])
        colors.append((0.1, 1.0, 0.25, 1.0))
    if target_pos is not None:
        points.append(target_pos[:])
        colors.append((1.0, 0.15, 0.1, 1.0))

    if not points:
        return

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')

    if source_pos is not None and target_pos is not None and source_pos != target_pos:
        line_batch = batch_for_shader(shader, 'LINES', {"pos": [source_pos[:], target_pos[:]]})
        gpu.state.line_width_set(3.0)
        shader.bind()
        shader.uniform_float("color", (1.0, 0.85, 0.0, 0.75))
        line_batch.draw(shader)
        gpu.state.line_width_set(1.0)

    for point, color in zip(points, colors):
        point_batch = batch_for_shader(shader, 'POINTS', {"pos": [point]})
        gpu.state.point_size_set(24.0)
        shader.bind()
        shader.uniform_float("color", color)
        point_batch.draw(shader)

    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')


def enable_path_selection_markers():
    """Enable temporary viewport markers for selected shortest-path nodes."""
    global _PATH_MARKER_HANDLE
    if _PATH_MARKER_HANDLE is None:
        _PATH_MARKER_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _draw_path_selection_markers, (), 'WINDOW', 'POST_VIEW'
        )


def disable_path_selection_markers():
    """Disable temporary viewport markers for selected shortest-path nodes."""
    global _PATH_MARKER_HANDLE
    if _PATH_MARKER_HANDLE is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_PATH_MARKER_HANDLE, 'WINDOW')
        _PATH_MARKER_HANDLE = None


class SCIGRAPHS_OT_select_node_tool(bpy.types.Operator):
    """Click to select the nearest graph node."""
    bl_idname = "scigraphs.select_node_tool"
    bl_label = "Select Node"
    bl_description = "Click on a node to select it"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handle = None
    _highlight_pos = None
    _highlight_idx = None
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and "num_nodes" in obj
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        obj = context.active_object
        if not obj or "num_nodes" not in obj:
            self._cleanup(context)
            return {'CANCELLED'}
        
        if event.type == 'MOUSEMOVE':
            # Highlight nearest node
            idx, pos = _get_nearest_node_index(context, event, obj)
            self._highlight_idx = idx
            self._highlight_pos = pos
        
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            idx, pos = _get_nearest_node_index(context, event, obj)
            if idx is not None:
                # Store selection
                obj["selected_node"] = idx
                self.report({'INFO'}, f"Selected node {idx}")
            self._cleanup(context)
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self._cleanup(context)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, _event):
        if context.area.type != 'VIEW_3D':
            return {'CANCELLED'}
        
        # Add draw handler
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback, (context,), 'WINDOW', 'POST_VIEW'
        )
        
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    
    def _draw_callback(self, _context):
        if self._highlight_pos:
            # Draw highlight sphere
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            gpu.state.blend_set('ALPHA')
            gpu.state.depth_test_set('LESS_EQUAL')
            
            # Draw a point marker
            batch = batch_for_shader(shader, 'POINTS', {"pos": [self._highlight_pos[:]]})
            gpu.state.point_size_set(15.0)
            shader.bind()
            shader.uniform_float("color", (1.0, 0.5, 0.0, 1.0))
            batch.draw(shader)
            
            gpu.state.blend_set('NONE')
            gpu.state.depth_test_set('NONE')
    
    def _cleanup(self, context):
        if self._draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            self._draw_handle = None
        context.area.tag_redraw()


class SCIGRAPHS_OT_pick_path_node(bpy.types.Operator):
    """Pick a graph node in the viewport and assign it to shortest-path inputs."""
    bl_idname = "scigraphs.pick_path_node"
    bl_label = "Pick Path Node"
    bl_description = "Click a graph node in the viewport to use it for shortest path"
    bl_options = {'REGISTER', 'UNDO'}

    target: bpy.props.EnumProperty(
        name="Target",
        items=[
            ('SOURCE', "Source", "Set shortest path source node"),
            ('TARGET', "Target", "Set shortest path target node"),
        ],
        default='SOURCE',
        options={'SKIP_SAVE'},
    )

    _draw_handle = None
    _highlight_idx = None
    _highlight_pos = None

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and "num_nodes" in obj

    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Node picking must be started from a 3D View")
            return {'CANCELLED'}

        self._highlight_idx = None
        self._highlight_pos = None
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback, (context,), 'WINDOW', 'POST_VIEW'
        )
        enable_path_selection_markers()
        context.window_manager.modal_handler_add(self)

        label = "source" if self.target == 'SOURCE' else "target"
        self.report({'INFO'}, f"Click a graph node to set shortest path {label}. ESC to cancel.")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        context.area.tag_redraw()

        obj = context.active_object
        if not obj or "num_nodes" not in obj:
            self._cleanup(context)
            return {'CANCELLED'}

        if event.type == 'MOUSEMOVE':
            self._highlight_idx, self._highlight_pos = _get_nearest_node_index(context, event, obj)

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            idx, _pos = _get_nearest_node_index(context, event, obj)
            if idx is None:
                self.report({'WARNING'}, "No graph node near the click")
                return {'RUNNING_MODAL'}

            props = context.scene.scigraphs
            if self.target == 'SOURCE':
                props.pathfinding_source = str(idx)
                label = "source"
            else:
                props.pathfinding_target = str(idx)
                label = "target"

            self.report({'INFO'}, f"Shortest path {label}: node {idx}")
            self._cleanup(context)
            return {'FINISHED'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self._cleanup(context)
            self.report({'INFO'}, "Node picking cancelled")
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def _draw_callback(self, context):
        if self._highlight_pos is None:
            return

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('LESS_EQUAL')

        color = (0.0, 1.0, 0.25, 1.0) if self.target == 'SOURCE' else (1.0, 0.15, 0.1, 1.0)
        batch = batch_for_shader(shader, 'POINTS', {"pos": [self._highlight_pos[:]]})
        gpu.state.point_size_set(18.0)
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)

        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')

    def _cleanup(self, context):
        if self._draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            self._draw_handle = None
        disable_path_selection_markers()
        context.area.tag_redraw()


class SCIGRAPHS_OT_path_tool(bpy.types.Operator):
    """Select source and target for shortest path visualization."""
    bl_idname = "scigraphs.path_tool"
    bl_label = "Shortest Path Tool"
    bl_description = "Click two nodes to visualize shortest path"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handle = None
    _source_idx = None
    _source_pos = None
    _highlight_idx = None
    _highlight_pos = None
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and "num_nodes" in obj
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        obj = context.active_object
        if not obj or "num_nodes" not in obj:
            self._cleanup(context)
            return {'CANCELLED'}
        
        if event.type == 'MOUSEMOVE':
            idx, pos = _get_nearest_node_index(context, event, obj)
            self._highlight_idx = idx
            self._highlight_pos = pos
        
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            idx, pos = _get_nearest_node_index(context, event, obj)
            if idx is not None:
                if self._source_idx is None:
                    # First click: set source
                    self._source_idx = idx
                    self._source_pos = pos
                    self.report({'INFO'}, f"Source: node {idx}. Click target node.")
                else:
                    # Second click: set target and compute path
                    props = context.scene.scigraphs
                    props.pathfinding_source = str(self._source_idx)
                    props.pathfinding_target = str(idx)
                    obj["path_source"] = self._source_idx
                    obj["path_target"] = idx
                    self.report({'INFO'}, f"Path: {self._source_idx} -> {idx}")
                    
                    # Try to compute and visualize path
                    try:
                        bpy.ops.scigraphs.find_shortest_path()
                    except Exception:
                        pass
                    
                    self._cleanup(context)
                    return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self._cleanup(context)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            return {'CANCELLED'}
        
        self._source_idx = None
        self._source_pos = None
        
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback, (context,), 'WINDOW', 'POST_VIEW'
        )
        enable_path_selection_markers()
        
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Click source node...")
        return {'RUNNING_MODAL'}
    
    def _draw_callback(self, context):
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        
        # Draw source if set
        if self._source_pos:
            batch = batch_for_shader(shader, 'POINTS', {"pos": [self._source_pos[:]]})
            gpu.state.point_size_set(20.0)
            shader.bind()
            shader.uniform_float("color", (0.0, 1.0, 0.0, 1.0))
            batch.draw(shader)
        
        # Draw highlight
        if self._highlight_pos:
            color = (1.0, 0.0, 0.0, 1.0) if self._source_idx is not None else (0.0, 1.0, 0.0, 1.0)
            batch = batch_for_shader(shader, 'POINTS', {"pos": [self._highlight_pos[:]]})
            gpu.state.point_size_set(15.0)
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)
            
            # Draw line from source to highlight
            if self._source_pos:
                vertices = [self._source_pos[:], self._highlight_pos[:]]
                batch = batch_for_shader(shader, 'LINES', {"pos": vertices})
                shader.bind()
                shader.uniform_float("color", (1.0, 1.0, 0.0, 0.5))
                batch.draw(shader)
        
        gpu.state.blend_set('NONE')
    
    def _cleanup(self, context):
        if self._draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            self._draw_handle = None
        disable_path_selection_markers()
        context.area.tag_redraw()


class SCIGRAPHS_OT_move_node_tool(bpy.types.Operator):
    """Click and drag to move a graph node."""
    bl_idname = "scigraphs.move_node_tool"
    bl_label = "Move Node"
    bl_description = "Click and drag a node to move it"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handle = None
    _dragging = False
    _drag_idx = None
    _highlight_idx = None
    _highlight_pos = None
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and "num_nodes" in obj
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        obj = context.active_object
        if not obj or "num_nodes" not in obj:
            self._cleanup(context)
            return {'CANCELLED'}
        
        region = context.region
        rv3d = context.region_data
        
        if event.type == 'MOUSEMOVE':
            if self._dragging and self._drag_idx is not None:
                # Move node to new position
                coord = (event.mouse_region_x, event.mouse_region_y)
                
                # Get depth from current node position
                pos_flat = obj.get("node_positions", [])
                positions = np.array(pos_flat).reshape(-1, 3)
                old_pos = positions[self._drag_idx]
                world_old = obj.matrix_world @ Vector(old_pos)
                
                # Project new 2D position to 3D at same depth
                depth_vec = rv3d.view_matrix @ world_old.to_4d()
                new_world = view3d_utils.region_2d_to_location_3d(
                    region, rv3d, coord, Vector((0, 0, -depth_vec.z))
                )
                
                # Transform back to local space
                new_local = obj.matrix_world.inverted() @ new_world
                
                # Update position
                positions[self._drag_idx] = [new_local.x, new_local.y, new_local.z]
                obj["node_positions"] = positions.flatten().tolist()
                
                # Update mesh
                geometry.update_node_positions_from_property(obj)
                geometry.rebuild_edges(obj)
                
                self._highlight_pos = new_world
            else:
                idx, pos = _get_nearest_node_index(context, event, obj)
                self._highlight_idx = idx
                self._highlight_pos = pos
        
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                idx, pos = _get_nearest_node_index(context, event, obj)
                if idx is not None:
                    self._dragging = True
                    self._drag_idx = idx
                    self.report({'INFO'}, f"Dragging node {idx}")
            elif event.value == 'RELEASE':
                if self._dragging:
                    self.report({'INFO'}, f"Node {self._drag_idx} moved")
                    self._dragging = False
                    self._drag_idx = None
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self._cleanup(context)
            return {'FINISHED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            return {'CANCELLED'}

        # Show node handles automatically while the move tool is active.
        try:
            from . import gizmos
            gizmos.enable_gizmos()
        except Exception:
            pass
        
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback, (context,), 'WINDOW', 'POST_VIEW'
        )
        
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Click and drag nodes to move them. Right-click to finish.")
        return {'RUNNING_MODAL'}
    
    def _draw_callback(self, context):
        if self._highlight_pos:
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            gpu.state.blend_set('ALPHA')
            
            color = (0.0, 0.8, 1.0, 1.0) if self._dragging else (1.0, 0.5, 0.0, 1.0)
            size = 20.0 if self._dragging else 15.0
            
            batch = batch_for_shader(shader, 'POINTS', {"pos": [self._highlight_pos[:]]})
            gpu.state.point_size_set(size)
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)
            
            gpu.state.blend_set('NONE')
    
    def _cleanup(self, context):
        if self._draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            self._draw_handle = None
        context.area.tag_redraw()


class SCIGRAPHS_OT_lasso_select_tool(bpy.types.Operator):
    """Draw a lasso to select multiple nodes."""
    bl_idname = "scigraphs.lasso_select_tool"
    bl_label = "Lasso Select"
    bl_description = "Draw a lasso to select multiple nodes"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handle = None
    _lasso_points = []
    _drawing = False
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and "num_nodes" in obj
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        obj = context.active_object
        if not obj or "num_nodes" not in obj:
            self._cleanup(context)
            return {'CANCELLED'}
        
        if event.type == 'MOUSEMOVE' and self._drawing:
            self._lasso_points.append((event.mouse_region_x, event.mouse_region_y))
        
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self._drawing = True
                self._lasso_points = [(event.mouse_region_x, event.mouse_region_y)]
            elif event.value == 'RELEASE':
                if self._drawing and len(self._lasso_points) > 2:
                    selected = self._select_nodes_in_lasso(context, obj)
                    self.report({'INFO'}, f"Selected {len(selected)} nodes")
                self._cleanup(context)
                return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self._cleanup(context)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def _select_nodes_in_lasso(self, context, obj):
        """Select nodes inside the lasso polygon."""
        from mathutils.geometry import intersect_point_tri_2d
        
        region = context.region
        rv3d = context.region_data
        
        pos_flat = obj.get("node_positions", [])
        if not pos_flat:
            return []
        
        positions = np.array(pos_flat).reshape(-1, 3)
        selected = []
        
        # Simple point-in-polygon test
        lasso = self._lasso_points
        n = len(lasso)
        
        for i, pos in enumerate(positions):
            world_pos = obj.matrix_world @ Vector(pos)
            screen_pos = view3d_utils.location_3d_to_region_2d(region, rv3d, world_pos)
            
            if screen_pos:
                # Ray casting point-in-polygon
                inside = False
                j = n - 1
                for k in range(n):
                    if ((lasso[k][1] > screen_pos[1]) != (lasso[j][1] > screen_pos[1]) and
                        screen_pos[0] < (lasso[j][0] - lasso[k][0]) * (screen_pos[1] - lasso[k][1]) / (lasso[j][1] - lasso[k][1]) + lasso[k][0]):
                        inside = not inside
                    j = k
                
                if inside:
                    selected.append(i)
        
        # Store selection
        obj["selected_nodes"] = selected
        return selected
    
    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            return {'CANCELLED'}
        
        self._lasso_points = []
        self._drawing = False
        
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback, (context,), 'WINDOW', 'POST_PIXEL'
        )
        
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Draw lasso around nodes to select")
        return {'RUNNING_MODAL'}
    
    def _draw_callback(self, context):
        if len(self._lasso_points) > 1:
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            gpu.state.blend_set('ALPHA')
            gpu.state.line_width_set(2.0)
            
            # Draw lasso line
            vertices = [(p[0], p[1]) for p in self._lasso_points]
            if len(vertices) > 1:
                batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": vertices})
                shader.bind()
                shader.uniform_float("color", (1.0, 1.0, 0.0, 0.8))
                batch.draw(shader)
            
            gpu.state.blend_set('NONE')
            gpu.state.line_width_set(1.0)
    
    def _cleanup(self, context):
        if self._draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            self._draw_handle = None
        self._lasso_points = []
        context.area.tag_redraw()


_TOOL_CLASSES = [
    SCIGRAPHS_OT_select_node_tool,
    SCIGRAPHS_OT_pick_path_node,
    SCIGRAPHS_OT_path_tool,
    SCIGRAPHS_OT_move_node_tool,
    SCIGRAPHS_OT_lasso_select_tool,
]


def register():
    for cls in _TOOL_CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    disable_path_selection_markers()
    for cls in reversed(_TOOL_CLASSES):
        bpy.utils.unregister_class(cls)
