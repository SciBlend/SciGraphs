# SciGraphs Modal Visual Operators
# Interactive operators with real-time visual feedback

import bpy
import numpy as np
import gpu
from gpu_extras.batch import batch_for_shader
import blf
from bpy_extras import view3d_utils
from mathutils import Vector


class SCIGRAPHS_OT_highlight_communities(bpy.types.Operator):
    """Interactively highlight communities on mouse hover."""
    bl_idname = "scigraphs.highlight_communities"
    bl_label = "Highlight Communities"
    bl_description = "Hover over nodes to highlight their community"
    bl_options = {'REGISTER'}
    
    _draw_handle = None
    _current_cluster = None
    _cluster_nodes = {}
    _node_positions = None
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and "num_nodes" in obj and "cluster_id" in obj.data.attributes
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        obj = context.active_object
        if not obj or "num_nodes" not in obj:
            self._cleanup(context)
            return {'CANCELLED'}
        
        if event.type == 'MOUSEMOVE':
            self._update_hover(context, event, obj)
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self._cleanup(context)
            return {'FINISHED'}
        
        return {'RUNNING_MODAL'}
    
    def _update_hover(self, context, event, obj):
        """Find which cluster the mouse is hovering over."""
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        
        if self._node_positions is None:
            return
        
        # Find nearest node
        min_dist = float('inf')
        nearest_cluster = None
        
        for i, pos in enumerate(self._node_positions):
            world_pos = obj.matrix_world @ Vector(pos)
            screen_pos = view3d_utils.location_3d_to_region_2d(region, rv3d, world_pos)
            
            if screen_pos:
                dist = (screen_pos[0] - coord[0])**2 + (screen_pos[1] - coord[1])**2
                if dist < min_dist and dist < 2500:  # 50 pixel threshold
                    min_dist = dist
                    if i in self._cluster_nodes:
                        for cluster_id, nodes in self._cluster_nodes.items():
                            if i in nodes:
                                nearest_cluster = cluster_id
                                break
        
        self._current_cluster = nearest_cluster
    
    def invoke(self, context, event):
        obj = context.active_object
        
        # Build cluster map
        mesh = obj.data
        cluster_attr = mesh.attributes.get("cluster_id")
        if not cluster_attr:
            self.report({'ERROR'}, "No cluster_id attribute found. Run clustering first.")
            return {'CANCELLED'}
        
        # Get cluster assignments
        cluster_ids = [int(cluster_attr.data[i].value) for i in range(len(cluster_attr.data))]
        
        # Build cluster -> node mapping
        self._cluster_nodes = {}
        for i, cid in enumerate(cluster_ids):
            if cid not in self._cluster_nodes:
                self._cluster_nodes[cid] = []
            self._cluster_nodes[cid].append(i)
        
        # Get node positions
        pos_flat = obj.get("node_positions", [])
        if pos_flat:
            self._node_positions = np.array(pos_flat).reshape(-1, 3)
        else:
            # Fall back to vertex positions
            self._node_positions = np.array([v.co[:] for v in mesh.vertices[:obj.get("num_nodes", 0)]])
        
        # Add draw handler
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback, (context,), 'WINDOW', 'POST_VIEW'
        )
        
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, f"Hover to highlight {len(self._cluster_nodes)} communities. Right-click to exit.")
        return {'RUNNING_MODAL'}
    
    def _draw_callback(self, context):
        if self._current_cluster is None or self._current_cluster not in self._cluster_nodes:
            return
        
        obj = context.active_object
        if not obj:
            return
        
        nodes = self._cluster_nodes[self._current_cluster]
        
        # Draw highlighted nodes
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('ALWAYS')
        
        # Generate cluster color
        hue = (self._current_cluster * 0.618) % 1.0  # Golden ratio for color distribution
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
        color = (*rgb, 0.9)
        
        # Draw points for cluster nodes
        positions = []
        for i in nodes:
            if i < len(self._node_positions):
                world_pos = obj.matrix_world @ Vector(self._node_positions[i])
                positions.append(world_pos[:])
        
        if positions:
            batch = batch_for_shader(shader, 'POINTS', {"pos": positions})
            gpu.state.point_size_set(12.0)
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)
        
        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('NONE')
        
        # Draw info text
        self._draw_cluster_info(context, self._current_cluster, len(nodes))
    
    def _draw_cluster_info(self, context, cluster_id, node_count):
        """Draw cluster info text."""
        font_id = 0
        blf.size(font_id, 16)
        blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
        blf.position(font_id, 20, 50, 0)
        blf.draw(font_id, f"Cluster {cluster_id}: {node_count} nodes")
    
    def _cleanup(self, context):
        if self._draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            self._draw_handle = None
        self._cluster_nodes = {}
        self._node_positions = None
        self._current_cluster = None
        context.area.tag_redraw()


class SCIGRAPHS_OT_visualize_centrality_interactive(bpy.types.Operator):
    """Interactive centrality visualization with threshold slider."""
    bl_idname = "scigraphs.visualize_centrality_interactive"
    bl_label = "Interactive Centrality View"
    bl_description = "Adjust centrality threshold interactively"
    bl_options = {'REGISTER'}
    
    _draw_handle = None
    _threshold = 0.5
    _centrality_values = None
    _node_positions = None
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or "num_nodes" not in obj:
            return False
        # Check for any centrality attribute
        mesh = obj.data
        return any(a.name.startswith("centrality_") for a in mesh.attributes)
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        obj = context.active_object
        if not obj:
            self._cleanup(context)
            return {'CANCELLED'}
        
        if event.type == 'WHEELUPMOUSE':
            self._threshold = min(1.0, self._threshold + 0.05)
        elif event.type == 'WHEELDOWNMOUSE':
            self._threshold = max(0.0, self._threshold - 0.05)
        elif event.type == 'MOUSEMOVE' and event.shift:
            # Fine control with shift+mouse
            delta = event.mouse_x - event.mouse_prev_x
            self._threshold = max(0.0, min(1.0, self._threshold + delta * 0.001))
        elif event.type in {'RIGHTMOUSE', 'ESC', 'RET'}:
            self._cleanup(context)
            return {'FINISHED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        obj = context.active_object
        mesh = obj.data
        
        # Find centrality attribute
        centrality_attr = None
        for attr in mesh.attributes:
            if attr.name.startswith("centrality_"):
                centrality_attr = attr
                break
        
        if not centrality_attr:
            self.report({'ERROR'}, "No centrality attribute found")
            return {'CANCELLED'}
        
        # Get values
        num_nodes = obj.get("num_nodes", len(mesh.vertices))
        self._centrality_values = np.array([
            centrality_attr.data[i].value for i in range(min(len(centrality_attr.data), num_nodes))
        ])
        
        # Normalize to 0-1
        vmin, vmax = self._centrality_values.min(), self._centrality_values.max()
        if vmax > vmin:
            self._centrality_values = (self._centrality_values - vmin) / (vmax - vmin)
        
        # Get positions
        pos_flat = obj.get("node_positions", [])
        if pos_flat:
            self._node_positions = np.array(pos_flat).reshape(-1, 3)
        else:
            self._node_positions = np.array([v.co[:] for v in mesh.vertices[:num_nodes]])
        
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback, (context,), 'WINDOW', 'POST_VIEW'
        )
        
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Scroll to adjust threshold. Enter/Right-click to finish.")
        return {'RUNNING_MODAL'}
    
    def _draw_callback(self, context):
        obj = context.active_object
        if not obj or self._centrality_values is None:
            return
        
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('ALWAYS')
        
        # Draw nodes above threshold in hot color, below in cool color
        hot_positions = []
        cold_positions = []
        
        for i, val in enumerate(self._centrality_values):
            if i < len(self._node_positions):
                world_pos = obj.matrix_world @ Vector(self._node_positions[i])
                if val >= self._threshold:
                    hot_positions.append(world_pos[:])
                else:
                    cold_positions.append(world_pos[:])
        
        # Draw cold nodes (small, blue)
        if cold_positions:
            batch = batch_for_shader(shader, 'POINTS', {"pos": cold_positions})
            gpu.state.point_size_set(4.0)
            shader.bind()
            shader.uniform_float("color", (0.2, 0.4, 0.8, 0.4))
            batch.draw(shader)
        
        # Draw hot nodes (large, red/orange)
        if hot_positions:
            batch = batch_for_shader(shader, 'POINTS', {"pos": hot_positions})
            gpu.state.point_size_set(12.0)
            shader.bind()
            shader.uniform_float("color", (1.0, 0.3, 0.1, 1.0))
            batch.draw(shader)
        
        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('NONE')
        
        # Draw threshold info
        self._draw_info(context)
    
    def _draw_info(self, context):
        font_id = 0
        blf.size(font_id, 16)
        blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
        blf.position(font_id, 20, 50, 0)
        
        above = np.sum(self._centrality_values >= self._threshold)
        total = len(self._centrality_values)
        blf.draw(font_id, f"Threshold: {self._threshold:.2f}  |  Above: {above}/{total} ({100*above/total:.1f}%)")
    
    def _cleanup(self, context):
        if self._draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            self._draw_handle = None
        self._centrality_values = None
        self._node_positions = None
        context.area.tag_redraw()


class SCIGRAPHS_OT_preview_layout(bpy.types.Operator):
    """Preview layout algorithm in real-time before applying."""
    bl_idname = "scigraphs.preview_layout"
    bl_label = "Preview Layout"
    bl_description = "Preview layout changes in real-time"
    bl_options = {'REGISTER'}
    
    _draw_handle = None
    _original_positions = None
    _preview_positions = None
    _iteration = 0
    _timer = None
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and "num_nodes" in obj
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            self._iterate_layout(context)
            context.area.tag_redraw()
        
        elif event.type == 'RET':
            # Apply the layout
            obj = context.active_object
            if obj and self._preview_positions is not None:
                obj["node_positions"] = self._preview_positions.flatten().tolist()
                from ..core import geometry
                geometry.update_node_positions_from_property(obj)
                geometry.rebuild_edges(obj)
                self.report({'INFO'}, "Layout applied")
            self._cleanup(context)
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            # Cancel - restore original
            obj = context.active_object
            if obj and self._original_positions is not None:
                obj["node_positions"] = self._original_positions.flatten().tolist()
            self._cleanup(context)
            self.report({'INFO'}, "Layout preview cancelled")
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def _iterate_layout(self, context):
        """Perform one layout iteration."""
        from ..core import layout
        
        obj = context.active_object
        if not obj:
            return
        
        props = context.scene.scigraphs
        
        # Simple force-directed iteration
        if self._preview_positions is not None:
            # Temporarily set positions for iteration
            obj["node_positions"] = self._preview_positions.flatten().tolist()
            
            success = layout.apply_graph_layout(
                obj,
                algorithm=props.layout_algorithm,
                iterations=1,
                scale=props.layout_scale,
                props=props
            )
            
            if success:
                pos_flat = obj.get("node_positions", [])
                if pos_flat:
                    self._preview_positions = np.array(pos_flat).reshape(-1, 3)
                self._iteration += 1
    
    def invoke(self, context, event):
        obj = context.active_object
        
        # Store original positions
        pos_flat = obj.get("node_positions", [])
        if pos_flat:
            self._original_positions = np.array(pos_flat).reshape(-1, 3).copy()
            self._preview_positions = self._original_positions.copy()
        else:
            self.report({'ERROR'}, "No node positions found")
            return {'CANCELLED'}
        
        # Add draw handler
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback, (context,), 'WINDOW', 'POST_VIEW'
        )
        
        # Add timer for animation
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Previewing layout... Enter to apply, Esc to cancel")
        return {'RUNNING_MODAL'}
    
    def _draw_callback(self, context):
        """Draw iteration count."""
        font_id = 0
        blf.size(font_id, 16)
        blf.color(font_id, 1.0, 1.0, 0.5, 1.0)
        blf.position(font_id, 20, 70, 0)
        blf.draw(font_id, f"Layout Preview - Iteration: {self._iteration}")
        blf.position(font_id, 20, 50, 0)
        blf.draw(font_id, "Enter: Apply | Esc: Cancel")
    
    def _cleanup(self, context):
        if self._draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            self._draw_handle = None
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        self._original_positions = None
        self._preview_positions = None
        context.area.tag_redraw()


_CLASSES = [
    SCIGRAPHS_OT_highlight_communities,
    SCIGRAPHS_OT_visualize_centrality_interactive,
    SCIGRAPHS_OT_preview_layout,
]


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
