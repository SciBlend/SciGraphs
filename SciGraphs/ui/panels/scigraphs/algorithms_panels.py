# Graph algorithms panels (traversal, pathfinding, spanning trees, etc.)

import bpy

class SCIGRAPHS_PT_algorithms(bpy.types.Panel):
    """Main panel for graph algorithms."""
    bl_label = "Graph Algorithms"
    bl_parent_id = "SCIGRAPHS_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            box = layout.box()
            box.label(text="No graph loaded", icon='ERROR')
            box.label(text="Create a graph first in Data panel")
            return
        
        layout.label(text="Algorithmic operations on graphs", icon='SCRIPT')


class SCIGRAPHS_PT_algorithms_traversal(bpy.types.Panel):
    """Graph traversal algorithms (BFS, DFS)."""
    bl_label = "Traversal"
    bl_parent_id = "SCIGRAPHS_PT_algorithms"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        obj = context.active_object
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        # Main traversal box
        box = layout.box()
        box.label(text="BFS / DFS Animation", icon='ORIENTATION_VIEW')
        
        # Algorithm and start nodes in columns
        row = box.row(align=True)
        row.prop(props, "traversal_algorithm", text="")
        row.prop(props, "traversal_start_mode", text="", icon='PINNED')
        
        # Manual node input if needed
        if props.traversal_start_mode == 'MANUAL':
            box.prop(props, "traversal_start_nodes", text="Nodes")
        
        # Animation mode and parameters
        box.separator()
        col = box.column(align=True)
        col.prop(props, "traversal_animation_mode", text="Mode")
        col.prop(props, "traversal_animation_speed", text="Speed")
        
        if props.traversal_animation_mode == 'CONTINUOUS':
            col.prop(props, "traversal_animation_smoothness", text="Smoothness")
        
        col.prop(props, "traversal_animation_loop", text="Loop")
        
        # Animate button
        box.separator()
        row = box.row()
        row.scale_y = 1.3
        row.operator("scigraphs.animate_traversal", text="Animate Traversal", icon='PLAY')
        
        # Show animation info if created
        if "traversal_max_order" in obj:
            info = box.box()
            info.scale_y = 0.7
            col = info.column(align=True)
            
            algo_str = obj.get("traversal_algorithm", "BFS")
            mode_str = obj.get("traversal_mode", "DISCRETE")
            col.label(text=f"{algo_str} ({mode_str.title()}) - {obj.get('traversal_visited_count', 0)} nodes")
            col.label(text="Attributes: traversal_activation, traversal_order")


class SCIGRAPHS_PT_algorithms_pathfinding(bpy.types.Panel):
    """Pathfinding algorithms (shortest paths, etc.)."""
    bl_label = "Pathfinding"
    bl_parent_id = "SCIGRAPHS_PT_algorithms"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        # Algorithm selection
        box = layout.box()
        box.label(text="Shortest Path", icon='CURVE_PATH')
        col = box.column(align=True)
        col.prop(props, "pathfinding_algorithm", text="Algorithm")

        row = col.row(align=True)
        row.prop(props, "pathfinding_source", text="Source Node")
        op = row.operator("scigraphs.pick_path_node", text="", icon='EYEDROPPER')
        op.target = 'SOURCE'

        row = col.row(align=True)
        row.prop(props, "pathfinding_target", text="Target Node")
        op = row.operator("scigraphs.pick_path_node", text="", icon='EYEDROPPER')
        op.target = 'TARGET'
        
        # Info box
        info = box.box()
        info.scale_y = 0.7
        algo = props.pathfinding_algorithm
        if algo == 'DIJKSTRA':
            info.label(text="Dijkstra: Optimal for non-negative weights")
        elif algo == 'ASTAR':
            info.label(text="A*: Uses position heuristic for speed")
        else:
            info.label(text="Bellman-Ford: Handles negative weights")
        
        # Execute button
        box.separator()
        row = box.row()
        row.scale_y = 1.3
        row.operator("scigraphs.find_shortest_path", text="Find Path", icon='PLAY')
        box.operator("scigraphs.path_tool", text="Pick Source and Target in Viewport", icon='TRACKING')


class SCIGRAPHS_PT_algorithms_spanning(bpy.types.Panel):
    """Spanning tree algorithms."""
    bl_label = "Spanning Trees"
    bl_parent_id = "SCIGRAPHS_PT_algorithms"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        # Algorithm selection
        box = layout.box()
        box.label(text="Spanning Tree", icon='OUTLINER_OB_FORCE_FIELD')
        box.prop(props, "spanning_algorithm", text="Algorithm")
        
        # Info
        info = box.box()
        info.scale_y = 0.7
        algo = props.spanning_algorithm
        if algo == 'KRUSKAL':
            info.label(text="Kruskal: Sort edges, union-find")
        elif algo == 'PRIM':
            info.label(text="Prim: Grow tree from start node")
        else:
            info.label(text="Maximum: Heaviest spanning tree")
        
        # Execute button
        box.separator()
        row = box.row()
        row.scale_y = 1.3
        row.operator("scigraphs.compute_mst", text="Compute MST", icon='PLAY')


class SCIGRAPHS_PT_algorithms_flow(bpy.types.Panel):
    """Network flow algorithms."""
    bl_label = "Network Flow"
    bl_parent_id = "SCIGRAPHS_PT_algorithms"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_directed", False)
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        # Node selection
        box = layout.box()
        box.label(text="Max Flow / Min Cut", icon='MOD_FLUIDSIM')
        col = box.column(align=True)
        col.prop(props, "flow_source", text="Source Node")
        col.prop(props, "flow_sink", text="Sink Node")
        
        # Info
        info = box.box()
        info.scale_y = 0.7
        info.label(text="Requires directed graph")
        info.label(text="Ford-Fulkerson algorithm")
        
        # Buttons
        box.separator()
        row = box.row(align=True)
        row.scale_y = 1.3
        row.operator("scigraphs.compute_max_flow", text="Max Flow", icon='PLAY')
        row.operator("scigraphs.compute_min_cut", text="Min Cut", icon='SELECT_DIFFERENCE')


def register():
    bpy.utils.register_class(SCIGRAPHS_PT_algorithms)
    bpy.utils.register_class(SCIGRAPHS_PT_algorithms_traversal)
    bpy.utils.register_class(SCIGRAPHS_PT_algorithms_pathfinding)
    bpy.utils.register_class(SCIGRAPHS_PT_algorithms_spanning)
    bpy.utils.register_class(SCIGRAPHS_PT_algorithms_flow)


def unregister():
    bpy.utils.unregister_class(SCIGRAPHS_PT_algorithms_flow)
    bpy.utils.unregister_class(SCIGRAPHS_PT_algorithms_spanning)
    bpy.utils.unregister_class(SCIGRAPHS_PT_algorithms_pathfinding)
    bpy.utils.unregister_class(SCIGRAPHS_PT_algorithms_traversal)
    bpy.utils.unregister_class(SCIGRAPHS_PT_algorithms)

