# SciGraphs Pie Menus
# Radial menus for quick access to common operations

import bpy


class SCIGRAPHS_MT_PIE_main(bpy.types.Menu):
    """Main SciGraphs pie menu."""
    bl_idname = "SCIGRAPHS_MT_PIE_main"
    bl_label = "SciGraphs"

    def draw(self, context):
        pie = self.layout.menu_pie()
        # 8 directions: W, E, S, N, NW, NE, SW, SE
        pie.operator("scigraphs.apply_layout", text="Apply Layout", icon='PLAY')
        pie.operator("scigraphs.calculate_centrality", text="Centrality", icon='LIGHT_POINT')
        pie.operator("scigraphs.setup_visualization", text="Setup Viz", icon='GEOMETRY_NODES')
        pie.operator("scigraphs.apply_clustering", text="Clustering", icon='GROUP_VERTEX')
        pie.operator("scigraphs.reset_layout", text="Reset Layout", icon='FILE_REFRESH')
        pie.operator_context = 'INVOKE_DEFAULT'
        pie.operator("scigraphs.update_appearance", text="Update Look", icon='MATERIAL')
        pie.operator_context = 'EXEC_DEFAULT'
        pie.operator("scigraphs.check_planarity", text="Planarity", icon='MESH_PLANE')
        pie.operator("scigraphs.export_gexf", text="Export GEXF", icon='EXPORT')


class SCIGRAPHS_MT_PIE_layout(bpy.types.Menu):
    """Layout operations pie menu."""
    bl_idname = "SCIGRAPHS_MT_PIE_layout"
    bl_label = "Layout"

    def draw(self, context):
        pie = self.layout.menu_pie()
        pie.operator("scigraphs.apply_layout", text="Apply", icon='PLAY')
        pie.operator("scigraphs.reset_layout", text="Reset", icon='FILE_REFRESH')
        pie.operator("scigraphs.execute_layout_step", text="Step", icon='FRAME_NEXT')
        pie.operator("scigraphs.bake_animation", text="Bake", icon='RENDER_ANIMATION')
        pie.operator("scigraphs.network_splitter_3d", text="Split 3D", icon='MESH_GRID')
        pie.operator("scigraphs.reset_splitter", text="Flatten", icon='MESH_PLANE')


class SCIGRAPHS_MT_PIE_analysis(bpy.types.Menu):
    """Analysis operations pie menu."""
    bl_idname = "SCIGRAPHS_MT_PIE_analysis"
    bl_label = "Analysis"

    def draw(self, context):
        pie = self.layout.menu_pie()
        pie.operator("scigraphs.calculate_centrality", text="Centrality", icon='LIGHT_POINT')
        pie.operator("scigraphs.apply_clustering", text="Clustering", icon='GROUP_VERTEX')
        pie.operator("scigraphs.calculate_clustering", text="Coeff.", icon='STICKY_UVS_LOC')
        pie.operator("scigraphs.find_sccs", text="SCCs", icon='LINKED')
        pie.operator("scigraphs.compute_mst", text="MST", icon='OUTLINER_OB_ARMATURE')
        pie.operator("scigraphs.detect_patterns", text="Patterns", icon='SHADERFX')
        pie.operator("scigraphs.check_planarity", text="Planarity", icon='MESH_PLANE')
        pie.operator("scigraphs.calculate_genus", text="Genus", icon='SURFACE_NSPHERE')


class SCIGRAPHS_MT_PIE_visualization(bpy.types.Menu):
    """Visualization pie menu."""
    bl_idname = "SCIGRAPHS_MT_PIE_visualization"
    bl_label = "Visualization"

    def draw(self, context):
        pie = self.layout.menu_pie()
        pie.operator("scigraphs.setup_visualization", text="Setup", icon='GEOMETRY_NODES')
        pie.operator_context = 'INVOKE_DEFAULT'
        pie.operator("scigraphs.update_appearance", text="Update", icon='MATERIAL')
        pie.operator_context = 'EXEC_DEFAULT'
        pie.operator("scigraphs.apply_edge_style_preset", text="Edge Style", icon='IPO_BEZIER')
        pie.operator("scigraphs.apply_rendering_preset", text="Render Preset", icon='SCENE')
        pie.operator("scigraphs.setup_lighting", text="Lighting", icon='LIGHT_SUN')
        pie.operator("scigraphs.reset_edge_style", text="Reset Edges", icon='LOOP_BACK')


class SCIGRAPHS_MT_PIE_topology(bpy.types.Menu):
    """Topology pie menu."""
    bl_idname = "SCIGRAPHS_MT_PIE_topology"
    bl_label = "Topology"

    def draw(self, context):
        pie = self.layout.menu_pie()
        pie.operator("scigraphs.check_planarity", text="Planarity", icon='MESH_PLANE')
        pie.operator("scigraphs.calculate_genus", text="Genus", icon='SURFACE_NSPHERE')
        pie.operator("scigraphs.compute_faces", text="Faces", icon='FACESEL')
        pie.operator("scigraphs.visualize_surface", text="Surface", icon='SURFACE_DATA')
        pie.operator("scigraphs.create_dual_graph", text="Dual", icon='MOD_WIREFRAME')
        pie.operator("scigraphs.toggle_dual_graph", text="Toggle Dual", icon='HIDE_OFF')
        pie.operator("scigraphs.toggle_topo_surface", text="Toggle Surface", icon='HIDE_OFF')
        pie.operator("scigraphs.validate_crossings", text="Crossings", icon='PIVOT_CURSOR')


class SCIGRAPHS_OT_pie_main(bpy.types.Operator):
    """Open main SciGraphs pie menu."""
    bl_idname = "scigraphs.pie_main"
    bl_label = "SciGraphs Pie Menu"

    def execute(self, context):
        bpy.ops.wm.call_menu_pie(name="SCIGRAPHS_MT_PIE_main")
        return {'FINISHED'}


class SCIGRAPHS_OT_pie_layout(bpy.types.Operator):
    """Open layout pie menu."""
    bl_idname = "scigraphs.pie_layout"
    bl_label = "Layout Pie Menu"

    def execute(self, context):
        bpy.ops.wm.call_menu_pie(name="SCIGRAPHS_MT_PIE_layout")
        return {'FINISHED'}


class SCIGRAPHS_OT_pie_analysis(bpy.types.Operator):
    """Open analysis pie menu."""
    bl_idname = "scigraphs.pie_analysis"
    bl_label = "Analysis Pie Menu"

    def execute(self, context):
        bpy.ops.wm.call_menu_pie(name="SCIGRAPHS_MT_PIE_analysis")
        return {'FINISHED'}


class SCIGRAPHS_OT_pie_visualization(bpy.types.Operator):
    """Open visualization pie menu."""
    bl_idname = "scigraphs.pie_visualization"
    bl_label = "Visualization Pie Menu"

    def execute(self, context):
        bpy.ops.wm.call_menu_pie(name="SCIGRAPHS_MT_PIE_visualization")
        return {'FINISHED'}


class SCIGRAPHS_OT_pie_topology(bpy.types.Operator):
    """Open topology pie menu."""
    bl_idname = "scigraphs.pie_topology"
    bl_label = "Topology Pie Menu"

    def execute(self, context):
        bpy.ops.wm.call_menu_pie(name="SCIGRAPHS_MT_PIE_topology")
        return {'FINISHED'}


_PIE_CLASSES = [
    SCIGRAPHS_MT_PIE_main,
    SCIGRAPHS_MT_PIE_layout,
    SCIGRAPHS_MT_PIE_analysis,
    SCIGRAPHS_MT_PIE_visualization,
    SCIGRAPHS_MT_PIE_topology,
    SCIGRAPHS_OT_pie_main,
    SCIGRAPHS_OT_pie_layout,
    SCIGRAPHS_OT_pie_analysis,
    SCIGRAPHS_OT_pie_visualization,
    SCIGRAPHS_OT_pie_topology,
]

_KEYMAP_ITEMS = []


def register():
    for cls in _PIE_CLASSES:
        bpy.utils.register_class(cls)
    
    # Register keymaps
    wm = bpy.context.window_manager
    if wm.keyconfigs.addon:
        km = wm.keyconfigs.addon.keymaps.new(name='3D View', space_type='VIEW_3D')
        
        # Q = Main pie menu
        kmi = km.keymap_items.new("scigraphs.pie_main", 'Q', 'PRESS', shift=True)
        _KEYMAP_ITEMS.append((km, kmi))
        
        # Shift+L = Layout pie
        kmi = km.keymap_items.new("scigraphs.pie_layout", 'L', 'PRESS', shift=True, ctrl=True)
        _KEYMAP_ITEMS.append((km, kmi))
        
        # Shift+A already used, use Ctrl+Shift+A = Analysis pie
        kmi = km.keymap_items.new("scigraphs.pie_analysis", 'A', 'PRESS', shift=True, ctrl=True)
        _KEYMAP_ITEMS.append((km, kmi))
        
        # Ctrl+Shift+V = Visualization pie
        kmi = km.keymap_items.new("scigraphs.pie_visualization", 'V', 'PRESS', shift=True, ctrl=True)
        _KEYMAP_ITEMS.append((km, kmi))
        
        # Ctrl+Shift+T = Topology pie
        kmi = km.keymap_items.new("scigraphs.pie_topology", 'T', 'PRESS', shift=True, ctrl=True)
        _KEYMAP_ITEMS.append((km, kmi))


def unregister():
    # Remove keymaps
    for km, kmi in _KEYMAP_ITEMS:
        km.keymap_items.remove(kmi)
    _KEYMAP_ITEMS.clear()
    
    for cls in reversed(_PIE_CLASSES):
        bpy.utils.unregister_class(cls)
