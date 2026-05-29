# Graph analysis panels (centrality, community detection, directed analysis, etc.)

import bpy

class SCIGRAPHS_PT_analysis(bpy.types.Panel):
    """Main analysis panel."""
    bl_label = "Analysis"
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
        
        layout.label(text="Analyze graph structure and properties", icon='VIEWZOOM')


class SCIGRAPHS_PT_analysis_centrality(bpy.types.Panel):
    """Centrality metrics analysis."""
    bl_label = "Centrality Metrics"
    bl_parent_id = "SCIGRAPHS_PT_analysis"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        # Centrality analysis
        box = layout.box()
        box.label(text="Node Importance Metrics", icon='DRIVER')
        col = box.column(align=True)
        col.prop(props, "centrality_method", text="Method")
        
        col.separator()
        row = col.row()
        row.scale_y = 1.3
        row.operator("scigraphs.calculate_centrality", text="Calculate", icon='PLAY')
        
        # Node clustering coefficient
        layout.separator()
        box = layout.box()
        box.label(text="Local Clustering", icon='MESH_UVSPHERE')
        box.label(text="Calculate clustering coefficient per node")
        row = box.row()
        row.scale_y = 1.3
        row.operator("scigraphs.calculate_clustering", text="Calculate", icon='PLAY')


class SCIGRAPHS_PT_analysis_community(bpy.types.Panel):
    """Community detection algorithms."""
    bl_label = "Community Detection"
    bl_parent_id = "SCIGRAPHS_PT_analysis"
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
        box.label(text="Clustering Algorithm", icon='GROUP')
        box.prop(props, "clustering_algorithm", text="")
        
        # Algorithm info
        layout.separator()
        box = layout.box()
        box.label(text="Algorithm Info", icon='INFO')
        info = box.box()
        info.scale_y = 0.7
        
        algo = props.clustering_algorithm
        if algo == 'cpm':
            info.label(text="Clique Percolation Method")
            info.label(text="Traag et al., Phys. Rev. E 84 (2011)")
        elif algo == 'infomap':
            info.label(text="Map equation framework")
            info.label(text="Rosvall & Bergstrom, PNAS 105 (2008)")
        elif algo == 'rb':
            info.label(text="Reichardt-Bornholdt Potts model")
            info.label(text="Phys. Rev. E 74 (2006)")
        elif algo == 'rn':
            info.label(text="Resolution-free Potts model")
            info.label(text="Ronhovde & Nussinov, Phys. Rev. E 81 (2010)")
        elif algo == 'rnsc':
            info.label(text="Restricted Neighbourhood Search")
            info.label(text="King et al., Bioinformatics 20 (2004)")
        elif algo == 'scluster':
            info.label(text="Hierarchical clustering (Jerarca)")
            info.label(text="Aldecoa & Marin, PLoS ONE 5 (2010)")
        elif algo == 'uvcluster':
            info.label(text="Iterative cluster analysis (Jerarca)")
            info.label(text="Arnau et al., Bioinformatics 21 (2005)")
        
        info.separator()
        info.label(text="Surprise metric computed automatically")
        
        # Apply button
        layout.separator()
        row = layout.row()
        row.scale_y = 1.5
        row.operator("scigraphs.apply_clustering", text="Detect Communities", icon='PLAY')


class SCIGRAPHS_PT_analysis_directed(bpy.types.Panel):
    """Directed graph specific analysis."""
    bl_label = "Directed Analysis"
    bl_parent_id = "SCIGRAPHS_PT_analysis"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_directed", False)
    
    def draw(self, context):
        layout = self.layout
        
        # Graph type indicator
        box = layout.box()
        box.label(text="Directed Graph Mode Active", icon='FORWARD')


class SCIGRAPHS_PT_analysis_directed_centrality(bpy.types.Panel):
    """Directed centrality metrics."""
    bl_label = "Directed Centrality"
    bl_parent_id = "SCIGRAPHS_PT_analysis_directed"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        # Directed centrality metrics
        box = layout.box()
        box.label(text="Directed Importance", icon='DRIVER')
        col = box.column(align=True)
        col.prop(props, "directed_centrality_method", text="")
        
        col.separator()
        row = col.row()
        row.scale_y = 1.3
        row.operator("scigraphs.calculate_directed_centrality", text="Calculate", icon='PLAY')
        
        # Info about metrics
        info = box.box()
        info.scale_y = 0.7
        metric = props.directed_centrality_method
        if metric == 'pagerank':
            info.label(text="Google's algorithm - web importance")
        elif metric == 'hub_score':
            info.label(text="Points to many authorities")
        elif metric == 'authority_score':
            info.label(text="Pointed to by many hubs")
        elif metric == 'in_degree':
            info.label(text="Popularity - incoming connections")
        elif metric == 'out_degree':
            info.label(text="Influence - outgoing connections")


class SCIGRAPHS_PT_analysis_directed_structure(bpy.types.Panel):
    """Directed graph structure analysis."""
    bl_label = "Structure Analysis"
    bl_parent_id = "SCIGRAPHS_PT_analysis_directed"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        
        # Pattern detection
        box = layout.box()
        box.label(text="Pattern Detection", icon='MESH_DATA')
        row = box.row()
        row.scale_y = 1.3
        row.operator("scigraphs.detect_patterns", text="Detect Patterns", icon='VIEWZOOM')
        
        # Show detected patterns if available
        if "pattern_is_dag" in obj:
            info = box.box()
            col = info.column(align=True)
            col.scale_y = 0.8
            
            if obj.get("pattern_is_dag", False):
                col.label(text="✓ DAG (Acyclic)", icon='CHECKMARK')
            if obj.get("pattern_has_cycles", False):
                num_cycles = obj.get("pattern_num_cycles", 0)
                col.label(text=f"⚠ {num_cycles} cycles found", icon='ERROR')
            if obj.get("pattern_is_strongly_connected", False):
                col.label(text="✓ Strongly connected", icon='LINKED')
            
            num_sccs = obj.get("pattern_num_strongly_connected_components", 0)
            col.label(text=f"SCCs: {num_sccs}")
        
        # Strong connected components
        layout.separator()
        box = layout.box()
        box.label(text="Strongly Connected Components", icon='FILE_REFRESH')
        row = box.row()
        row.scale_y = 1.3
        row.operator("scigraphs.find_sccs", text="Find SCCs", icon='STICKY_UVS_LOC')
        
        # Show SCC info if available
        if "num_sccs" in obj:
            info = box.box()
            col = info.column(align=True)
            col.scale_y = 0.8
            col.label(text=f"Components: {obj.get('num_sccs', 0)}")
            col.label(text=f"Largest: {obj.get('largest_scc', 0)} nodes")


class SCIGRAPHS_PT_analysis_directed_flow(bpy.types.Panel):
    """Directed graph flow analysis and animation."""
    bl_label = "Flow Analysis"
    bl_parent_id = "SCIGRAPHS_PT_analysis_directed"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        obj = context.active_object
        
        # Flow analysis
        box = layout.box()
        box.label(text="Node Roles", icon='MOD_FLUIDSIM')
        row = box.row()
        row.scale_y = 1.3
        row.operator("scigraphs.analyze_flow", text="Analyze Flow", icon='ANIM')
        
        # Show flow info if available
        if "flow_sources" in obj:
            info = box.box()
            col = info.column(align=True)
            col.scale_y = 0.8
            col.label(text=f"Sources: {obj.get('flow_sources', 0)}", icon='TRACKING_FORWARDS_SINGLE')
            col.label(text=f"Sinks: {obj.get('flow_sinks', 0)}", icon='TRACKING_BACKWARDS_SINGLE')
            col.label(text=f"Intermediaries: {obj.get('flow_intermediaries', 0)}", icon='ARROW_LEFTRIGHT')
        
        # Flow animation
        layout.separator()
        box = layout.box()
        box.label(text="Flow Animation", icon='ANIM')
        
        info_box = box.box()
        info_box.scale_y = 0.7
        info_box.label(text="Animate information flow through the network", icon='INFO')
        
        # Mode selection
        box.prop(props, "flow_animation_mode", text="Mode")
        
        # Mode description
        mode_info = box.box()
        mode_info.scale_y = 0.7
        if props.flow_animation_mode == 'DISCRETE':
            mode_info.label(text="Binary: nodes switch 0→1 instantly")
        else:
            mode_info.label(text="Smooth: gradient wave propagation")
        
        # Parameters
        col = box.column(align=True)
        col.prop(props, "flow_animation_speed")
        
        if props.flow_animation_mode == 'CONTINUOUS':
            col.prop(props, "flow_animation_smoothness")
        
        col.prop(props, "flow_animation_loop")
        
        # Loop explanation
        if props.flow_animation_loop:
            loop_info = box.box()
            loop_info.scale_y = 0.6
            timeline_range = context.scene.frame_end - context.scene.frame_start
            num_cycles = max(1, int(timeline_range / props.flow_animation_speed))
            loop_info.label(text=f"Will create {num_cycles} cycles in timeline")
            loop_info.label(text=f"({timeline_range} frames total)")
        
        row = box.row()
        row.scale_y = 1.4
        row.operator("scigraphs.animate_flow", text="Create Flow Animation", icon='PLAY')
        
        # Show animation info if created
        if "flow_max_distance" in obj:
            info = box.box()
            col = info.column(align=True)
            col.scale_y = 0.8
            
            mode_str = obj.get("flow_mode", "DISCRETE")
            col.label(text=f"Mode: {mode_str.title()}")
            col.label(text=f"Max Distance: {obj.get('flow_max_distance', 0)} steps")
            
            usage = info.box()
            usage.scale_y = 0.7
            usage.label(text="Attributes (visible in Spreadsheet):", icon='NODE')
            if mode_str == "DISCRETE":
                usage.label(text="- flow_activation: 0 or 1 (binary)")
            else:
                usage.label(text="- flow_activation: 0.0 to 1.0 (gradient)")
            usage.label(text="- flow_distance: static distances")
            usage.label(text="Press Play to see animation!")


class SCIGRAPHS_PT_analysis_statistical(bpy.types.Panel):
    """Statistical analysis of graph properties."""
    bl_label = "Statistical Analysis"
    bl_parent_id = "SCIGRAPHS_PT_analysis"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        
        # Calculate button
        box = layout.box()
        box.label(text="Global Graph Metrics", icon='GRAPH')
        row = box.row()
        row.scale_y = 1.3
        row.operator("scigraphs.calculate_global_statistics", text="Calculate Statistics", icon='PLAY')
        
        # Show results if available
        if obj and "stat_density" in obj:
            layout.separator()
            box = layout.box()
            box.label(text="Results:", icon='INFO')
            col = box.column(align=True)
            col.scale_y = 0.8
            
            col.label(text=f"Density: {obj.get('stat_density', 0):.4f}")
            col.label(text=f"Global Clustering: {obj.get('stat_global_clustering', 0):.4f}")
            col.label(text=f"Diameter: {obj.get('stat_diameter', 0)}")
            col.label(text=f"Avg Path Length: {obj.get('stat_avg_path_length', 0):.4f}")
            col.label(text=f"Assortativity: {obj.get('stat_assortativity', 0):.4f}")
            
            layout.separator()
            col = box.column(align=True)
            col.label(text="Degree Distribution:")
            col.label(text=f"  Mean: {obj.get('stat_degree_mean', 0):.2f}")
            col.label(text=f"  Median: {obj.get('stat_degree_median', 0):.0f}")
            col.label(text=f"  Std: {obj.get('stat_degree_std', 0):.2f}")
            col.label(text=f"  Range: [{obj.get('stat_degree_min', 0)}, {obj.get('stat_degree_max', 0)}]")


def register():
    bpy.utils.register_class(SCIGRAPHS_PT_analysis)
    bpy.utils.register_class(SCIGRAPHS_PT_analysis_centrality)
    bpy.utils.register_class(SCIGRAPHS_PT_analysis_community)
    bpy.utils.register_class(SCIGRAPHS_PT_analysis_directed)
    bpy.utils.register_class(SCIGRAPHS_PT_analysis_directed_centrality)
    bpy.utils.register_class(SCIGRAPHS_PT_analysis_directed_structure)
    bpy.utils.register_class(SCIGRAPHS_PT_analysis_directed_flow)
    bpy.utils.register_class(SCIGRAPHS_PT_analysis_statistical)


def unregister():
    bpy.utils.unregister_class(SCIGRAPHS_PT_analysis_statistical)
    bpy.utils.unregister_class(SCIGRAPHS_PT_analysis_directed_flow)
    bpy.utils.unregister_class(SCIGRAPHS_PT_analysis_directed_structure)
    bpy.utils.unregister_class(SCIGRAPHS_PT_analysis_directed_centrality)
    bpy.utils.unregister_class(SCIGRAPHS_PT_analysis_directed)
    bpy.utils.unregister_class(SCIGRAPHS_PT_analysis_community)
    bpy.utils.unregister_class(SCIGRAPHS_PT_analysis_centrality)
    bpy.utils.unregister_class(SCIGRAPHS_PT_analysis)

