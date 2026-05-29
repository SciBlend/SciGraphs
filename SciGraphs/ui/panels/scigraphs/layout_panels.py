# Layout and positioning panels

import bpy

class SCIGRAPHS_PT_layout(bpy.types.Panel):
    """Main layout panel for graph positioning."""
    bl_label = "Layout & Positioning"
    bl_parent_id = "SCIGRAPHS_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        obj = context.active_object
        
        # Quick info and status
        if obj and "num_nodes" in obj:
            num_nodes = obj["num_nodes"]
            num_edges = len(obj.get("edges_data", "").split(",")) // 2
            
            box = layout.box()
            row = box.row(align=True)
            row.label(text=f"{num_nodes:,} nodes", icon='MESH_CIRCLE')
            row.label(text=f"{num_edges:,} edges", icon='CURVE_PATH')
            
            # Smart recommendations based on graph size
            if num_nodes > 10000:
                split = box.split(factor=0.7)
                split.label(text="Large Graph Detected", icon='INFO')
                if props.layout_algorithm in ['SPRING', 'SPRING_3D']:
                    row = box.row()
                    row.alert = True
                    row.label(text="Current: Very Slow!", icon='ERROR')
                    box.label(text="Recommended: DrL, Yifan Hu, or LGL", icon='FUND')
            elif num_nodes > 1000:
                if props.layout_algorithm in ['SPRING', 'SPRING_3D']:
                    box.label(text="Medium graph - Consider faster algorithms", icon='INFO')
        else:
            box = layout.box()
            box.label(text="No graph loaded", icon='ERROR')
            box.label(text="Create a graph first in Data panel")


class SCIGRAPHS_PT_layout_algorithm(bpy.types.Panel):
    """Algorithm selection for graph layout."""
    bl_label = "Algorithm Selection"
    bl_parent_id = "SCIGRAPHS_PT_layout"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        
        # Algorithm selector with icon
        layout.label(text="Choose Layout Algorithm:", icon='SORTSIZE')
        layout.prop(props, "layout_algorithm", text="")
        
        # Algorithm info card
        algo = props.layout_algorithm
        box = layout.box()
        box.label(text="Algorithm Info:", icon='INFO')
        
        # Dimension indicator
        dimensions = {
            'GRID': '2D', 'SPRING': '2D', 'FORCEATLAS2': '2D',
            'IGRAPH_DH': '2D', 'IGRAPH_GRAPHOPT': '2D',
            'SUGIYAMA': '2D', 'CIRCULAR_HIERARCHY': '2D',
            'RANDOM': '3D', 'SPHERE': '3D', 'SPIRAL_3D': '3D', 
            'HELIX': '3D', 'CUBE': '3D',
            'SPECTRAL_3D': '3D', 'MDS_3D': '3D', 'HIERARCHICAL_3D': '3D', 
            'BIPARTITE_3D': '3D',
            'YIFAN_HU': '3D', 'IGRAPH_DRL': '3D', 'IGRAPH_FR': '3D',
            'IGRAPH_KK': '3D', 'IGRAPH_LGL': '3D', 'SPRING_3D': '3D',
            'GRAPHVIZ_DOT': '2D', 'GRAPHVIZ_NEATO': '2D', 'GRAPHVIZ_FDP': '2D',
            'GRAPHVIZ_SFDP': '2D', 'GRAPHVIZ_TWOPI': '2D', 'GRAPHVIZ_CIRCO': '2D',
            'GRAPHVIZ_OSAGE': '2D', 'GRAPHVIZ_PATCHWORK': '2D',
        }
        
        # Speed indicator
        speed_icons = {
            'RANDOM': ('CHECKMARK', 'Instant'),
            'GRID': ('CHECKMARK', 'Instant'),
            'SPHERE': ('CHECKMARK', 'Instant'),
            'SPIRAL_3D': ('CHECKMARK', 'Instant'),
            'HELIX': ('CHECKMARK', 'Instant'),
            'CUBE': ('CHECKMARK', 'Instant'),
            'SPECTRAL_3D': ('KEYTYPE_KEYFRAME_VEC', 'Fast'),
            'MDS_3D': ('TIME', 'Medium'),
            'HIERARCHICAL_3D': ('KEYTYPE_KEYFRAME_VEC', 'Fast'),
            'BIPARTITE_3D': ('KEYTYPE_KEYFRAME_VEC', 'Fast'),
            'YIFAN_HU': ('KEYTYPE_KEYFRAME_VEC', 'Fast'),
            'IGRAPH_DRL': ('KEYTYPE_EXTREME_VEC', 'Very Fast'),
            'IGRAPH_FR': ('KEYTYPE_KEYFRAME_VEC', 'Fast'),
            'IGRAPH_KK': ('TIME', 'Medium'),
            'IGRAPH_LGL': ('KEYTYPE_KEYFRAME_VEC', 'Fast'),
            'IGRAPH_DH': ('TIME', 'Medium'),
            'IGRAPH_GRAPHOPT': ('KEYTYPE_KEYFRAME_VEC', 'Fast'),
            'FORCEATLAS2': ('TIME', 'Medium'),
            'SPRING': ('ERROR', 'Slow'),
            'SPRING_3D': ('ERROR', 'Very Slow'),
            'SUGIYAMA': ('KEYTYPE_KEYFRAME_VEC', 'Fast'),
            'CIRCULAR_HIERARCHY': ('KEYTYPE_KEYFRAME_VEC', 'Fast'),
            'GRAPHVIZ_DOT': ('KEYTYPE_KEYFRAME_VEC', 'Fast'),
            'GRAPHVIZ_NEATO': ('KEYTYPE_KEYFRAME_VEC', 'Fast'),
            'GRAPHVIZ_FDP': ('KEYTYPE_KEYFRAME_VEC', 'Fast'),
            'GRAPHVIZ_SFDP': ('KEYTYPE_KEYFRAME_VEC', 'Fast'),
            'GRAPHVIZ_TWOPI': ('KEYTYPE_KEYFRAME_VEC', 'Fast'),
            'GRAPHVIZ_CIRCO': ('KEYTYPE_KEYFRAME_VEC', 'Fast'),
            'GRAPHVIZ_OSAGE': ('KEYTYPE_KEYFRAME_VEC', 'Fast'),
            'GRAPHVIZ_PATCHWORK': ('KEYTYPE_KEYFRAME_VEC', 'Fast'),
        }
        
        icon, speed_text = speed_icons.get(algo, ('QUESTION', 'Unknown'))
        dimension = dimensions.get(algo, '?D')
        
        row = box.row(align=True)
        row.label(text=f"{dimension}", icon='EMPTY_AXIS')
        row.label(text=f"Speed: {speed_text}", icon=icon)
        
        # Use case
        use_cases = {
            'RANDOM': 'Quick test, initial positions',
            'GRID': '2D regular arrangement',
            'SPHERE': 'Spherical distribution',
            'SPIRAL_3D': 'Temporal/sequential data',
            'HELIX': 'DNA-like, paired data',
            'CUBE': 'Bounded 3D space',
            'SPECTRAL_3D': 'Community detection',
            'MDS_3D': 'Distance preservation',
            'HIERARCHICAL_3D': 'Tree-like structures',
            'BIPARTITE_3D': 'Two-set graphs',
            'YIFAN_HU': 'Large graphs, best quality',
            'IGRAPH_DRL': 'Massive graphs (100k+)',
            'IGRAPH_FR': 'General purpose',
            'IGRAPH_KK': 'Reproducible layouts',
            'IGRAPH_LGL': 'Very large sparse graphs',
            'IGRAPH_DH': 'High quality optimization',
            'IGRAPH_GRAPHOPT': 'Energy minimization',
            'FORCEATLAS2': 'Gephi compatibility',
            'SPRING': 'Classic 2D (slow)',
            'SPRING_3D': 'Classic 3D (very slow)',
            'SUGIYAMA': 'DAGs, workflows, processes',
            'CIRCULAR_HIERARCHY': 'Hierarchies from roots',
            'GRAPHVIZ_DOT': 'Hierarchical directed graphs',
            'GRAPHVIZ_NEATO': 'General undirected graphs',
            'GRAPHVIZ_FDP': 'Force-directed graphs',
            'GRAPHVIZ_SFDP': 'Large force-directed graphs',
            'GRAPHVIZ_TWOPI': 'Radial structures',
            'GRAPHVIZ_CIRCO': 'Circular structures',
            'GRAPHVIZ_OSAGE': 'Clustered graphs',
            'GRAPHVIZ_PATCHWORK': 'Area-style layouts',
        }
        
        use_case = use_cases.get(algo, 'General purpose')
        box.label(text=f"Best for: {use_case}", icon='INFO')


class SCIGRAPHS_PT_layout_settings(bpy.types.Panel):
    """Quick settings and one-click layout application."""
    bl_label = "Quick Settings"
    bl_parent_id = "SCIGRAPHS_PT_layout"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        
        # Only global scale setting
        layout.use_property_split = True
        layout.prop(props, "layout_scale", text="Scale")
        
        # Apply layout button - one-click execution
        layout.separator()
        box = layout.box()
        box.label(text="Calculate layout once:", icon='INFO')
        row = box.row()
        row.scale_y = 2.0
        row.operator("scigraphs.apply_layout", text="Apply Layout Now", icon='PLAY')


class SCIGRAPHS_PT_layout_interactive(bpy.types.Panel):
    """Interactive Gephi-style layout execution (real-time animation)."""
    bl_label = "Interactive Mode"
    bl_parent_id = "SCIGRAPHS_PT_layout"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw_header(self, context):
        self.layout.label(text="", icon='TIME')
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        obj = context.active_object
        scene = context.scene
        
        # Info card
        box = layout.box()
        box.label(text="Real-time layout calculation (Gephi-style)", icon='INFO')
        box.label(text="Computes layout iteratively over frames.")
        box.label(text="You can stop anytime to inspect results.")
        
        # Timeline configuration
        layout.separator()
        box = layout.box()
        box.label(text="Timeline", icon='TIME')
        num_frames = scene.frame_end - scene.frame_start + 1
        box.label(text=f"Frames: {scene.frame_start} to {scene.frame_end} ({num_frames} total)")
        
        # Execution settings
        layout.separator()
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        col = layout.column(align=True)
        col.prop(props, "iterations_per_frame", text="Iterations/Frame")
        col.prop(props, "execution_speed", text="Speed (sec/frame)")
        col.prop(props, "auto_stop_threshold", text="Auto-Stop Energy")
        
        layout.separator()
        
        # Display options
        row = layout.row(align=True)
        row.prop(props, "update_viewport", text="Live Update", toggle=True, icon='RESTRICT_VIEW_OFF')
        row.prop(props, "show_forces", text="Show Forces", toggle=True, icon='FORCE_FORCE')
        
        # Main controls
        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("scigraphs.execute_layout_step", text="Start Execution", icon='PLAY')
        row.operator("scigraphs.reset_layout", text="Reset", icon='FILE_REFRESH')
        
        row = layout.row()
        row.operator("scigraphs.bake_animation", text="Bake to Animation", icon='REC')
        
        # Current status
        if obj and "layout_iteration" in obj:
            box = layout.box()
            box.label(text="Current Status", icon='INFO')
            
            iteration = obj["layout_iteration"]
            col = box.column(align=True)
            col.label(text=f"Iteration: {iteration}")
            
            if "layout_energy" in obj:
                energy = obj["layout_energy"]
                col.label(text=f"Energy: {energy:.4f}")
                
                # Energy visualization (pseudo progress bar)
                if iteration > 1:
                    prev_energy = obj.get("prev_energy", energy)
                    if prev_energy > 0:
                        change_pct = abs((energy - prev_energy) / prev_energy * 100)
                        if change_pct < 1:
                            col.label(text="Converging...", icon='CHECKMARK')
                        else:
                            col.label(text="Computing...", icon='TIME')


class SCIGRAPHS_PT_layout_algorithm_params(bpy.types.Panel):
    """Algorithm-specific parameters (dynamic based on selection)."""
    bl_label = "Algorithm Parameters"
    bl_parent_id = "SCIGRAPHS_PT_layout"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        """Only show panel for algorithms with custom parameters."""
        props = context.scene.scigraphs
        return props.layout_algorithm in [
            'SPRING', 'SPRING_3D', 'FORCEATLAS2', 'IGRAPH_FR', 'IGRAPH_KK', 
            'IGRAPH_DRL', 'IGRAPH_DRL_2D', 'IGRAPH_LGL', 'IGRAPH_DH', 'IGRAPH_GRAPHOPT',
            'YIFAN_HU', 'GRAPHVIZ_DOT', 'GRAPHVIZ_NEATO', 'GRAPHVIZ_FDP',
            'GRAPHVIZ_SFDP', 'GRAPHVIZ_TWOPI', 'GRAPHVIZ_CIRCO',
            'GRAPHVIZ_OSAGE', 'GRAPHVIZ_PATCHWORK'
        ]
    
    def draw(self, context):
        props = context.scene.scigraphs
        layout = self.layout
        algo = props.layout_algorithm
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        if algo in ['SPRING', 'SPRING_3D']:
            self._draw_spring(layout, props, algo)
        elif algo == 'FORCEATLAS2':
            self._draw_fa2(layout, props)
        elif algo == 'IGRAPH_FR':
            self._draw_igraph_fr(layout, props)
        elif algo == 'IGRAPH_KK':
            self._draw_igraph_kk(layout, props)
        elif algo in ['IGRAPH_DRL', 'IGRAPH_DRL_2D']:
            self._draw_igraph_drl(layout, props)
        elif algo == 'IGRAPH_LGL':
            self._draw_igraph_lgl(layout, props)
        elif algo == 'IGRAPH_DH':
            self._draw_igraph_dh(layout, props)
        elif algo == 'IGRAPH_GRAPHOPT':
            self._draw_igraph_graphopt(layout, props)
        elif algo == 'YIFAN_HU':
            self._draw_yifan_hu(layout, props)
        elif algo.startswith('GRAPHVIZ_'):
            self._draw_graphviz(layout, props, algo)
    
    def _draw_spring(self, layout, props, algo):
        """Spring (NetworkX) parameters - only for SPRING and SPRING_3D."""
        dimension = "2D" if algo == 'SPRING' else "3D"
        box = layout.box()
        box.label(text=f"Spring Layout {dimension} Settings", icon='FORCE_LENNARDJONES')
        
        # Iterations
        col = box.column(align=True)
        col.prop(props, "iterations", text="Iterations")
        
        box.separator()
        
        # Force parameters
        col = box.column(align=True)
        col.label(text="Force Dynamics:")
        col.prop(props, "repulsion_strength", text="Repulsion")
        col.prop(props, "attraction_strength", text="Attraction")
        col.prop(props, "gravity_strength", text="Gravity")
        col.prop(props, "edge_distance", text="Edge Distance")
        
        box.separator()
        
        # Cooling parameters
        col = box.column(align=True)
        col.label(text="Convergence Control:")
        col.prop(props, "initial_temperature", text="Initial Temp")
        col.prop(props, "cooling_factor", text="Cooling Factor")
    
    def _draw_fa2(self, layout, props):
        """ForceAtlas2 parameters."""
        box = layout.box()
        box.label(text="ForceAtlas2 Settings", icon='FORCE_FORCE')
        
        col = box.column(align=True)
        col.prop(props, "fa2_scaling_ratio")
        col.prop(props, "fa2_gravity")
        col.prop(props, "fa2_strong_gravity")
        col.prop(props, "fa2_lin_log_mode")
        
        box.separator()
        col = box.column(align=True)
        col.prop(props, "fa2_barnes_hut_optimize")
        if props.fa2_barnes_hut_optimize:
            col.prop(props, "fa2_barnes_hut_theta")
        col.prop(props, "fa2_jitter_tolerance")
        col.prop(props, "fa2_edge_weight_influence")
    
    def _draw_igraph_fr(self, layout, props):
        """Fruchterman-Reingold parameters."""
        box = layout.box()
        box.label(text="Fruchterman-Reingold Settings", icon='FORCE_LENNARDJONES')
        
        # Info about parameter usage
        info_box = box.box()
        info_box.label(text="Note: These parameters only work in", icon='INFO')
        info_box.label(text="Interactive Mode (Gephi-style)")
        info_box.label(text="'Apply Layout Now' uses default values")
        
        box.separator()
        
        col = box.column(align=True)
        col.prop(props, "igraph_fr_start_temp")
        col.prop(props, "igraph_fr_coolexp")
        col.prop(props, "igraph_fr_maxdelta")
        col.prop(props, "igraph_fr_area")
        col.prop(props, "igraph_fr_repulserad")
    
    def _draw_igraph_kk(self, layout, props):
        """Kamada-Kawai parameters."""
        box = layout.box()
        box.label(text="Kamada-Kawai Settings", icon='DRIVER_DISTANCE')
        
        col = box.column(align=True)
        col.prop(props, "igraph_kk_maxiter")
        col.prop(props, "igraph_kk_epsilon")
        col.prop(props, "igraph_kk_kkconst")
    
    def _draw_igraph_drl(self, layout, props):
        """DrL parameters — all 6 phases exposed."""
        box = layout.box()
        box.label(text="DrL Settings", icon='MOD_PARTICLES')
        
        # Global parameter
        col = box.column(align=True)
        col.prop(props, "igraph_drl_edge_cut")
        
        box.separator()
        
        # Per-phase parameters
        phases = [
            ("Init", "init", "PLAY"),
            ("Liquid", "liquid", "MOD_FLUIDSIM"),
            ("Expansion", "expansion", "FULLSCREEN_ENTER"),
            ("Cooldown", "cooldown", "FREEZE"),
            ("Crunch", "crunch", "MESH_ICOSPHERE"),
            ("Simmer", "simmer", "LIGHT_SUN"),
        ]
        for label, phase, icon in phases:
            phase_box = box.box()
            phase_box.label(text=label, icon=icon)
            col = phase_box.column(align=True)
            col.prop(props, f"igraph_drl_{phase}_iterations")
            col.prop(props, f"igraph_drl_{phase}_temperature")
            col.prop(props, f"igraph_drl_{phase}_attraction")
            col.prop(props, f"igraph_drl_{phase}_damping_mult")
    
    def _draw_igraph_lgl(self, layout, props):
        """LGL parameters."""
        box = layout.box()
        box.label(text="LGL Settings", icon='STICKY_UVS_LOC')
        
        col = box.column(align=True)
        col.prop(props, "igraph_lgl_maxiter")
        col.prop(props, "igraph_lgl_maxdelta")
        col.prop(props, "igraph_lgl_area")
        col.prop(props, "igraph_lgl_coolexp")
        col.prop(props, "igraph_lgl_repulserad")
        col.prop(props, "igraph_lgl_cellsize")
    
    def _draw_igraph_dh(self, layout, props):
        """Davidson-Harel parameters."""
        box = layout.box()
        box.label(text="Davidson-Harel Settings", icon='FORCE_HARMONIC')
        
        col = box.column(align=True)
        col.prop(props, "igraph_dh_maxiter")
        col.prop(props, "igraph_dh_fineiter")
        col.prop(props, "igraph_dh_cool_fact")
        
        box.separator()
        col = box.column(align=True)
        col.label(text="Weight Parameters:")
        col.prop(props, "igraph_dh_weight_node_dist")
        col.prop(props, "igraph_dh_weight_border")
        col.prop(props, "igraph_dh_weight_edge_lengths")
        col.prop(props, "igraph_dh_weight_edge_crossings")
        col.prop(props, "igraph_dh_weight_node_edge_dist")
    
    def _draw_igraph_graphopt(self, layout, props):
        """GraphOpt parameters."""
        box = layout.box()
        box.label(text="GraphOpt Settings", icon='GRAPH')
        
        col = box.column(align=True)
        col.prop(props, "igraph_graphopt_niter")
        col.prop(props, "igraph_graphopt_node_charge")
        col.prop(props, "igraph_graphopt_node_mass")
        col.prop(props, "igraph_graphopt_spring_length")
        col.prop(props, "igraph_graphopt_spring_constant")
        col.prop(props, "igraph_graphopt_max_sa_movement")


    def _draw_yifan_hu(self, layout, props):
        """Yifan Hu / sfdp parameters."""
        box = layout.box()
        box.label(text="Yifan Hu (sfdp) Settings", icon='FORCE_VORTEX')
        
        # Dimension mode
        col = box.column(align=True)
        col.prop(props, "sfdp_dim")
        col.prop(props, "graphviz_quiet")
        
        # Z generation (only for 2D+Z mode)
        if props.sfdp_dim == '2Z':
            z_box = box.box()
            z_box.label(text="Z Depth Generation", icon='ORIENTATION_LOCAL')
            col = z_box.column(align=True)
            col.prop(props, "sfdp_z_method")
            col.prop(props, "sfdp_z_scale")
        
        box.separator()
        
        # Core params
        core_box = box.box()
        core_box.label(text="Force Parameters", icon='FORCE_CHARGE')
        col = core_box.column(align=True)
        col.prop(props, "sfdp_k")
        col.prop(props, "sfdp_repulsive_force")
        col.prop(props, "sfdp_maxiter")
        
        box.separator()
        
        # Multilevel & quality
        qual_box = box.box()
        qual_box.label(text="Quality", icon='MODIFIER')
        col = qual_box.column(align=True)
        col.prop(props, "sfdp_smoothing")
        col.prop(props, "sfdp_quadtree")
        col.prop(props, "sfdp_levels")
        col.prop(props, "sfdp_beautify")
        
        box.separator()
        
        # Overlap
        ov_box = box.box()
        ov_box.label(text="Overlap", icon='SELECT_SUBTRACT')
        col = ov_box.column(align=True)
        col.prop(props, "sfdp_overlap")
        if props.sfdp_overlap == 'prism':
            col.prop(props, "sfdp_overlap_scaling")

        self._draw_graphviz_advanced_attrs(box, props)

    def _draw_graphviz(self, layout, props, algo):
        """Graphviz/scigraphs-utils parameters."""
        engine = algo.replace('GRAPHVIZ_', '').lower()
        box = layout.box()
        box.label(text=f"Graphviz {engine} Settings", icon='GRAPH')

        col = box.column(align=True)
        col.prop(props, "graphviz_quiet")

        if engine in {'neato', 'fdp', 'sfdp'}:
            col.prop(props, "graphviz_dimension")
            box.label(text="Native 3D requires scigraphs-utils 0.1.1 or newer.", icon='INFO')

        if engine == 'dot':
            dot_box = box.box()
            dot_box.label(text="Dot Parameters", icon='SORT_ASC')
            col = dot_box.column(align=True)
            col.prop(props, "graphviz_dot_directed")
            col.prop(props, "graphviz_dot_rankdir")
            col.prop(props, "graphviz_dot_ranksep")
            col.prop(props, "graphviz_dot_nodesep")
            col.prop(props, "graphviz_dot_splines")
        elif engine == 'neato':
            neato_box = box.box()
            neato_box.label(text="Neato Parameters", icon='FORCE_LENNARDJONES')
            col = neato_box.column(align=True)
            col.prop(props, "graphviz_neato_mode")
            col.prop(props, "graphviz_neato_model")
            col.prop(props, "graphviz_neato_start")
            col.prop(props, "graphviz_neato_maxiter")
        elif engine == 'fdp':
            fdp_box = box.box()
            fdp_box.label(text="FDP Parameters", icon='FORCE_FORCE')
            col = fdp_box.column(align=True)
            col.prop(props, "graphviz_fdp_start")
            col.prop(props, "sfdp_k", text="K")
            col.prop(props, "sfdp_maxiter", text="Max Iterations")
            col.prop(props, "sfdp_overlap", text="Overlap")
            if props.sfdp_overlap == 'prism':
                col.prop(props, "sfdp_overlap_scaling")
        elif engine == 'sfdp':
            sfdp_box = box.box()
            sfdp_box.label(text="SFDP Parameters", icon='FORCE_VORTEX')
            col = sfdp_box.column(align=True)
            col.prop(props, "sfdp_k")
            col.prop(props, "sfdp_repulsive_force")
            col.prop(props, "sfdp_maxiter")
            col.prop(props, "sfdp_smoothing")
            col.prop(props, "sfdp_quadtree")
            col.prop(props, "sfdp_levels")
            col.prop(props, "sfdp_beautify")
            col.prop(props, "sfdp_overlap")
            if props.sfdp_overlap == 'prism':
                col.prop(props, "sfdp_overlap_scaling")
        elif engine == 'twopi':
            twopi_box = box.box()
            twopi_box.label(text="Twopi Parameters", icon='ORIENTATION_VIEW')
            col = twopi_box.column(align=True)
            col.prop(props, "graphviz_twopi_root")
            col.prop(props, "graphviz_twopi_ranksep")
        elif engine == 'circo':
            circo_box = box.box()
            circo_box.label(text="Circo Parameters", icon='MESH_CIRCLE')
            circo_box.column(align=True).prop(props, "graphviz_circo_mindist")
        elif engine == 'osage':
            osage_box = box.box()
            osage_box.label(text="Osage Parameters", icon='PACKAGE')
            col = osage_box.column(align=True)
            col.prop(props, "graphviz_osage_pack")
            col.prop(props, "graphviz_osage_packmode")
        elif engine == 'patchwork':
            info_box = box.box()
            info_box.label(text="Patchwork exposes no dedicated wrapper parameters.", icon='INFO')

        self._draw_graphviz_advanced_attrs(box, props)

    def _draw_graphviz_advanced_attrs(self, box, props):
        """Draw pass-through Graphviz attribute fields."""
        box.separator()
        adv_box = box.box()
        adv_box.label(text="Advanced Graphviz Attributes", icon='SETTINGS')
        col = adv_box.column(align=True)
        col.prop(props, "graphviz_extra_graph_attrs")
        col.prop(props, "graphviz_node_attrs")
        col.prop(props, "graphviz_edge_attrs")


class SCIGRAPHS_PT_layout_splitter(bpy.types.Panel):
    """Network Splitter 3D - Split layouts into Z-layers."""
    bl_label = "Network Splitter 3D"
    bl_parent_id = "SCIGRAPHS_PT_layout"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw_header(self, context):
        self.layout.label(text="", icon='SNAP_VERTEX')
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        obj = context.active_object
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        # Info box
        box = layout.box()
        box.label(text="Split layout into distinct Z-layers", icon='INFO')
        box.label(text="Apply after any 2D/3D layout algorithm")
        
        # Current status
        if obj and "splitter_num_layers" in obj:
            row = box.row()
            row.alert = False
            row.label(text=f"Current: {obj['splitter_num_layers']} layers ({obj.get('splitter_criterion', 'N/A')})", icon='CHECKMARK')
        
        # Split Criterion
        layout.separator()
        box = layout.box()
        box.label(text="Split Criterion", icon='FILTER')
        
        col = box.column(align=True)
        col.prop(props, "splitter_criterion", text="")
        
        # Criterion-specific settings
        if props.splitter_criterion == 'COMMUNITY':
            col.separator()
            col.prop(props, "splitter_community_algorithm", text="Algorithm")
            col.prop(props, "splitter_community_resolution", text="Resolution")
            
        elif props.splitter_criterion == 'ATTRIBUTE':
            col.separator()
            col.prop(props, "splitter_attribute", text="Attribute")
            
        elif props.splitter_criterion == 'DEGREE':
            col.separator()
            col.prop(props, "splitter_degree_bins", text="Bins")
            
        elif props.splitter_criterion == 'CENTRALITY':
            col.separator()
            col.prop(props, "splitter_centrality_bins", text="Bins")
        
        # Layer Settings
        layout.separator()
        box = layout.box()
        box.label(text="Layer Settings", icon='OUTLINER_OB_EMPTY')
        
        col = box.column(align=True)
        col.prop(props, "splitter_layer_height", text="Layer Height")
        col.prop(props, "splitter_base_z", text="Base Z")
        col.prop(props, "splitter_layer_order", text="Order")
        
        # Options
        layout.separator()
        box = layout.box()
        box.label(text="Options", icon='OPTIONS')
        
        col = box.column(align=True)
        col.prop(props, "splitter_preserve_xy", text="Preserve XY Positions")
        col.prop(props, "splitter_center_layers", text="Center Each Layer")
        col.prop(props, "splitter_scale_by_size", text="Scale by Layer Size")
        
        # Edge handling
        col.separator()
        col.prop(props, "splitter_inter_layer_edges", text="Inter-Layer Edges")
        
        # Action buttons
        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("scigraphs.network_splitter_3d", text="Split Network", icon='SNAP_VERTEX')
        row.operator("scigraphs.reset_splitter", text="", icon='LOOP_BACK')
        
        # Layer info
        if obj and "splitter_num_layers" in obj:
            layout.separator()
            info_box = layout.box()
            info_box.scale_y = 0.8
            info_box.label(text=f"Layers: {obj['splitter_num_layers']}")
            if "splitter_criterion" in obj:
                info_box.label(text=f"Criterion: {obj['splitter_criterion']}")


def register():
    bpy.utils.register_class(SCIGRAPHS_PT_layout)
    bpy.utils.register_class(SCIGRAPHS_PT_layout_algorithm)
    bpy.utils.register_class(SCIGRAPHS_PT_layout_settings)
    bpy.utils.register_class(SCIGRAPHS_PT_layout_algorithm_params)
    bpy.utils.register_class(SCIGRAPHS_PT_layout_interactive)
    bpy.utils.register_class(SCIGRAPHS_PT_layout_splitter)


def unregister():
    bpy.utils.unregister_class(SCIGRAPHS_PT_layout_splitter)
    bpy.utils.unregister_class(SCIGRAPHS_PT_layout_interactive)
    bpy.utils.unregister_class(SCIGRAPHS_PT_layout_algorithm_params)
    bpy.utils.unregister_class(SCIGRAPHS_PT_layout_settings)
    bpy.utils.unregister_class(SCIGRAPHS_PT_layout_algorithm)
    bpy.utils.unregister_class(SCIGRAPHS_PT_layout)

