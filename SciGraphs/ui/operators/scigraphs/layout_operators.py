# Layout and positioning operators

import bpy
import numpy as np
from ....core import layout, geometry
from ....properties.layout_properties import LAYOUT_PROPERTIES

try:
    from ... import gpu_preview
except Exception:  # noqa: BLE001 - preview module is optional
    gpu_preview = None


LAYOUT_PARAMETER_NAMES = tuple(LAYOUT_PROPERTIES.keys())


def _as_operator_property(property_def):
    """Clone a PropertyDeferred for operator use without animation support."""
    factory = getattr(property_def, "function", None)
    keywords = getattr(property_def, "keywords", None)
    if factory is None or keywords is None:
        return property_def

    keywords = dict(keywords)
    options = set(keywords.get("options", set()))
    options.discard('ANIMATABLE')
    keywords["options"] = options
    return factory(**keywords)


OPERATOR_LAYOUT_PROPERTIES = {
    name: _as_operator_property(property_def)
    for name, property_def in LAYOUT_PROPERTIES.items()
}


class SCIGRAPHS_OT_ApplyLayout(bpy.types.Operator):
    """Apply selected layout algorithm to graph."""
    bl_idname = "scigraphs.apply_layout"
    bl_label = "Apply Layout"
    bl_description = "Recalculate node positions using the selected layout algorithm"
    bl_options = {'REGISTER', 'UNDO'}
    
    algorithm: bpy.props.EnumProperty(
        name="Algorithm",
        options={'SKIP_SAVE'},
        items=[
            # === 2D LAYOUTS ===
            ('GRID', "Grid (2D)", "Arrange nodes in a 2D grid (instant)"),
            ('SPRING', "Spring (2D - NetworkX)", "Force-directed 2D layout (slow)"),
            ('FORCEATLAS2', "ForceAtlas2 (2D)", "Gephi's algorithm, requires optional fa2 package (medium)"),
            ('IGRAPH_DRL_2D', "DrL (2D - igraph)", "Distributed Recursive Layout 2D, very fast for huge graphs (very fast)"),
            ('IGRAPH_DH', "Davidson-Harel (2D - igraph)", "Simulated annealing approach (medium)"),
            ('IGRAPH_GRAPHOPT', "Graphopt (2D - igraph)", "Energy-based optimization (fast)"),
            ('CIRCLE_PACKING', "Circle Packing (2D - Koebe)", "Koebe theorem: tangent circles for planar graphs (medium)"),

            # === 3D GEOMETRIC LAYOUTS ===
            ('RANDOM', "Random (3D)", "Distribute nodes randomly in 3D space (instant)"),
            ('SPHERE', "Sphere (3D)", "Distribute nodes on sphere surface using Fibonacci algorithm (instant)"),
            ('SPIRAL_3D', "Spiral (3D)", "Arrange nodes in upward spiral pattern (instant)"),
            ('HELIX', "Helix (3D)", "Double helix pattern like DNA structure (instant)"),
            ('CUBE', "Cube (3D)", "Distribute nodes in and on a cube (instant)"),

            # === 3D GRAPH-BASED LAYOUTS ===
            ('SPECTRAL_3D', "Spectral (3D)", "Use graph Laplacian eigenvectors for 3D positioning (fast)"),
            ('MDS_3D', "MDS (3D)", "Multidimensional scaling using shortest path distances (medium)"),
            ('HIERARCHICAL_3D', "Hierarchical (3D)", "Tree-like hierarchy in layers (fast)"),
            ('BIPARTITE_3D', "Bipartite (3D)", "Two parallel planes for bipartite graphs (fast)"),

            # === 3D FORCE-DIRECTED LAYOUTS ===
            ('YIFAN_HU', "Yifan Hu (3D)", "Scalable force-directed placement via bundled scigraphs-utils"),
            ('IGRAPH_DRL', "DrL (3D - igraph)", "Distributed Recursive Layout for huge graphs 100k+ (very fast)"),
            ('IGRAPH_FR', "Fruchterman-Reingold (3D - igraph)", "Classic force-directed in 3D (fast)"),
            ('IGRAPH_KK', "Kamada-Kawai (3D - igraph)", "Deterministic 3D layout, reproducible (medium)"),
            ('IGRAPH_LGL', "LGL (3D - igraph)", "Large Graph Layout, optimized for massive graphs (fast)"),
            ('SPRING_3D', "Spring (3D - NetworkX)", "Force-directed 3D layout (very slow)"),

            # === GRAPHVIZ LAYOUTS (scigraphs-utils) ===
            ('GRAPHVIZ_DOT', "Graphviz Dot (2D)", "Hierarchical layout via bundled scigraphs-utils"),
            ('GRAPHVIZ_NEATO', "Graphviz Neato (2D/3D)", "Spring model layout via bundled scigraphs-utils"),
            ('GRAPHVIZ_FDP', "Graphviz FDP (2D/3D)", "Force-directed placement via bundled scigraphs-utils"),
            ('GRAPHVIZ_SFDP', "Graphviz SFDP (2D/3D)", "Scalable force-directed placement via bundled scigraphs-utils"),
            ('GRAPHVIZ_TWOPI', "Graphviz Twopi (2D)", "Radial layout via bundled scigraphs-utils"),
            ('GRAPHVIZ_CIRCO', "Graphviz Circo (2D)", "Circular layout via bundled scigraphs-utils"),
            ('GRAPHVIZ_OSAGE', "Graphviz Osage (2D)", "Cluster layout via bundled scigraphs-utils"),
            ('GRAPHVIZ_PATCHWORK', "Graphviz Patchwork (2D)", "Patchwork layout via bundled scigraphs-utils"),

            # === DIRECTED GRAPH LAYOUTS ===
            ('SUGIYAMA', "Sugiyama/Layered (2D - Directed)", "Hierarchical DAG layout, minimizes crossings (fast)"),
            ('CIRCULAR_HIERARCHY', "Circular Hierarchy (2D - Directed)", "Concentric circles from roots (fast)"),
        ],
        default='YIFAN_HU',
    )
    
    scale: bpy.props.FloatProperty(
        name="Scale",
        default=5.0,
        min=0.1,
        max=100.0,
        options={'SKIP_SAVE'},
    )
    
    iterations: bpy.props.IntProperty(
        name="Iterations",
        default=50,
        min=1,
        max=1000,
        options={'SKIP_SAVE'},
    )

    show_full_parameter_dialog: bpy.props.BoolProperty(
        default=False,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    
    def invoke(self, context, event):
        props = context.scene.scigraphs
        self.algorithm = props.layout_algorithm
        self.scale = props.layout_scale
        self.iterations = props.iterations
        self._copy_layout_parameters_from_scene(props)
        self.show_full_parameter_dialog = True
        return context.window_manager.invoke_props_dialog(self, width=620)

    def draw(self, context):
        props = self
        ui_layout = self.layout
        ui_layout.use_property_split = True
        ui_layout.use_property_decorate = False

        box = ui_layout.box()
        box.use_property_decorate = False
        box.label(text="Layout", icon='MOD_PARTICLES')
        col = box.column(align=True)
        col.use_property_decorate = False
        col.prop(self, "algorithm")
        col.prop(self, "scale")
        col.prop(self, "iterations")

        self._draw_algorithm_parameters(ui_layout, props, self.algorithm)

    def _copy_layout_parameters_from_scene(self, props):
        for prop_name in LAYOUT_PARAMETER_NAMES:
            if hasattr(props, prop_name):
                setattr(self, prop_name, getattr(props, prop_name))

    def _sync_layout_parameters_to_scene(self, props):
        props.layout_algorithm = self.algorithm
        props.layout_scale = self.scale
        props.iterations = self.iterations
        for prop_name in LAYOUT_PARAMETER_NAMES:
            if hasattr(props, prop_name) and hasattr(self, prop_name):
                setattr(props, prop_name, getattr(self, prop_name))

    def _draw_algorithm_parameters(self, ui_layout, props, algorithm):
        if algorithm in ['SPRING', 'SPRING_3D']:
            self._draw_props_box(
                ui_layout, "Spring Parameters", 'FORCE_LENNARDJONES', props,
                ["repulsion_strength", "attraction_strength", "gravity_strength",
                 "edge_distance", "initial_temperature", "cooling_factor"]
            )
        elif algorithm == 'FORCEATLAS2':
            self._draw_props_box(
                ui_layout, "ForceAtlas2 Parameters", 'FORCE_FORCE', props,
                ["fa2_scaling_ratio", "fa2_gravity", "fa2_strong_gravity",
                 "fa2_lin_log_mode", "fa2_barnes_hut_optimize"]
            )
            if props.fa2_barnes_hut_optimize:
                self._draw_props_box(ui_layout, "Barnes-Hut", 'MOD_PARTICLES', props, ["fa2_barnes_hut_theta"])
            self._draw_props_box(ui_layout, "ForceAtlas2 Quality", 'SETTINGS', props,
                                 ["fa2_jitter_tolerance", "fa2_edge_weight_influence"])
        elif algorithm == 'IGRAPH_FR':
            self._draw_props_box(
                ui_layout, "Fruchterman-Reingold Parameters", 'FORCE_LENNARDJONES', props,
                ["igraph_fr_start_temp", "igraph_fr_coolexp", "igraph_fr_maxdelta",
                 "igraph_fr_area", "igraph_fr_repulserad"]
            )
        elif algorithm == 'IGRAPH_KK':
            self._draw_props_box(
                ui_layout, "Kamada-Kawai Parameters", 'DRIVER_DISTANCE', props,
                ["igraph_kk_maxiter", "igraph_kk_epsilon", "igraph_kk_kkconst"]
            )
        elif algorithm in ['IGRAPH_DRL', 'IGRAPH_DRL_2D']:
            self._draw_igraph_drl_parameters(ui_layout, props)
        elif algorithm == 'IGRAPH_LGL':
            self._draw_props_box(
                ui_layout, "LGL Parameters", 'STICKY_UVS_LOC', props,
                ["igraph_lgl_maxiter", "igraph_lgl_maxdelta", "igraph_lgl_area",
                 "igraph_lgl_coolexp", "igraph_lgl_repulserad", "igraph_lgl_cellsize"]
            )
        elif algorithm == 'IGRAPH_DH':
            self._draw_props_box(
                ui_layout, "Davidson-Harel Parameters", 'FORCE_HARMONIC', props,
                ["igraph_dh_maxiter", "igraph_dh_fineiter", "igraph_dh_cool_fact",
                 "igraph_dh_weight_node_dist", "igraph_dh_weight_border",
                 "igraph_dh_weight_edge_lengths", "igraph_dh_weight_edge_crossings",
                 "igraph_dh_weight_node_edge_dist"]
            )
        elif algorithm == 'IGRAPH_GRAPHOPT':
            self._draw_props_box(
                ui_layout, "GraphOpt Parameters", 'GRAPH', props,
                ["igraph_graphopt_niter", "igraph_graphopt_node_charge",
                 "igraph_graphopt_node_mass", "igraph_graphopt_spring_length",
                 "igraph_graphopt_spring_constant", "igraph_graphopt_max_sa_movement"]
            )
        elif algorithm == 'YIFAN_HU':
            self._draw_yifan_hu_parameters(ui_layout, props)
        elif algorithm.startswith('GRAPHVIZ_'):
            self._draw_graphviz_parameters(ui_layout, props, algorithm)

    def _draw_props_box(self, ui_layout, label, icon, props, prop_names):
        box = ui_layout.box()
        box.use_property_decorate = False
        box.label(text=label, icon=icon)
        col = box.column(align=True)
        col.use_property_decorate = False
        for prop_name in prop_names:
            col.prop(props, prop_name)

    def _draw_igraph_drl_parameters(self, ui_layout, props):
        self._draw_props_box(ui_layout, "DrL Global Parameters", 'MOD_PARTICLES', props, ["igraph_drl_edge_cut"])
        for label, phase, icon in [
            ("Init", "init", 'PLAY'),
            ("Liquid", "liquid", 'MOD_FLUIDSIM'),
            ("Expansion", "expansion", 'FULLSCREEN_ENTER'),
            ("Cooldown", "cooldown", 'FREEZE'),
            ("Crunch", "crunch", 'MESH_ICOSPHERE'),
            ("Simmer", "simmer", 'LIGHT_SUN'),
        ]:
            self._draw_props_box(
                ui_layout, f"DrL {label}", icon, props,
                [f"igraph_drl_{phase}_iterations", f"igraph_drl_{phase}_temperature",
                 f"igraph_drl_{phase}_attraction", f"igraph_drl_{phase}_damping_mult"]
            )

    def _draw_yifan_hu_parameters(self, ui_layout, props):
        self._draw_props_box(ui_layout, "Yifan Hu Dimensions", 'ORIENTATION_LOCAL', props, ["sfdp_dim", "graphviz_quiet"])
        if props.sfdp_dim == '2Z':
            self._draw_props_box(ui_layout, "Z Depth Generation", 'EMPTY_AXIS', props, ["sfdp_z_method", "sfdp_z_scale"])
        self._draw_sfdp_parameters(ui_layout, props)
        self._draw_graphviz_advanced_parameters(ui_layout, props)

    def _draw_graphviz_parameters(self, ui_layout, props, algorithm):
        engine = algorithm.replace('GRAPHVIZ_', '').lower()
        common = ["graphviz_quiet"]
        if engine in {'neato', 'fdp', 'sfdp'}:
            common.append("graphviz_dimension")
        self._draw_props_box(ui_layout, f"Graphviz {engine} Common", 'GRAPH', props, common)

        if engine == 'dot':
            self._draw_props_box(
                ui_layout, "Dot Parameters", 'SORT_ASC', props,
                ["graphviz_dot_directed", "graphviz_dot_rankdir", "graphviz_dot_ranksep",
                 "graphviz_dot_nodesep", "graphviz_dot_splines"]
            )
        elif engine == 'neato':
            self._draw_props_box(
                ui_layout, "Neato Parameters", 'FORCE_LENNARDJONES', props,
                ["graphviz_neato_mode", "graphviz_neato_model",
                 "graphviz_neato_start", "graphviz_neato_maxiter"]
            )
        elif engine == 'fdp':
            self._draw_props_box(ui_layout, "FDP Parameters", 'FORCE_FORCE', props,
                                 ["graphviz_fdp_start", "sfdp_k", "sfdp_maxiter", "sfdp_overlap"])
            if props.sfdp_overlap == 'prism':
                self._draw_props_box(ui_layout, "FDP Overlap", 'SELECT_SUBTRACT', props, ["sfdp_overlap_scaling"])
        elif engine == 'sfdp':
            self._draw_sfdp_parameters(ui_layout, props)
        elif engine == 'twopi':
            self._draw_props_box(ui_layout, "Twopi Parameters", 'ORIENTATION_VIEW', props,
                                 ["graphviz_twopi_root", "graphviz_twopi_ranksep"])
        elif engine == 'circo':
            self._draw_props_box(ui_layout, "Circo Parameters", 'MESH_CIRCLE', props, ["graphviz_circo_mindist"])
        elif engine == 'osage':
            self._draw_props_box(ui_layout, "Osage Parameters", 'PACKAGE', props,
                                 ["graphviz_osage_pack", "graphviz_osage_packmode"])
        elif engine == 'patchwork':
            box = ui_layout.box()
            box.label(text="Patchwork exposes no dedicated wrapper parameters", icon='INFO')

        self._draw_graphviz_advanced_parameters(ui_layout, props)

    def _draw_sfdp_parameters(self, ui_layout, props):
        self._draw_props_box(ui_layout, "SFDP Force Parameters", 'FORCE_CHARGE', props,
                             ["sfdp_k", "sfdp_repulsive_force", "sfdp_maxiter"])
        self._draw_props_box(ui_layout, "SFDP Quality", 'MODIFIER', props,
                             ["sfdp_smoothing", "sfdp_quadtree", "sfdp_levels", "sfdp_beautify"])
        self._draw_props_box(ui_layout, "SFDP Overlap", 'SELECT_SUBTRACT', props, ["sfdp_overlap"])
        if props.sfdp_overlap == 'prism':
            self._draw_props_box(ui_layout, "Prism Overlap", 'SELECT_SUBTRACT', props, ["sfdp_overlap_scaling"])

    def _draw_graphviz_advanced_parameters(self, ui_layout, props):
        self._draw_props_box(
            ui_layout, "Advanced Graphviz Attributes", 'SETTINGS', props,
            ["graphviz_extra_graph_attrs", "graphviz_node_attrs", "graphviz_edge_attrs"]
        )
    
    def execute(self, context):
        self.show_full_parameter_dialog = False
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}

        self._sync_layout_parameters_to_scene(props)
        
        success = layout.apply_graph_layout(
            obj,
            algorithm=self.algorithm,
            iterations=self.iterations,
            scale=self.scale,
            props=self
        )
        
        if not success:
            self.report({'ERROR'}, "Layout calculation failed")
            return {'CANCELLED'}
        
        geometry.update_node_positions_from_property(obj)
        geometry.rebuild_edges(obj)
        
        self.report({'INFO'}, f"Layout '{self.algorithm}' applied")
        return {'FINISHED'}


SCIGRAPHS_OT_ApplyLayout.__annotations__.update(OPERATOR_LAYOUT_PROPERTIES)


class SCIGRAPHS_OT_ExecuteLayoutStep(bpy.types.Operator):
    """Execute layout iterations frame-by-frame in Gephi style."""
    bl_idname = "scigraphs.execute_layout_step"
    bl_label = "Execute"
    bl_description = "Execute layout iterations and cache them frame by frame in the timeline range (Gephi-style)"
    
    _timer = None
    _current_frame = 0
    _start_frame = 0
    _end_frame = 0
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            props = context.scene.scigraphs
            obj = context.active_object
            
            if not obj or "num_nodes" not in obj:
                self.cancel(context)
                self.report({'ERROR'}, "Graph object lost during execution")
                return {'CANCELLED'}
            
            # Check if we've reached the end frame
            if self._current_frame > self._end_frame:
                self.cancel(context)
                self.report({'INFO'}, f"Layout complete: {self._end_frame - self._start_frame + 1} frames cached")
                # Play animation
                bpy.ops.screen.animation_play()
                return {'FINISHED'}
            
            # Execute one iteration with advanced parameters
            success, energy = layout.execute_layout_iteration(
                obj,
                algorithm=props.layout_algorithm,
                scale=props.layout_scale,
                current_frame=self._current_frame,
                repulsion=props.repulsion_strength,
                attraction=props.attraction_strength,
                gravity=props.gravity_strength,
                cooling=props.cooling_factor,
                initial_temp=props.initial_temperature,
                edge_dist=props.edge_distance,
                auto_stop=props.auto_stop_threshold
            )
            
            if not success:
                self.cancel(context)
                iteration = obj.get("layout_iteration", 0)
                self.report({'INFO'}, f"Layout computation complete at iteration {iteration}")
                # Play animation
                bpy.ops.screen.animation_play()
                return {'FINISHED'}
            
            # Update visualization if enabled
            if props.update_viewport:
                geometry.update_node_positions_from_property(obj)

                # Live preview mode: with the GPU preview active we only push
                # the new node positions and let the preview redraw. Edge
                # topology does not change during a layout, so we skip the
                # costly per-frame edge rebuild (and any Geometry Nodes work).
                live_preview = gpu_preview is not None and gpu_preview.is_enabled()
                if live_preview:
                    gpu_preview.invalidate(obj)
                else:
                    geometry.rebuild_edges(obj)

                # Force viewport update
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
            
            # Show progress with energy
            progress = ((self._current_frame - self._start_frame) / 
                       (self._end_frame - self._start_frame + 1) * 100)
            iteration = obj.get("layout_iteration", 0)
            self.report({'INFO'}, f"Frame {self._current_frame}/{self._end_frame} - Iter {iteration} - Energy {energy:.2f} ({progress:.0f}%)")
            
            # Move to next frame
            self._current_frame += 1
            
            return {'RUNNING_MODAL'}
        
        elif event.type == 'ESC':
            self.cancel(context)
            self.report({'INFO'}, "Layout execution cancelled")
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        # Get timeline range
        scene = context.scene
        self._start_frame = scene.frame_start
        self._end_frame = scene.frame_end
        self._current_frame = self._start_frame
        
        # Set to start frame
        scene.frame_set(self._start_frame)
        
        # Reset layout iteration counter
        obj["layout_iteration"] = 0
        
        # Add timer with configurable speed
        wm = context.window_manager
        self._timer = wm.event_timer_add(props.execution_speed, window=context.window)
        wm.modal_handler_add(self)
        
        num_frames = self._end_frame - self._start_frame + 1
        self.report({'INFO'}, f"Starting layout execution for {num_frames} frames...")
        
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        if self._timer:
            wm = context.window_manager
            wm.event_timer_remove(self._timer)


class SCIGRAPHS_OT_ResetLayout(bpy.types.Operator):
    """Reset layout to initial random positions."""
    bl_idname = "scigraphs.reset_layout"
    bl_label = "Reset"
    bl_description = "Reset layout to initial random positions and clear timeline cache"
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        # Reset iteration counter
        obj["layout_iteration"] = 0
        
        # Generate new random positions
        num_nodes = obj["num_nodes"]
        scale = props.layout_scale
        random_pos = np.random.rand(num_nodes, 3) * scale
        obj["node_positions"] = random_pos.flatten().tolist()
        
        # Update mesh
        geometry.update_node_positions_from_property(obj)
        geometry.rebuild_edges(obj)
        
        # Clear all keyframes for this object
        if obj.animation_data and obj.animation_data.action:
            obj.animation_data_clear()
        
        if obj.data.animation_data and obj.data.animation_data.action:
            obj.data.animation_data_clear()
        
        # Reset to frame 1
        context.scene.frame_set(1)
        
        self.report({'INFO'}, "Layout reset to random positions")
        return {'FINISHED'}


class SCIGRAPHS_OT_BakeAnimation(bpy.types.Operator):
    """Create animation of layout simulation."""
    bl_idname = "scigraphs.bake_animation"
    bl_label = "Bake Animation"
    bl_description = "Create an animation of the layout simulation (automatic)"
    
    _timer = None
    _frame = 0
    _max_frames = 100
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            obj = context.active_object
            props = context.scene.scigraphs
            
            if self._frame >= self._max_frames:
                self.cancel(context)
                self.report({'INFO'}, "Animation baked")
                return {'FINISHED'}
            
            layout.apply_graph_layout(
                obj,
                algorithm=props.layout_algorithm,
                iterations=1,
                scale=props.layout_scale
            )
            
            geometry.update_node_positions_from_property(obj)
            
            obj.keyframe_insert(data_path='["node_positions"]', frame=self._frame)
            
            self._frame += 1
            context.scene.frame_set(self._frame)
        
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        
        self._frame = 0
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)


class SCIGRAPHS_OT_NetworkSplitter3D(bpy.types.Operator):
    """Split network layout into distinct Z-layers."""
    bl_idname = "scigraphs.network_splitter_3d"
    bl_label = "Network Splitter 3D"
    bl_description = "Split network layout into Z-layers by community, degree, attribute, etc."
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and "num_nodes" in obj and obj.get("node_positions")
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        success, num_layers, layer_info = layout.apply_network_splitter_3d(
            obj,
            criterion=props.splitter_criterion,
            attribute=props.splitter_attribute if props.splitter_criterion == 'ATTRIBUTE' else None,
            layer_height=props.splitter_layer_height,
            layer_order=props.splitter_layer_order,
            degree_bins=props.splitter_degree_bins,
            centrality_bins=props.splitter_centrality_bins,
            community_algorithm=props.splitter_community_algorithm,
            resolution=props.splitter_community_resolution,
            preserve_xy=props.splitter_preserve_xy,
            center_layers=props.splitter_center_layers,
            scale_by_size=props.splitter_scale_by_size,
            base_z=props.splitter_base_z,
        )
        
        if success:
            # Update mesh visualization
            geometry.update_node_positions_from_property(obj)
            geometry.rebuild_edges(obj)
            
            self.report({'INFO'}, f"Network split into {num_layers} Z-layers")
            
            # Force viewport update
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        else:
            self.report({'ERROR'}, "Network splitting failed")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class SCIGRAPHS_OT_ResetSplitter(bpy.types.Operator):
    """Reset Z positions to flat plane."""
    bl_idname = "scigraphs.reset_splitter"
    bl_label = "Flatten Z"
    bl_description = "Reset all Z positions to zero (flatten the layout)"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and "num_nodes" in obj and obj.get("node_positions")
    
    def execute(self, context):
        obj = context.active_object
        
        pos_flat = obj.get("node_positions", [])
        if not pos_flat:
            self.report({'ERROR'}, "No positions found")
            return {'CANCELLED'}
        
        positions = np.array(pos_flat).reshape(-1, 3)
        positions[:, 2] = 0  # Set all Z to 0
        
        obj["node_positions"] = positions.flatten().tolist()
        
        # Clear splitter metadata
        if "splitter_criterion" in obj:
            del obj["splitter_criterion"]
        if "splitter_num_layers" in obj:
            del obj["splitter_num_layers"]
        if "splitter_layer_assignments" in obj:
            del obj["splitter_layer_assignments"]
        
        # Update mesh
        geometry.update_node_positions_from_property(obj)
        geometry.rebuild_edges(obj)
        
        self.report({'INFO'}, "Layout flattened to Z=0")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(SCIGRAPHS_OT_ApplyLayout)
    bpy.utils.register_class(SCIGRAPHS_OT_ExecuteLayoutStep)
    bpy.utils.register_class(SCIGRAPHS_OT_ResetLayout)
    bpy.utils.register_class(SCIGRAPHS_OT_BakeAnimation)
    bpy.utils.register_class(SCIGRAPHS_OT_NetworkSplitter3D)
    bpy.utils.register_class(SCIGRAPHS_OT_ResetSplitter)


def unregister():
    bpy.utils.unregister_class(SCIGRAPHS_OT_ResetSplitter)
    bpy.utils.unregister_class(SCIGRAPHS_OT_NetworkSplitter3D)
    bpy.utils.unregister_class(SCIGRAPHS_OT_BakeAnimation)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ResetLayout)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ExecuteLayoutStep)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ApplyLayout)

