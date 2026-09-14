# Layout and positioning operators

import bpy
import numpy as np
from scigraphs_core import layout
from ....core import geometry
from scigraphs_core.mesh.mesh_utils import layout_edge_pairs
from ....properties.layout_properties import LAYOUT_PROPERTIES

try:
    from ... import gpu_render as gpu_preview
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
    bl_idname = "scigraphs.apply_layout"
    bl_label = "Apply Layout"
    bl_description = "Recalculate node positions using the selected layout algorithm"
    bl_options = {'REGISTER', 'UNDO'}
    
    algorithm: bpy.props.EnumProperty(
        name="Algorithm",
        options={'SKIP_SAVE'},
        items=[
            ('GRID', "Grid (2D)", "Arrange nodes in a 2D grid (instant)"),
            ('SPRING', "Spring (2D - NetworkX)", "Force-directed 2D layout (slow)"),
            ('FORCEATLAS2', "ForceAtlas2 (2D)", "Gephi's algorithm, requires optional fa2 package (medium)"),
            ('IGRAPH_DRL_2D', "DrL (2D - igraph)", "Distributed Recursive Layout 2D, very fast for huge graphs (very fast)"),
            ('IGRAPH_DH', "Davidson-Harel (2D - igraph)", "Simulated annealing approach (medium)"),
            ('IGRAPH_GRAPHOPT', "Graphopt (2D - igraph)", "Energy-based optimization (fast)"),
            ('CIRCLE_PACKING', "Circle Packing (2D - Koebe)", "Koebe theorem: tangent circles for planar graphs (medium)"),

            ('RANDOM', "Random (3D)", "Distribute nodes randomly in 3D space (instant)"),
            ('SPHERE', "Sphere (3D)", "Distribute nodes on sphere surface using Fibonacci algorithm (instant)"),
            ('SPIRAL_3D', "Spiral (3D)", "Arrange nodes in upward spiral pattern (instant)"),
            ('HELIX', "Helix (3D)", "Double helix pattern like DNA structure (instant)"),
            ('CUBE', "Cube (3D)", "Distribute nodes in and on a cube (instant)"),

            ('SPECTRAL_3D', "Spectral (3D)", "Graph Laplacian eigenvectors. Solved per connected component"),
            ('MDS_3D', "MDS (3D)", "Multidimensional scaling using shortest path distances (medium)"),
            ('HIERARCHICAL_3D', "Hierarchical (3D)", "Tree-like hierarchy in layers (fast)"),
            ('BIPARTITE_3D', "Bipartite (3D)", "Two parallel planes for bipartite graphs (fast)"),

            ('YIFAN_HU', "Yifan Hu (3D)", "Scalable force-directed placement via bundled scigraphs-utils"),
            ('IGRAPH_DRL', "DrL (3D - igraph)", "Distributed Recursive Layout for huge graphs 100k+ (very fast)"),
            ('IGRAPH_FR', "Fruchterman-Reingold (3D - igraph)", "Classic force-directed in 3D (fast)"),
            ('IGRAPH_KK', "Kamada-Kawai (3D - igraph)", "Deterministic 3D layout, reproducible (medium)"),
            ('IGRAPH_LGL', "LGL (2D - igraph)", "Large Graph Layout, planar output, optimized for massive graphs"),
            ('SPRING_3D', "Spring (3D - NetworkX)", "Force-directed 3D layout (very slow)"),

            ('GRAPHVIZ_DOT', "Graphviz Dot (2D)", "Hierarchical layout via bundled scigraphs-utils"),
            ('GRAPHVIZ_NEATO', "Graphviz Neato (2D/3D)", "Spring model layout via bundled scigraphs-utils"),
            ('GRAPHVIZ_FDP', "Graphviz FDP (2D/3D)", "Force-directed placement via bundled scigraphs-utils"),
            ('GRAPHVIZ_SFDP', "Graphviz SFDP (2D/3D)", "Scalable force-directed placement via bundled scigraphs-utils"),
            ('GRAPHVIZ_TWOPI', "Graphviz Twopi (2D)", "Radial layout via bundled scigraphs-utils"),
            ('GRAPHVIZ_CIRCO', "Graphviz Circo (2D)", "Circular layout via bundled scigraphs-utils"),
            ('GRAPHVIZ_OSAGE', "Graphviz Osage (2D)", "Cluster layout via bundled scigraphs-utils"),
            ('GRAPHVIZ_PATCHWORK', "Graphviz Patchwork (2D)", "Patchwork layout via bundled scigraphs-utils"),

            ('SUGIYAMA', "Sugiyama/Layered (2D - Directed)", "Layered DAG drawing with heuristic crossing reduction. Slower than the force layouts"),
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
            props=self,
            # The layout package may not read a mesh, so pass the edges in.
            # An object that keeps its topology in mesh.edges rather than in
            # edges_data otherwise arrives edgeless and returns isolated points.
            edge_pairs=layout_edge_pairs(obj),
        )
        
        if not success:
            self.report({'ERROR'}, "Layout calculation failed")
            return {'CANCELLED'}
        
        geometry.update_node_positions_from_property(obj)
        geometry.rebuild_edges(obj)

        # After rebuild_edges, which replaces the mesh on an edges_data object.
        values = (geometry.store_packing_radii(obj)
                  if self.algorithm == 'CIRCLE_PACKING' else None)
        note = ""
        if values is not None:
            note = (self._show_packing_radii(context.scene, values)
                    + self._packing_note(obj, values))

        self.report({'INFO'}, f"Layout '{self.algorithm}' applied{note}")
        return {'FINISHED'}

    @staticmethod
    def _show_packing_radii(scene, values):
        """Draw the nodes at the radii the packing just produced, and say so.

        The preview sizes a node as ``base * (1 + norm * (mult - 1))`` over the
        attribute's own range. That is affine, not proportional, so it only
        reproduces the packing for one choice of the two: ``mult = hi / lo``
        cancels the constant term and ``base = lo`` fixes the scale. ``mult``
        stops at 32, so a wider spread than that draws the large circles short
        of tangency rather than silently claiming to be exact.

        Size gets its own attribute here rather than the shared one, which color
        and the filters also read. Sharing it meant that coloring the packing by
        anything at all resized the circles by that instead, and the tangency
        the layout had just solved for went with it.
        """
        if not hasattr(scene, "scigraphs_preview_size_attr_name"):
            return ""
        scene.scigraphs_preview_size_attr_name = "circle_radius"
        scene.scigraphs_preview_size_by_attr = True

        lo, hi = float(np.min(values)), float(np.max(values))
        if lo <= 0.0 or hi <= lo:
            return ", sized by circle_radius"

        scene.scigraphs_preview_impostor_radius = lo
        scene.scigraphs_preview_size_max_mult = hi / lo
        # Relative, since the property stores a 32-bit float and reading it back
        # differs from the double by more than any absolute epsilon worth using.
        if scene.scigraphs_preview_size_max_mult < (hi / lo) * (1.0 - 1e-4):
            return (f", radii clamped: spread is {hi / lo:.0f}x, over the size "
                    f"multiplier's ceiling")
        return ""

    @staticmethod
    def _packing_note(obj, values):
        """What the run actually produced, measured on the mesh."""
        err = geometry.packing_tangency(obj, values)
        if err is None:
            return ""
        if err < 0.01:
            return ", circles tangent"
        return (f", NOT a packing: this graph is not planar, so the layout fell "
                f"back to force relaxation ({err * 100:.0f}% median gap)")


SCIGRAPHS_OT_ApplyLayout.__annotations__.update(OPERATOR_LAYOUT_PROPERTIES)


class SCIGRAPHS_OT_ExecuteLayoutStep(bpy.types.Operator):
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
            
            if self._current_frame > self._end_frame:
                self.cancel(context)
                self.report({'INFO'}, f"Layout complete: {self._end_frame - self._start_frame + 1} frames cached")
                bpy.ops.screen.animation_play()
                return {'FINISHED'}

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
                auto_stop=props.auto_stop_threshold,
                # The layout package is Blender-free, so the scene properties are
                # passed in. Reaching for `bpy.context` there raises a NameError a
                # bare except swallows, and the algorithm runs on hard-coded defaults.
                props=props,
                # Recomputed per iteration: `rebuild_edges` replaces them each frame.
                edge_pairs=layout_edge_pairs(obj),
            )
            
            if not success:
                self.cancel(context)
                iteration = obj.get("layout_iteration", 0)
                self.report({'INFO'}, f"Layout computation complete at iteration {iteration}")
                bpy.ops.screen.animation_play()
                return {'FINISHED'}

            if props.update_viewport:
                geometry.update_node_positions_from_property(obj)

                # A layout never changes edge topology, so with the GPU preview up
                # the per-frame edge rebuild is wasted work.
                live_preview = gpu_preview is not None and gpu_preview.is_enabled()
                if live_preview:
                    # Moving nodes is the one change the batch and tree signatures
                    # cannot see, so invalidate by hand.
                    gpu_preview.invalidate_geometry(obj)
                else:
                    geometry.rebuild_edges(obj)

                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()

            progress = ((self._current_frame - self._start_frame) / 
                       (self._end_frame - self._start_frame + 1) * 100)
            iteration = obj.get("layout_iteration", 0)
            self.report({'INFO'}, f"Frame {self._current_frame}/{self._end_frame} - Iter {iteration} - Energy {energy:.2f} ({progress:.0f}%)")

            self._current_frame += 1
            
            return {'RUNNING_MODAL'}
        
        elif event.type == 'ESC':
            self.cancel(context)
            self.report({'INFO'}, "Layout execution canceled")
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        scene = context.scene
        self._start_frame = scene.frame_start
        self._end_frame = scene.frame_end
        self._current_frame = self._start_frame

        scene.frame_set(self._start_frame)
        obj["layout_iteration"] = 0

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
    bl_idname = "scigraphs.reset_layout"
    bl_label = "Reset"
    bl_description = "Reset layout to initial random positions and clear timeline cache"
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        obj["layout_iteration"] = 0

        num_nodes = obj["num_nodes"]
        scale = props.layout_scale
        random_pos = np.random.rand(num_nodes, 3) * scale
        obj["node_positions"] = random_pos.flatten().tolist()

        geometry.update_node_positions_from_property(obj)
        geometry.rebuild_edges(obj)

        # Both the object and its data can carry keyframes; clear each.
        if obj.animation_data and obj.animation_data.action:
            obj.animation_data_clear()
        
        if obj.data.animation_data and obj.data.animation_data.action:
            obj.data.animation_data_clear()
        
        context.scene.frame_set(1)
        
        self.report({'INFO'}, "Layout reset to random positions")
        return {'FINISHED'}


def bake_layout_animation(obj, scene, stages=24, iterations_per_stage=1,
                          algorithm=None, scale=None):
    """Run the layout in stages and store them as shape keys.

    Returns `(stage_count, mean_step)`. A mean step of 0 means the layout did
    not move, which happens when `apply_graph_layout` ignores `iterations`.

    Shape keys rather than keyframes on `node_positions`: Blender evaluates
    them before the modifier stack, they cost one curve per stage instead of
    three per node, and they need no handler or open window.
    
    """
    from ....api import anim as sg_anim

    props = scene.scigraphs
    algorithm = algorithm or props.layout_algorithm
    scale = props.layout_scale if scale is None else scale
    edge_pairs = layout_edge_pairs(obj)

    def positions():
        count = len(obj.data.vertices)
        flat = np.empty(count * 3, dtype=np.float64)
        obj.data.vertices.foreach_get("co", flat)
        return flat.reshape(count, 3)

    poses = [positions().copy()]
    for _ in range(max(1, int(stages)) - 1):
        layout.apply_graph_layout(
            obj,
            algorithm=algorithm,
            iterations=max(1, int(iterations_per_stage)),
            scale=scale,
            edge_pairs=edge_pairs,
        )
        geometry.update_node_positions_from_property(obj)
        poses.append(positions().copy())

    steps = [float(np.linalg.norm(b - a, axis=1).mean())
             for a, b in zip(poses, poses[1:])]
    mean_step = float(np.mean(steps)) if steps else 0.0

    first, last = scene.frame_start, scene.frame_end
    span = max(last - first, 1)
    keys = [(first + int(round(i * span / max(len(poses) - 1, 1))), pose)
            for i, pose in enumerate(poses)]
    sg_anim.positions(obj, keys)
    return len(poses), mean_step


class SCIGRAPHS_OT_BakeAnimation(bpy.types.Operator):
    bl_idname = "scigraphs.bake_animation"
    bl_label = "Bake Animation"
    bl_description = ("Bake the layout simulation into shape keys across the "
                      "scene's frame range")
    bl_options = {'REGISTER', 'UNDO'}

    stages: bpy.props.IntProperty(
        name="Stages",
        description="Poses to store. The animation interpolates between them, "
                    "so this is not the frame count",
        default=24, min=2, soft_max=120,
    )

    iterations_per_stage: bpy.props.IntProperty(
        name="Iterations per Stage",
        description="Layout iterations run between one stored pose and the next",
        default=1, min=1, soft_max=50,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and "num_nodes" in obj

    def execute(self, context):
        obj = context.active_object
        if obj is None or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}

        count, mean_step = bake_layout_animation(
            obj, context.scene,
            stages=self.stages,
            iterations_per_stage=self.iterations_per_stage)

        if mean_step <= 1e-9:
            self.report(
                {'WARNING'},
                f"{count} stages baked but the layout did not move "
                f"(mean step {mean_step:.2e}) - this algorithm may ignore the "
                f"iteration count")
            return {'FINISHED'}

        self.report({'INFO'},
                    f"Baked {count} stages, mean step {mean_step:.4f}")
        return {'FINISHED'}


class SCIGRAPHS_OT_NetworkSplitter3D(bpy.types.Operator):
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
            # Every criterion but ATTRIBUTE is a function of the edges; without
            # them the splitter reports one component per node.
            edge_pairs=layout_edge_pairs(obj),
        )
        
        if success:
            geometry.update_node_positions_from_property(obj)
            geometry.rebuild_edges(obj)

            self.report({'INFO'}, f"Network split into {num_layers} Z-layers")

            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        else:
            self.report({'ERROR'}, "Network splitting failed")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class SCIGRAPHS_OT_ResetSplitter(bpy.types.Operator):
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
        
        if "splitter_criterion" in obj:
            del obj["splitter_criterion"]
        if "splitter_num_layers" in obj:
            del obj["splitter_num_layers"]
        if "splitter_layer_assignments" in obj:
            del obj["splitter_layer_assignments"]

        geometry.update_node_positions_from_property(obj)
        geometry.rebuild_edges(obj)
        
        self.report({'INFO'}, "Layout flattened to Z=0")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(SCIGRAPHS_OT_ApplyLayout)
    bpy.utils.register_class(SCIGRAPHS_OT_ExecuteLayoutStep)
    bpy.utils.register_class(SCIGRAPHS_OT_ResetLayout)
    bpy.utils.register_class(SCIGRAPHS_OT_BakeAnimation)
    bpy.utils.register_class(SCIGRAPHS_OT_ImportAnimation)
    bpy.utils.register_class(SCIGRAPHS_OT_ImportGraphAnimation)
    bpy.utils.register_class(SCIGRAPHS_OT_ClearAnimation)
    bpy.utils.register_class(SCIGRAPHS_OT_NetworkSplitter3D)
    bpy.utils.register_class(SCIGRAPHS_OT_ResetSplitter)


def unregister():
    bpy.utils.unregister_class(SCIGRAPHS_OT_ResetSplitter)
    bpy.utils.unregister_class(SCIGRAPHS_OT_NetworkSplitter3D)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ClearAnimation)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ImportGraphAnimation)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ImportAnimation)
    bpy.utils.unregister_class(SCIGRAPHS_OT_BakeAnimation)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ResetLayout)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ExecuteLayoutStep)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ApplyLayout)



class SCIGRAPHS_OT_ImportAnimation(bpy.types.Operator):
    """Load node positions over time from a CSV and store them as shape keys.

    One row per node per stage, not per frame:

        stage,node,x,y,z
        0,0,-2.0,0.0,0.0
        1,0,-2.0,0.0,0.0

    `node` is a vertex index or a node name. A stage that misses a node is an
    error. Optional `frame` and `size` columns pin the stage to a frame and
    multiply the node radius.
    
    """

    bl_idname = "scigraphs.import_animation"
    bl_label = "Import Trajectory"
    bl_description = ("Load a CSV of node positions over time onto the active "
                      "graph as shape keys")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH' and "num_nodes" in obj

    def execute(self, context):
        import csv
        import os
        from collections import defaultdict

        from ....api import anim as sg_anim

        obj = context.active_object
        props = context.scene.scigraphs
        path = bpy.path.abspath(props.animation_filepath or "")
        if not path or not os.path.isfile(path):
            self.report({'ERROR'}, "No trajectory file set")
            return {'CANCELLED'}

        by_stage = defaultdict(dict)
        frames = {}
        try:
            with open(path, newline="") as handle:
                reader = csv.DictReader(handle)
                missing = {"stage", "node", "x", "y", "z"} - set(
                    reader.fieldnames or ())
                if missing:
                    self.report({'ERROR'},
                                f"{os.path.basename(path)} is missing "
                                f"column(s): {sorted(missing)}")
                    return {'CANCELLED'}
                from ....core.mesh.geometry import node_names
                by_name = {n: i for i, n in enumerate(node_names(obj))}

                sizes = {}
                for row in reader:
                    stage = int(float(row["stage"]))
                    raw = (row["node"] or "").strip()
                    try:
                        node = int(float(raw))
                    except ValueError:
                        if raw not in by_name:
                            self.report(
                                {'ERROR'},
                                f"node {raw!r} is not in this graph; the file "
                                f"names nodes but the object does not know "
                                f"that name")
                            return {'CANCELLED'}
                        node = by_name[raw]
                    by_stage[stage][node] = (
                        float(row["x"]), float(row["y"]), float(row["z"]))
                    if row.get("size") not in (None, ""):
                        sizes.setdefault(stage, {})[node] = float(row["size"])
                    if row.get("frame") not in (None, ""):
                        frames[stage] = int(float(row["frame"]))
        except (OSError, ValueError, KeyError) as exc:
            self.report({'ERROR'}, f"Could not read trajectory: {exc}")
            return {'CANCELLED'}

        count = len(obj.data.vertices)
        poses = []
        for stage in sorted(by_stage):
            nodes = by_stage[stage]
            if set(nodes) != set(range(count)):
                short = sorted(set(range(count)) - set(nodes))[:5]
                self.report(
                    {'ERROR'},
                    f"Stage {stage} covers {len(nodes)} of {count} vertices "
                    f"(missing {short}{'...' if len(short) == 5 else ''}); a "
                    f"shape key needs every vertex, not most of them")
                return {'CANCELLED'}
            poses.append(np.array([nodes[i] for i in range(count)],
                                  dtype=np.float64))

        if len(poses) < 2:
            self.report({'ERROR'},
                        f"{len(poses)} stage(s) found; an animation needs at "
                        f"least two")
            return {'CANCELLED'}

        scene = context.scene
        first, last = scene.frame_start, scene.frame_end
        span = max(last - first, 1)
        keys = []
        for index, pose in enumerate(poses):
            stage = sorted(by_stage)[index]
            frame = frames.get(
                stage, first + int(round(index * span / (len(poses) - 1))))
            keys.append((frame, pose))

        sg_anim.ensure_handler()
        sg_anim.positions(obj, keys)

        if sizes:
            ordered = sorted(by_stage)
            short = [s for s in ordered if set(sizes.get(s, {})) != set(range(count))]
            if short:
                self.report({'ERROR'},
                            f"stages {short[:5]} have an incomplete 'size' "
                            f"column while others have one")
                return {'CANCELLED'}
            from ....core.mesh.geometry import NODE_SCALE_ATTRIBUTE
            sg_anim.attribute(
                obj, NODE_SCALE_ATTRIBUTE,
                [(frame, [sizes[stage][i] for i in range(count)])
                 for (frame, _), stage in zip(keys, ordered)])

        moved = max(float(np.abs(b - a).max())
                    for a, b in zip(poses, poses[1:]))
        if moved <= 1e-9:
            self.report({'WARNING'},
                        f"{len(poses)} stages loaded but no node moves between "
                        f"them - the trajectory is a single repeated pose")
            return {'FINISHED'}

        self.report({'INFO'},
                    f"Loaded {len(poses)} stages over frames "
                    f"{keys[0][0]}-{keys[-1][0]}, largest move {moved:.3f}")
        return {'FINISHED'}


class SCIGRAPHS_OT_ClearAnimation(bpy.types.Operator):
    bl_idname = "scigraphs.clear_animation"
    bl_label = "Clear Animation"
    bl_description = "Remove the shape keys and attribute keys this graph carries"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        from ....api import anim as sg_anim
        sg_anim.clear(context.active_object)
        self.report({'INFO'}, "Animation cleared")
        return {'FINISHED'}


class SCIGRAPHS_OT_ImportGraphAnimation(bpy.types.Operator):
    """Build a graph and its animation from one `.sgraphs` file.

    `edge` rows give the topology, `pos` rows the motion, and the `meta`
    header says what wrote the file and how big it should be. Nodes are named,
    so the file is not tied to one object's vertex order.

    See `core.data_io.graph_animation` for the format.
    
    """

    bl_idname = "scigraphs.import_graph_animation"
    bl_label = "Import Graph + Motion"
    bl_description = ("Create a graph and its node animation from a single "
                      "file that carries both")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import os

        import networkx as nx

        from ....api import anim as sg_anim
        from ....core.city2graph import morphology
        from ....core.data_io import graph_animation as ga

        props = context.scene.scigraphs
        path = bpy.path.abspath(getattr(props, "graph_animation_filepath", ""))
        if not path or not os.path.isfile(path):
            self.report({'ERROR'}, "No graph animation file set")
            return {'CANCELLED'}

        try:
            data = ga.read(path)
        except ga.GraphAnimationError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        except OSError as exc:
            self.report({'ERROR'}, f"Could not read the file: {exc}")
            return {'CANCELLED'}

        choice = getattr(props, "graph_animation_direction", 'FILE')
        if choice == 'DIRECTED':
            directed = True
        elif choice == 'UNDIRECTED':
            directed = False
        else:
            directed = (data.meta.get("directed", "").lower() == "true")
        graph = nx.DiGraph() if directed else nx.Graph()
        first = data.stages[0]
        for node in data.nodes:
            graph.add_node(node, pos=tuple(first[node]),
                           **data.node_attrs.get(node, {}))
        for source, target, attrs in data.edges:
            graph.add_edge(source, target, **attrs)

        name = os.path.splitext(os.path.basename(path))[0]
        obj = morphology.create_graph_from_networkx(graph, name=name,
                                                    use_positions=True)
        if obj is None:
            self.report({'ERROR'}, "The graph could not be created")
            return {'CANCELLED'}

        context.view_layer.objects.active = obj
        obj.select_set(True)

        verts = obj.data.vertices
        if len(verts) != len(data.nodes):
            self.report({'ERROR'},
                        f"{len(data.nodes)} nodes in the file but "
                        f"{len(verts)} vertices in the mesh")
            return {'CANCELLED'}

        drift = max(
            max(abs(a - b) for a, b in zip(verts[i].co, first[node]))
            for i, node in enumerate(data.nodes))
        if drift > 1e-4:
            self.report({'ERROR'},
                        f"the mesh does not sit where the file's first stage "
                        f"puts it (off by {drift:.4f}); the node order cannot "
                        f"be trusted, so the animation is not applied")
            return {'CANCELLED'}

        from ....core.mesh.geometry import NODE_NAMES_KEY
        obj[NODE_NAMES_KEY] = ",".join(str(n) for n in data.nodes)

        for name in {k for attrs in data.node_attrs.values() for k in attrs}:
            values = [data.node_attrs.get(n, {}).get(name) for n in data.nodes]
            if any(v is None for v in values):
                continue
            numeric = all(isinstance(v, (int, float)) for v in values)
            if not numeric:
                continue
            kind = ('INT' if all(isinstance(v, int) for v in values)
                    else 'FLOAT')
            layer = obj.data.attributes.get(name)
            if layer is None:
                layer = obj.data.attributes.new(name, kind, 'POINT')
            for i, value in enumerate(values):
                layer.data[i].value = value

        keys = []
        for stage, frame in zip(data.stages,
                                data.frames_over(context.scene.frame_start,
                                                 context.scene.frame_end)):
            keys.append((frame, np.array([stage[n] for n in data.nodes],
                                         dtype=np.float64)))

        sg_anim.ensure_handler()
        sg_anim.positions(obj, keys)

        if data.sizes is not None:
            from ....core.mesh.geometry import NODE_SCALE_ATTRIBUTE
            sg_anim.attribute(
                obj, NODE_SCALE_ATTRIBUTE,
                [(frame, [sizes[n] for n in data.nodes])
                 for (frame, _), sizes in zip(keys, data.sizes)])

        try:
            bpy.ops.scigraphs.setup_visualization()
        except RuntimeError:
            pass

        moved = max(float(np.abs(b - a).max())
                    for a, b in zip([k[1] for k in keys],
                                    [k[1] for k in keys[1:]]))
        if moved <= 1e-9:
            self.report({'WARNING'},
                        f"{len(keys)} stages loaded but nothing moves between "
                        f"them")
            return {'FINISHED'}

        made_by = data.meta.get("generator", "an unknown version")
        if data.meta.get("format", "").endswith("/0"):
            self.report({'WARNING'},
                        f"{os.path.basename(path)} has no header: it predates "
                        f"the format and cannot be version-checked")
        self.report({'INFO'},
                    f"{len(data.nodes)} nodes, {len(data.edges)} edges, "
                    f"{len(keys)} stages over frames "
                    f"{keys[0][0]}-{keys[-1][0]} - written by {made_by}")
        return {'FINISHED'}
