import bpy

from . import dynamic, filter_gpu, filters, playback, shaders, state


class SCIGRAPHS_UL_filter_stack(bpy.types.UIList):
    """One row per clause. Ranges are in native units: the normalized sliders
    say nothing about a particular graph."""

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_prop, index):
        row = layout.row(align=True)
        row.prop(item, "mute", text="", emboss=False,
                 icon='CHECKBOX_DEHLT' if item.mute else 'CHECKBOX_HLT')
        label = filters.CHANNEL_LABELS.get(item.channel, item.channel)
        if item.channel in ('ATTR', 'EDGE_ATTR') and item.attr_name:
            label = item.attr_name
        row.label(text=label)
        sub = row.row()
        sub.active = not item.mute
        sub.alignment = 'RIGHT'
        sub.label(text=_range_text(context, item))
        if item.invert:
            row.label(text="", icon='ARROW_LEFTRIGHT')


# Reads as an answer to the channel name: "Bridge: yes".
_FLAG_TEXT = {(False, False): "neither", (True, True): "both",
              (False, True): "yes", (True, False): "no"}


def _range_text(context, slot):
    if slot.channel in filters.FLAG_CHANNELS:
        low, high = float(slot.range_min), float(slot.range_max)
        inside = (low <= 0.0 <= high, low <= 1.0 <= high)
        if slot.invert:
            inside = (not inside[0], not inside[1])
        return _FLAG_TEXT[inside]
    obj = getattr(context, "active_object", None)
    if state.is_graph_object(obj):
        try:
            values, low, high = filters.channel(
                obj, context.scene, slot.channel, slot.attr_name)[:3]
        except Exception:  # noqa: BLE001 - the panel must draw regardless
            values = None
        if values is not None and high > low:
            lo = low + slot.range_min * (high - low)
            hi = low + slot.range_max * (high - low)
            fmt = "{:,.0f}" if high - low >= 8 else "{:,.3g}"
            return (fmt.format(lo) + " - " + fmt.format(hi)
                    + ("  of " + fmt.format(high)))
    return f"{slot.range_min:.2f} - {slot.range_max:.2f}"


class SCIGRAPHS_PT_gpu_preview(bpy.types.Panel):
    bl_label = "GPU Preview (Fast)"
    bl_parent_id = "SCIGRAPHS_PT_visualization"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        self.layout.prop(context.scene, "scigraphs_preview_enabled", text="")

    @staticmethod
    def _draw_attr_row(col, scene):
        row = col.row(align=True)
        row.prop(scene, "scigraphs_preview_attr_name", text="Attribute")
        row.operator("scigraphs.set_preview_attr", text="", icon='DOWNARROW_HLT')

    def _draw_filter_stack(self, layout, scene, obj, enabled):
        """The stack, plus a survivor count, since an empty viewport looks broken."""
        box = layout.box()
        box.enabled = enabled
        header = box.row(align=True)
        header.label(text="Filters", icon='FILTER')
        header.operator("scigraphs.filter_add", text="", icon='ADD')
        header.operator("scigraphs.filter_remove", text="", icon='REMOVE')
        sub = header.row(align=True)
        sub.operator("scigraphs.filter_move", text="", icon='TRIA_UP').direction = 'UP'
        sub.operator("scigraphs.filter_move", text="",
                     icon='TRIA_DOWN').direction = 'DOWN'

        slots = scene.scigraphs_filters
        if not slots:
            box.label(text="Everything is drawn", icon='CHECKMARK')
            return

        column = box.column()
        column.use_property_split = False
        column.template_list("SCIGRAPHS_UL_filter_stack", "", scene,
                             "scigraphs_filters", scene,
                             "scigraphs_filters_index", rows=3)

        index = scene.scigraphs_filters_index
        if not 0 <= index < len(slots):
            return
        slot = slots[index]
        detail = box.column()
        detail.use_property_split = True
        detail.prop(slot, "channel", text="On")
        # Naming the field turns "unavailable" into "no edge weight".
        if slot.channel in ('ATTR', 'EDGE_ATTR'):
            row = detail.row(align=True)
            row.prop(slot, "attr_name", text="Attribute")
            if slot.channel == 'ATTR':
                row.operator("scigraphs.set_preview_attr", text="",
                             icon='DOWNARROW_HLT')
        elif slot.channel in filters.NEEDS_GROUPS:
            detail.prop(slot, "attr_name", text="Grouped By")
        elif slot.channel in filters.NEEDS_WEIGHT:
            detail.prop(slot, "attr_name", text="Weight")
        elif slot.channel in filters.NEEDS_SEEDS:
            row = detail.row(align=True)
            row.operator("scigraphs.filter_seed", icon='RESTRICT_SELECT_OFF')
            row.operator("scigraphs.filter_seed", text="",
                         icon='X').clear = True
            seeded = filters.SEED_ATTR in obj.data.attributes
            count = obj.get(filters.SEED_COUNT, 0) if seeded else 0
            if not seeded:
                text = "No seed yet: select nodes and press the button"
            elif count:
                text = f"{count:,} nodes seeded"
            else:
                text = "A seed is set"
            detail.label(text=text,
                         icon='CHECKMARK' if seeded else 'INFO')
        row = detail.row(align=True)
        row.prop(slot, "range_min", text="Min", slider=True)
        row.prop(slot, "range_max", text="Max", slider=True)
        detail.prop(slot, "invert")

        values = filters.channel(obj, scene, slot.channel, slot.attr_name)[0]
        if values is None:
            detail.label(text=self._unavailable(slot.channel), icon='ERROR')
            return
        kept = self._surviving(obj, scene)
        if kept is not None:
            nodes, edges = kept
            detail.label(
                text=f"{nodes:,} nodes and {edges:,} edges left",
                icon='CHECKMARK' if nodes else 'ERROR')

        # On the GPU the slider scrubs; on the CPU it stutters, so say which.
        live, why = filter_gpu.eligible(obj, scene)
        if live:
            detail.label(text="Evaluated on the GPU: a threshold is a uniform",
                         icon='SHADING_RENDERED')
        elif why != "no filter":
            detail.label(text=f"Rebuilt on the CPU: {why}", icon='TIME')

    @staticmethod
    def _unavailable(name):
        if name in filters.NEEDS_WEIGHT:
            return "This graph has no edge weight to measure"
        if name in filters.NEEDS_GROUPS:
            return "This graph has no grouping attribute"
        if name in filters.NEEDS_SEEDS:
            return "Seed some nodes first"
        return "This channel is not available on this graph"

    @staticmethod
    def _surviving(obj, scene):
        try:
            node_mask, edge_mask, active = filters.masks(obj, scene)
        except Exception:  # noqa: BLE001 - the panel must draw regardless
            return None
        if not active:
            return None
        return int(node_mask.sum()), int(edge_mask.sum())

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        obj = context.active_object

        layout.use_property_split = True
        layout.use_property_decorate = False

        if not state.is_graph_object(obj):
            layout.label(text="Select a graph object", icon='INFO')
            return

        enabled = scene.scigraphs_preview_enabled
        entry = state.CACHE.get(obj.as_pointer()) if enabled else None
        shown = entry["visible_count"] if entry else len(obj.data.vertices)
        if entry is not None and entry.get("filter") is not None:
            # The shader cuts the batches, so the build recorded no count.
            kept = self._surviving(obj, scene)
            if kept is not None:
                shown = kept[0]
        info = layout.row()
        info.enabled = enabled
        info.label(
            text=f"{shown:,}/{len(obj.data.vertices):,} nodes  |  {len(obj.data.edges):,} edges",
            icon='SNAP_VERTEX',
        )

        box = layout.box()
        box.enabled = enabled
        head = box.row(align=True)
        head.label(text="Animated Layout", icon='PLAY')
        running = playback.is_running(obj)
        head.operator(
            "scigraphs.animate_layout",
            text="Stop" if running else "Animate",
            icon='PAUSE' if running else 'PLAY',
        )
        props = getattr(scene, "scigraphs", None)
        if running:
            pb = playback.get(obj)
            frame_stats = pb.stats()
            on_gpu = type(pb.source).__name__ == "ComputeSource"
            box.label(
                text=(f"{pb.algorithm}: eased transition" if pb.interpolated
                      else f"{pb.algorithm} ({pb.model}) on "
                           + ("GPU" if on_gpu else "CPU")),
                icon='FORCE_FORCE',
            )
            box.label(
                text=f"{frame_stats['frames']} frames recorded  |  {frame_stats['mb']:.1f} MB"
                     + ("  |  full" if frame_stats['full'] else ""),
                icon='RENDER_ANIMATION',
            )
            box.label(text="Press Play; scrub to replay", icon='INFO')
            box.operator("scigraphs.reset_animation", text="Discard",
                         icon='TRASH')
        else:
            ok, why = dynamic.eligible(obj, scene) if enabled else (False, "")
            if enabled and not ok:
                box.label(text=why.capitalize(), icon='ERROR')
            sub = box.column(align=True)
            sub.enabled = ok
            if props is not None:
                sub.prop(props, "layout_algorithm", text="Algorithm")
                model, algo = playback.algorithm_model(scene)
                if model is None and playback.target_is_safe(algo):
                    sub.label(text="No iterative form: eased transition",
                              icon='IPO_EASE_IN_OUT')
                elif model is None:
                    sub.label(text="Not animatable: simulates with FA2 instead",
                              icon='ERROR')
                else:
                    sub.label(text=f"Force model: {model}", icon='FORCE_FORCE')
            sub.prop(scene, "scigraphs_preview_animate_steps", text="Iter/Frame")
            sub.prop(scene, "scigraphs_preview_animate_max_frames",
                     text="Record Frames")
            sub.prop(scene, "scigraphs_preview_animate_gpu", text="Simulate on GPU")

            if props is not None and playback.algorithm_model(scene)[0]:
                forces = box.column(align=True)
                forces.enabled = ok
                forces.label(text="Forces", icon='PHYSICS')
                forces.prop(props, "repulsion_strength", text="Repulsion")
                forces.prop(props, "attraction_strength", text="Attraction")
                forces.prop(props, "gravity_strength", text="Gravity")
                forces.prop(props, "layout_scale", text="Scale")
                # 1.0 gives the size Scale asks for; the default predates that.
                if props.gravity_strength < 0.5:
                    forces.label(text="Gravity < 0.5 spreads past Scale",
                                 icon='INFO')

        col = layout.column(align=True)
        col.enabled = enabled
        col.label(text="Nodes", icon='MESH_CIRCLE')
        col.prop(scene, "scigraphs_preview_node_style", text="Style")
        col.prop(scene, "scigraphs_preview_node_size", text="Point Size")
        if scene.scigraphs_preview_node_style in ('AUTO', 'SPHERE'):
            col.prop(scene, "scigraphs_preview_impostor_radius", text="Impostor Radius")
            if shaders.sphere_failed():
                col.label(text="Sphere impostor unsupported on this backend", icon='ERROR')
        col.prop(scene, "scigraphs_preview_round_points", text="Round Points")
        if scene.scigraphs_preview_round_points and shaders.round_point_failed():
            col.label(text="Round points unsupported on this backend", icon='ERROR')
        col.prop(scene, "scigraphs_preview_color_mode", text="Color")
        if scene.scigraphs_preview_color_mode == 'FLAT':
            col.prop(scene, "scigraphs_preview_node_color", text="")
        elif scene.scigraphs_preview_color_mode == 'ATTRIBUTE':
            self._draw_attr_row(col, scene)
            col.prop(scene, "scigraphs_preview_colormap", text="Colormap")
            col.prop(scene, "scigraphs_preview_reverse_colormap", text="Reverse")

        col = layout.column(align=True)
        col.enabled = enabled
        col.prop(scene, "scigraphs_preview_size_by_attr", text="Size by Attribute")
        if scene.scigraphs_preview_size_by_attr:
            if scene.scigraphs_preview_color_mode != 'ATTRIBUTE':
                self._draw_attr_row(col, scene)
            col.prop(scene, "scigraphs_preview_size_max_mult", text="Max Multiplier")

        col = layout.column(align=True)
        col.enabled = enabled
        col.label(text="Edges", icon='CURVE_PATH')
        col.prop(scene, "scigraphs_preview_show_edges", text="Show Edges")
        sub = col.column(align=True)
        sub.enabled = scene.scigraphs_preview_show_edges
        sub.prop(scene, "scigraphs_preview_edge_style", text="Style")
        sub.prop(scene, "scigraphs_preview_edge_radius_scale",
                 text="Radius Scale")
        sub.prop(scene, "scigraphs_preview_edge_width", text="Width")
        sub.prop(scene, "scigraphs_preview_edge_color", text="Color")
        sub.prop(scene, "scigraphs_preview_additive_edges", text="Additive (Density)")
        sub.prop(scene, "scigraphs_preview_edge_arrows", text="Arrows")
        if scene.scigraphs_preview_edge_arrows != 'OFF':
            sub.prop(scene, "scigraphs_preview_edge_arrow_style",
                     text="Arrow Style")
            sub.prop(scene, "scigraphs_preview_edge_arrow_size",
                     text="Arrow Size")
        sub.prop(scene, "scigraphs_preview_edge_size_by_attr",
                 text="Width by Attribute")
        if scene.scigraphs_preview_edge_size_by_attr:
            obj = context.active_object
            if obj is not None and obj.type == 'MESH':
                sub.prop_search(scene, "scigraphs_preview_edge_attr_name",
                                obj.data, "attributes", text="Attribute")
            else:
                sub.prop(scene, "scigraphs_preview_edge_attr_name",
                         text="Attribute")
            sub.prop(scene, "scigraphs_preview_edge_size_max_mult",
                     text="Max Multiplier")
            sub.prop(scene, "scigraphs_preview_edge_size_scale", text="Scale")

            # An unweighted graph still has the per-edge magnitudes the filter
            # stack computes from its structure.
            has_edge_attr = bool(scene.scigraphs_preview_edge_attr_name) and (
                obj is not None and obj.type == 'MESH'
                and scene.scigraphs_preview_edge_attr_name in obj.data.attributes)
            if not has_edge_attr:
                warn = sub.box()
                warn.label(text="No edge attribute selected", icon='INFO')
                warn.label(text="Bake one from the graph's own structure:")
                warn.operator_menu_enum("scigraphs.bake_edge_channel", "channel",
                                        text="Bake Edge Channel", icon='PLUS')
            else:
                sub.operator_menu_enum("scigraphs.bake_edge_channel", "channel",
                                       text="Bake Another Channel", icon='PLUS')

            # Every failure to reach pixels looks the same: one flat thickness.
            from . import batches as _b
            applied, why = _b.edge_width_status(obj, scene)
            sub.label(text=why.capitalize(),
                      icon='CHECKMARK' if applied else 'ERROR')

        sub.prop(scene, "scigraphs_preview_edge_taper", text="Taper")
        if scene.scigraphs_preview_edge_taper != 'NONE':
            sub.prop(scene, "scigraphs_preview_edge_taper_amount",
                     text="Taper Amount")
            if scene.scigraphs_preview_edge_style == 'LINE':
                sub.label(text="Taper needs tubes; set Edge Style to Ribbon",
                          icon='INFO')
        sub.prop(scene, "scigraphs_preview_edge_styles_gpu",
                 text="GPU Edge Styles")
        if scene.scigraphs_preview_edge_styles_gpu:
            if props is not None:
                box = sub.box().column(align=True)
                box.prop(props, "edge_style_type", text="Shape")
                style = props.edge_style_type
                if style in ('CURVED', 'QUADRATIC', 'ARC', 'TAPERED', 'BUNDLED'):
                    box.prop(props, "edge_curvature", text="Curvature")
                    box.prop(props, "edge_segments", text="Segments")
                    box.prop(props, "edge_curve_direction", text="Direction")
                if style == 'ORTHOGONAL':
                    box.prop(props, "edge_orthogonal_style", text="Routing")
                if style == 'TAPERED':
                    box.prop(props, "edge_taper_start", text="Taper Start")
                    box.prop(props, "edge_taper_end", text="Taper End")
                if style == 'BUNDLED':
                    box.prop(props, "edge_bundle_strength", text="Strength")
                    box.prop(props, "edge_bundle_iterations", text="Iterations")
                    box.prop(props, "edge_bundle_compatibility_threshold",
                             text="Compatibility")
                if style == 'HIERARCHICAL':
                    box.prop_search(scene, "scigraphs_preview_coarsen_attr",
                                    obj.data, "attributes", text="Clusters")
                    box.prop(props, "edge_heb_beta", text="Bundling",
                             slider=True)
                    box.prop(props, "edge_segments", text="Segments")
                    box.prop(props, "edge_heb_remove_lca",
                             text="Skip Common Ancestor")
                    box.prop(props, "edge_heb_fade_long", text="Fade Long",
                             slider=True)
                    box.prop(props, "edge_heb_gradient", text="Gradient")
                    box.prop(scene, "scigraphs_preview_heb_vertex_shader",
                             text="Bundle on the GPU")
                    levels = entry.get("hierarchy") if entry else None
                    if levels:
                        sizes = " / ".join(f"{lv['centers'].shape[0]:,}"
                                           for lv in levels)
                        box.label(text=f"Routed through {sizes}",
                                  icon='OUTLINER')
                    if entry and entry.get("reps", {}).get("bundle") is not None:
                        box.label(text="Strength is live: no rebuild",
                                  icon='SHADING_RENDERED')
                    elif enabled:
                        box.label(text="No cluster tree: edges stay straight",
                                  icon='ERROR')
                        box.label(text="Run Clustering, then pick its attribute")
                if style == 'FDEB':
                    box.prop(props, "edge_segments", text="Segments")
                    box.prop(props, "edge_bundle_strength", text="Strength")
                    box.prop(props, "edge_fdeb_cycles", text="Cycles")
                    box.prop(props, "edge_bundle_iterations",
                             text="Iterations per Cycle")
                    box.prop(props, "edge_fdeb_threshold",
                             text="Compatibility")
                    box.prop(props, "edge_fdeb_radius", text="Radius")
                    box.prop(props, "edge_fdeb_visibility",
                             text="Visibility Test")
                if style == 'MINGLE':
                    box.prop(props, "edge_segments", text="Segments")
                    box.prop(props, "edge_mingle_neighbours", text="Candidates")
                    box.prop(props, "edge_mingle_rounds", text="Rounds")
                    box.prop(props, "edge_mingle_min_gain", text="Min Saving")
                if style == 'SBEB':
                    box.prop(props, "edge_segments", text="Segments")
                    box.prop(props, "edge_sbeb_resolution", text="Resolution")
                    box.prop(props, "edge_sbeb_iterations", text="Iterations")
                    box.prop(props, "edge_sbeb_clusters", text="Clusters")
                    box.prop(props, "edge_sbeb_threshold", text="Shape Cutoff")
                    box.prop(props, "edge_sbeb_attraction", text="Attraction")
                    box.prop(props, "edge_sbeb_smooth", text="Smoothing")
                    box.prop(props, "edge_sbeb_recluster",
                             text="Recluster Each Round")
                if style == 'ROUTED':
                    box.prop(props, "edge_segments", text="Segments")
                    box.prop(props, "edge_routed_resolution", text="Grid")
                    box.prop(props, "edge_routed_iterations", text="Rounds")
                    box.prop(props, "edge_routed_reinforce",
                             text="Reinforcement")
                    box.prop(props, "edge_routed_avoid_nodes",
                             text="Avoid Crowding")
                if style in ('HIERARCHICAL', 'FDEB', 'SBEB', 'ROUTED', 'MINGLE'):
                    # One block for the settings that act on any control polygon.
                    box.separator()
                    if style != 'HIERARCHICAL':
                        box.prop(props, "edge_heb_beta", text="Bundling",
                                 slider=True)
                    box.prop(props, "edge_bundle_turn_limit",
                             text="Turning Limit")
                    box.prop(props, "edge_bundle_adaptive_beta",
                             text="Follow Length", slider=True)
                    box.prop(props, "edge_bundle_company",
                             text="Need Company", slider=True)
                    box.prop(props, "edge_bundle_view_beta",
                             text="Follow Zoom", slider=True)
                    box.prop(props, "edge_bundle_density_opacity",
                             text="Fade Crowded", slider=True)
                box.prop(props, "edge_auto_offset_parallel",
                         text="Offset Parallel Edges")
                if props.edge_auto_offset_parallel:
                    box.prop(props, "edge_parallel_offset", text="Offset")

        self._draw_filter_stack(layout, scene, obj, enabled)

        col = layout.column(align=True)
        col.enabled = enabled
        col.label(text="Simplification", icon='MOD_DECIM')
        col.prop(scene, "scigraphs_preview_backbone_mode", text="Edges")
        bb = scene.scigraphs_preview_backbone_mode
        if bb != 'ALL':
            if bb in ('TOPK', 'MST'):
                col.prop(scene, "scigraphs_preview_backbone_k", text="Top-k")
            if bb == 'DISPARITY':
                col.prop(scene, "scigraphs_preview_backbone_alpha",
                         text="Alpha")
            if bb == 'SAMPLE':
                col.prop(scene, "scigraphs_preview_backbone_sample",
                         text="Fraction")
            if obj is not None and obj.type == 'MESH':
                col.prop_search(scene, "scigraphs_preview_backbone_attr",
                                obj.data, "attributes", text="Weight")
            stats = entry.get("simplify_stats") if entry else None
            if stats:
                col.label(
                    text=(f"{stats['kept']:,}/{stats['total']:,} edges - "
                          f"{stats['weight_frac'] * 100.0:.0f}% weight kept"),
                    icon='CHECKMARK',
                )
        col.prop(scene, "scigraphs_preview_coarsen",
                 text="Coarsen by Community")
        if scene.scigraphs_preview_coarsen:
            if obj is not None and obj.type == 'MESH':
                col.prop_search(scene, "scigraphs_preview_coarsen_attr",
                                obj.data, "attributes", text="Attribute")
            else:
                col.prop(scene, "scigraphs_preview_coarsen_attr",
                         text="Attribute")
            col.prop(scene, "scigraphs_preview_coarsen_size_mode",
                     text="Size By")
            if scene.scigraphs_preview_coarsen_size_mode == 'ATTRIBUTE':
                if obj is not None and obj.type == 'MESH':
                    col.prop_search(scene,
                                    "scigraphs_preview_coarsen_size_attr",
                                    obj.data, "attributes", text="Size Attr")
                else:
                    col.prop(scene, "scigraphs_preview_coarsen_size_attr",
                             text="Size Attr")
                col.prop(scene, "scigraphs_preview_coarsen_size_agg",
                         text="Aggregate")
            if scene.scigraphs_preview_coarsen_size_mode not in ('EXTENT',):
                col.prop(scene, "scigraphs_preview_coarsen_size_mult",
                         text="Max Size x")
            col.prop(scene, "scigraphs_preview_coarsen_px",
                     text="Coarsen Below (px)")
            coarse = entry.get("coarse") if entry else None
            if coarse:
                active = " (active)" if entry.get("_coarse_active") else ""
                col.label(
                    text=(f"{coarse['n_comm']:,} communities - "
                          f"{coarse['n_super']:,} superedges{active}"),
                    icon='OUTLINER_OB_POINTCLOUD',
                )
                if coarse.get("size_missing"):
                    col.label(text="Size attribute not found - uniform sizes",
                              icon='ERROR')
                elif coarse.get("size_range"):
                    lo, hi = coarse["size_range"]
                    col.label(text=f"Size range {lo:,.4g} - {hi:,.4g}",
                              icon='FIXED_SIZE')
            elif entry is not None:
                col.label(text="Community attribute not found", icon='ERROR')

            col.prop(scene, "scigraphs_preview_adaptive",
                     text="Adaptive Cut (camera)")
            if scene.scigraphs_preview_adaptive:
                col.prop(scene, "scigraphs_preview_adaptive_algorithm",
                         text="Levels By")
                col.prop(scene, "scigraphs_preview_adaptive_predicted",
                         text="Split Above", slider=True)
                col.prop(scene, "scigraphs_preview_adaptive_min_px",
                         text="Split Above (px)")
                col.prop(scene, "scigraphs_preview_adaptive_collapse",
                         text="Merge Below", slider=True)
                col.prop(scene, "scigraphs_preview_adaptive_merge_loss",
                         text="Fidelity Traded", slider=True)
                col.operator("scigraphs.refine_cut",
                             text="Refine from Image", icon='SHADERFX')
                col.prop(scene, "scigraphs_preview_adaptive_freeze",
                         text="Freeze Cut", icon='FREEZE')
                hierarchy = entry.get("hierarchy") if entry else None
                if hierarchy:
                    sizes = " / ".join(f"{lvl['centers'].shape[0]:,}"
                                       for lvl in hierarchy)
                    col.label(text=f"Levels: {sizes}", icon='OUTLINER')
                    drawn = entry.get("_adaptive_drawn")
                    if drawn is not None and entry.get("_adaptive_active"):
                        total = sum(int(d.sum()) for d in drawn)
                        singles = int(drawn[len(hierarchy)].sum())
                        col.label(
                            text=(f"Drawing {total:,} elements, {singles:,} "
                                  f"individual nodes"),
                            icon='CHECKMARK',
                        )
                elif entry is not None:
                    col.label(text="No hierarchy above the first level",
                              icon='ERROR')
        col.prop(scene, "scigraphs_preview_density_fallback",
                 text="Density Fallback")

        col = layout.column(align=True)
        col.enabled = enabled
        col.label(text="Density Cloud (3D)", icon='OUTLINER_OB_VOLUME')
        col.prop(scene, "scigraphs_preview_volume_mode", text="Field")
        if scene.scigraphs_preview_volume_mode != 'OFF':
            if shaders.volume_failed():
                col.label(text="3D textures unsupported on this backend",
                          icon='ERROR')
            col.prop(scene, "scigraphs_preview_volume_when", text="Shown")
            if scene.scigraphs_preview_volume_when == 'HYBRID':
                col.prop(scene, "scigraphs_preview_volume_hybrid_pct",
                         text="Cloud Above (pct)")
            if scene.scigraphs_preview_volume_mode == 'WEIGHTED':
                self._draw_attr_row(col, scene)
            if scene.scigraphs_preview_volume_mode != 'COLOR':
                col.prop(scene, "scigraphs_preview_colormap", text="Colormap")
            col.prop(scene, "scigraphs_preview_volume_res", text="Resolution")
            col.prop(scene, "scigraphs_preview_volume_smooth", text="Smoothing")
            col.prop(scene, "scigraphs_preview_volume_threshold",
                     text="Threshold", slider=True)
            col.prop(scene, "scigraphs_preview_volume_falloff", text="Falloff")
            col.prop(scene, "scigraphs_preview_volume_gain", text="Opacity Gain")
            col.prop(scene, "scigraphs_preview_volume_slices", text="Slices")
            col.prop(scene, "scigraphs_preview_volume_levels",
                     text="Isosurfaces")
            if scene.scigraphs_preview_volume_levels > 0:
                col.prop(scene, "scigraphs_preview_volume_shell",
                         text="Shell Width", slider=True)
            stats = entry.get("volume_stats") if entry else None
            if stats:
                nz, ny, nx = stats["shape"]
                active = " (active)" if entry.get("_volume_active") else ""
                col.label(
                    text=(f"{nx}x{ny}x{nz} voxels - "
                          f"{stats['bytes'] / 1e6:.1f} MB{active}"),
                    icon='MESH_GRID',
                )
                col.label(
                    text=(f"Peak {stats['dtrue']:,.4g} nodes/unit3 "
                          f"(scale {stats['dmax']:,.4g})"),
                    icon='FIXED_SIZE',
                )
                if stats["dropped"]:
                    col.label(
                        text=(f"{stats['dropped']:,} crowded nodes drawn as "
                              f"cloud, {stats['nodes'] - stats['dropped']:,} "
                              f"kept discrete"),
                        icon='CHECKMARK',
                    )
                if stats["weighted_missing"]:
                    col.label(text="No attribute selected - counting nodes",
                              icon='ERROR')
            elif entry is not None:
                col.label(text="Density field not built", icon='ERROR')

        col = layout.column(align=True)
        col.enabled = enabled
        col.prop(scene, "scigraphs_preview_lod_enabled", text="Level of Detail")
        if scene.scigraphs_preview_lod_enabled:
            col.prop(scene, "scigraphs_preview_lod_max_points", text="Max Points")

        col = layout.column(align=True)
        col.enabled = enabled
        col.prop(scene, "scigraphs_preview_use_blocks", text="Spatial Blocks (Large)")
        if scene.scigraphs_preview_use_blocks:
            col.prop(scene, "scigraphs_preview_block_count", text="Target Blocks")
            col.prop(scene, "scigraphs_preview_node_budget", text="Node Budget")
            if entry is not None and entry.get("blocks") is not None:
                drawn = entry.get("_blocks_drawn")
                total = entry["blocks"]["count"]
                if drawn is not None:
                    col.label(text=f"Blocks drawn: {drawn:,}/{total:,}", icon='MESH_GRID')

        col = layout.column(align=True)
        is_engine = getattr(scene.render, "engine", "") == 'SCIGRAPHS'
        if not is_engine:
            col.label(text="Set Render Engine to 'SciGraphs' to render", icon='INFO')
        else:
            col.label(text="Render / Lighting / DoF: Properties editor", icon='PROPERTIES')

        col = layout.column(align=True)
        col.enabled = enabled
        col.prop(scene, "scigraphs_preview_depth_test", text="Depth Test")
        col.prop(scene, "scigraphs_preview_hide_mesh", text="Hide Native Mesh")
        if enabled and not scene.scigraphs_preview_hide_mesh:
            col.row().label(text="Mesh drawn twice (slower)", icon='ERROR')

        col = layout.column(align=True)
        col.enabled = enabled
        col.operator("scigraphs.measure_visibility", icon='HIDE_OFF')
        col.operator("scigraphs.benchmark_visibility", text="Benchmark",
                     icon='TIME')

        layout.separator()
        row = layout.row(align=True)
        row.enabled = enabled
        row.operator("scigraphs.pick_node", icon='RESTRICT_SELECT_OFF')
        row.operator("scigraphs.refresh_gpu_preview", text="", icon='FILE_REFRESH')

        picked = obj.get("scigraphs_preview_picked", None)
        if picked is not None:
            layout.label(text=f"Picked node: #{picked}", icon='PINNED')


# Shown only under the SciGraphs engine; DoF is Blender's own camera panel.

class SCIGRAPHS_RENDER_PT_engine(bpy.types.Panel):
    bl_label = "SciGraphs Render"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'
    COMPAT_ENGINES = {'SCIGRAPHS'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        scene = context.scene
        col = layout.column(align=True)
        col.prop(scene, "scigraphs_preview_render_bg", text="Background")
        col.prop(scene, "scigraphs_preview_render_aa", text="AA Samples")
        col.prop(scene, "scigraphs_preview_depth_test", text="Depth Test")
        col.prop(scene, "scigraphs_preview_render_id_pass", text="Node ID Pass")
        col.prop(scene, "scigraphs_preview_render_overdraw",
                 text="Overdraw Pass")
        col.prop(scene, "scigraphs_preview_render_labels", text="Text Labels")
        if scene.scigraphs_preview_render_labels:
            sub = col.column(align=True)
            sub.prop(scene, "scigraphs_preview_labels_declutter",
                     text="Declutter Labels")
            sub.label(text="Style: SciGraphs sidebar - Text Labels",
                      icon='SMALL_CAPS')

        if scene.camera is not None and scene.camera.type == 'CAMERA':
            dof = scene.camera.data.dof
            info = layout.row()
            info.label(
                text=("Depth of Field: ON (camera)" if dof.use_dof
                      else "Depth of Field: off - see Camera properties"),
                icon='CAMERA_DATA',
            )
            if dof.use_dof:
                col = layout.column(align=True)
                col.prop(scene, "scigraphs_preview_dof_highlights",
                         text="Bokeh Highlights")


class SCIGRAPHS_RENDER_PT_lighting(bpy.types.Panel):
    bl_label = "Lighting"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'
    bl_parent_id = "SCIGRAPHS_RENDER_PT_engine"
    COMPAT_ENGINES = {'SCIGRAPHS'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        scene = context.scene
        col = layout.column(align=True)
        col.prop(scene, "scigraphs_preview_use_scene_lights", text="Use Scene Lights")
        col.prop(scene, "scigraphs_preview_light_strength", text="Strength")
        col.prop(scene, "scigraphs_preview_ambient", text="Ambient")
        col.prop(scene, "scigraphs_preview_rim", text="Rim")
