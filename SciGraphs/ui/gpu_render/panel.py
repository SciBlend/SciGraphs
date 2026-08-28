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


def _ctx(context):
    """What every section needs, or None when there is no graph to describe.

    Each panel resolves this for itself. That is a dict lookup per section and
    it buys independence: a sub-panel draws correctly whether or not its parent
    is expanded, which is not true of state threaded down from one draw.
    """
    scene = context.scene
    obj = context.active_object
    if not state.is_graph_object(obj):
        return None
    enabled = bool(scene.scigraphs_preview_enabled)
    return {
        "scene": scene,
        "obj": obj,
        "enabled": enabled,
        "entry": state.CACHE.get(obj.as_pointer()) if enabled else None,
        "props": getattr(scene, "scigraphs", None),
    }


def _no_graph(layout):
    layout.label(text="Select a graph object", icon='INFO')


def _body(layout, enabled):
    """A split, undecorated column that greys out with the preview."""
    layout.use_property_split = True
    layout.use_property_decorate = False
    col = layout.column(align=True)
    col.enabled = enabled
    return col


def _draw_attr_row(col, scene):
    row = col.row(align=True)
    row.prop(scene, "scigraphs_preview_attr_name", text="Attribute")
    row.operator("scigraphs.set_preview_attr", text="", icon='DOWNARROW_HLT')


def _draw_view_transform(col, scene):
    """Colormap colors are written scene-linear and the view transform runs
    on top of them. Standard and Raw are a round trip; AgX and Filmic are
    tone mappers, so no pre-correction on our side can undo them."""
    view = getattr(scene, "view_settings", None)
    transform = getattr(view, "view_transform", "Standard")
    if transform in ('Standard', 'Raw'):
        return
    warn = col.box().column(align=True)
    warn.label(text=f"View transform is {transform}", icon='ERROR')
    warn.label(text="It tone-maps and desaturates the ramp, so a color")
    warn.label(text="no longer reads back as its value. Set it to")
    warn.label(text="Standard in Render Properties > Color Management")


def _draw_normalization(col, scene):
    col.prop(scene, "scigraphs_preview_norm_mode", text="Normalization")
    col.prop(scene, "scigraphs_preview_norm_gamma", text="Gamma")
    row = col.row(align=True)
    row.prop(scene, "scigraphs_preview_clip_low_pct", text="Clip")
    row.prop(scene, "scigraphs_preview_clip_high_pct", text="")


def _graph_extent(obj):
    """Longest side of the graph's bounding box, in object units."""
    import numpy as np
    mesh = getattr(obj, "data", None)
    n = len(mesh.vertices) if mesh is not None else 0
    if not n:
        return 0.0
    co = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    return float(np.ptp(co, axis=0).max())


def _draw_forces(box, scene, props, enabled, model=None, obj=None):
    """Drawn in both branches: a running simulation is retuned from these,
    so hiding them while animating put the feature out of reach. ``model``
    is the running source's, which outlives a change to the menu."""
    if props is None or not (model or playback.algorithm_model(scene)[0]):
        return
    forces = box.column(align=True)
    forces.enabled = enabled
    forces.label(text="Forces", icon='PHYSICS')
    forces.prop(props, "repulsion_strength", text="Repulsion")
    forces.prop(props, "attraction_strength", text="Attraction")
    forces.prop(props, "gravity_strength", text="Gravity")
    forces.prop(props, "layout_scale", text="Scale")
    extent = _graph_extent(obj) if obj is not None else 0.0
    scale = float(getattr(props, "layout_scale", 0.0) or 0.0)
    if props.gravity_strength < 0.05 and scale > 0.0 and extent > scale * 4.0:
        warn = forces.box().column(align=True)
        warn.label(text=f"Graph is {extent / scale:.0f}x wider than Scale",
                   icon='ERROR')
        warn.label(text="with Gravity at 0 nothing pulls it in, so the")
        warn.label(text="layout will barely appear to move. Raise Gravity.")
    elif props.gravity_strength < 0.5:
        forces.label(text="Gravity < 0.5 spreads past Scale", icon='INFO')


def _unavailable(name):
    if name in filters.NEEDS_WEIGHT:
        return "This graph has no edge weight to measure"
    if name in filters.NEEDS_GROUPS:
        return "This graph has no grouping attribute"
    if name in filters.NEEDS_SEEDS:
        return "Seed some nodes first"
    return "This channel is not available on this graph"


def _surviving(obj, scene):
    try:
        node_mask, edge_mask, active = filters.masks(obj, scene)
    except Exception:  # noqa: BLE001 - the panel must draw regardless
        return None
    if not active:
        return None
    return int(node_mask.sum()), int(edge_mask.sum())


def _shown_count(obj, scene, entry):
    shown = entry["visible_count"] if entry else len(obj.data.vertices)
    if entry is not None and entry.get("filter") is not None:
        # The shader cuts the batches, so the build recorded no count.
        kept = _surviving(obj, scene)
        if kept is not None:
            shown = kept[0]
    return shown


class _EngineTab:
    """Top level in the Render tab, alongside Nodes, Edges and the rest.

    These were sub-panels of the sidebar's GPU Preview, which put half the
    engine's settings in the viewport and half in the Properties editor. All of
    it lives in one place now; the sidebar keeps only what is on screen and the
    buttons that act on it.
    """

    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'SCIGRAPHS'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES


class SCIGRAPHS_PT_gpu_animation(_EngineTab, bpy.types.Panel):
    bl_label = "Animated Layout"
    bl_order = 8

    def draw(self, context):
        layout = self.layout
        ctx = _ctx(context)
        if ctx is None:
            _no_graph(layout)
            return
        scene, obj = ctx["scene"], ctx["obj"]
        props, enabled = ctx["props"], ctx["enabled"]
        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.column()
        box.enabled = enabled
        running = playback.is_running(obj)
        box.operator(
            "scigraphs.animate_layout",
            text="Stop" if running else "Animate",
            icon='PAUSE' if running else 'PLAY',
        )
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
            entry = ctx["entry"]
            if entry is not None and entry.get("dynamic") is None:
                warn = box.column(align=True)
                warn.label(text="Drawn on the static path", icon='ERROR')
                warn.label(text="positions cannot update live; the view only")
                warn.label(text="catches up when you press Stop")
            elif entry is not None and entry.get("upload_failed"):
                box.label(text="Position texture failed to upload",
                          icon='ERROR')
            if frame_stats.get("stalled"):
                warn = box.column(align=True)
                warn.label(text="Positions are not reaching the GPU",
                           icon='ERROR')
                warn.label(text=f"{frame_stats.get('rebuilds', 0)} rebuilds; "
                                f"the view is showing a stale frame")
            elif frame_stats.get("rebuilds"):
                box.label(text=f"{frame_stats['rebuilds']} rebuild(s) during "
                               f"playback", icon='INFO')
            if frame_stats["live"]:
                box.label(text="Past the recording: still simulating, but "
                               "these frames do not scrub", icon='ERROR')
                box.label(text="and will not render the same way twice")
            else:
                box.label(text="Press Play; scrub to replay", icon='INFO')
            if pb.interpolated:
                box.label(text="Eased transition: nothing to retune",
                          icon='IPO_EASE_IN_OUT')
            else:
                _draw_forces(box, scene, props, True, model=pb.model, obj=obj)
                if frame_stats["changes"]:
                    box.label(
                        text=(f"{frame_stats['changes']} live change(s); "
                              "each resimulated the frames after it"),
                        icon='FORCE_FORCE')
                if frame_stats["note"]:
                    box.label(text=f"Fixed until restart: {frame_stats['note']}",
                              icon='INFO')
            box.operator("scigraphs.reset_animation", text="Discard",
                         icon='TRASH')
            return

        ok, why = dynamic.eligible(obj, scene) if enabled else (False, "")
        if enabled and not ok:
            box.label(text=why.capitalize(), icon='ERROR')
        else:
            box.label(text="Press Animate first; Play alone does nothing",
                      icon='INFO')
        sub = box.column(align=True)
        sub.enabled = ok
        if props is not None:
            sub.prop(scene, "scigraphs_preview_animate_algorithm",
                     text="Algorithm")
            model, algo = playback.algorithm_model(scene)
            if model is None and playback.target_is_safe(algo):
                sub.label(text="No iterative form: eased transition",
                          icon='IPO_EASE_IN_OUT')
            elif model is None:
                sub.label(text="Not animatable: simulates with FA2 instead",
                          icon='ERROR')
            else:
                note = playback.animated_note(algo)
                if note:
                    sub.label(text=note, icon='FORCE_FORCE')
                else:
                    sub.label(text=f"Force model: {model}", icon='FORCE_FORCE')
        sub.prop(scene, "scigraphs_preview_repulsion", text="Repulsion")
        if scene.scigraphs_preview_repulsion == 'TREE':
            sub.prop(scene, "scigraphs_preview_theta")
            if not playback.quadtree_available():
                sub.label(text="No quadtree in scigraphs-utils; using grid",
                          icon='INFO')
            elif bool(scene.scigraphs_preview_animate_gpu):
                sub.label(text="Simulate on GPU keeps the grid; the tree is "
                               "the CPU path", icon='INFO')
        sub.prop(scene, "scigraphs_preview_animate_steps", text="Iter/Frame")
        sub.prop(scene, "scigraphs_preview_animate_max_frames",
                 text="Record Frames")
        sub.prop(scene, "scigraphs_preview_animate_gpu", text="Simulate on GPU")

        _draw_forces(box, scene, props, ok, obj=obj)


class SCIGRAPHS_PT_gpu_structure(_EngineTab, bpy.types.Panel):
    """The spatial structure the simulation uses, drawn over the graph.

    Nested under Animated Layout, which is the only thing that builds it. That
    was a five-deep panel back when this lived in the sidebar; here it is one
    level under the tab, the same as Edges > Shape.
    """

    bl_label = "Show Structure"
    bl_parent_id = "SCIGRAPHS_PT_gpu_animation"

    def draw_header(self, context):
        self.layout.prop(context.scene, "scigraphs_preview_grid_show", text="")

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        col = _body(layout, bool(scene.scigraphs_preview_grid_show))
        col.prop(scene, "scigraphs_preview_grid_mode", text="")
        if scene.scigraphs_preview_grid_mode in ('NEAR', 'BOTH') \
                and not bool(scene.scigraphs_preview_animate_gpu):
            col.label(text="Near field bins are GPU only", icon='INFO')
        if scene.scigraphs_preview_grid_mode == 'TREE':
            col.prop(scene, "scigraphs_preview_tree_depth")
        col.prop(scene, "scigraphs_preview_grid_width")
        col.prop(scene, "scigraphs_preview_grid_color", text="")
        col.prop(scene, "scigraphs_preview_grid_by_count")
        if bool(scene.scigraphs_preview_grid_by_count):
            col.prop(scene, "scigraphs_preview_grid_hot", text="")


class SCIGRAPHS_PT_gpu_filters(_EngineTab, bpy.types.Panel):
    bl_label = "Filters"
    bl_order = 9

    def draw(self, context):
        """The stack, plus a survivor count, since an empty viewport looks broken."""
        layout = self.layout
        ctx = _ctx(context)
        if ctx is None:
            _no_graph(layout)
            return
        scene, obj = ctx["scene"], ctx["obj"]
        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.column()
        box.enabled = ctx["enabled"]
        header = box.row(align=True)
        header.operator("scigraphs.filter_add", text="Add", icon='ADD')
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
            detail.label(text=_unavailable(slot.channel), icon='ERROR')
            return
        kept = _surviving(obj, scene)
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


class SCIGRAPHS_PT_gpu_simplify(_EngineTab, bpy.types.Panel):
    bl_label = "Simplification"
    bl_order = 10

    def draw(self, context):
        layout = self.layout
        ctx = _ctx(context)
        if ctx is None:
            _no_graph(layout)
            return
        scene, obj, entry = ctx["scene"], ctx["obj"], ctx["entry"]
        col = _body(layout, ctx["enabled"])

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


class SCIGRAPHS_PT_gpu_performance(_EngineTab, bpy.types.Panel):
    """What to trade away when the graph stops being interactive."""

    bl_label = "Performance"
    bl_order = 11

    def draw(self, context):
        layout = self.layout
        ctx = _ctx(context)
        if ctx is None:
            _no_graph(layout)
            return
        scene, entry, enabled = ctx["scene"], ctx["entry"], ctx["enabled"]
        col = _body(layout, enabled)

        col.prop(scene, "scigraphs_preview_lod_enabled", text="Level of Detail")
        if scene.scigraphs_preview_lod_enabled:
            col.prop(scene, "scigraphs_preview_lod_max_points", text="Max Points")

        col.separator()
        col.prop(scene, "scigraphs_preview_use_blocks",
                 text="Spatial Blocks (Large)")
        if scene.scigraphs_preview_use_blocks:
            col.prop(scene, "scigraphs_preview_block_count", text="Target Blocks")
            col.prop(scene, "scigraphs_preview_node_budget", text="Node Budget")
            if entry is not None and entry.get("blocks") is not None:
                drawn = entry.get("_blocks_drawn")
                total = entry["blocks"]["count"]
                if drawn is not None:
                    col.label(text=f"Blocks drawn: {drawn:,}/{total:,}",
                              icon='MESH_GRID')

        col.separator()
        col.prop(scene, "scigraphs_preview_depth_test", text="Depth Test")
        col.prop(scene, "scigraphs_preview_hide_mesh", text="Hide Native Mesh")
        if enabled and not scene.scigraphs_preview_hide_mesh:
            col.label(text="Mesh drawn twice (slower)", icon='ERROR')

        col.separator()
        col.operator("scigraphs.measure_visibility", icon='HIDE_OFF')
        col.operator("scigraphs.benchmark_visibility", text="Benchmark",
                     icon='TIME')


class _Render:
    """Top level in the Render tab, not nested under one root.

    Nesting cost a click on every section and buried the list one level down,
    where the eye reads "SciGraphs Render" as the whole engine rather than as
    its output settings. Flat, the tab reads as the sections themselves: Nodes,
    Edges, Density Cloud. bl_order fixes the sequence, since registration order
    is not a promise.
    """

    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'SCIGRAPHS'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES


# Shown only under the SciGraphs engine; DoF is Blender's own camera panel.

class SCIGRAPHS_RENDER_PT_engine(bpy.types.Panel):
    """Output, not the root: it is a sibling of Nodes and Edges now, so the
    label says what it holds rather than naming the whole engine."""

    bl_label = "Render Output"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'
    bl_order = 0
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
        col.prop(scene, "scigraphs_preview_render_id_pass", text="Node ID Pass")
        col.prop(scene, "scigraphs_preview_render_overdraw",
                 text="Overdraw Pass")

        if scene.camera is not None and scene.camera.type == 'CAMERA':
            dof = scene.camera.data.dof
            info = layout.row()
            info.label(
                text=("Depth of Field: ON (camera)" if dof.use_dof
                      else "Depth of Field: off - see Camera properties"),
                icon='CAMERA_DATA',
            )


class SCIGRAPHS_RENDER_PT_nodes(_Render, bpy.types.Panel):
    bl_label = "Nodes"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        entry = (_ctx(context) or {}).get("entry")
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column(align=True)

        col.prop(scene, "scigraphs_preview_node_style", text="Style")
        col.prop(scene, "scigraphs_preview_node_size", text="Point Size")
        if scene.scigraphs_preview_node_style in ('AUTO', 'SPHERE'):
            col.prop(scene, "scigraphs_preview_impostor_radius",
                     text="Impostor Radius")
            if shaders.sphere_failed():
                col.label(text="Sphere impostor unsupported on this backend",
                          icon='ERROR')
        col.prop(scene, "scigraphs_preview_round_points", text="Round Points")
        if scene.scigraphs_preview_round_points and shaders.round_point_failed():
            col.label(text="Round points unsupported on this backend",
                      icon='ERROR')

        col.separator()
        col.prop(scene, "scigraphs_preview_color_mode", text="Color")
        if scene.scigraphs_preview_color_mode == 'FLAT':
            col.prop(scene, "scigraphs_preview_node_color", text="")
        elif scene.scigraphs_preview_color_mode == 'ATTRIBUTE':
            _draw_attr_row(col, scene)
            col.prop(scene, "scigraphs_preview_colormap", text="Colormap")
            col.prop(scene, "scigraphs_preview_reverse_colormap", text="Reverse")
            _draw_normalization(col, scene)
            missing = entry.get("unmeasured", 0) if entry else 0
            if missing:
                col.label(
                    text=(f"{missing:,} nodes have no value: drawn in the "
                          f"NaN color at base size"),
                    icon='INFO',
                )
            _draw_view_transform(col, scene)

        col.separator()
        col.prop(scene, "scigraphs_preview_size_by_attr",
                 text="Size by Attribute")
        if scene.scigraphs_preview_size_by_attr:
            if scene.scigraphs_preview_color_mode != 'ATTRIBUTE':
                _draw_attr_row(col, scene)
            col.prop(scene, "scigraphs_preview_size_max_mult",
                     text="Max Multiplier")


class SCIGRAPHS_RENDER_PT_edges(_Render, bpy.types.Panel):
    bl_label = "Edges"
    bl_order = 2

    def draw_header(self, context):
        self.layout.prop(context.scene, "scigraphs_preview_show_edges", text="")

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column(align=True)
        col.enabled = scene.scigraphs_preview_show_edges

        col.prop(scene, "scigraphs_preview_edge_style", text="Style")
        col.prop(scene, "scigraphs_preview_edge_radius_scale",
                 text="Radius Scale")
        col.prop(scene, "scigraphs_preview_edge_width", text="Width")
        col.prop(scene, "scigraphs_preview_edge_color", text="Color")
        col.prop(scene, "scigraphs_preview_additive_edges",
                 text="Additive (Density)")
        col.prop(scene, "scigraphs_preview_edge_xray", text="Occluded Edges")

        col.separator()
        col.prop(scene, "scigraphs_preview_edge_arrows", text="Arrows")
        if scene.scigraphs_preview_edge_arrows != 'OFF':
            col.prop(scene, "scigraphs_preview_edge_arrow_style",
                     text="Arrow Style")
            col.prop(scene, "scigraphs_preview_edge_arrow_size",
                     text="Arrow Size")

        col.separator()
        col.prop(scene, "scigraphs_preview_edge_taper", text="Taper")
        if scene.scigraphs_preview_edge_taper != 'NONE':
            col.prop(scene, "scigraphs_preview_edge_taper_amount",
                     text="Taper Amount")
            if scene.scigraphs_preview_edge_style == 'LINE':
                col.label(text="Taper needs tubes; set Edge Style to Ribbon",
                          icon='INFO')


class SCIGRAPHS_RENDER_PT_edge_width(bpy.types.Panel):
    """Width driven by an attribute, and whether it is reaching pixels."""

    bl_label = "Width by Attribute"
    bl_parent_id = "SCIGRAPHS_RENDER_PT_edges"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'SCIGRAPHS'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw_header(self, context):
        self.layout.prop(context.scene, "scigraphs_preview_edge_size_by_attr",
                         text="")

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        obj = context.active_object
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column(align=True)
        col.enabled = bool(scene.scigraphs_preview_edge_size_by_attr)

        if obj is not None and obj.type == 'MESH':
            col.prop_search(scene, "scigraphs_preview_edge_attr_name",
                            obj.data, "attributes", text="Attribute")
        else:
            col.prop(scene, "scigraphs_preview_edge_attr_name",
                     text="Attribute")
        col.prop(scene, "scigraphs_preview_edge_size_max_mult",
                 text="Max Multiplier")
        col.prop(scene, "scigraphs_preview_edge_size_scale", text="Scale")

        # An unweighted graph still has the per-edge magnitudes the filter
        # stack computes from its structure.
        has_edge_attr = bool(scene.scigraphs_preview_edge_attr_name) and (
            obj is not None and obj.type == 'MESH'
            and scene.scigraphs_preview_edge_attr_name in obj.data.attributes)
        if not has_edge_attr:
            warn = col.box()
            warn.label(text="No edge attribute selected", icon='INFO')
            warn.label(text="Bake one from the graph's own structure:")
            warn.operator_menu_enum("scigraphs.bake_edge_channel", "channel",
                                    text="Bake Edge Channel", icon='PLUS')
        else:
            col.operator_menu_enum("scigraphs.bake_edge_channel", "channel",
                                   text="Bake Another Channel", icon='PLUS')

        # Every failure to reach pixels looks the same: one flat thickness.
        from . import batches as _b
        applied, why = _b.edge_width_status(obj, scene)
        col.label(text=why.capitalize(),
                  icon='CHECKMARK' if applied else 'ERROR')


class SCIGRAPHS_RENDER_PT_edge_shape(bpy.types.Panel):
    """The GPU edge styles. One shape at a time, and only its own settings:
    the twelve shapes share a pool of parameters most of them ignore."""

    bl_label = "Shape"
    bl_parent_id = "SCIGRAPHS_RENDER_PT_edges"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'SCIGRAPHS'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw_header(self, context):
        self.layout.prop(context.scene, "scigraphs_preview_edge_styles_gpu",
                         text="")

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        obj = context.active_object
        entry = (_ctx(context) or {}).get("entry")
        props = getattr(scene, "scigraphs", None)
        layout.use_property_split = True
        layout.use_property_decorate = False
        if props is None:
            return
        box = layout.column(align=True)
        box.enabled = bool(scene.scigraphs_preview_edge_styles_gpu)

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
            if obj is not None and obj.type == 'MESH':
                box.prop_search(scene, "scigraphs_preview_coarsen_attr",
                                obj.data, "attributes", text="Clusters")
            box.prop(props, "edge_heb_beta", text="Bundling", slider=True)
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
                box.label(text=f"Routed through {sizes}", icon='OUTLINER')
            if entry and entry.get("reps", {}).get("bundle") is not None:
                box.label(text="Strength is live: no rebuild",
                          icon='SHADING_RENDERED')
            elif entry is not None:
                box.label(text="No cluster tree: edges stay straight",
                          icon='ERROR')
                box.label(text="Run Clustering, then pick its attribute")
        if style == 'FDEB':
            box.prop(props, "edge_segments", text="Segments")
            box.prop(props, "edge_bundle_strength", text="Strength")
            box.prop(props, "edge_fdeb_cycles", text="Cycles")
            box.prop(props, "edge_bundle_iterations",
                     text="Iterations per Cycle")
            box.prop(props, "edge_fdeb_threshold", text="Compatibility")
            box.prop(props, "edge_fdeb_radius", text="Radius")
            box.prop(props, "edge_fdeb_visibility", text="Visibility Test")
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
            box.prop(props, "edge_sbeb_recluster", text="Recluster Each Round")
        if style == 'ROUTED':
            box.prop(props, "edge_segments", text="Segments")
            box.prop(props, "edge_routed_resolution", text="Grid")
            box.prop(props, "edge_routed_iterations", text="Rounds")
            box.prop(props, "edge_routed_reinforce", text="Reinforcement")
            box.prop(props, "edge_routed_avoid_nodes", text="Avoid Crowding")
        if style in ('HIERARCHICAL', 'FDEB', 'SBEB', 'ROUTED', 'MINGLE'):
            # One block for the settings that act on any control polygon.
            box.separator()
            if style != 'HIERARCHICAL':
                box.prop(props, "edge_heb_beta", text="Bundling", slider=True)
            box.prop(props, "edge_bundle_turn_limit", text="Turning Limit")
            box.prop(props, "edge_bundle_adaptive_beta",
                     text="Follow Length", slider=True)
            box.prop(props, "edge_bundle_company",
                     text="Need Company", slider=True)
            box.prop(props, "edge_bundle_view_beta",
                     text="Follow Zoom", slider=True)
            box.prop(props, "edge_bundle_density_opacity",
                     text="Fade Crowded", slider=True)
        box.separator()
        box.prop(props, "edge_auto_offset_parallel",
                 text="Offset Parallel Edges")
        if props.edge_auto_offset_parallel:
            box.prop(props, "edge_parallel_offset", text="Offset")


class SCIGRAPHS_RENDER_PT_density(_Render, bpy.types.Panel):
    bl_label = "Density Cloud"
    bl_order = 3

    def draw_header(self, context):
        self.layout.prop(context.scene, "scigraphs_preview_volume_show",
                         text="")

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        entry = (_ctx(context) or {}).get("entry")
        col = _body(layout, scene.scigraphs_preview_volume_mode != 'OFF')

        col.prop(scene, "scigraphs_preview_volume_mode", text="Field")
        if scene.scigraphs_preview_volume_mode == 'OFF':
            return
        if shaders.volume_failed():
            col.label(text="3D textures unsupported on this backend",
                      icon='ERROR')
        col.prop(scene, "scigraphs_preview_volume_when", text="Shown")
        if scene.scigraphs_preview_volume_when == 'HYBRID':
            col.prop(scene, "scigraphs_preview_volume_hybrid_pct",
                     text="Cloud Above (pct)")
        if scene.scigraphs_preview_volume_mode == 'WEIGHTED':
            _draw_attr_row(col, scene)
        if scene.scigraphs_preview_volume_mode != 'COLOR':
            col.prop(scene, "scigraphs_preview_colormap", text="Colormap")
            _draw_view_transform(col, scene)
        col.prop(scene, "scigraphs_preview_volume_res", text="Resolution")
        col.prop(scene, "scigraphs_preview_volume_smooth", text="Smoothing")
        col.prop(scene, "scigraphs_preview_volume_threshold",
                 text="Threshold", slider=True)
        col.prop(scene, "scigraphs_preview_volume_falloff", text="Falloff")
        col.prop(scene, "scigraphs_preview_volume_gain", text="Opacity Gain")
        col.prop(scene, "scigraphs_preview_volume_slices", text="Slices")
        col.prop(scene, "scigraphs_preview_volume_levels", text="Isosurfaces")
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


class SCIGRAPHS_RENDER_PT_labels(_Render, bpy.types.Panel):
    bl_label = "Text Labels"
    bl_order = 4

    def draw_header(self, context):
        self.layout.prop(context.scene, "scigraphs_preview_render_labels",
                         text="")

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        col = _body(layout, bool(scene.scigraphs_preview_render_labels))
        col.prop(scene, "scigraphs_preview_labels_declutter",
                 text="Declutter Labels")
        if scene.scigraphs_preview_labels_declutter:
            col.prop(scene, "scigraphs_preview_labels_priority", text="Keep")
            if scene.scigraphs_preview_labels_priority in ('ATTR', 'EDGE_ATTR'):
                col.prop(scene, "scigraphs_preview_labels_priority_attr",
                         text="Attribute")

        props = getattr(scene, "scigraphs", None)
        if props is not None:
            from ..panels.scigraphs.visualization_panels import draw_text_style
            body = layout.column()
            body.enabled = bool(scene.scigraphs_preview_render_labels)
            body.use_property_split = True
            body.use_property_decorate = False
            draw_text_style(body, props, context.active_object)


class SCIGRAPHS_RENDER_PT_legend(_Render, bpy.types.Panel):
    bl_label = "Color Key"
    bl_order = 5

    def draw_header(self, context):
        self.layout.prop(context.scene, "scigraphs_preview_legend", text="")

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        col = _body(layout, bool(scene.scigraphs_preview_legend))
        col.prop(scene, "scigraphs_preview_legend_anchor", text="Corner")
        col.prop(scene, "scigraphs_preview_legend_orient", text="Layout")
        place = col.row(align=True)
        place.prop(scene, "scigraphs_preview_legend_x", text="Nudge X")
        place.prop(scene, "scigraphs_preview_legend_y", text="Y")
        col.prop(scene, "scigraphs_preview_legend_scale", text="Size")
        col.prop(scene, "scigraphs_preview_legend_box")

        mode = scene.scigraphs_preview_color_mode
        cp = getattr(scene, "scigraphs_coloring", None)
        if mode == 'FLAT':
            col.label(text="Nothing to key: color mode is Flat", icon='INFO')
        elif mode == 'VERTEX' and not getattr(cp, "attribute_name", ""):
            col.label(text="Nothing to key: pick an attribute in Coloring",
                      icon='INFO')
        elif mode == 'VERTEX':
            col.label(text=f"Keyed from the Coloring bake: {cp.attribute_name}",
                      icon='INFO')


class SCIGRAPHS_RENDER_PT_lighting(_Render, bpy.types.Panel):
    bl_label = "Lighting"
    bl_order = 6

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        scene = context.scene
        col = layout.column(align=True)
        col.prop(scene, "scigraphs_preview_use_scene_lights",
                 text="Use Scene Lights")
        col.prop(scene, "scigraphs_preview_light_strength", text="Strength")
        col.prop(scene, "scigraphs_preview_ambient", text="Ambient")
        col.prop(scene, "scigraphs_preview_rim", text="Rim")

        cam = scene.camera
        if cam is not None and cam.type == 'CAMERA' and cam.data.dof.use_dof:
            col.separator()
            col.prop(scene, "scigraphs_preview_dof_highlights",
                     text="Bokeh Highlights")
