"""
Metapath analysis UI panels for heterogeneous graph analysis.

The panel mirrors the city2graph metapath workflow:
``amenities + street dual graph -> bridge_nodes -> add_metapaths``. Shared
inputs live in the root panel; the two materialization strategies
(``add_metapaths`` by hops and ``add_metapaths_by_weight`` by cost) are
separate subpanels, with result inspection and the raw step-by-step operators
in their own subpanels.
"""

import bpy


def _amenities_is_valid(amenities_obj):
    """True if the object is a usable amenity source for metapaths."""
    return bool(
        amenities_obj
        and (amenities_obj.get("is_osm_features") or amenities_obj.get("is_city2graph"))
    )


class SCIGRAPHS_PT_c2g_metapaths(bpy.types.Panel):
    """Metapath Analysis root panel: shared inputs for both strategies."""
    bl_label = "Metapath Analysis"
    bl_parent_id = "SCIGRAPHS_PT_city2graph_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.city2graph
        obj = context.active_object

        layout.use_property_split = True
        layout.use_property_decorate = False

        net_box = layout.box()
        net_box.label(text="Street Network (active object)", icon='FORCE_CURVE')
        net_row = net_box.row()
        net_row.scale_y = 0.85
        if obj and obj.get("is_osmnx"):
            net_row.label(text=f"{obj.name}", icon='CHECKMARK')
        elif obj and obj.get("is_street_dual"):
            net_row.label(text=f"Dual: {obj.name}", icon='CHECKMARK')
        else:
            net_row.alert = True
            net_row.label(text="Activate an OSMnx network", icon='ERROR')

        amen_box = layout.box()
        amen_box.label(text="Amenities (endpoints)", icon='STICKY_UVS_LOC')
        col = amen_box.column(align=True)
        col.prop(props, "metapath_amenities_object", text="")

        amenities_obj = props.metapath_amenities_object
        if _amenities_is_valid(amenities_obj):
            info_row = col.row()
            info_row.scale_y = 0.8
            info_row.label(
                text=f"{amenities_obj.get('feature_count', 0)} features",
                icon='CHECKMARK',
            )
        elif amenities_obj:
            warn = col.row()
            warn.scale_y = 0.8
            warn.alert = True
            warn.label(text="Not a valid features object", icon='ERROR')

        graph_box = layout.box()
        graph_box.label(text="Graph Construction", icon='OUTLINER_OB_CURVE')
        gcol = graph_box.column(align=True)
        gcol.prop(props, "metapath_k_neighbors", text="Bridge K")
        gcol.prop(props, "metapath_amenity_limit", text="Amenity Limit")
        hint = graph_box.row()
        hint.scale_y = 0.7
        hint.label(
            text="Dual graph + KNN bridge are built automatically",
            icon='INFO',
        )


class SCIGRAPHS_PT_c2g_metapaths_hops(bpy.types.Panel):
    """Materialize metapaths by a fixed number of street hops."""
    bl_label = "Materialize by Hops"
    bl_parent_id = "SCIGRAPHS_PT_c2g_metapaths"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'

    def draw(self, context):
        layout = self.layout
        props = context.scene.city2graph

        layout.use_property_split = True
        layout.use_property_decorate = False

        seq_box = layout.box()
        seq_box.scale_y = 0.7
        seq_box.label(
            text="amenity -> segment (xN) -> amenity",
            icon='ARROW_LEFTRIGHT',
        )

        col = layout.column(align=True)
        col.prop(props, "metapath_hops", text="Street Hops")

        viz = layout.column(align=True)
        viz.prop(props, "metapath_visualize_limit", text="Viz Limit")
        viz.prop(props, "metapath_curve_thickness", text="Curve Thickness")

        layout.separator()
        run = layout.row()
        run.scale_y = 1.4
        run.operator(
            "scigraphs.compute_metapaths_wizard",
            icon='PLAY',
            text="Materialize Metapaths",
        )


class SCIGRAPHS_PT_c2g_metapaths_weight(bpy.types.Panel):
    """Materialize metapaths by an accumulated cost threshold (Dijkstra)."""
    bl_label = "Materialize by Weight"
    bl_parent_id = "SCIGRAPHS_PT_c2g_metapaths"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.city2graph

        layout.use_property_split = True
        layout.use_property_decorate = False

        seq_box = layout.box()
        seq_box.scale_y = 0.7
        seq_box.label(
            text="Connect endpoints reachable within a cost band",
            icon='DRIVER_DISTANCE',
        )

        col = layout.column(align=True)
        col.prop(props, "metapath_weight_attr", text="Cost Attribute")
        col.prop(props, "metapath_weight_threshold", text="Max Cost")
        col.prop(props, "metapath_weight_min_threshold", text="Min Cost")
        col.prop(props, "metapath_endpoint_type", text="Endpoint Type")

        layout.separator()
        run = layout.row()
        run.scale_y = 1.4
        run.operator(
            "scigraphs.compute_metapaths_by_weight",
            icon='PLAY',
            text="Materialize by Weight",
        )


class SCIGRAPHS_PT_c2g_metapaths_result(bpy.types.Panel):
    """Inspect the active dual graph / metapath result."""
    bl_label = "Result & Inspect"
    bl_parent_id = "SCIGRAPHS_PT_c2g_metapaths"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return bool(
            obj
            and (
                obj.get("is_street_dual")
                or obj.get("is_metapath_result")
                or obj.get("is_metapath_mesh")
            )
        )

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        info_box = layout.box()
        info_box.scale_y = 0.85

        if obj.get("is_street_dual"):
            num_nodes = obj.get('num_nodes', 0)
            num_edges = obj.get('num_edges', 0)
            info_box.label(text="Street Dual Graph", icon='MESH_DATA')
            info_box.label(text=f"Segments: {num_nodes:,} | Connections: {num_edges:,}")

            if obj.get("has_metapath_bridges"):
                num_bridges = obj.get('num_bridges', 0)
                info_box.label(text=f"Bridges: {num_bridges} connections")
                info_box.label(text="Ready to materialize metapaths")

        elif obj.get("is_metapath_result"):
            raw_count = obj.get("num_metapaths_raw", 0)
            unique_count = obj.get("num_metapaths_unique", 0)
            hops = obj.get("metapath_hops", "?")
            num_viz = obj.get("num_visualized", 0)
            avg_mult = obj.get("avg_multiplicity", 0)
            min_mult = obj.get("min_multiplicity", 0)
            max_mult = obj.get("max_multiplicity", 0)

            info_box.label(text="Metapath Result", icon='CURVE_DATA')
            info_box.label(text=f"Total: {raw_count:,} | Unique: {unique_count:,} ({hops}-hop)")
            if avg_mult > 0:
                info_box.label(text=f"Multiplicity: {min_mult}-{max_mult} (avg {avg_mult:.1f}x)")
            info_box.label(text=f"Visualized: {num_viz:,} curves")

            info_box.separator()
            info_box.operator(
                "scigraphs.convert_metapaths_to_mesh",
                icon='MESH_DATA',
                text="Convert to Mesh (Spreadsheet)",
            )
            hint = info_box.box()
            hint.scale_y = 0.6
            hint.label(text="Convert to mesh to read 'multiplicity'", icon='INFO')

        elif obj.get("is_metapath_mesh"):
            raw_count = obj.get("num_metapaths_raw", 0)
            unique_count = obj.get("num_metapaths_unique", 0)
            hops = obj.get("metapath_hops", "?")

            info_box.label(text="Metapath Mesh", icon='MESH_DATA')
            info_box.label(text=f"Total: {raw_count:,} | Unique: {unique_count:,} ({hops}-hop)")

            if obj.data and obj.data.attributes.get("multiplicity"):
                info_box.label(text="Attribute 'multiplicity' available", icon='CHECKMARK')
                hint = info_box.box()
                hint.scale_y = 0.6
                hint.label(text="Spreadsheet -> domain 'Edge'", icon='INFO')


class SCIGRAPHS_PT_c2g_metapaths_advanced(bpy.types.Panel):
    """Run the individual pipeline operators (debugging / fine control)."""
    bl_label = "Advanced (Step-by-Step)"
    bl_parent_id = "SCIGRAPHS_PT_c2g_metapaths"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout

        info = layout.box()
        info.scale_y = 0.7
        info.label(text="Run stages individually on the active object", icon='INFO')

        col = layout.column(align=True)
        col.label(text="1 · Street dual graph (from OSMnx)")
        col.operator(
            "scigraphs.create_street_dual_graph",
            icon='MESH_DATA',
            text="Create Street Dual Graph",
        )
        col.separator()
        col.label(text="2 · Bridge amenities (on the dual graph)")
        col.operator(
            "scigraphs.bridge_amenities",
            icon='STICKY_UVS_LOC',
            text="Bridge Amenities",
        )
        col.separator()
        col.label(text="3 · Compute metapaths (by hops)")
        col.operator(
            "scigraphs.compute_metapaths",
            icon='CURVE_DATA',
            text="Compute Metapaths",
        )


classes = [
    SCIGRAPHS_PT_c2g_metapaths,
    SCIGRAPHS_PT_c2g_metapaths_hops,
    SCIGRAPHS_PT_c2g_metapaths_weight,
    SCIGRAPHS_PT_c2g_metapaths_result,
    SCIGRAPHS_PT_c2g_metapaths_advanced,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
