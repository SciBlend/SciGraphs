import bpy


def focus_graph_in_top_view(context, obj):
    """Select an imported graph, switch 3D views to top view, and frame it."""
    if obj is None:
        return

    view_layer = context.view_layer
    for selected in list(context.selected_objects):
        selected.select_set(False)
    obj.select_set(True)
    view_layer.objects.active = obj

    for area in context.screen.areas:
        if area.type != 'VIEW_3D':
            continue

        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        space = next((s for s in area.spaces if s.type == 'VIEW_3D'), None)
        if region is None or space is None:
            continue

        with context.temp_override(area=area, region=region, space_data=space):
            try:
                bpy.ops.view3d.view_axis(type='TOP', align_active=False)
                bpy.ops.view3d.view_selected(use_all_regions=False)
            except RuntimeError:
                continue
