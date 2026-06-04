"""Shared UI for feature imports."""


def draw_feature_selector(layout, props, title="Feature Selector"):
    """Draw the shared feature source and tag selector."""
    box = layout.box()
    box.label(text=title, icon='OUTLINER_OB_MESH')

    col = box.column(align=True)
    col.prop(props, "feat_source", text="Source")
    col.prop(props, "feat_type", text="Feature")
    if props.feat_type == 'CUSTOM':
        col.prop(props, "feat_custom_tags", text="Tags")
    col.prop(props, "feat_limit", text="Max Features")
    if props.feat_source == 'OSMNX':
        col.prop(props, "feat_nodes_only", text="Nodes Only")

    info = box.box()
    info.scale_y = 0.7
    if props.feat_source == 'OVERTURE':
        info.label(text="Only features Overture serves are listed (buildings, places, segments, water, land).")
        info.label(text="POI presets filter the Overture places layer by category.")
    else:
        info.label(text="OSMnx queries OpenStreetMap/Overpass with the selected preset or tags.")
        info.label(text="Nodes Only matches the notebook point-feature workflow.")
