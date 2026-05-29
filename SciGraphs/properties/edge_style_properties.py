from bpy.props import (
    StringProperty,
    EnumProperty,
    BoolProperty,
    IntProperty,
    FloatProperty,
)

EDGE_STYLE_PROPERTIES = {
    # ========================================
    # EDGE STYLE PROPERTIES
    # ========================================

    'edge_style_type': EnumProperty(
        name="Edge Style",
        description="Visual style for graph edges",
        items=[
            ('STRAIGHT', "Straight", "Direct straight lines between nodes"),
            ('CURVED', "Curved (Bezier)", "Smooth cubic Bezier curves"),
            ('QUADRATIC', "Quadratic", "Quadratic Bezier curves (simpler, faster)"),
            ('ARC', "Arc", "Circular arc segments"),
            ('BUNDLED', "Bundled", "Edge bundling for dense graphs (groups similar paths)"),
            ('TAPERED', "Tapered", "Variable thickness along edge (for directed graphs)"),
            ('ORTHOGONAL', "Orthogonal", "Right-angle (90°) connections"),
        ],
        default='STRAIGHT',
    ),

    'edge_style_preset': EnumProperty(
        name="Preset",
        description="Pre-configured edge style settings",
        items=[
            ('CUSTOM', "Custom", "Use manual settings"),
            ('GEPHI_DEFAULT', "Gephi Default", "Gephi-style curved edges"),
            ('CYTOSCAPE_BEZIER', "Cytoscape Bezier", "Cytoscape-style Bezier curves"),
            ('SCHEMATIC', "Schematic", "Technical diagram style with orthogonal edges"),
            ('BUNDLED_DENSE', "Bundled (Dense)", "Strong bundling for very dense graphs"),
            ('FLOW_DIAGRAM', "Flow Diagram", "Tapered edges showing direction"),
            ('MINIMAL', "Minimal", "Clean straight lines"),
        ],
        default='CUSTOM',
    ),

    'edge_curvature': FloatProperty(
        name="Curvature",
        description="Intensity of edge curvature (0 = straight, 1 = maximum curve)",
        default=0.3,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    ),

    'edge_segments': IntProperty(
        name="Segments",
        description="Number of segments for curved edges (more = smoother but heavier)",
        default=8,
        min=2,
        max=32,
    ),

    'edge_curve_direction': EnumProperty(
        name="Curve Direction",
        description="Direction of the curve offset",
        items=[
            ('AUTO', "Auto", "Automatic based on node positions"),
            ('CLOCKWISE', "Clockwise", "Curve bends clockwise"),
            ('COUNTER_CLOCKWISE', "Counter-Clockwise", "Curve bends counter-clockwise"),
            ('ALTERNATING', "Alternating", "Alternate direction for parallel edges"),
        ],
        default='AUTO',
    ),

    'edge_parallel_offset': FloatProperty(
        name="Parallel Offset",
        description="Offset distance for parallel/multi-edges between same nodes",
        default=0.05,
        min=0.0,
        max=0.5,
    ),

    'edge_auto_offset_parallel': BoolProperty(
        name="Auto-Offset Parallel",
        description="Automatically offset parallel edges to avoid overlap",
        default=True,
    ),

    'edge_bundle_strength': FloatProperty(
        name="Bundle Strength",
        description="Strength of edge bundling (0 = no bundling, 1 = maximum)",
        default=0.6,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    ),

    'edge_bundle_iterations': IntProperty(
        name="Bundle Iterations",
        description="Number of iterations for edge bundling algorithm",
        default=6,
        min=1,
        max=20,
    ),

    'edge_bundle_compatibility_threshold': FloatProperty(
        name="Compatibility Threshold",
        description="Minimum compatibility score for edges to be bundled together",
        default=0.6,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    ),

    'edge_taper_start': FloatProperty(
        name="Taper Start",
        description="Relative thickness at edge start (for tapered edges)",
        default=1.0,
        min=0.1,
        max=3.0,
    ),

    'edge_taper_end': FloatProperty(
        name="Taper End",
        description="Relative thickness at edge end (for tapered edges)",
        default=0.3,
        min=0.1,
        max=3.0,
    ),

    'edge_self_loop_radius': FloatProperty(
        name="Self-Loop Radius",
        description="Radius of self-loop edges (edges connecting a node to itself)",
        default=0.2,
        min=0.05,
        max=1.0,
    ),

    'edge_orthogonal_style': EnumProperty(
        name="Orthogonal Style",
        description="Style of orthogonal (right-angle) edges",
        items=[
            ('HORIZONTAL_FIRST', "Horizontal First", "Go horizontal then vertical"),
            ('VERTICAL_FIRST', "Vertical First", "Go vertical then horizontal"),
            ('SHORTEST', "Shortest Path", "Choose based on shortest total length"),
            ('CENTERED', "Centered", "Meet at midpoint with two bends"),
        ],
        default='CENTERED',
    ),

    # ========================================
    # NODE ATTRIBUTE IMPORT
    # ========================================

    'node_attr_filepath': StringProperty(
        name="Node Attribute File",
        description="Path to a file containing vertex-only attributes (node_name, value columns)",
        subtype='FILE_PATH',
    ),

    'node_attr_delimiter': EnumProperty(
        name="Delimiter",
        description="Column separator for the node attribute file",
        items=[
            ('\t', "Tab", "Tab-separated values"),
            (',', "Comma (,)", "Comma-separated values"),
            (';', "Semicolon (;)", "Semicolon-separated values"),
            (' ', "Space", "Space-separated values"),
        ],
        default='\t',
    ),

    'node_attr_has_header': BoolProperty(
        name="Has Header Row",
        description="Whether the first row contains column names",
        default=False,
    ),

    'edge_style_affect_existing': BoolProperty(
        name="Affect Existing Geometry",
        description="Apply style to existing edge geometry (vs only new edges)",
        default=True,
    ),

    'edge_style_preserve_osmnx': BoolProperty(
        name="Preserve OSMnx Curves",
        description="Keep original street geometry for OSMnx graphs instead of overwriting",
        default=True,
    ),
}
