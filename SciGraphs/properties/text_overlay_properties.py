from .callbacks import get_attribute_items, get_system_fonts
from bpy.props import (
    StringProperty,
    EnumProperty,
    BoolProperty,
    IntProperty,
    FloatProperty,
    FloatVectorProperty,
)

TEXT_OVERLAY_PROPERTIES = {
    # ========================================
    # TEXT OVERLAY PROPERTIES
    # ========================================

    'text_overlay_enabled': BoolProperty(
        name="Enable Text Overlay",
        description="Enable text labels overlay for graph nodes",
        default=False,
    ),

    'text_source': EnumProperty(
        name="Text Source",
        description="Source of text for node labels",
        items=[
            ('NODE_ID', "Node ID", "Use node identifier as label"),
            ('ATTRIBUTE', "Attribute", "Use a custom attribute as label"),
        ],
        default='NODE_ID',
    ),

    'text_attribute': EnumProperty(
        name="Text Attribute",
        description="Attribute to use as label (when Text Source is Attribute)",
        items=get_attribute_items,
    ),

    'text_size_mode': EnumProperty(
        name="Size Mode",
        description="How text size is determined",
        items=[
            ('FIXED', "Fixed", "Fixed size in pixels, always legible"),
            ('PROPORTIONAL', "Proportional", "Size scales with distance like 3D objects"),
            ('ADAPTIVE', "Adaptive", "Minimum size guaranteed plus distance scaling"),
        ],
        default='ADAPTIVE',
    ),

    'text_size_fixed': IntProperty(
        name="Fixed Size",
        description="Text size in pixels (for Fixed and Adaptive modes)",
        default=14,
        min=6,
        max=72,
    ),

    'text_size_scale': FloatProperty(
        name="Size Scale",
        description="Scale factor for proportional sizing",
        default=1.0,
        min=0.1,
        max=10.0,
    ),

    'text_max_distance': FloatProperty(
        name="Max Distance",
        description="Maximum distance from camera to show labels (0 = no limit)",
        default=100.0,
        min=0.0,
        max=10000.0,
    ),

    'text_filter_enabled': BoolProperty(
        name="Enable Attribute Filter",
        description="Filter visible labels based on an attribute value",
        default=False,
    ),

    'text_filter_attribute': EnumProperty(
        name="Filter Attribute",
        description="Attribute to use for filtering visibility",
        items=get_attribute_items,
    ),

    'text_filter_operator': EnumProperty(
        name="Filter Operator",
        description="Comparison operator for filtering",
        items=[
            ('GREATER', "Greater Than (>)", "Show nodes with attribute > value"),
            ('LESS', "Less Than (<)", "Show nodes with attribute < value"),
            ('EQUAL', "Equal (=)", "Show nodes with attribute = value"),
            ('NOT_EQUAL', "Not Equal (!=)", "Show nodes with attribute != value"),
            ('GREATER_EQUAL', "Greater or Equal (>=)", "Show nodes with attribute >= value"),
            ('LESS_EQUAL', "Less or Equal (<=)", "Show nodes with attribute <= value"),
        ],
        default='GREATER',
    ),

    'text_filter_value': FloatProperty(
        name="Filter Value",
        description="Threshold value for attribute filtering",
        default=0.0,
    ),

    'text_color': FloatVectorProperty(
        name="Text Color",
        description="Color of the text labels",
        subtype='COLOR',
        default=(1.0, 1.0, 1.0),
        min=0.0,
        max=1.0,
        size=3,
    ),

    'text_background_enabled': BoolProperty(
        name="Text Background",
        description="Add semi-transparent background behind text",
        default=True,
    ),

    'text_background_color': FloatVectorProperty(
        name="Background Color",
        description="Background color for text labels",
        subtype='COLOR',
        default=(0.0, 0.0, 0.0),
        min=0.0,
        max=1.0,
        size=3,
    ),

    'text_background_alpha': FloatProperty(
        name="Background Alpha",
        description="Opacity of text background",
        default=0.7,
        min=0.0,
        max=1.0,
    ),

    'text_depth_occlusion': BoolProperty(
        name="Depth Occlusion",
        description="Hide labels for nodes occluded by geometry",
        default=True,
    ),

    # --- Attribute Format Properties ---

    'text_format_type': EnumProperty(
        name="Format Type",
        description="How to format numeric attribute values",
        items=[
            ('AUTO', "Auto", "Automatically detect integer or float"),
            ('INTEGER', "Integer", "Display as integer (no decimals)"),
            ('FLOAT', "Float", "Display as decimal number"),
            ('SCIENTIFIC', "Scientific", "Scientific notation (e.g., 1.23e+05)"),
            ('PERCENTAGE', "Percentage", "Display as percentage (value * 100)%"),
        ],
        default='AUTO',
    ),

    'text_float_decimals': IntProperty(
        name="Decimal Places",
        description="Number of decimal places for float display",
        default=2,
        min=0,
        max=10,
    ),

    'text_format_prefix': StringProperty(
        name="Prefix",
        description="Text to add before the value",
        default="",
    ),

    'text_format_suffix': StringProperty(
        name="Suffix",
        description="Text to add after the value",
        default="",
    ),

    'text_thousands_separator': BoolProperty(
        name="Thousands Separator",
        description="Add comma as thousands separator",
        default=False,
    ),

    # --- Font Properties ---

    'text_font_source': EnumProperty(
        name="Font Source",
        description="Where to get the font from",
        items=[
            ('SYSTEM', "System Font", "Use a font installed on your system"),
            ('CUSTOM', "Custom File", "Select a custom font file"),
        ],
        default='SYSTEM',
    ),

    'text_font_system': EnumProperty(
        name="System Font",
        description="Select a font from your system",
        items=get_system_fonts,
    ),

    'text_font_custom': StringProperty(
        name="Custom Font",
        description="Path to a custom font file (.ttf, .otf)",
        subtype='FILE_PATH',
        default="",
    ),

    # --- Auto-Update Properties ---

    'text_auto_update': BoolProperty(
        name="Auto Update",
        description="Automatically update overlay when view changes",
        default=False,
    ),

    'text_auto_update_interval': FloatProperty(
        name="Update Interval",
        description="Seconds between auto-updates (lower = more responsive but heavier)",
        default=0.5,
        min=0.1,
        max=5.0,
    ),
}
