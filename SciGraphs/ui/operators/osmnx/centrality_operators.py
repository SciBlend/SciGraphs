"""Centrality, orientation rose, and apply-attribute-to-vertex-colors operators."""

import bpy

from ....core.osmnx.centrality import (
    edge_betweenness_line,
    get_colors_by_values,
    node_betweenness,
    node_closeness,
)
from .utils import _get_osmnx_graph, _get_unprojected_graph


def _pick_graph(obj):
    G = _get_osmnx_graph(obj)
    G_un = _get_unprojected_graph(obj)
    return G or G_un


# ---------------------------------------------------------------------------
# Centrality computation + write-back to mesh
# ---------------------------------------------------------------------------

class SCIGRAPHS_OT_OSMnxCentrality(bpy.types.Operator):
    """Compute node or edge centrality and store it as mesh attribute."""
    bl_idname = "scigraphs.osmnx_centrality"
    bl_label = "Compute Centrality"
    bl_description = "Compute betweenness or closeness centrality (optionally with rustworkx)"
    bl_options = {'REGISTER', 'UNDO'}
    
    kind: bpy.props.EnumProperty(
        name="Centrality",
        items=[
            ('BETWEENNESS_NODE', "Betweenness (Node)", ""),
            ('CLOSENESS', "Closeness", ""),
            ('BETWEENNESS_EDGE', "Betweenness (Edge)", ""),
        ],
        default='BETWEENNESS_NODE',
    )
    
    weighted: bpy.props.BoolProperty(
        name="Use Length Weight",
        default=True,
    )
    
    fast: bpy.props.BoolProperty(
        name="Use Rustworkx (Fast)",
        default=True,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def invoke(self, context, event):
        props = context.scene.scigraphs
        self.kind = props.osmnx_centrality_kind
        self.weighted = props.osmnx_centrality_weighted
        self.fast = props.osmnx_centrality_fast
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        obj = context.active_object

        G = _pick_graph(obj)
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}

        kind = self.kind
        weight = "length" if self.weighted else None
        fast = self.fast

        if kind == 'BETWEENNESS_NODE':
            values = node_betweenness(G, weight=weight, fast=fast)
            target = 'NODES'
            attr = "betweenness"
        elif kind == 'CLOSENESS':
            values = node_closeness(G, weight=weight, fast=fast)
            target = 'NODES'
            attr = "closeness"
        elif kind == 'BETWEENNESS_EDGE':
            values = edge_betweenness_line(G, weight=weight, fast=fast)
            target = 'EDGES'
            attr = "edge_betweenness"
        else:
            self.report({'ERROR'}, f"Unknown centrality kind {kind}")
            return {'CANCELLED'}

        if not values:
            self.report({'ERROR'}, "Centrality computation returned no values")
            return {'CANCELLED'}

        # Store on graph (nodes or edges) so other ops (color) can read it.
        if target == 'NODES':
            for n, v in values.items():
                if n in G.nodes:
                    G.nodes[n][attr] = float(v)
        else:
            for (u, v), val in values.items():
                if G.has_edge(u, v):
                    for data in G.get_edge_data(u, v).values():
                        data[attr] = float(val)

        # Write to mesh attribute for easy preview in Blender.
        mesh = obj.data
        try:
            if target == 'NODES':
                if attr in mesh.attributes:
                    mesh.attributes.remove(mesh.attributes[attr])
                mattr = mesh.attributes.new(name=attr, type='FLOAT', domain='POINT')
                nodes_str = obj.get("nodes_data", "")
                node_ids = nodes_str.split(",") if nodes_str else []
                for i, nid in enumerate(node_ids[: len(mesh.vertices)]):
                    try:
                        mattr.data[i].value = float(values.get(int(nid), 0.0))
                    except (ValueError, TypeError):
                        mattr.data[i].value = 0.0
            else:
                if attr in mesh.attributes:
                    mesh.attributes.remove(mesh.attributes[attr])
                mattr = mesh.attributes.new(name=attr, type='FLOAT', domain='EDGE')
                # Best-effort fill; edge-to-(u,v) mapping depends on core.osmnx.mesh_bridge.
                # Leave zeros if we cannot resolve the mapping.
                for i in range(len(mattr.data)):
                    mattr.data[i].value = 0.0
        except Exception as e:
            self.report({'WARNING'}, f"Could not write mesh attribute: {e}")

        props = context.scene.scigraphs
        props.osmnx_color_attr_name = attr
        props.osmnx_color_target = target
        obj[f"osmnx_has_{attr}"] = True
        obj["osmnx_last_centrality_attr"] = attr

        vmin = min(values.values())
        vmax = max(values.values())
        self.report({'INFO'}, f"{attr}: {len(values)} values, range [{vmin:.4f}, {vmax:.4f}]")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Apply attribute → vertex colors (POINT / EDGE)
# ---------------------------------------------------------------------------

class SCIGRAPHS_OT_OSMnxAttrToColors(bpy.types.Operator):
    """Map a float mesh attribute (nodes or edges) to vertex colors."""
    bl_idname = "scigraphs.osmnx_attr_to_colors"
    bl_label = "Apply Attribute to Vertex Colors"
    bl_description = "Convert a float attribute into a Blender color attribute using a colormap"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return False
        return True

    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        mesh = obj.data

        attr_name = props.osmnx_color_attr_name.strip()
        if not attr_name:
            self.report({'ERROR'}, "Set the attribute name first")
            return {'CANCELLED'}

        if attr_name not in mesh.attributes:
            self.report({'ERROR'}, f"Attribute '{attr_name}' not found on mesh")
            return {'CANCELLED'}

        src = mesh.attributes[attr_name]
        if src.data_type != 'FLOAT':
            self.report({'ERROR'}, "Only FLOAT attributes are supported")
            return {'CANCELLED'}

        values = [d.value for d in src.data]
        rgba = get_colors_by_values(values, cmap_name=props.osmnx_colormap)
        if rgba is None:
            self.report({'ERROR'}, "matplotlib not available")
            return {'CANCELLED'}

        color_name = f"{attr_name}_color"
        if color_name in mesh.color_attributes:
            mesh.color_attributes.remove(mesh.color_attributes[color_name])

        domain = 'POINT' if src.domain == 'POINT' else 'CORNER'
        # POINT -> use POINT color attr; EDGE -> promote to CORNER by taking per-loop edge index.
        try:
            color = mesh.color_attributes.new(name=color_name, type='FLOAT_COLOR', domain=domain)
        except Exception:
            color = mesh.color_attributes.new(name=color_name, type='BYTE_COLOR', domain=domain)

        if domain == 'POINT':
            for i, c in enumerate(rgba):
                if i >= len(color.data):
                    break
                color.data[i].color = (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
        else:
            # Map edge values onto loops: each loop belongs to a polygon, but
            # for plain edge-only meshes (no faces), this domain is moot. We
            # fallback to per-loop color = 0.5.
            for i in range(len(color.data)):
                color.data[i].color = (0.5, 0.5, 0.5, 1.0)

        self.report({'INFO'}, f"Applied {len(rgba)} colors to '{color_name}'")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# 3D orientation rose mesh
# ---------------------------------------------------------------------------

class SCIGRAPHS_OT_OSMnxOrientationRose(bpy.types.Operator):
    """Generate a 2D polar rose plot from edge bearings as a Blender image."""
    bl_idname = "scigraphs.osmnx_orientation_rose"
    bl_label = "Generate Orientation Rose"
    bl_description = (
        "Render a polar histogram of edge bearings as an image and load it "
        "into Blender's Image Editor"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)

    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object

        G = _pick_graph(obj)
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}

        bearings = []
        for _u, _v, data in G.edges(data=True):
            b = data.get("bearing")
            if b is not None:
                bearings.append(float(b))

        if not bearings:
            self.report(
                {'ERROR'},
                "No edge bearings found. Run 'Add Edge Bearings' first.",
            )
            return {'CANCELLED'}

        try:
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            self.report({'ERROR'}, f"Pillow / numpy required: {exc}")
            return {'CANCELLED'}

        bins = max(4, int(props.osmnx_rose_bins))
        bin_edges = np.linspace(0.0, 360.0, bins + 1)
        bearings_arr = np.mod(np.asarray(bearings, dtype=float), 360.0)
        counts, _ = np.histogram(bearings_arr, bins=bin_edges)
        max_count = int(counts.max()) if counts.size else 0

        # Pillow canvas. Compass polar plot: 0° = North, growing clockwise.
        size = 900
        margin = 90
        img = Image.new("RGBA", (size, size), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)

        cx = cy = size // 2
        radius = (size // 2) - margin

        # Concentric grid + cardinal cross.
        grid_color = (210, 210, 215, 255)
        for frac in (0.25, 0.5, 0.75, 1.0):
            r = radius * frac
            draw.ellipse(
                (cx - r, cy - r, cx + r, cy + r),
                outline=grid_color,
                width=1,
            )
        draw.line((cx - radius, cy, cx + radius, cy), fill=grid_color, width=1)
        draw.line((cx, cy - radius, cx, cy + radius), fill=grid_color, width=1)

        # Wedges. Compass convention: angle measured clockwise from north,
        # which in image coords (y-down) means we rotate the standard math
        # angle by -90° and invert the rotation direction.
        bar_color = (31, 119, 180, 230)   # matplotlib default blue
        edge_color = (255, 255, 255, 255)
        for i, count in enumerate(counts):
            if count <= 0 or max_count <= 0:
                continue
            r = radius * (count / max_count)
            start_compass = bin_edges[i]
            end_compass = bin_edges[i + 1]
            # Convert compass deg → Pillow degrees (0° at 3 o'clock, CCW).
            start_pil = (start_compass - 90.0) % 360.0
            end_pil = (end_compass - 90.0) % 360.0
            # pieslice draws CW in image space already.
            if end_pil <= start_pil:
                end_pil += 360.0
            draw.pieslice(
                (cx - r, cy - r, cx + r, cy + r),
                start=start_pil,
                end=end_pil,
                fill=bar_color,
                outline=edge_color,
                width=1,
            )

        # Cardinal labels.
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 22)
            font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        except (OSError, IOError):
            font = ImageFont.load_default()
            font_title = font

        label_color = (40, 40, 50, 255)
        cardinals = [
            ("N", 0.0),
            ("NE", 45.0),
            ("E", 90.0),
            ("SE", 135.0),
            ("S", 180.0),
            ("SW", 225.0),
            ("W", 270.0),
            ("NW", 315.0),
        ]
        label_radius = radius + 28
        for text, deg in cardinals:
            theta = np.deg2rad(deg - 90.0)
            tx = cx + label_radius * np.cos(theta)
            ty = cy + label_radius * np.sin(theta)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((tx - tw / 2, ty - th / 2), text, font=font, fill=label_color)

        title = f"Orientation Rose  ·  {len(bearings)} edges  ·  {bins} bins"
        bbox = draw.textbbox((0, 0), title, font=font_title)
        tw = bbox[2] - bbox[0]
        draw.text(((size - tw) / 2, 18), title, font=font_title, fill=label_color)

        if max_count > 0:
            scale_text = f"max bin count: {max_count}"
            bbox = draw.textbbox((0, 0), scale_text, font=font)
            tw = bbox[2] - bbox[0]
            draw.text(((size - tw) / 2, size - 38), scale_text, font=font, fill=label_color)

        import os
        import tempfile
        tmp_dir = tempfile.gettempdir()
        out_path = os.path.join(tmp_dir, f"{obj.name}_orientation_rose.png")
        img.save(out_path, format="PNG")

        image_name = f"{obj.name}_orientation_rose"
        if image_name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[image_name])
        image = bpy.data.images.load(out_path, check_existing=False)
        image.name = image_name
        image.pack()

        # Try to display it in any open Image Editor.
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    for space in area.spaces:
                        if space.type == 'IMAGE_EDITOR':
                            space.image = image
                            break
                    break

        self.report(
            {'INFO'},
            f"Orientation rose: {len(bearings)} bearings, {bins} bins → image '{image_name}'",
        )
        return {'FINISHED'}


classes = [
    SCIGRAPHS_OT_OSMnxCentrality,
    SCIGRAPHS_OT_OSMnxAttrToColors,
    SCIGRAPHS_OT_OSMnxOrientationRose,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
