# Text overlay for graph node labels: projects 3D node positions to screen,
# tests depth occlusion, and writes a PNG of the labels.

import bpy
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from mathutils import Vector

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from scigraphs_core.logger import log


@dataclass
class TextOverlaySettings:
    """Configuration for text overlay generation."""
    size_mode: str  # 'FIXED', 'PROPORTIONAL', 'ADAPTIVE'
    fixed_size: int
    size_scale: float
    max_distance: float
    text_color: Tuple[float, float, float]
    background_enabled: bool
    background_color: Tuple[float, float, float]
    background_alpha: float
    depth_occlusion: bool
    filter_enabled: bool
    filter_attribute: str
    filter_operator: str
    filter_value: float
    format_type: str = 'AUTO'  # 'AUTO', 'INTEGER', 'FLOAT', 'SCIENTIFIC', 'PERCENTAGE'
    float_decimals: int = 2
    format_prefix: str = ""
    format_suffix: str = ""
    thousands_separator: bool = False
    font_path: str = ""


@dataclass
class ProjectedNode:
    """A node projected to screen coordinates."""
    name: str
    x: float  # Pixel X coordinate
    y: float  # Pixel Y coordinate
    distance: float  # Distance from camera
    visible: bool
    occluded: bool
    attribute_value: Optional[float]


def resolve_node_names(obj) -> Optional[List[str]]:
    """Resolve the node names of a graph object from its stored metadata.

    ``obj["node_names"]`` (a JSON list) first, then ``obj["nodes_data"]`` (the
    comma-separated form geospatial graphs use), then names built from vertex
    indices. Aligned with mesh vertex order; None if the object is not a mesh.
    """
    if obj is None or obj.type != 'MESH':
        return None

    mesh = obj.data
    node_names = None

    node_names_json = obj.get("node_names")
    nodes_data = obj.get("nodes_data")

    if node_names_json:
        try:
            node_names = json.loads(node_names_json)
        except json.JSONDecodeError:
            log("Warning: Could not parse node_names JSON")

    if node_names is None and nodes_data:
        node_names = [n.strip() for n in nodes_data.split(",")]

    if node_names is None:
        log("Warning: No node names found, using vertex indices")
        node_names = [f"Node_{i}" for i in range(len(mesh.vertices))]

    return node_names


def get_node_positions_from_object(obj) -> Dict[str, Vector]:
    """Map node name to world position, reading vertex positions off the mesh."""
    if obj is None or obj.type != 'MESH':
        return {}
    
    mesh = obj.data
    node_names = resolve_node_names(obj)
    if node_names is None:
        return {}
    
    positions = {}
    world_matrix = obj.matrix_world
    
    for i, vert in enumerate(mesh.vertices):
        if i < len(node_names):
            world_pos = world_matrix @ vert.co
            positions[node_names[i]] = world_pos
    
    return positions


def get_node_attribute_values(obj, attribute_name: str) -> Dict[str, float]:
    """Map node name to attribute value, from mesh attributes or object properties."""
    if obj is None or not attribute_name:
        return {}
    
    mesh = obj.data
    node_names = resolve_node_names(obj)
    
    if not node_names:
        return {}
    
    values = {}
    
    if attribute_name in mesh.attributes:
        attr = mesh.attributes[attribute_name]
        if attr.domain == 'POINT':
            for i, data in enumerate(attr.data):
                if i < len(node_names):
                    if hasattr(data, 'value'):
                        values[node_names[i]] = float(data.value)
                    elif hasattr(data, 'vector'):
                        values[node_names[i]] = data.vector.length
    
    elif f"attr_{attribute_name}" in obj:
        attr_data = obj[f"attr_{attribute_name}"]
        if isinstance(attr_data, (list, tuple)):
            for i, val in enumerate(attr_data):
                if i < len(node_names):
                    values[node_names[i]] = float(val)
    
    return values


def project_nodes_to_screen(
    obj,
    camera: bpy.types.Object,
    scene: bpy.types.Scene,
    render_resolution: Tuple[int, int]
) -> List[ProjectedNode]:
    """Project 3D node positions onto 2D screen pixels.

    Reproduces Blender's own camera projection: focal length, sensor size and
    fit, lens shift, and the camera transform. ``render_resolution`` is
    (width, height) in pixels.
    """
    import mathutils
    
    if camera is None or camera.type != 'CAMERA':
        log("Error: No valid camera provided")
        return []
    
    positions = get_node_positions_from_object(obj)
    if not positions:
        return []
    
    width, height = render_resolution
    camera_pos = camera.matrix_world.translation
    
    cam_data = camera.data
    focal_length = cam_data.lens  # in mm
    sensor_width = cam_data.sensor_width  # in mm
    sensor_height = cam_data.sensor_height  # in mm
    shift_x = cam_data.shift_x
    shift_y = cam_data.shift_y
    
    aspect_ratio = width / height
    sensor_aspect = sensor_width / sensor_height
    
    if cam_data.sensor_fit == 'AUTO':
        if width >= height:
            sensor_fit = 'HORIZONTAL'
        else:
            sensor_fit = 'VERTICAL'
    else:
        sensor_fit = cam_data.sensor_fit
    
    # View dimensions per sensor fit, matching Blender's internal calculation.
    if sensor_fit == 'HORIZONTAL':
        view_fac = width / sensor_width
        sensor_size = sensor_width
        pixel_aspect_x = 1.0
        pixel_aspect_y = height / width * sensor_aspect
    else:  # VERTICAL
        view_fac = height / sensor_height
        sensor_size = sensor_height
        pixel_aspect_x = width / height / sensor_aspect
        pixel_aspect_y = 1.0
    
    log(f"Camera: focal_length={focal_length}mm, sensor={sensor_width}x{sensor_height}mm, fit={sensor_fit}")
    log(f"Camera shift: x={shift_x}, y={shift_y}")
    log(f"Render resolution: {width}x{height}, aspect={aspect_ratio:.3f}")
    
    modelview_matrix = camera.matrix_world.inverted()
    
    projected = []
    
    for name, world_pos in positions.items():
        cam_co = modelview_matrix @ world_pos
        
        # cam_co.z is negative in front of the camera.
        if cam_co.z >= 0:
            projected.append(ProjectedNode(
                name=name, x=0, y=0, distance=0,
                visible=False, occluded=True, attribute_value=None
            ))
            continue
        
        depth = -cam_co.z  # Make positive (distance along view axis)
        
        # Sensor-plane projection: (cam_coord * focal_length) / depth, then
        # normalized to sensor size.
        
        if sensor_fit == 'HORIZONTAL':
            # Horizontal fit: sensor_width matches image width
            proj_x = (cam_co.x * focal_length) / (depth * sensor_width / 2.0)
            proj_y = (cam_co.y * focal_length) / (depth * sensor_width / 2.0) * aspect_ratio
        else:
            # Vertical fit: sensor_height matches image height
            proj_x = (cam_co.x * focal_length) / (depth * sensor_height / 2.0) / aspect_ratio
            proj_y = (cam_co.y * focal_length) / (depth * sensor_height / 2.0)
        
        # Lens shift is in sensor units, about -0.5 to 0.5, and moves the center.
        proj_x += shift_x * 2.0
        proj_y += shift_y * 2.0
        
        # Convert from projection space [-1, 1] to normalized [0, 1]
        norm_x = (proj_x + 1.0) / 2.0
        norm_y = (proj_y + 1.0) / 2.0
        
        visible = (
            0.0 <= norm_x <= 1.0 and
            0.0 <= norm_y <= 1.0
        )
        
        # Convert to pixel coordinates (flip Y for image coordinates)
        x_px = norm_x * width
        y_px = (1.0 - norm_y) * height
        
        distance = (world_pos - camera_pos).length
        
        projected.append(ProjectedNode(
            name=name,
            x=x_px,
            y=y_px,
            distance=distance,
            visible=visible,
            occluded=False,  # Will be set by occlusion test
            attribute_value=None
        ))
    
    return projected


def test_depth_occlusion(
    context: bpy.types.Context,
    obj: bpy.types.Object,
    projected_nodes: List[ProjectedNode],
    camera: bpy.types.Object
) -> List[ProjectedNode]:
    """Flag the nodes hidden by geometry, raycasting from the camera to each one.

    Returns the same list with ``occluded`` set.
    """
    if camera is None:
        return projected_nodes
    
    depsgraph = context.evaluated_depsgraph_get()
    camera_pos = camera.matrix_world.translation

    # Tolerance must clear the node's own glyph: the ray is aimed at the center
    # of a sphere that really exists, so it hits the near surface one radius
    # early. A fixed 0.1 only worked while glyphs kept the 0.02 default radius.
    try:
        glyph_radius = float(obj.get("scigraphs_node_size", 0.0) or 0.0)
    except (TypeError, ValueError):
        glyph_radius = 0.0
    tolerance = max(0.1, glyph_radius * 1.5)

    positions = get_node_positions_from_object(obj)
    
    for node in projected_nodes:
        if not node.visible:
            node.occluded = True
            continue
        
        node_pos = positions.get(node.name)
        if node_pos is None:
            node.occluded = True
            continue
        
        direction = (node_pos - camera_pos).normalized()
        node_distance = (node_pos - camera_pos).length
        
        # Small offset to avoid self-intersection
        ray_origin = camera_pos + direction * 0.01
        
        hit, location, normal, index, hit_obj, matrix = context.scene.ray_cast(
            depsgraph, ray_origin, direction
        )
        
        if hit:
            hit_distance = (location - camera_pos).length
            # Node is occluded if the hit occurs before reaching the node
            node.occluded = hit_distance < (node_distance - tolerance)
        else:
            node.occluded = False
    
    return projected_nodes


def declutter_labels(
    projected_nodes: List[ProjectedNode],
    font_size: int,
) -> List[ProjectedNode]:
    """Drop labels whose box would overlap one already accepted.

    The renderer has no collision pass and simply overdraws, so a clustered
    layout turns into stacked, unreadable text. Greedy in the order it is given,
    so sort by importance first: the first label to claim a region keeps it.

    The box is estimated, not measured. Asking Pillow for metrics would mean
    loading the font a second time and duplicating `calculate_text_size`; the
    estimate only has to keep neighbors apart, and it errs wide.
    """
    accepted: List[ProjectedNode] = []
    boxes: List[Tuple[float, float, float, float]] = []
    half_h = max(font_size, 1) * 0.62

    for node in projected_nodes:
        half_w = 0.30 * font_size * max(len(node.name), 1)
        box = (node.x - half_w, node.y - half_h,
               node.x + half_w, node.y + half_h)
        if any(box[0] < b[2] and b[0] < box[2] and
               box[1] < b[3] and b[1] < box[3] for b in boxes):
            continue
        boxes.append(box)
        accepted.append(node)

    return accepted


def apply_distance_filter(
    projected_nodes: List[ProjectedNode],
    max_distance: float
) -> List[ProjectedNode]:
    """Drop nodes beyond ``max_distance``. 0 means no limit."""
    if max_distance <= 0:
        return projected_nodes
    
    return [n for n in projected_nodes if n.distance <= max_distance]


def apply_attribute_filter(
    projected_nodes: List[ProjectedNode],
    obj: bpy.types.Object,
    settings: TextOverlaySettings
) -> List[ProjectedNode]:
    """Keep the nodes whose filter attribute passes the comparison in ``settings``."""
    if not settings.filter_enabled or not settings.filter_attribute:
        return projected_nodes
    
    attr_values = get_node_attribute_values(obj, settings.filter_attribute)
    
    if not attr_values:
        log(f"Warning: Attribute '{settings.filter_attribute}' not found")
        return projected_nodes
    
    for node in projected_nodes:
        node.attribute_value = attr_values.get(node.name)
    
    filtered = []
    for node in projected_nodes:
        if node.attribute_value is None:
            continue
        
        val = node.attribute_value
        threshold = settings.filter_value
        
        passes_filter = False
        if settings.filter_operator == 'GREATER':
            passes_filter = val > threshold
        elif settings.filter_operator == 'LESS':
            passes_filter = val < threshold
        elif settings.filter_operator == 'EQUAL':
            passes_filter = abs(val - threshold) < 0.0001
        elif settings.filter_operator == 'NOT_EQUAL':
            passes_filter = abs(val - threshold) >= 0.0001
        elif settings.filter_operator == 'GREATER_EQUAL':
            passes_filter = val >= threshold
        elif settings.filter_operator == 'LESS_EQUAL':
            passes_filter = val <= threshold
        
        if passes_filter:
            filtered.append(node)
    
    return filtered


def calculate_text_size(
    node: ProjectedNode,
    settings: TextOverlaySettings,
    camera: bpy.types.Object
) -> int:
    """Text size in pixels for one node, from the size mode and its distance."""
    if settings.size_mode == 'FIXED':
        return settings.fixed_size
    
    # Base size that would appear at distance 1.0
    base_size = settings.fixed_size * settings.size_scale
    
    if settings.size_mode == 'PROPORTIONAL':
        if node.distance > 0:
            size = int(base_size * 10.0 / node.distance)
        else:
            size = int(base_size)
        return max(4, min(size, 200))  # Clamp to reasonable range
    
    elif settings.size_mode == 'ADAPTIVE':
        # Minimum size guaranteed, plus scaling
        if node.distance > 0:
            scaled_size = int(base_size * 5.0 / node.distance)
        else:
            scaled_size = int(base_size)
        return max(settings.fixed_size, min(scaled_size, 200))
    
    return settings.fixed_size


def format_value(value: Any, settings: TextOverlaySettings) -> str:
    """Format a value for display, following the overlay's format settings.

    Covers integer/float detection, decimal places, scientific notation,
    percentages, thousands separators and prefix/suffix.
    """
    # A string that will not parse as a number passes through untouched.
    if isinstance(value, str):
        try:
            value = float(value)
        except (ValueError, TypeError):
            return f"{settings.format_prefix}{value}{settings.format_suffix}"
    
    if value is None:
        return ""
    
    result = ""
    
    if settings.format_type == 'AUTO':
        # Detect if value is effectively an integer
        if isinstance(value, float) and value.is_integer():
            result = str(int(value))
        elif isinstance(value, int):
            result = str(value)
        else:
            result = f"{value:.{settings.float_decimals}f}"
            
    elif settings.format_type == 'INTEGER':
        result = str(int(round(value)))
        
    elif settings.format_type == 'FLOAT':
        result = f"{value:.{settings.float_decimals}f}"
        
    elif settings.format_type == 'SCIENTIFIC':
        result = f"{value:.{settings.float_decimals}e}"
        
    elif settings.format_type == 'PERCENTAGE':
        percentage = value * 100
        result = f"{percentage:.{settings.float_decimals}f}%"
    
    if settings.thousands_separator and settings.format_type != 'SCIENTIFIC':
        if '.' in result:
            int_part, dec_part = result.split('.')
            int_part = f"{int(int_part.replace(',', '')):,}"
            result = f"{int_part}.{dec_part}"
        elif '%' in result:
            num_part = result.rstrip('%')
            if '.' in num_part:
                int_part, dec_part = num_part.split('.')
                int_part = f"{int(int_part.replace(',', '')):,}"
                result = f"{int_part}.{dec_part}%"
            else:
                result = f"{int(num_part):,}%"
        else:
            try:
                result = f"{int(result):,}"
            except ValueError:
                pass
    
    return f"{settings.format_prefix}{result}{settings.format_suffix}"


def load_font(font_path: str, size: int):
    """Load a TrueType font, falling back through the common system font paths."""
    if not PIL_AVAILABLE:
        return None
    
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except (IOError, OSError) as e:
            log(f"Could not load custom font {font_path}: {e}")
    
    fallback_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux
        "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Arch Linux
        "C:\\Windows\\Fonts\\arial.ttf",  # Windows
        "C:\\Windows\\Fonts\\segoeui.ttf",  # Windows
        "/Library/Fonts/Arial.ttf",  # macOS
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
    ]
    
    for fallback in fallback_fonts:
        if os.path.exists(fallback):
            try:
                return ImageFont.truetype(fallback, size)
            except (IOError, OSError):
                continue
    
    log("Warning: Using default bitmap font, text may look pixelated")
    return ImageFont.load_default()


def generate_text_image(
    projected_nodes: List[ProjectedNode],
    resolution: Tuple[int, int],
    settings: TextOverlaySettings,
    camera: bpy.types.Object,
    output_path: Optional[str] = None
) -> Optional[str]:
    """Draw the labels into a PNG with Pillow. Returns its path, or None."""
    if not PIL_AVAILABLE:
        log("Error: Pillow (PIL) is required for text overlay generation")
        return None
    
    width, height = resolution
    
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    text_color = tuple(int(c * 255) for c in settings.text_color) + (255,)
    bg_color = tuple(int(c * 255) for c in settings.background_color) + (int(settings.background_alpha * 255),)
    
    visible_nodes = [n for n in projected_nodes if n.visible and not n.occluded]
    
    # Sort by distance (furthest first, so closer nodes draw on top)
    visible_nodes.sort(key=lambda n: -n.distance)
    
    # Cache fonts by size to avoid reloading
    font_cache = {}
    
    for node in visible_nodes:
        text = format_value(node.name, settings)
        font_size = calculate_text_size(node, settings, camera)
        
        if font_size not in font_cache:
            font_cache[font_size] = load_font(settings.font_path, font_size)
        font = font_cache[font_size]
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text on node position
        x = node.x - text_width / 2
        y = node.y - text_height / 2
        
        if settings.background_enabled:
            padding = 3
            draw.rectangle(
                [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
                fill=bg_color
            )
        
        draw.text((x, y), text, font=font, fill=text_color)
    
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "scigraphs_text_overlay.png")
    
    image.save(output_path, 'PNG')
    log(f"Text overlay saved to: {output_path}")
    
    return output_path


def setup_compositor_overlay(scene: bpy.types.Scene, image_path: str) -> bool:
    """Wire the compositor to lay the text image over the render.

    Render Layers -> Alpha Over <- Image, scaled to render size. Works on
    Blender 5.0+, which uses ``compositing_node_group``, and on 4.x, which uses
    ``scene.node_tree``.
    """
    log(f"Setting up compositor overlay with image: {image_path}")
    
    tree = None
    is_blender_5 = hasattr(scene, 'compositing_node_group')
    
    if is_blender_5:
        # Blender 5.0+ API: use compositing_node_group
        log("Using Blender 5.0+ compositor API")
        
        if scene.compositing_node_group is not None:
            tree = scene.compositing_node_group
            log(f"Using existing compositor node group: {tree.name}")
        else:
            tree = bpy.data.node_groups.new("SciGraphs_Compositor", "CompositorNodeTree")
            scene.compositing_node_group = tree
            log("Created new compositor node group")
    else:
        # Blender 4.x API: use scene.node_tree
        log("Using Blender 4.x compositor API")
        scene.use_nodes = True
        tree = getattr(scene, 'node_tree', None)
    
    if tree is None:
        log("Error: Could not access or create compositor node tree")
        return False
    
    image_name = "SciGraphs_TextOverlay"
    
    if image_name in bpy.data.images:
        img = bpy.data.images[image_name]
        img.filepath = image_path
        img.reload()
    else:
        img = bpy.data.images.load(image_path, check_existing=False)
        img.name = image_name
    
    render_layers = None
    output_node = None
    alpha_over = None
    image_node = None
    
    for node in tree.nodes:
        if node.type == 'R_LAYERS':
            render_layers = node
        elif node.type == 'COMPOSITE' or node.type == 'GROUP_OUTPUT':
            output_node = node
        elif node.type == 'ALPHAOVER' and node.name == 'SciGraphs_TextAlphaOver':
            alpha_over = node
        elif node.type == 'IMAGE' and node.name == 'SciGraphs_TextImage':
            image_node = node
    
    if render_layers is None:
        render_layers = tree.nodes.new(type='CompositorNodeRLayers')
        render_layers.location = (0, 300)
    
    if output_node is None:
        if is_blender_5:
            # Blender 5.0 uses NodeGroupOutput
            output_node = tree.nodes.new(type='NodeGroupOutput')
            output_node.location = (600, 300)
            if not any(s.name == 'Image' for s in tree.interface.items_tree if hasattr(s, 'in_out') and s.in_out == 'OUTPUT'):
                tree.interface.new_socket(name='Image', in_out='OUTPUT', socket_type='NodeSocketColor')
        else:
            # Blender 4.x uses CompositorNodeComposite
            output_node = tree.nodes.new(type='CompositorNodeComposite')
            output_node.location = (600, 300)
    
    if alpha_over is None:
        alpha_over = tree.nodes.new(type='CompositorNodeAlphaOver')
        alpha_over.name = 'SciGraphs_TextAlphaOver'
        alpha_over.label = 'Text Overlay'
        alpha_over.location = (400, 300)
    
    if image_node is None:
        image_node = tree.nodes.new(type='CompositorNodeImage')
        image_node.name = 'SciGraphs_TextImage'
        image_node.label = 'Text Labels'
        image_node.location = (100, 100)
    
    image_node.image = img
    
    # Create or find Scale node to ensure text image matches render size
    scale_node = None
    for node in tree.nodes:
        if node.type == 'SCALE' and node.name == 'SciGraphs_TextScale':
            scale_node = node
            break
    
    if scale_node is None:
        scale_node = tree.nodes.new(type='CompositorNodeScale')
        scale_node.name = 'SciGraphs_TextScale'
        scale_node.label = 'Match Render Size'
        scale_node.location = (250, 100)
    
    # Always ensure Scale node is set to Render Size mode
    # Blender 4.x uses 'space' property, Blender 5.0+ uses inputs[1]
    scale_set = False
    
    if hasattr(scale_node, 'space'):
        try:
            scale_node.space = 'RENDER_SIZE'
            if scale_node.space == 'RENDER_SIZE':
                scale_set = True
                log("Scale node set to RENDER_SIZE via 'space' property")
        except (AttributeError, TypeError):
            pass  # Property exists but is read-only or deprecated in this version
    
    # Blender 5.0+ uses inputs[1] for the scale mode (enum socket)
    # Valid values: 'Relative', 'Absolute', 'Scene Size', 'Render Size'
    if not scale_set and len(scale_node.inputs) > 1:
        mode_input = scale_node.inputs[1]
        if hasattr(mode_input, 'default_value'):
            mode_input.default_value = 'Render Size'
            scale_set = True
            log("Scale node set to Render Size via inputs[1] (Blender 5.0+)")
    
    if not scale_set:
        log("Warning: Could not set Scale node to Render Size mode")
    
    for link in list(tree.links):
        if link.to_node == alpha_over:
            tree.links.remove(link)
        elif link.from_node == alpha_over and link.to_node == output_node:
            tree.links.remove(link)
        elif link.to_node == scale_node:
            tree.links.remove(link)
        elif link.from_node == scale_node:
            tree.links.remove(link)
    
    # Blender 5.0 names the Alpha Over inputs "Background" (the rendered scene),
    # "Foreground" (the labels) and "Factor". Connect by name where the name
    # exists, by index otherwise.
    if 'Background' in alpha_over.inputs:
        tree.links.new(render_layers.outputs['Image'], alpha_over.inputs['Background'])
    else:
        tree.links.new(render_layers.outputs['Image'], alpha_over.inputs[1])
    
    # Text Image -> Scale -> Alpha Over Foreground.
    tree.links.new(image_node.outputs['Image'], scale_node.inputs['Image'])
    
    if 'Foreground' in alpha_over.inputs:
        tree.links.new(scale_node.outputs['Image'], alpha_over.inputs['Foreground'])
    else:
        tree.links.new(scale_node.outputs['Image'], alpha_over.inputs[2])
    
    if output_node.type == 'GROUP_OUTPUT':
        if 'Image' in output_node.inputs:
            tree.links.new(alpha_over.outputs['Image'], output_node.inputs['Image'])
    else:
        tree.links.new(alpha_over.outputs['Image'], output_node.inputs['Image'])
    
    viewer_node = None
    for node in tree.nodes:
        if node.type == 'VIEWER':
            viewer_node = node
            break
    
    if viewer_node is None:
        viewer_node = tree.nodes.new(type='CompositorNodeViewer')
        viewer_node.location = (600, 100)
    
    tree.links.new(alpha_over.outputs['Image'], viewer_node.inputs['Image'])
    
    log("Compositor configured for text overlay")
    return True


def remove_compositor_overlay(scene: bpy.types.Scene) -> bool:
    """Remove the text overlay nodes from the compositor. Blender 4.x and 5.0+."""
    tree = None
    is_blender_5 = hasattr(scene, 'compositing_node_group')
    
    if is_blender_5:
        # Blender 5.0+ API
        tree = scene.compositing_node_group
    else:
        # Blender 4.x API
        if not getattr(scene, 'use_nodes', False):
            return True
        tree = getattr(scene, 'node_tree', None)
    
    if tree is None:
        return True
    
    nodes_to_remove = []
    for node in tree.nodes:
        if node.name in ('SciGraphs_TextAlphaOver', 'SciGraphs_TextImage', 'SciGraphs_TextScale'):
            nodes_to_remove.append(node)
    
    for node in nodes_to_remove:
        tree.nodes.remove(node)
    
    # Reconnect render layers to output directly
    render_layers = None
    output_node = None
    
    for node in tree.nodes:
        if node.type == 'R_LAYERS':
            render_layers = node
        elif node.type == 'COMPOSITE' or node.type == 'GROUP_OUTPUT':
            output_node = node
    
    if render_layers and output_node:
        if 'Image' in output_node.inputs:
            tree.links.new(render_layers.outputs['Image'], output_node.inputs['Image'])
    
    if "SciGraphs_TextOverlay" in bpy.data.images:
        bpy.data.images.remove(bpy.data.images["SciGraphs_TextOverlay"])
    
    log("Text overlay removed from compositor")
    return True


def get_available_attributes(obj) -> List[str]:
    """List a graph's POINT attribute names plus its ``attr_``-prefixed properties."""
    if obj is None or obj.type != 'MESH':
        return []
    
    attributes = []
    mesh = obj.data
    
    for attr in mesh.attributes:
        if attr.domain == 'POINT':
            attributes.append(attr.name)
    
    for key in obj.keys():
        if key.startswith("attr_") and key not in attributes:
            attr_name = key[5:]  # Remove "attr_" prefix
            attributes.append(attr_name)
    
    return sorted(set(attributes))


# ---- Settings snapshot helpers ----

def get_font_path(props) -> str:
    """Resolve the font path from the addon property group."""
    if props.text_font_source == 'CUSTOM' and props.text_font_custom:
        return props.text_font_custom
    elif props.text_font_source == 'SYSTEM' and props.text_font_system:
        if props.text_font_system != 'NONE':
            return props.text_font_system
    return ""


def get_settings_snapshot(context):
    """Create a hashable snapshot of all overlay-relevant settings.

    A change in the returned tuple signals that the overlay needs regeneration.
    """
    return get_settings_snapshot_with_object(context, context.active_object)


def get_settings_snapshot_with_object(context, obj):
    """Like :func:`get_settings_snapshot` but accepts an explicit *obj*."""
    props = context.scene.scigraphs
    scene = context.scene
    camera = scene.camera

    camera_matrix = tuple(camera.matrix_world.to_translation()) if camera else None
    camera_rotation = tuple(camera.matrix_world.to_euler()) if camera else None
    camera_lens = camera.data.lens if camera else None
    camera_shift = (camera.data.shift_x, camera.data.shift_y) if camera else None

    obj_matrix = tuple(obj.matrix_world.to_translation()) if obj else None

    render_res = (
        scene.render.resolution_x,
        scene.render.resolution_y,
        scene.render.resolution_percentage,
    )

    text_settings = (
        props.text_source,
        props.text_attribute,
        props.text_size_mode,
        props.text_size_fixed,
        props.text_size_scale,
        props.text_max_distance,
        props.text_depth_occlusion,
        props.text_filter_enabled,
        props.text_filter_attribute,
        props.text_filter_operator,
        props.text_filter_value,
        tuple(props.text_color),
        props.text_background_enabled,
        tuple(props.text_background_color),
        props.text_background_alpha,
        props.text_format_type,
        props.text_float_decimals,
        props.text_format_prefix,
        props.text_format_suffix,
        props.text_thousands_separator,
        props.text_font_source,
        props.text_font_system,
        props.text_font_custom,
    )

    return (
        camera_matrix,
        camera_rotation,
        camera_lens,
        camera_shift,
        obj_matrix,
        render_res,
        text_settings,
    )

