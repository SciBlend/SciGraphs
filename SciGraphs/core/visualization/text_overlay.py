# Text overlay generation for graph node labels
#
# This module handles projection of 3D node positions to 2D screen coordinates,
# depth occlusion testing, and PNG image generation with text labels.

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

from ...utils.logger import log


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
    # Number format settings
    format_type: str = 'AUTO'  # 'AUTO', 'INTEGER', 'FLOAT', 'SCIENTIFIC', 'PERCENTAGE'
    float_decimals: int = 2
    format_prefix: str = ""
    format_suffix: str = ""
    thousands_separator: bool = False
    # Font settings
    font_path: str = ""


@dataclass
class ProjectedNode:
    """A node projected to screen coordinates."""
    name: str
    x: float  # Pixel X coordinate
    y: float  # Pixel Y coordinate
    distance: float  # Distance from camera
    visible: bool  # Whether node is within camera view
    occluded: bool  # Whether node is hidden by geometry
    attribute_value: Optional[float]  # Value of filter attribute if applicable


def get_node_positions_from_object(obj) -> Dict[str, Vector]:
    """
    Extract node positions from a graph object.
    
    Reads vertex positions from mesh and maps them to node names
    stored in obj["node_names"] or obj["nodes_data"].
    
    Returns:
        Dictionary mapping node name to world position Vector
    """
    if obj is None or obj.type != 'MESH':
        return {}
    
    mesh = obj.data
    node_names = None
    
    # Try different sources for node names
    node_names_json = obj.get("node_names")
    nodes_data = obj.get("nodes_data")
    
    if node_names_json:
        try:
            node_names = json.loads(node_names_json)
        except json.JSONDecodeError:
            log("Warning: Could not parse node_names JSON")
    
    if node_names is None and nodes_data:
        # nodes_data is comma-separated string (used by geospatial graphs)
        node_names = [n.strip() for n in nodes_data.split(",")]
    
    if node_names is None:
        # Fallback: generate names from vertex indices
        log("Warning: No node names found, using vertex indices")
        node_names = [f"Node_{i}" for i in range(len(mesh.vertices))]
    
    positions = {}
    world_matrix = obj.matrix_world
    
    # Match vertices to node names by index
    for i, vert in enumerate(mesh.vertices):
        if i < len(node_names):
            world_pos = world_matrix @ vert.co
            positions[node_names[i]] = world_pos
    
    return positions


def get_node_attribute_values(obj, attribute_name: str) -> Dict[str, float]:
    """
    Get attribute values for each node from mesh attributes or object properties.
    
    Args:
        obj: Graph object
        attribute_name: Name of the attribute to retrieve
        
    Returns:
        Dictionary mapping node name to attribute value
    """
    if obj is None or not attribute_name:
        return {}
    
    mesh = obj.data
    node_names_json = obj.get("node_names")
    
    if not node_names_json:
        return {}
    
    try:
        node_names = json.loads(node_names_json)
    except json.JSONDecodeError:
        return {}
    
    values = {}
    
    # Check mesh attributes first
    if attribute_name in mesh.attributes:
        attr = mesh.attributes[attribute_name]
        if attr.domain == 'POINT':
            for i, data in enumerate(attr.data):
                if i < len(node_names):
                    # Handle different attribute types
                    if hasattr(data, 'value'):
                        values[node_names[i]] = float(data.value)
                    elif hasattr(data, 'vector'):
                        values[node_names[i]] = data.vector.length
    
    # Check object custom properties as fallback
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
    """
    Project 3D node positions to 2D screen coordinates.
    
    Uses Blender's camera projection matrix to correctly handle
    focal length, sensor size, shift, and camera transformation.
    
    Args:
        obj: Graph object containing nodes
        camera: Camera object to project from
        scene: Current scene
        render_resolution: (width, height) in pixels
        
    Returns:
        List of ProjectedNode with screen coordinates
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
    
    # Get camera data
    cam_data = camera.data
    focal_length = cam_data.lens  # in mm
    sensor_width = cam_data.sensor_width  # in mm
    sensor_height = cam_data.sensor_height  # in mm
    shift_x = cam_data.shift_x  # Lens shift X
    shift_y = cam_data.shift_y  # Lens shift Y
    
    # Determine sensor fit and calculate effective sensor dimensions
    aspect_ratio = width / height
    sensor_aspect = sensor_width / sensor_height
    
    if cam_data.sensor_fit == 'AUTO':
        if width >= height:
            sensor_fit = 'HORIZONTAL'
        else:
            sensor_fit = 'VERTICAL'
    else:
        sensor_fit = cam_data.sensor_fit
    
    # Calculate the view dimensions based on sensor fit
    # This matches Blender's internal calculation
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
    
    # Get camera view matrix (world to camera space)
    modelview_matrix = camera.matrix_world.inverted()
    
    projected = []
    
    for name, world_pos in positions.items():
        # Transform to camera space
        cam_co = modelview_matrix @ world_pos
        
        # cam_co.z is negative when in front of camera
        if cam_co.z >= 0:
            # Behind camera
            projected.append(ProjectedNode(
                name=name, x=0, y=0, distance=0,
                visible=False, occluded=True, attribute_value=None
            ))
            continue
        
        # Perspective projection matching Blender's camera
        depth = -cam_co.z  # Make positive (distance along view axis)
        
        # Project to sensor plane
        # The projection formula: screen_coord = (cam_coord * focal_length) / depth
        # Then normalize to sensor size
        
        if sensor_fit == 'HORIZONTAL':
            # Horizontal fit: sensor_width matches image width
            proj_x = (cam_co.x * focal_length) / (depth * sensor_width / 2.0)
            proj_y = (cam_co.y * focal_length) / (depth * sensor_width / 2.0) * aspect_ratio
        else:
            # Vertical fit: sensor_height matches image height
            proj_x = (cam_co.x * focal_length) / (depth * sensor_height / 2.0) / aspect_ratio
            proj_y = (cam_co.y * focal_length) / (depth * sensor_height / 2.0)
        
        # Apply lens shift (shift is in sensor units, typically -0.5 to 0.5)
        # Shift affects the projection center
        proj_x += shift_x * 2.0
        proj_y += shift_y * 2.0
        
        # Convert from projection space [-1, 1] to normalized [0, 1]
        norm_x = (proj_x + 1.0) / 2.0
        norm_y = (proj_y + 1.0) / 2.0
        
        # Check if node is within frame
        visible = (
            0.0 <= norm_x <= 1.0 and
            0.0 <= norm_y <= 1.0
        )
        
        # Convert to pixel coordinates (flip Y for image coordinates)
        x_px = norm_x * width
        y_px = (1.0 - norm_y) * height
        
        # Calculate actual distance from camera
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
    """
    Test which nodes are occluded by geometry using raycasting.
    
    Casts a ray from camera to each node and checks if it hits
    geometry before reaching the node.
    
    Args:
        context: Blender context
        obj: Graph object (excluded from raycast)
        projected_nodes: List of projected nodes to test
        camera: Camera to cast rays from
        
    Returns:
        Updated list with occluded flag set
    """
    if camera is None:
        return projected_nodes
    
    depsgraph = context.evaluated_depsgraph_get()
    camera_pos = camera.matrix_world.translation
    
    positions = get_node_positions_from_object(obj)
    
    for node in projected_nodes:
        if not node.visible:
            node.occluded = True
            continue
        
        node_pos = positions.get(node.name)
        if node_pos is None:
            node.occluded = True
            continue
        
        # Ray direction from camera to node
        direction = (node_pos - camera_pos).normalized()
        node_distance = (node_pos - camera_pos).length
        
        # Small offset to avoid self-intersection
        ray_origin = camera_pos + direction * 0.01
        
        # Cast ray
        hit, location, normal, index, hit_obj, matrix = context.scene.ray_cast(
            depsgraph, ray_origin, direction
        )
        
        if hit:
            hit_distance = (location - camera_pos).length
            # Node is occluded if hit occurs before reaching the node
            # Allow small tolerance for nodes at surface
            tolerance = 0.1
            node.occluded = hit_distance < (node_distance - tolerance)
        else:
            node.occluded = False
    
    return projected_nodes


def apply_distance_filter(
    projected_nodes: List[ProjectedNode],
    max_distance: float
) -> List[ProjectedNode]:
    """
    Filter out nodes beyond maximum distance.
    
    Args:
        projected_nodes: List of projected nodes
        max_distance: Maximum distance (0 = no limit)
        
    Returns:
        Filtered list of nodes
    """
    if max_distance <= 0:
        return projected_nodes
    
    return [n for n in projected_nodes if n.distance <= max_distance]


def apply_attribute_filter(
    projected_nodes: List[ProjectedNode],
    obj: bpy.types.Object,
    settings: TextOverlaySettings
) -> List[ProjectedNode]:
    """
    Filter nodes based on attribute value comparison.
    
    Args:
        projected_nodes: List of projected nodes
        obj: Graph object to get attributes from
        settings: Filter settings
        
    Returns:
        Filtered list of nodes
    """
    if not settings.filter_enabled or not settings.filter_attribute:
        return projected_nodes
    
    attr_values = get_node_attribute_values(obj, settings.filter_attribute)
    
    if not attr_values:
        log(f"Warning: Attribute '{settings.filter_attribute}' not found")
        return projected_nodes
    
    # Update nodes with attribute values
    for node in projected_nodes:
        node.attribute_value = attr_values.get(node.name)
    
    # Apply filter
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
    """
    Calculate text size for a node based on settings and distance.
    
    Args:
        node: Projected node
        settings: Text overlay settings
        camera: Camera for reference
        
    Returns:
        Text size in pixels
    """
    if settings.size_mode == 'FIXED':
        return settings.fixed_size
    
    # Base size that would appear at distance 1.0
    base_size = settings.fixed_size * settings.size_scale
    
    if settings.size_mode == 'PROPORTIONAL':
        # Size inversely proportional to distance
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
        # Ensure minimum readable size
        return max(settings.fixed_size, min(scaled_size, 200))
    
    return settings.fixed_size


def format_value(value: Any, settings: TextOverlaySettings) -> str:
    """
    Format a value according to the text overlay settings.
    
    Handles integer/float detection, decimal places, scientific notation,
    percentages, thousand separators, and prefix/suffix.
    
    Args:
        value: The value to format (can be string, int, or float)
        settings: Text overlay settings containing format options
        
    Returns:
        Formatted string representation
    """
    # If value is already a string that cannot be parsed as number, return as-is
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
    
    # Apply thousands separator
    if settings.thousands_separator and settings.format_type != 'SCIENTIFIC':
        # Split integer and decimal parts
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
    
    # Apply prefix and suffix
    return f"{settings.format_prefix}{result}{settings.format_suffix}"


def load_font(font_path: str, size: int):
    """
    Load a font from path, with fallback options.
    
    Args:
        font_path: Path to font file (can be empty)
        size: Font size in pixels
        
    Returns:
        PIL ImageFont object
    """
    if not PIL_AVAILABLE:
        return None
    
    # Try custom font path first
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except (IOError, OSError) as e:
            log(f"Could not load custom font {font_path}: {e}")
    
    # Fallback font paths
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
    
    # Last resort: default bitmap font
    log("Warning: Using default bitmap font, text may look pixelated")
    return ImageFont.load_default()


def generate_text_image(
    projected_nodes: List[ProjectedNode],
    resolution: Tuple[int, int],
    settings: TextOverlaySettings,
    camera: bpy.types.Object,
    output_path: Optional[str] = None
) -> Optional[str]:
    """
    Generate PNG image with text labels using Pillow.
    
    Args:
        projected_nodes: List of nodes with screen coordinates
        resolution: (width, height) in pixels
        settings: Text overlay configuration
        camera: Camera for size calculations
        output_path: Path to save image (auto-generated if None)
        
    Returns:
        Path to generated image, or None on failure
    """
    if not PIL_AVAILABLE:
        log("Error: Pillow (PIL) is required for text overlay generation")
        return None
    
    width, height = resolution
    
    # Create transparent RGBA image
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # Convert colors to 0-255 range
    text_color = tuple(int(c * 255) for c in settings.text_color) + (255,)
    bg_color = tuple(int(c * 255) for c in settings.background_color) + (int(settings.background_alpha * 255),)
    
    # Filter to visible, non-occluded nodes
    visible_nodes = [n for n in projected_nodes if n.visible and not n.occluded]
    
    # Sort by distance (furthest first, so closer nodes draw on top)
    visible_nodes.sort(key=lambda n: -n.distance)
    
    # Cache fonts by size to avoid reloading
    font_cache = {}
    
    for node in visible_nodes:
        # Format the text value
        text = format_value(node.name, settings)
        font_size = calculate_text_size(node, settings, camera)
        
        # Get or load font for this size
        if font_size not in font_cache:
            font_cache[font_size] = load_font(settings.font_path, font_size)
        font = font_cache[font_size]
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text on node position
        x = node.x - text_width / 2
        y = node.y - text_height / 2
        
        # Draw background rectangle if enabled
        if settings.background_enabled:
            padding = 3
            draw.rectangle(
                [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
                fill=bg_color
            )
        
        # Draw text
        draw.text((x, y), text, font=font, fill=text_color)
    
    # Generate output path if not provided
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "scigraphs_text_overlay.png")
    
    # Save image
    image.save(output_path, 'PNG')
    log(f"Text overlay saved to: {output_path}")
    
    return output_path


def setup_compositor_overlay(scene: bpy.types.Scene, image_path: str) -> bool:
    """
    Configure compositor to overlay text image on render.
    
    Creates or updates compositor node setup:
    Render Layers -> Alpha Over <- Image (text overlay) -> Output
    
    Compatible with Blender 5.0+ which uses compositing_node_group instead of node_tree.
    
    Args:
        scene: Scene to configure compositor for
        image_path: Path to text overlay image
        
    Returns:
        True on success, False on failure
    """
    log(f"Setting up compositor overlay with image: {image_path}")
    
    tree = None
    is_blender_5 = hasattr(scene, 'compositing_node_group')
    
    if is_blender_5:
        # Blender 5.0+ API: use compositing_node_group
        log("Using Blender 5.0+ compositor API")
        
        # Check if there's an existing node group
        if scene.compositing_node_group is not None:
            tree = scene.compositing_node_group
            log(f"Using existing compositor node group: {tree.name}")
        else:
            # Create new compositor node tree
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
    
    # Load or update image
    image_name = "SciGraphs_TextOverlay"
    
    if image_name in bpy.data.images:
        img = bpy.data.images[image_name]
        img.filepath = image_path
        img.reload()
    else:
        img = bpy.data.images.load(image_path, check_existing=False)
        img.name = image_name
    
    # Find or create nodes
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
    
    # Create render layers if missing
    if render_layers is None:
        render_layers = tree.nodes.new(type='CompositorNodeRLayers')
        render_layers.location = (0, 300)
    
    # Create output node if missing
    if output_node is None:
        if is_blender_5:
            # Blender 5.0 uses NodeGroupOutput
            output_node = tree.nodes.new(type='NodeGroupOutput')
            output_node.location = (600, 300)
            # Create output socket for the node group
            if not any(s.name == 'Image' for s in tree.interface.items_tree if hasattr(s, 'in_out') and s.in_out == 'OUTPUT'):
                tree.interface.new_socket(name='Image', in_out='OUTPUT', socket_type='NodeSocketColor')
        else:
            # Blender 4.x uses CompositorNodeComposite
            output_node = tree.nodes.new(type='CompositorNodeComposite')
            output_node.location = (600, 300)
    
    # Create alpha over node
    if alpha_over is None:
        alpha_over = tree.nodes.new(type='CompositorNodeAlphaOver')
        alpha_over.name = 'SciGraphs_TextAlphaOver'
        alpha_over.label = 'Text Overlay'
        alpha_over.location = (400, 300)
    
    # Create image node
    if image_node is None:
        image_node = tree.nodes.new(type='CompositorNodeImage')
        image_node.name = 'SciGraphs_TextImage'
        image_node.label = 'Text Labels'
        image_node.location = (100, 100)
    
    # Set image
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
    
    # Try Blender 4.x method first (space property)
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
    
    # Clear existing links to alpha over inputs and scale node
    for link in list(tree.links):
        if link.to_node == alpha_over:
            tree.links.remove(link)
        elif link.from_node == alpha_over and link.to_node == output_node:
            tree.links.remove(link)
        elif link.to_node == scale_node:
            tree.links.remove(link)
        elif link.from_node == scale_node:
            tree.links.remove(link)
    
    # Alpha Over node inputs in Blender 5.0:
    # - "Background" : bottom layer (the rendered scene)
    # - "Foreground" : top layer (text labels, drawn over using alpha)
    # - "Factor" : blend factor
    
    # Connect nodes using input names for Blender 5.0 compatibility:
    # Render Layers -> Alpha Over Background (the rendered scene as base)
    if 'Background' in alpha_over.inputs:
        tree.links.new(render_layers.outputs['Image'], alpha_over.inputs['Background'])
    else:
        # Fallback for older Blender versions
        tree.links.new(render_layers.outputs['Image'], alpha_over.inputs[1])
    
    # Text Image -> Scale -> Alpha Over Foreground
    # First connect image to scale node
    tree.links.new(image_node.outputs['Image'], scale_node.inputs['Image'])
    
    # Then connect scale to Alpha Over Foreground
    if 'Foreground' in alpha_over.inputs:
        tree.links.new(scale_node.outputs['Image'], alpha_over.inputs['Foreground'])
    else:
        # Fallback for older Blender versions
        tree.links.new(scale_node.outputs['Image'], alpha_over.inputs[2])
    
    # Alpha Over -> Output
    if output_node.type == 'GROUP_OUTPUT':
        # For Blender 5.0, connect to the Image input of GroupOutput
        if 'Image' in output_node.inputs:
            tree.links.new(alpha_over.outputs['Image'], output_node.inputs['Image'])
    else:
        # For Blender 4.x, connect to Composite node
        tree.links.new(alpha_over.outputs['Image'], output_node.inputs['Image'])
    
    # Find or create Viewer node and connect Alpha Over to it
    viewer_node = None
    for node in tree.nodes:
        if node.type == 'VIEWER':
            viewer_node = node
            break
    
    if viewer_node is None:
        viewer_node = tree.nodes.new(type='CompositorNodeViewer')
        viewer_node.location = (600, 100)
    
    # Connect Alpha Over output to Viewer
    tree.links.new(alpha_over.outputs['Image'], viewer_node.inputs['Image'])
    
    log("Compositor configured for text overlay")
    return True


def remove_compositor_overlay(scene: bpy.types.Scene) -> bool:
    """
    Remove text overlay nodes from compositor.
    
    Compatible with Blender 5.0+ and 4.x.
    
    Args:
        scene: Scene to modify
        
    Returns:
        True on success
    """
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
    
    # Remove image from data
    if "SciGraphs_TextOverlay" in bpy.data.images:
        bpy.data.images.remove(bpy.data.images["SciGraphs_TextOverlay"])
    
    log("Text overlay removed from compositor")
    return True


def get_available_attributes(obj) -> List[str]:
    """
    Get list of available attributes from a graph object.
    
    Args:
        obj: Graph object
        
    Returns:
        List of attribute names
    """
    if obj is None or obj.type != 'MESH':
        return []
    
    attributes = []
    mesh = obj.data
    
    # Mesh attributes
    for attr in mesh.attributes:
        if attr.domain == 'POINT':
            attributes.append(attr.name)
    
    # Object custom properties that look like attributes
    for key in obj.keys():
        if key.startswith("attr_") and key not in attributes:
            attr_name = key[5:]  # Remove "attr_" prefix
            attributes.append(attr_name)
    
    return sorted(set(attributes))


# ---------------------------------------------------------------------------
# Settings snapshot helpers (moved from text_overlay_operators.py)
# ---------------------------------------------------------------------------

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

