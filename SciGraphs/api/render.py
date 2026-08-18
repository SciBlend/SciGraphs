"""EEVEE render path for SciGraphs graphs (Geometry Nodes, not the GPU preview).

Requires `scene.scigraphs_display_engine = 'GEOMETRY_NODES'`; the GPU preview and
EEVEE are mutually exclusive.

    render.eevee(obj, "/tmp/out.png", color_attribute="node_betweenness",
                 clip_high_pct=98, look='blueprint')
"""

import math
import pathlib
import time

import bpy

from . import graphs, preview

LIGHT_COLLECTION = "SciGraphs_Lights"
KEY_LIGHT = "SciGraphs_Key"
FILL_LIGHT = "SciGraphs_Fill"
RIM_LIGHT = "SciGraphs_Rim"
WORLD_NAME = "SciGraphs_EEVEE_World"


def _material_name(obj):
    return f"{obj.name}_SciGraphsEEVEE"


# TOP_DOWN yields rotation_euler (0,0,0): +X right, +Y up (north up).
TOP_DOWN = (0.0, 0.0, 1.0)
OBLIQUE = (0.42, -0.70, 0.58)

VIEWS = {
    'TOP': {"direction": TOP_DOWN, "projection": 'ORTHO'},
    'OBLIQUE': {"direction": OBLIQUE, "projection": 'PERSP'},
}


# Looks: a preset's colormap may not approach its own backdrop at either end.
# Measured by rendering a 100-node lattice on each backdrop and counting nodes
# that clip: magma on near-black 'ink' put its worst node 4/765 from the
# background, turbo on mid-gray 'terrain' scored 52/765. That rules magma and
# inferno out of the dark looks and the pale-ended maps out of 'paper'.
# relief = terrain colors with low-sun lamps (azimuth/elevation).
PRESETS = {
    'slate': {
        "background": (0.055, 0.060, 0.075),
        "ambient": (0.42, 0.45, 0.52),
        "ambient_strength": 0.45,
        "colormap": "viridis",
        "color": (0.55, 0.60, 0.68),
        "edge_color": (0.16, 0.17, 0.21),
        "key": 2.1, "fill": 0.55, "rim": 0.75,
    },
    'paper': {
        "background": (0.93, 0.93, 0.91),
        "ambient": (0.62, 0.63, 0.66),
        "ambient_strength": 0.35,
        "colormap": "plasma",
        "color": (0.36, 0.39, 0.46),
        "edge_color": (0.22, 0.23, 0.26),
        "key": 1.7, "fill": 0.45, "rim": 0.55,
    },
    'ink': {
        "background": (0.015, 0.015, 0.020),
        "ambient": (0.30, 0.31, 0.38),
        "ambient_strength": 0.40,
        "colormap": "turbo",
        "color": (0.50, 0.52, 0.58),
        "edge_color": (0.13, 0.12, 0.15),
        "key": 2.2, "fill": 0.55, "rim": 0.85,
    },
    'blueprint': {
        "background": (0.035, 0.065, 0.115),
        "ambient": (0.34, 0.44, 0.62),
        "ambient_strength": 0.50,
        "colormap": "cividis",
        "color": (0.45, 0.56, 0.70),
        "edge_color": (0.14, 0.20, 0.30),
        "key": 2.0, "fill": 0.60, "rim": 0.80,
    },
    'terrain': {
        "background": (0.270, 0.250, 0.220),
        "ambient": (0.50, 0.48, 0.44),
        "ambient_strength": 0.55,
        "colormap": "inferno",
        "color": (0.62, 0.58, 0.52),
        "edge_color": (0.22, 0.20, 0.18),
        "key": 2.0, "fill": 0.60, "rim": 0.70,
    },
    'relief': {
        "background": (0.270, 0.250, 0.220),
        "ambient": (0.46, 0.47, 0.52),
        "ambient_strength": 0.22,
        "colormap": "inferno",
        "color": (0.80, 0.76, 0.70),
        "edge_color": (0.26, 0.24, 0.21),
        "key": 3.0, "fill": 0.35, "rim": 0.30,
        "azimuth": 315.0, "elevation": 22.0, "softness": 2.0,
    },
}

DEFAULT_PRESET = 'slate'


def preset(name=DEFAULT_PRESET):
    """Named look as a shallow copy (mutating one key does not edit PRESETS)."""
    return dict(PRESETS.get(str(name or DEFAULT_PRESET).lower(),
                            PRESETS[DEFAULT_PRESET]))


# Same fraction as preview.autoscale; world-space radii for GN spheres/tubes.
NODE_FRACTION = preview._NODE_FRACTION  # 0.35
EDGE_RATIO_MAX = 0.25
EDGE_RATIO_MIN = 0.05


def node_cloud(obj):
    """Node positions in object space as `(positions, source)`, preferring
    `node_positions`, else `is_intersection` (OSMnx street meshes), else all vertices.
    """
    import numpy as np

    positions = obj.get("node_positions")
    if positions is not None and len(positions) >= 3:
        return (np.asarray(list(positions), dtype=np.float32).reshape(-1, 3),
                "node_positions")

    coords = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", coords)
    coords = coords.reshape(-1, 3)

    marker = obj.data.attributes.get("is_intersection")
    if marker is not None and getattr(marker, "domain", "") == 'POINT':
        flags = np.empty(len(obj.data.vertices), dtype=np.int32)
        try:
            marker.data.foreach_get("value", flags)
        except (RuntimeError, TypeError, ValueError):
            return coords, "vertices"
        nodes = coords[flags != 0]
        if len(nodes) >= 2:
            return nodes, "is_intersection"
    return coords, "vertices"


def measure(obj):
    """Like `preview.extent(obj)`, but over `node_cloud()` (OSMnx-safe)."""
    import numpy as np

    cloud, source = node_cloud(obj)
    if source == "node_positions" or len(cloud) == 0:
        return preview.extent(obj), source

    lo, hi = cloud.min(axis=0), cloud.max(axis=0)
    size = hi - lo
    diagonal = float(np.linalg.norm(size))

    sample = cloud
    if len(cloud) > 2000:
        sample = cloud[::len(cloud) // 2000][:2000]
    d = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    positive = nn[np.isfinite(nn) & (nn > 0)]
    median_nn = float(np.median(positive)) if len(positive) else diagonal / 50.0

    scale = diagonal if diagonal > 0 else 1.0
    median_nn = float(np.clip(median_nn or scale / 50.0,
                              scale / 800.0, scale / 6.0))
    return ((lo + hi) / 2.0, size, diagonal, median_nn), source


def autoscale_geometry(obj, node_fraction=NODE_FRACTION, node_resolution=16,
                       edge_resolution=8, edge_ratio=None, verbose=True):
    """Size GN glyphs from median nearest-neighbor distance and rebuild the tree.

    The object props it writes are authoritative; the scene viz knobs are best
    effort, since they clamp and their sockets may be missing.
    """
    measurements, source = measure(obj)
    if measurements is None:
        return {}
    _center, _size, diagonal, median_nn = measurements

    node_radius = median_nn * float(node_fraction)

    edge_count = float(obj.get("num_edges") or 0)
    node_count = float(obj.get("num_nodes") or 1)
    mean_degree = 2.0 * edge_count / max(node_count, 1.0)
    if edge_ratio is None:
        edge_ratio = max(EDGE_RATIO_MIN,
                         min(EDGE_RATIO_MAX, 1.2 / max(mean_degree, 1.0)))
    edge_ratio = float(edge_ratio)
    edge_radius = node_radius * edge_ratio

    obj["scigraphs_node_size"] = float(node_radius)
    obj["scigraphs_node_resolution"] = int(node_resolution)
    obj["scigraphs_node_shade_smooth"] = True
    obj["scigraphs_edge_thickness"] = float(edge_radius)
    obj["scigraphs_edge_resolution"] = int(edge_resolution)

    viz = getattr(bpy.context.scene, "scigraphs_viz", None)
    if viz is not None:
        try:
            viz.node_scale = float(node_radius)
            viz.edge_thickness = float(edge_radius)
            viz.node_resolution = int(node_resolution)
            viz.edge_resolution = int(edge_resolution)
        except (TypeError, ValueError):
            pass

    values = {
        "diagonal": diagonal,
        "median_nearest_neighbor": median_nn,
        "node_radius": node_radius,
        "edge_radius": edge_radius,
        "edge_ratio": edge_ratio,
        "mean_degree": mean_degree,
        "node_source": source,
    }
    if verbose:
        print(f"  graph extent           {diagonal:.4f} Blender units")
        print(f"  nearest neighbor      {median_nn:.5f} (median, "
              f"over {source})")
        print(f"  node radius            {node_radius:.5f}")
        print(f"  edge radius            {edge_radius:.5f} "
              f"({edge_ratio:.2f} of the node, mean degree {mean_degree:.1f})")
    return values


def geometry_nodes(obj, rebuild=True):
    """Switch to the Geometry Nodes path, rebuilding SciGraphs_Viz when sizes
    change, and set `show_render`, which `apply_display_engine` leaves alone.
    """
    scene = bpy.context.scene
    graphs.activate(obj)

    mod = obj.modifiers.get("SciGraphs_Viz")
    if mod is None or rebuild:
        bpy.ops.scigraphs.setup_visualization(target='FULL')
        mod = obj.modifiers.get("SciGraphs_Viz")
        _restore_coloring(obj)

    if scene.scigraphs_display_engine != 'GEOMETRY_NODES':
        scene.scigraphs_display_engine = 'GEOMETRY_NODES'
    if getattr(scene, "scigraphs_preview_enabled", False):
        scene.scigraphs_preview_enabled = False

    if mod is not None:
        mod.show_viewport = True
        mod.show_render = True

    # GPU preview may leave display_type='BOUNDS' (viewport-only; looks broken).
    if obj.display_type == 'BOUNDS':
        obj.display_type = 'TEXTURED'
    return mod


def _restore_coloring(obj):
    """Re-wire coloring after viz rebuild (scigraphs_is_node is otherwise lost)."""
    if obj is None or not obj.get("scigraphs_color_attr"):
        return False
    try:
        from SciGraphs.ui.coloring.functions import reapply_coloring_after_viz_rebuild
    except Exception:  # noqa: BLE001 - coloring is optional
        return False
    try:
        return bool(reapply_coloring_after_viz_rebuild(obj))
    except Exception:  # noqa: BLE001 - never fail a render over colors
        return False


def attribute_domain(obj, attribute):
    """`'POINT'`, `'EDGE'`, ... or None if absent."""
    if obj is None or obj.type != 'MESH' or not attribute:
        return None
    attr = obj.data.attributes.get(attribute)
    if attr is None:
        return None
    return (getattr(attr, "domain", "") or "POINT").upper()


def protect_attribute(obj, attribute):
    """Set `scigraphs_color_attr` so GN rebuild does not strip this attribute."""
    if obj is None or not attribute:
        return False
    obj["scigraphs_color_attr"] = str(attribute)
    return True


def color_graph(obj, attribute, colormap="viridis", reverse=False, norm=None,
                gamma=None, vmin=None, vmax=None, clip_low_pct=None,
                clip_high_pct=None, nodes_only=None, edge_color=None,
                verbose=True):
    """Map a mesh attribute through a colormap for EEVEE, returning its name or None.

    On POINT, nodes_only gates tubes to edge_color; on EDGE the gate is off and
    values are promoted to points. Prefer clip_*_pct for heavy-tailed attributes.
    """
    domain = attribute_domain(obj, attribute)
    if domain is None:
        if verbose:
            names = [a.name for a in obj.data.attributes] if obj and obj.type == 'MESH' else []
            print(f"  no attribute '{attribute}' on this mesh; have {names}")
        return None

    props = getattr(bpy.context.scene, "scigraphs_coloring", None)
    if props is None:
        if verbose:
            print("  coloring properties are not registered")
        return None

    graphs.activate(obj)
    protect_attribute(obj, attribute)

    props.colormap = str(colormap)
    props.reverse = bool(reverse)
    props.color_domain = 'AUTO'
    props.color_attribute_name = ""
    props.auto_setup_material = True
    props.opacity = 1.0
    if norm is not None:
        props.color_norm = str(norm).upper()
    if gamma is not None:
        props.color_gamma = float(gamma)
    props.clip_low_pct = float(clip_low_pct) if clip_low_pct is not None else 0.0
    props.clip_high_pct = float(clip_high_pct) if clip_high_pct is not None else 100.0

    if vmin is None and vmax is None:
        props.auto_range = True
    else:
        props.auto_range = False
        if vmin is not None:
            props.vmin = float(vmin)
        if vmax is not None:
            props.vmax = float(vmax)

    if nodes_only is None:
        nodes_only = (domain == 'POINT')
    props.nodes_only = bool(nodes_only)
    if edge_color is not None:
        rgb = tuple(float(v) for v in tuple(edge_color)[:3])
        props.edge_color = (*rgb, 1.0)

    # Use the operator so it matches the registered PropertyGroup (not a stale import).
    result = bpy.ops.scigraphs.color_set_attribute(attribute=attribute)
    if 'FINISHED' not in result:
        if verbose:
            print(f"  coloring '{attribute}' was {'/'.join(result)}")
        return None
    if verbose:
        print(f"  color                 {attribute} ({domain.lower()}) "
              f"-> {colormap}{' reversed' if reverse else ''} "
              f"[{props.last_vmin:.4g} … {props.last_vmax:.4g}]")
    return attribute


def color_range(obj):
    """Mapped range after clipping/log, or None."""
    props = getattr(bpy.context.scene, "scigraphs_coloring", None)
    if props is None or obj is None:
        return None
    if not obj.get("scigraphs_last_color_attribute"):
        return None
    return (float(props.last_vmin), float(props.last_vmax))


def _color_attribute(obj):
    """Color layer name, or None (only when no color_apply material in slot 0)."""
    if obj is None or obj.type != 'MESH':
        return None
    recorded = obj.get("scigraphs_last_color_attribute") or ""
    layers = getattr(obj.data, "color_attributes", None)
    if layers is None:
        return None
    if recorded and recorded in layers:
        return recorded
    active = getattr(layers, "active_color", None)
    if active is not None:
        return active.name
    return layers[0].name if len(layers) else None


def material(obj, color=(0.55, 0.60, 0.68), roughness=0.42, metallic=0.0):
    """Ensure a material, never overwriting a slot 0 this module does not own.

    Reads the color layer through an Attribute node, not Vertex Color, because
    only the former survives Realize Instances.
    """
    existing = None
    if obj.data.materials and obj.data.materials[0] is not None:
        existing = obj.data.materials[0]
    if existing is not None and not existing.name.startswith(_material_name(obj)):
        return existing

    mat = bpy.data.materials.get(_material_name(obj))
    if mat is None:
        mat = bpy.data.materials.new(_material_name(obj))
    mat.use_nodes = True

    tree = mat.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)

    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (300, 0)
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    bsdf.inputs['Roughness'].default_value = float(roughness)
    bsdf.inputs['Metallic'].default_value = float(metallic)
    tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    layer = _color_attribute(obj)
    if layer:
        attr = tree.nodes.new("ShaderNodeAttribute")
        attr.attribute_type = 'GEOMETRY'
        attr.attribute_name = layer
        attr.location = (-260, 0)
        tree.links.new(attr.outputs['Color'], bsdf.inputs['Base Color'])
    else:
        bsdf.inputs['Base Color'].default_value = (*color, 1.0)

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    _wire_material(obj, mat)
    return mat


def _wire_material(obj, mat):
    """Fill empty Set Material sockets (None overrides slot 0 with default gray)."""
    mod = obj.modifiers.get("SciGraphs_Viz")
    group = getattr(mod, "node_group", None) if mod else None
    if group is None:
        return False
    wired = False
    for node in group.nodes:
        if getattr(node, "bl_idname", "") != 'GeometryNodeSetMaterial':
            continue
        socket = node.inputs.get('Material')
        if socket is not None and socket.default_value is None:
            socket.default_value = mat
            wired = True
    return wired


def _aim(obj, direction):
    """Rotate a light so it shines from `direction` toward the origin."""
    from mathutils import Vector

    d = Vector(direction).normalized()
    obj.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    return d


def sun_direction(azimuth_deg, elevation_deg):
    """Unit vector toward the sun; azimuth clockwise from north (+Y)."""
    a = math.radians(float(azimuth_deg))
    e = math.radians(float(elevation_deg))
    return (math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e))


def light_rig(center, diagonal, key=2.1, fill=0.55, rim=0.75, softness=6.0,
              azimuth=None, elevation=None):
    """Create or update three named SUN lamps (irradiance, so scale-independent).

    Pass azimuth and elevation for hillshade, e.g. 315/22. Energies near
    2.1/0.55/0.75 avoid clipping under the Standard view transform.
    """
    from mathutils import Vector

    coll = graphs.collection(LIGHT_COLLECTION)
    center = Vector(tuple(float(v) for v in center))
    distance = max(float(diagonal), 1e-6) * 1.5

    if azimuth is None or elevation is None:
        setup = (
            (KEY_LIGHT, (-0.45, -0.55, 0.70), key, softness),
            (FILL_LIGHT, (0.65, -0.25, 0.35), fill, softness * 2.0),
            (RIM_LIGHT, (0.10, 0.80, 0.30), rim, softness),
        )
    else:
        setup = (
            (KEY_LIGHT, sun_direction(azimuth, elevation), key, softness),
            (FILL_LIGHT, sun_direction(azimuth + 180.0,
                                       min(75.0, float(elevation) + 45.0)),
             fill, softness * 2.0),
            (RIM_LIGHT, sun_direction(azimuth + 100.0,
                                      max(8.0, float(elevation) - 6.0)),
             rim, softness),
        )

    lights = []
    for name, direction, energy, angle in setup:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != 'LIGHT':
            data = bpy.data.lights.get(name) or bpy.data.lights.new(name, type='SUN')
            data.type = 'SUN'
            obj = bpy.data.objects.new(name, data)
        if obj.name not in coll.objects:
            coll.objects.link(obj)
        obj.data.type = 'SUN'
        obj.data.energy = float(energy)
        obj.data.angle = math.radians(float(angle))
        obj.data.color = (1.0, 1.0, 1.0)
        d = _aim(obj, direction)
        obj.location = center + d * distance
        lights.append(obj)
    return lights


def world(background=(0.055, 0.060, 0.075), ambient=(0.42, 0.45, 0.52),
          ambient_strength=0.45):
    """World that splits the camera backdrop from the lighting through Light Path
    / Is Camera Ray, so no film_transparent pass is needed.
    """
    wld = bpy.data.worlds.get(WORLD_NAME) or bpy.data.worlds.new(WORLD_NAME)
    wld.use_nodes = True
    tree = wld.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)

    output = tree.nodes.new("ShaderNodeOutputWorld")
    output.location = (400, 0)

    view = tree.nodes.new("ShaderNodeBackground")
    view.location = (140, -140)
    view.inputs['Color'].default_value = (*tuple(background)[:3], 1.0)
    view.inputs['Strength'].default_value = 1.0

    light = tree.nodes.new("ShaderNodeBackground")
    light.location = (140, 60)
    light.inputs['Color'].default_value = (*tuple(ambient)[:3], 1.0)
    light.inputs['Strength'].default_value = float(ambient_strength)

    path = tree.nodes.new("ShaderNodeLightPath")
    path.location = (-120, 220)

    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.location = (280, 0)
    tree.links.new(path.outputs['Is Camera Ray'], mix.inputs['Fac'])
    tree.links.new(light.outputs['Background'], mix.inputs[1])
    tree.links.new(view.outputs['Background'], mix.inputs[2])
    tree.links.new(mix.outputs['Shader'], output.inputs['Surface'])

    bpy.context.scene.world = wld
    return wld


def _is_straight_down(direction, tolerance=1e-6):
    """True when direction is +Z (camera looks down -Z)."""
    x, y, z = (float(v) for v in tuple(direction)[:3])
    return z > 0.0 and (x * x + y * y) <= tolerance * max(z * z, 1.0)


def _fit_points(obj, pad):
    """World-space points for camera_fit_coords: nodes plus padded AABB corners."""
    import numpy as np

    local, _source = node_cloud(obj)
    if len(local) == 0:
        return None

    matrix = np.array(obj.matrix_world.transposed())
    world = local @ matrix[:3, :3] + matrix[3, :3]

    lo, hi = world.min(axis=0) - pad, world.max(axis=0) + pad
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return np.vstack([world, corners])


def _frame_extent(cam, scene=None):
    """(width, height) of an orthographic frame in Blender units."""
    scene = scene or bpy.context.scene
    width_px = float(scene.render.resolution_x)
    height_px = float(scene.render.resolution_y)
    scale = float(cam.data.ortho_scale)
    if width_px >= height_px:
        return scale, scale * height_px / max(width_px, 1.0)
    return scale * width_px / max(height_px, 1.0), scale


def _clamp_to_ground(cam, ground, points, verbose=False):
    """Shrink ortho frame to ground coverage; warn if the graph no longer fits."""
    from mathutils import Vector

    corners = [ground.matrix_world @ Vector(c) for c in ground.bound_box]
    gx = [c.x for c in corners]
    gy = [c.y for c in corners]

    scene = bpy.context.scene
    before = float(cam.data.ortho_scale)
    width, height = _frame_extent(cam, scene)
    aspect = width / height if height else 1.0

    room_x = 2.0 * min(cam.location.x - min(gx), max(gx) - cam.location.x)
    room_y = 2.0 * min(cam.location.y - min(gy), max(gy) - cam.location.y)
    allowed = min(room_x, room_y * aspect)   # as a *width*
    allowed_scale = allowed if scene.render.resolution_x >= scene.render.resolution_y \
        else allowed / aspect

    covered_x = max(0.0, min(max(gx), cam.location.x + width / 2)
                    - max(min(gx), cam.location.x - width / 2))
    covered_y = max(0.0, min(max(gy), cam.location.y + height / 2)
                    - max(min(gy), cam.location.y - height / 2))
    bare = max(0.0, 1.0 - (covered_x * covered_y) / max(width * height, 1e-12))

    if 0 < allowed_scale < before:
        cam.data.ortho_scale = allowed_scale
    after = float(cam.data.ortho_scale)

    width, height = _frame_extent(cam, scene)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    fits = (lo[0] >= cam.location.x - width / 2 - 1e-9
            and hi[0] <= cam.location.x + width / 2 + 1e-9
            and lo[1] >= cam.location.y - height / 2 - 1e-9
            and hi[1] <= cam.location.y + height / 2 + 1e-9)

    if verbose or not fits:
        print(f"  ground clamp           {before:.4f} -> {after:.4f} BU "
              f"({bare * 100:.1f}% of the frame would have been bare backdrop)")
    if not fits:
        print("  WARNING: the ground is too small to cover a frame that holds "
              "the graph, the figure is now cropping the network. Build the "
              "ground wider; see FRAME_MARGIN.")
    return {"scale_before": before, "scale_after": after,
            "bare_fraction": float(bare), "graph_fits": bool(fits)}


# Ortho frame wider than graph per axis; ground must cover aspect*margin (see ground=).
FRAME_MARGIN = 1.06


def frame_camera(obj, direction=None, margin=FRAME_MARGIN, lens=50.0,
                 projection=None, view='TOP', pushback=1.0, ground=None,
                 verbose=False):
    """Frame the camera for EEVEE, orthographic top-down by default.

    Roll is leveled for a straight-down view, which otherwise flips north, and
    `ground` clamps the ortho frame to the terrain's coverage.
    """
    from mathutils import Vector

    scene = bpy.context.scene

    spec = VIEWS.get(str(view).upper(), VIEWS['TOP'])
    if direction is None:
        direction = spec["direction"]
    if projection is None:
        projection = spec["projection"]
    ortho = str(projection).upper().startswith('ORTHO')

    node_radius = float(obj.get("scigraphs_node_size", 0.0) or 0.0)

    cam = bpy.data.objects.get("SciGraphs_Camera")
    if cam is None or cam.type != 'CAMERA':
        cam = bpy.data.objects.new("SciGraphs_Camera",
                                   bpy.data.cameras.new("SciGraphs_Camera"))
        scene.collection.objects.link(cam)
    cam.data.type = 'ORTHO' if ortho else 'PERSP'

    saved = getattr(scene, "scigraphs_preview_impostor_radius", None)
    try:
        if saved is not None:
            scene.scigraphs_preview_impostor_radius = node_radius
        cam = preview.frame_camera(obj, direction=direction, margin=margin,
                                lens=lens)
    finally:
        if saved is not None:
            scene.scigraphs_preview_impostor_radius = saved

    if cam is None:
        return cam

    # track_quat for +Z rolls 180°; force north-up for overhead.
    if _is_straight_down(direction):
        cam.rotation_euler = (0.0, 0.0, 0.0)

    if not ortho:
        return cam

    measurements, _source = measure(obj)
    if measurements is None:
        return cam
    center, _size, diagonal, _nn = measurements

    points = _fit_points(obj, node_radius)
    if points is None:
        return cam

    depsgraph = bpy.context.evaluated_depsgraph_get()
    flat = points.ravel().tolist()

    # Ortho: derive scale from camera-space extent (camera_fit_coords depends on prior state).
    import numpy as np

    location, _fitted = cam.camera_fit_coords(depsgraph, flat)

    basis = cam.matrix_world.to_3x3().normalized()
    right, up = basis.col[0], basis.col[1]
    pts = points.reshape(-1, 3)
    xs = pts @ np.array([right.x, right.y, right.z])
    ys = pts @ np.array([up.x, up.y, up.z])
    width = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())

    render = scene.render
    aspect = ((render.resolution_x * render.pixel_aspect_x)
              / max(render.resolution_y * render.pixel_aspect_y, 1e-9))
    needed = max(width, height * aspect) if aspect >= 1.0 \
        else max(width / aspect, height)

    cam.data.ortho_scale = max(needed * float(margin), 1e-6)

    center_offset = (right * ((xs.max() + xs.min()) * 0.5 - Vector(location).dot(right))
                     + up * ((ys.max() + ys.min()) * 0.5 - Vector(location).dot(up)))
    location = Vector(location) + center_offset

    d = Vector(tuple(float(v) for v in direction)).normalized()
    cam.location = Vector(location) + d * (diagonal * float(pushback))

    if ground is not None and _is_straight_down(direction):
        _clamp_to_ground(cam, ground, points, verbose=verbose)

    distance = (cam.location - Vector(tuple(float(v) for v in center))).length
    cam.data.clip_start = max(distance * 1e-4, 1e-5)
    cam.data.clip_end = max((distance + diagonal) * 4.0, 100.0)
    return cam


def eevee_settings(scene, samples=32, shadows=True, raytracing=False,
                   ao_distance=None):
    """Set EEVEE Next (5.x) properties through a guarded setattr, since the 4.x names raise.

    Keep `ao_distance` down to a few node radii or ambient occlusion muddies the
    whole graph.
    """
    def _set(target, name, value):
        if hasattr(target, name):
            try:
                setattr(target, name, value)
                return True
            except (TypeError, ValueError, AttributeError):
                return False
        return False

    applied = {}
    eevee = scene.eevee

    applied['taa_render_samples'] = _set(eevee, "taa_render_samples", int(samples))
    applied['taa_samples'] = _set(eevee, "taa_samples", int(samples))

    applied['use_shadows'] = _set(eevee, "use_shadows", bool(shadows))
    if shadows:
        _set(eevee, "shadow_ray_count", 2)
        _set(eevee, "shadow_step_count", 6)
        _set(eevee, "shadow_resolution_scale", 1.0)

    # AO only (not GI): color bleeding between categorical nodes lies about the data.
    applied['use_fast_gi'] = _set(eevee, "use_fast_gi", True)
    _set(eevee, "fast_gi_method", 'AMBIENT_OCCLUSION')
    _set(eevee, "fast_gi_ray_count", 2)
    _set(eevee, "fast_gi_step_count", 8)
    if ao_distance is not None:
        applied['fast_gi_distance'] = _set(eevee, "fast_gi_distance",
                                           float(ao_distance))

    applied['use_raytracing'] = _set(eevee, "use_raytracing", bool(raytracing))
    _set(eevee, "clamp_surface_indirect", 10.0)
    return applied


def eevee(obj, filename, resolution=(1000, 750), samples=32,
          direction=None, background=None,
          ambient=None, ambient_strength=None,
          node_fraction=NODE_FRACTION, node_resolution=16,
          edge_resolution=8, edge_ratio=None, autosize=True,
          isolate=True, hide=None,
          shadows=True, raytracing=False, view_transform='Standard',
          exposure=0.0, color=None, lens=50.0,
          verbose=True, shrink=True,
          look=DEFAULT_PRESET, view='TOP', projection=None,
          pushback=1.0, margin=FRAME_MARGIN, ground=None,
          color_attribute=None, colormap=None, color_reverse=False,
          color_norm=None, color_gamma=None, vmin=None, vmax=None,
          clip_low_pct=None, clip_high_pct=None, nodes_only=None,
          edge_color=None,
          key=None, fill=None, rim=None,
          azimuth=None, elevation=None, softness=None):
    """Render the graph with EEVEE and return the output path.

    view_transform defaults to Standard because AgX desaturates the ends of a
    colormap. `isolate` and `hide` work as in `preview.render`.
    """
    scene = bpy.context.scene
    path = pathlib.Path(filename)
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    chosen = preset(look)
    if background is None:
        background = chosen["background"]
    if ambient is None:
        ambient = chosen["ambient"]
    if ambient_strength is None:
        ambient_strength = chosen["ambient_strength"]
    if color is None:
        color = chosen["color"]
    if colormap is None:
        colormap = chosen["colormap"]
    if edge_color is None:
        edge_color = chosen["edge_color"]
    if key is None:
        key = chosen["key"]
    if fill is None:
        fill = chosen["fill"]
    if rim is None:
        rim = chosen["rim"]
    if azimuth is None:
        azimuth = chosen.get("azimuth")
    if elevation is None:
        elevation = chosen.get("elevation")
    if softness is None:
        softness = chosen.get("softness", 6.0)

    # Frame is sized for obj; other meshes left visible are composed into that frame.
    others = []
    if isolate:
        for other in bpy.data.objects:
            if other.type == 'MESH' and other is not obj and not other.hide_render:
                others.append(other)
                other.hide_render = True
    for other in (hide or ()):
        if other is not None and not other.hide_render:
            others.append(other)
            other.hide_render = True

    if autosize:
        autoscale_geometry(obj, node_fraction=node_fraction,
                           node_resolution=node_resolution,
                           edge_resolution=edge_resolution,
                           edge_ratio=edge_ratio, verbose=verbose)

    # Protect before rebuild so GN never adds Remove Attribute for this attr.
    if color_attribute:
        protect_attribute(obj, color_attribute)

    # Material before tree build, or Set Material bakes None and overrides slot 0.
    material(obj, color=color)
    geometry_nodes(obj)

    # Color after rebuild so scigraphs_is_node exists for the shader gate.
    if color_attribute:
        color_graph(obj, color_attribute, colormap=colormap,
                    reverse=color_reverse, norm=color_norm, gamma=color_gamma,
                    vmin=vmin, vmax=vmax, clip_low_pct=clip_low_pct,
                    clip_high_pct=clip_high_pct, nodes_only=nodes_only,
                    edge_color=edge_color, verbose=verbose)

    material(obj, color=color)

    measurements, _source = measure(obj)
    center = measurements[0] if measurements else (0.0, 0.0, 0.0)
    diagonal = measurements[2] if measurements else 1.0
    node_radius = float(obj.get("scigraphs_node_size", 0.0) or 0.0)

    light_rig(center, diagonal, key=key, fill=fill, rim=rim,
              softness=float(softness), azimuth=azimuth, elevation=elevation)
    world(background=background, ambient=ambient,
          ambient_strength=ambient_strength)

    # Resolution before camera: camera_fit_coords uses current render aspect.
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.film_transparent = False
    scene.render.filepath = str(path)

    frame_camera(obj, direction=direction, lens=lens, view=view,
                 projection=projection, pushback=pushback, margin=margin,
                 ground=ground, verbose=verbose)

    scene.view_settings.view_transform = view_transform
    scene.view_settings.exposure = float(exposure)

    scene.render.engine = 'BLENDER_EEVEE'
    eevee_settings(scene, samples=samples, shadows=shadows,
                   raytracing=raytracing,
                   ao_distance=max(node_radius * 6.0, diagonal * 1e-3))

    try:
        bpy.ops.render.render(write_still=True)
    finally:
        for other in others:
            other.hide_render = False

    elapsed = time.time() - started
    # shrink_png palettes differ per image; use shrink=False for pixelwise compares.
    if shrink:
        preview.shrink_png(path)
    if verbose:
        print(f"  EEVEE {resolution[0]}x{resolution[1]} @{samples} samples "
              f"in {elapsed:.1f} s -> {path}")
    return path


def _luminance(path):
    import numpy as np
    from PIL import Image

    rgb = np.asarray(Image.open(str(path)).convert("RGB"), dtype=np.float32)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def relief_contrast(path):
    """Luminance std and spread, on a 0 to 255 scale, of a ground-only frame."""
    import numpy as np

    try:
        y = _luminance(path)
    except ImportError:
        return None
    low, high = np.percentile(y, (5, 95))
    return {"std": float(y.std()), "spread": float(high - low),
            "mean": float(y.mean()), "min": float(y.min()),
            "max": float(y.max())}


def differing_fraction(path, reference, threshold=24, sign='any'):
    """Fraction of pixels differing from a reference frame; sign='brighter'
    excludes shadows cast by a low sun.
    """
    import numpy as np

    try:
        from PIL import Image
    except ImportError:
        return None
    a = np.asarray(Image.open(str(path)).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(str(reference)).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        raise ValueError(f"{path} is {a.shape} and "
                         f"{reference} is {b.shape}")
    delta = a - b
    hit = np.abs(delta).sum(axis=2) > threshold
    if str(sign).lower() == 'brighter':
        hit &= delta.sum(axis=2) > 0
    elif str(sign).lower() == 'darker':
        hit &= delta.sum(axis=2) < 0
    return float(hit.mean())


__all__ = [
    "DEFAULT_PRESET",
    "FRAME_MARGIN",
    "NODE_FRACTION",
    "OBLIQUE",
    "PRESETS",
    "TOP_DOWN",
    "VIEWS",
    "attribute_domain",
    "autoscale_geometry",
    "color_graph",
    "color_range",
    "differing_fraction",
    "eevee_settings",
    "frame_camera",
    "geometry_nodes",
    "light_rig",
    "material",
    "measure",
    "node_cloud",
    "preset",
    "protect_attribute",
    "relief_contrast",
    "eevee",
    "sun_direction",
    "world",
]
