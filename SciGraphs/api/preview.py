"""GPU preview render: graphs only, via the SciGraphs engine.

Take every scale from `extent()` rather than a constant, since a geospatial
graph runs about 1 Blender unit per km.

    preview.autoscale(obj)
    preview.frame_camera(obj)
    preview.render(obj, "/tmp/streets.png")
"""

import pathlib

import bpy

from . import graphs


def extent(obj):
    """Graph bounding box: (center, size, diagonal, median_nn)."""
    import numpy as np

    # Prefer node_positions; mesh verts can be road-centerline samples.
    positions = obj.get("node_positions")
    if positions is not None and len(positions) >= 3:
        coords = np.asarray(list(positions), dtype=np.float32).reshape(-1, 3)
    else:
        coords = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", coords)
        coords = coords.reshape(-1, 3)
    if len(coords) == 0:
        return None

    lo, hi = coords.min(axis=0), coords.max(axis=0)
    size = hi - lo
    diagonal = float(np.linalg.norm(size))

    # Cap the NN sample at 2000; full pairwise is O(n^2).
    sample = coords
    if len(coords) > 2000:
        step = len(coords) // 2000
        sample = coords[::step][:2000]
    d = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)

    # Ignore zero NN (coincident nodes); else median is 0 and nothing draws.
    positive = nn[np.isfinite(nn) & (nn > 0)]
    if len(positive):
        median_nn = float(np.median(positive))
    else:
        median_nn = diagonal / 50.0

    # Clamp so near-duplicates / tiny graphs still get a visible radius.
    scale = diagonal if diagonal > 0 else 1.0
    median_nn = float(np.clip(median_nn or scale / 50.0,
                              scale / 800.0, scale / 6.0))

    return (lo + hi) / 2.0, size, diagonal, median_nn


# Node radius / median NN; 0.35 balances lattice vs street networks.
_NODE_FRACTION = 0.35


def autoscale(obj, node_fraction=_NODE_FRACTION, verbose=True):
    """Set preview node radius from median nearest-neighbor distance."""
    measurements = extent(obj)
    if measurements is None:
        return {}
    center, size, diagonal, median_nn = measurements

    node_radius = median_nn * node_fraction

    scene = bpy.context.scene
    scene.scigraphs_preview_impostor_radius = node_radius
    # Leave edge_radius_scale; it tracks node radius as a fraction.

    values = {
        "diagonal": diagonal,
        "median_nearest_neighbor": median_nn,
        "impostor_radius": node_radius,
        "default_would_have_been": 0.1,
        "times_larger": 0.1 / node_radius if node_radius else float("inf"),
    }
    if verbose:
        print(f"  graph extent           {diagonal:.4f} Blender units")
        print(f"  nearest neighbor      {median_nn:.5f} (median)")
        print(f"  node radius chosen     {node_radius:.5f}")
        print(f"  the default (0.1) would have been "
              f"{values['times_larger']:.0f}x larger")
    return values


def frame_camera(obj, direction=(0.0, -0.35, 1.0), margin=1.06, lens=50.0):
    """Frame scene camera on the graph via camera_fit_coords."""
    import numpy as np
    from mathutils import Vector

    measurements = extent(obj)
    if measurements is None:
        return None
    center, size, diagonal, _ = measurements

    cam_obj = bpy.data.objects.get("SciGraphs_Camera")
    if cam_obj is None or cam_obj.type != 'CAMERA':
        cam_obj = bpy.data.objects.new(
            "SciGraphs_Camera", bpy.data.cameras.new("SciGraphs_Camera"))
        bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    cam_obj.data.lens = float(lens)

    # Orient before camera_fit_coords (solves along current axis).
    d = Vector(direction).normalized()
    cam_obj.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.view_layer.update()

    positions = obj.get("node_positions")
    if positions is not None and len(positions) >= 3:
        local_coords = np.asarray(list(positions), dtype=np.float32).reshape(-1, 3)
    else:
        local_coords = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", local_coords)
        local_coords = local_coords.reshape(-1, 3)
    matrix = np.array(obj.matrix_world.transposed())
    world_coords = local_coords @ matrix[:3, :3] + matrix[3, :3]

    # Pad by impostor radius so spheres aren't clipped at the edge.
    node_radius = float(getattr(bpy.context.scene, "scigraphs_preview_impostor_radius", 0.0))
    lo, hi = world_coords.min(axis=0) - node_radius, world_coords.max(axis=0) + node_radius
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    points = np.vstack([world_coords, corners])

    depsgraph = bpy.context.evaluated_depsgraph_get()
    location, _scale = cam_obj.camera_fit_coords(depsgraph, points.ravel().tolist())
    cam_obj.location = Vector(location) + d * (diagonal * (margin - 1.0))

    # Extend clip planes for large/geospatial distances.
    distance = (cam_obj.location - Vector(center)).length
    cam_obj.data.clip_start = max(distance * 1e-4, 1e-5)
    cam_obj.data.clip_end = max((distance + diagonal) * 4.0, 100.0)
    return cam_obj


def backbone(mode='TOPK', k=3, alpha=0.05, attribute=""):
    """Sparsify edges on GPU: TOPK, DISPARITY, MST, SAMPLE, or ALL."""
    scene = bpy.context.scene
    scene.scigraphs_preview_backbone_mode = mode
    scene.scigraphs_preview_backbone_k = int(k)
    scene.scigraphs_preview_backbone_alpha = float(alpha)
    if attribute:
        scene.scigraphs_preview_backbone_attr = attribute
    return mode


def render(obj, filename, resolution=(1280, 960), node_style='AUTO',
           edge_style='AUTO', background=(0.05, 0.05, 0.08, 1.0),
           aa=2, direction=(0.0, -0.35, 1.0), autosize=True,
           isolate=True, verbose=True):
    """Render with the SciGraphs engine, preferring SPHERE nodes on Vulkan."""
    scene = bpy.context.scene
    path = pathlib.Path(filename)
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Hide other meshes so framing matches only `obj`.
    others = []
    if isolate:
        for other in bpy.data.objects:
            if other.type == 'MESH' and other is not obj and not other.hide_render:
                others.append(other)
                other.hide_render = True

    if autosize:
        autoscale(obj, verbose=verbose)

    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(path)

    scene.scigraphs_display_engine = 'GPU'
    scene.scigraphs_preview_enabled = True
    scene.scigraphs_preview_node_style = node_style
    scene.scigraphs_preview_render_bg = background
    scene.scigraphs_preview_render_aa = int(aa)

    if edge_style == 'AUTO':
        measurements = extent(obj)
        edge_radius = (scene.scigraphs_preview_impostor_radius
                       * scene.scigraphs_preview_edge_radius_scale)
        pixels = 0.0
        if measurements and measurements[2] > 0:
            pixels = 2.0 * edge_radius / measurements[2] * resolution[0]
        if pixels < 1.5:
            scene.scigraphs_preview_edge_style = 'LINE'
            scene.scigraphs_preview_edge_width = max(
                1.0, min(4.0, resolution[0] / 400.0))
            if verbose:
                print(f"  edges as lines ({scene.scigraphs_preview_edge_width:.1f} px): "
                      f"the tube would be {pixels:.2f} px and would not be visible")
        else:
            scene.scigraphs_preview_edge_style = 'RIBBON'
            # Thin tubes on dense graphs so nodes stay readable.
            edge_count = float(obj.get("num_edges") or 0)
            node_count = float(obj.get("num_nodes") or 1)
            mean_degree = 2.0 * edge_count / max(node_count, 1.0)
            scene.scigraphs_preview_edge_radius_scale = max(0.08, min(0.45, 1.4 / max(mean_degree, 1.0)))
            if verbose:
                print(f"  edges as tubes, thickness {scene.scigraphs_preview_edge_radius_scale:.2f}"
                      f" of the node radius (mean degree {mean_degree:.1f})")
    else:
        scene.scigraphs_preview_edge_style = edge_style

    frame_camera(obj, direction=direction)
    graphs.activate(obj)

    scene.render.engine = 'SCIGRAPHS'
    try:
        bpy.ops.render.render(write_still=True)
    finally:
        for other in others:
            other.hide_render = False

    shrink_png(path)
    if verbose:
        print(f"  render -> {path}")
    return path


def shrink_png(path, colors=256):
    """Palette-quantize PNG in place when smaller; no-op if Pillow missing."""
    try:
        from PIL import Image
    except ImportError:
        return path

    import io

    try:
        path = pathlib.Path(path)
        before = path.stat().st_size
        buffer = io.BytesIO()
        with Image.open(path) as image:
            image.convert("RGB").convert(
                "P", palette=Image.ADAPTIVE, colors=int(colors)
            ).save(buffer, "PNG", optimize=True)
        if buffer.tell() < before:
            path.write_bytes(buffer.getvalue())
    except Exception:  # noqa: BLE001 - never fail a notebook over file size
        return path
    return path


def snapshot(filename, samples=32, resolution=(1280, 720)):
    """Quick scene render with the current engine; returns output path."""
    scene = bpy.context.scene
    path = pathlib.Path(filename)
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = (scene.render.filepath, scene.render.image_settings.file_format,
                scene.render.resolution_x, scene.render.resolution_y)
    try:
        scene.render.filepath = str(path)
        scene.render.image_settings.file_format = 'PNG'
        scene.render.resolution_x, scene.render.resolution_y = resolution
        if scene.render.engine.startswith('BLENDER_EEVEE'):
            try:
                scene.eevee.taa_render_samples = samples
            except AttributeError:
                pass
        bpy.ops.render.render(write_still=True)
    finally:
        (scene.render.filepath, scene.render.image_settings.file_format,
         scene.render.resolution_x, scene.render.resolution_y) = previous
    return path
