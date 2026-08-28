import numpy as np



def pick_nearest_node(context, obj, mouse_x, mouse_y, radius_px=20.0,
                      region=None, rv3d=None):
    """Node index, or None when nothing is inside ``radius_px``.

    ``region``/``rv3d`` are explicit for callers that are not running in the
    region they mean. A modal operator started from a timer has neither on its
    context, and reading them off it there silently picks nothing.
    """
    region = region if region is not None else getattr(context, "region", None)
    rv3d = rv3d if rv3d is not None else getattr(context, "region_data", None)
    if region is None or rv3d is None:
        return None

    mesh = obj.data
    num_verts = len(mesh.vertices)
    coords = np.empty(num_verts * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", coords)
    coords = coords.reshape(num_verts, 3)

    mat = np.array(rv3d.perspective_matrix @ obj.matrix_world, dtype=np.float64)
    homog = np.empty((num_verts, 4), dtype=np.float64)
    homog[:, :3] = coords
    homog[:, 3] = 1.0
    clip = homog @ mat.T

    w = clip[:, 3]
    in_front = w > 1e-6
    if not in_front.any():
        return None

    ndc_x = np.full(num_verts, np.inf)
    ndc_y = np.full(num_verts, np.inf)
    ndc_x[in_front] = clip[in_front, 0] / w[in_front]
    ndc_y[in_front] = clip[in_front, 1] / w[in_front]

    sx = (ndc_x * 0.5 + 0.5) * region.width
    sy = (ndc_y * 0.5 + 0.5) * region.height

    dx = sx - mouse_x
    dy = sy - mouse_y
    dist2 = dx * dx + dy * dy
    dist2[~in_front] = np.inf

    nearest = int(np.argmin(dist2))
    if dist2[nearest] > radius_px * radius_px:
        return None
    return nearest
