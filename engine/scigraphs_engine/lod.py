# What gets drawn at all, decided in screen pixels rather than raw 3D distance:
# screen_radius (px) ~= world_radius / distance * focal_scale.

import numpy as np

# Thresholds are in pixels of a 1080-tall image, not of the real buffer.
REF_HEIGHT = 1080.0


def px_scale(height):
    return (float(height) / REF_HEIGHT) if height else 1.0


# Node representation tiers, keyed by on-screen node radius in reference px.
NODE_HIDE = 0
NODE_POINT = 1
NODE_DISK = 2
NODE_SPHERE = 3

EDGE_HIDE = 0
EDGE_LINE = 1
EDGE_RIBBON = 2


def _project_points(points, persp_mat):
    n = points.shape[0]
    homog = np.empty((n, 4), dtype=np.float64)
    homog[:, :3] = points
    homog[:, 3] = 1.0
    clip = homog @ np.asarray(persp_mat, dtype=np.float64).T
    w = clip[:, 3]
    return clip, w


def frustum_cull_spheres(centers, radii, persp_mat):
    """Conservative mask of bounding spheres that might be in the frustum."""
    clip, w = _project_points(centers, persp_mat)
    margin = np.abs(radii)[:, None]
    keep = np.ones(centers.shape[0], dtype=bool)
    aw = np.abs(w) + 1e-9
    for axis in range(3):
        ndc = clip[:, axis] / aw
        r_ndc = margin[:, 0] / aw
        keep &= (ndc + r_ndc >= -1.0) & (ndc - r_ndc <= 1.0)
    # Behind-camera: keep if the sphere could still straddle the near plane.
    keep &= (clip[:, 2] + margin[:, 0] >= -aw)
    return keep


def projected_pixel_radius(centers, radii, persp_mat, region_height):
    _, w = _project_points(centers, persp_mat)
    aw = np.abs(w) + 1e-9
    # NDC half-extent ~= radius / w, and NDC spans 2 over the height.
    ndc_radius = np.asarray(radii, dtype=np.float64) / aw
    return ndc_radius * 0.5 * region_height


def node_tier(pixel_radius, sphere_min=20.0, disk_min=4.0, point_min=1.0):
    if pixel_radius >= sphere_min:
        return NODE_SPHERE
    if pixel_radius >= disk_min:
        return NODE_DISK
    if pixel_radius >= point_min:
        return NODE_POINT
    return NODE_HIDE


def edge_tier(pixel_width, ribbon_min=4.0, line_min=1.0):
    if pixel_width >= ribbon_min:
        return EDGE_RIBBON
    if pixel_width >= line_min:
        return EDGE_LINE
    return EDGE_HIDE


def apply_budget(order_key, counts, budget):
    """Mask of blocks to draw within ``budget``, greedy in descending
    ``order_key``. A ``budget`` of 0 or less means no limit."""
    if budget <= 0:
        return np.ones(counts.shape[0], dtype=bool)
    order = np.argsort(-np.asarray(order_key))
    cumulative = np.cumsum(counts[order])
    allowed = cumulative <= budget
    mask = np.zeros(counts.shape[0], dtype=bool)
    mask[order[allowed]] = True
    # Never return an empty mask: the top block always gets through.
    if not mask.any() and counts.size:
        mask[order[0]] = True
    return mask
