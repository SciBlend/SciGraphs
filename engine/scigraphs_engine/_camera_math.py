# Transcribed from backends/wgpu/camera.py, held float32-exact against it by
# tests/test_camera.py. Depth is z in [0, 1] (WebGPU), not OpenGL's.

import numpy as np


def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """World -> view, right-handed, looking down -z. +Z up like Blender."""
    eye = np.asarray(eye, dtype=np.float64)
    f = _normalize(np.asarray(target, dtype=np.float64) - eye)
    up = _normalize(up)
    if abs(float(np.dot(f, up))) > 0.9999:
        up = np.array([0.0, 1.0, 0.0]) if abs(f[1]) < 0.9 else \
            np.array([1.0, 0.0, 0.0])
    s = _normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float64)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3] = -float(np.dot(s, eye))
    m[1, 3] = -float(np.dot(u, eye))
    m[2, 3] = float(np.dot(f, eye))
    return m.astype(np.float32)


def perspective(fov_y_deg, aspect, near, far):
    """View -> clip with z in [0, 1]."""
    f = 1.0 / np.tan(np.radians(float(fov_y_deg)) * 0.5)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / float(aspect)
    m[1, 1] = f
    m[2, 2] = far / (near - far)
    m[2, 3] = (far * near) / (near - far)
    m[3, 2] = -1.0
    return m.astype(np.float32)


def bounding_sphere(coords):
    """(center, radius) of the box's circumsphere. Over-frames a flat layout."""
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
    if coords.size == 0:
        return np.zeros(3, dtype=np.float32), 1.0
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    center = (lo + hi) * 0.5
    radius = float(np.linalg.norm(hi - lo)) * 0.5
    return center, (radius if radius > 1e-6 else 1.0)


def fit_view(coords, size, fov_y_deg=45.0, margin=1.15,
             direction=(0.0, -1.0, 0.35), up=(0.0, 0.0, 1.0)):
    """A resolved api.Camera framing all of ``coords`` at ``size``."""
    from .api import Camera

    width, height = size
    center, radius = bounding_sphere(coords)
    aspect = float(width) / float(height)
    # Fit the tighter axis: vertical-only fitting crops a portrait render.
    half_v = np.radians(float(fov_y_deg)) * 0.5
    half_h = np.arctan(np.tan(half_v) * aspect)
    dist = radius * float(margin) / np.sin(min(half_v, half_h))

    eye = center - _normalize(direction).astype(np.float32) * dist
    near = max(dist - radius * 2.0, dist * 1e-3)
    far = dist + radius * 4.0
    return Camera(look_at(eye, center, up),
                  perspective(fov_y_deg, aspect, near, far), near, far)


def look_at_camera(eye, target, up, fov_y_deg, size, near=None, far=None):
    from .api import Camera

    width, height = size
    eye = np.asarray(eye, dtype=np.float64)
    span = float(np.linalg.norm(np.asarray(target, dtype=np.float64) - eye))
    near = float(near) if near is not None else max(span * 1e-3, 1e-4)
    far = float(far) if far is not None else max(span * 10.0, near * 100.0)
    return Camera(look_at(eye, target, up),
                  perspective(fov_y_deg, float(width) / float(height),
                              near, far), near, far)
