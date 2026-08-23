# Camera matrices and the 224-byte uniform block every shader shares.
#
# Blender's `gpu` uses OpenGL depth, z in [-1, 1]; WebGPU and Vulkan use
# [0, 1]. An OpenGL projection matrix does not fail here, it folds half the
# depth range behind the near plane, and nothing shows that until two objects
# overlap. numpy is row-major and WGSL's mat4x4 is column-major, so the
# transpose happens in `Camera.block()`.

import numpy as np

# 3 mat4 (192) + 2 vec4 (32), no std140 padding needed.
CAMERA_BLOCK_SIZE = 224


def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """World -> view, right-handed, looking down -z. ``up`` defaults to +Z."""
    eye = np.asarray(eye, dtype=np.float64)
    f = _normalize(np.asarray(target, dtype=np.float64) - eye)
    up = _normalize(up)
    if abs(float(np.dot(f, up))) > 0.9999:
        # Looking along `up`: pick another axis rather than emit a NaN matrix.
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
    """View -> clip, z in [0, 1]."""
    f = 1.0 / np.tan(np.radians(float(fov_y_deg)) * 0.5)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / float(aspect)
    m[1, 1] = f
    m[2, 2] = far / (near - far)
    m[2, 3] = (far * near) / (near - far)
    m[3, 2] = -1.0
    return m.astype(np.float32)


def orthographic(half_width, half_height, near, far):
    """Parallel view -> clip, z in [0, 1]."""
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = 1.0 / float(half_width)
    m[1, 1] = 1.0 / float(half_height)
    m[2, 2] = 1.0 / (near - far)
    m[2, 3] = near / (near - far)
    return m.astype(np.float32)


# Maps OpenGL's z in [-1, 1] onto WebGPU's [0, 1]: z' = 0.5 z + 0.5 w.
_GL_TO_WGPU = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0.5, 0.5],
                        [0, 0, 0, 1]], dtype=np.float32)


def from_gl_projection(proj):
    """Remap an OpenGL projection matrix to z in [0, 1]. Use it on anything
    from Blender's `region_data.window_matrix` or `gpu.matrix`."""
    return (_GL_TO_WGPU @ np.asarray(proj, dtype=np.float32)).astype(np.float32)


def fit_view(coords, size, fov_y_deg=45.0, margin=1.15, direction=(0.0, -1.0, 0.35),
             up=(0.0, 0.0, 1.0)):
    """A Camera framing all of ``coords`` at viewport ``size`` (w, h). Fits
    the bounding sphere, so a flat layout comes out over-framed."""
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
    width, height = size
    if coords.size == 0:
        center = np.zeros(3, dtype=np.float32)
        radius = 1.0
    else:
        lo = coords.min(axis=0)
        hi = coords.max(axis=0)
        center = (lo + hi) * 0.5
        radius = float(np.linalg.norm(hi - lo)) * 0.5
        if radius < 1e-6:
            radius = 1.0

    aspect = float(width) / float(height)
    # Fit the tighter axis: vertical-only fitting crops a portrait render.
    half_v = np.radians(float(fov_y_deg)) * 0.5
    half_h = np.arctan(np.tan(half_v) * aspect)
    dist = radius * float(margin) / np.sin(min(half_v, half_h))

    eye = center - _normalize(direction).astype(np.float32) * dist
    near = max(dist - radius * 2.0, dist * 1e-3)
    far = dist + radius * 4.0
    return Camera(look_at(eye, center, up),
                  perspective(fov_y_deg, aspect, near, far),
                  near=near, far=far)


class Camera:
    """view + proj, and the bytes for binding 0. ``view_proj`` is precomputed:
    doing it in WGSL costs a matrix multiply per vertex to save 64 bytes."""

    __slots__ = ("view", "proj", "near", "far")

    def __init__(self, view, proj, near=0.1, far=1000.0):
        self.view = np.asarray(view, dtype=np.float32).reshape(4, 4)
        self.proj = np.asarray(proj, dtype=np.float32).reshape(4, 4)
        self.near = float(near)
        self.far = float(far)

    @property
    def view_proj(self):
        return (self.proj @ self.view).astype(np.float32)

    def block(self, width, height):
        """The 224 bytes of the `camera` uniform, float32. Layout in README 1.3."""
        buf = np.zeros(CAMERA_BLOCK_SIZE // 4, dtype=np.float32)
        buf[0:16] = self.view.T.ravel()
        buf[16:32] = self.proj.T.ravel()
        buf[32:48] = self.view_proj.T.ravel()
        buf[48:52] = (float(width), float(height),
                      1.0 / float(width), 1.0 / float(height))
        buf[52:56] = (self.near, self.far,
                      float(width) / float(height), 0.0)
        return buf

    def project(self, points, width, height):
        """(N, 3) world points to (N, 3) pixels: x right, y down, z in [0, 1].
        Nothing draws with this; it lets a test predict where a node lands."""
        p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        h = np.concatenate([p, np.ones((p.shape[0], 1), np.float32)], axis=1)
        clip = h @ self.view_proj.T
        w = clip[:, 3:4]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        ndc = clip[:, :3] / w
        out = np.empty_like(ndc)
        out[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * width
        # NDC +y is up, pixel row 0 is the top. Same flip the rasterizer does.
        out[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
        out[:, 2] = ndc[:, 2]
        return out
