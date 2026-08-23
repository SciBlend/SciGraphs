# The offscreen render target, and getting the pixels back out.

import numpy as np
import wgpu

# rgba8unorm: the golden baseline PNGs are 8-bit. Not -srgb, see to_display.
COLOR_FORMAT = wgpu.TextureFormat.rgba8unorm

# depth32float rather than depth24plus: the impostor shaders write frag_depth
# directly, and a banded depth buffer shows as rings on intersecting spheres.
DEPTH_FORMAT = wgpu.TextureFormat.depth32float


# Applied on readback, not through an rgba8unorm-srgb target: that encodes per
# fragment before blending, where Blender blends in linear and encodes once.
# The two differ wherever anything is translucent, here every edge. Over
# fifteen flat add-on backgrounds under 'Standard': 0.02-0.05 LSB with this,
# every pixel differing without. Table in tests/crossbackend/wgpu_producer.py.
def to_display(linear):
    """Scene-referred linear to display-referred sRGB (IEC 61966-2-1). Any
    shape, values in [0, 1], RGB only; alpha is not encoded."""
    a = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    return np.where(a <= 0.0031308, a * 12.92,
                    1.055 * np.power(np.maximum(a, 1e-12), 1.0 / 2.4) - 0.055
                    ).astype(np.float32)


class Target:
    __slots__ = ("ctx", "width", "height", "color", "depth",
                 "color_view", "depth_view")

    def __init__(self, ctx, width, height):
        width, height = int(width), int(height)
        if width < 1 or height < 1:
            raise ValueError(f"target size must be positive, got {width}x{height}")
        cap = ctx.limit("max-texture-dimension-2d", 8192)
        if width > cap or height > cap:
            raise ValueError(
                f"{width}x{height} exceeds max-texture-dimension-2d {cap}")

        self.ctx = ctx
        self.width = width
        self.height = height
        self.color = ctx.device.create_texture(
            label="scigraphs-color", size=(width, height, 1),
            format=COLOR_FORMAT, usage=(wgpu.TextureUsage.RENDER_ATTACHMENT
                                        | wgpu.TextureUsage.COPY_SRC
                                        | wgpu.TextureUsage.TEXTURE_BINDING))
        self.depth = ctx.device.create_texture(
            label="scigraphs-depth", size=(width, height, 1),
            format=DEPTH_FORMAT, usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
        self.color_view = self.color.create_view()
        self.depth_view = self.depth.create_view()

    @property
    def size(self):
        return (self.width, self.height)

    def attachments(self, clear_color=(0.0, 0.0, 0.0, 1.0), clear=True):
        """([color], depth) dicts for begin_render_pass. ``clear`` is per pass:
        a multi-pass render clears once, then loads."""
        load = "clear" if clear else "load"
        color = {
            "view": self.color_view,
            "resolve_target": None,
            "clear_value": tuple(float(v) for v in clear_color),
            "load_op": load,
            "store_op": "store",
        }
        depth = {
            "view": self.depth_view,
            "depth_clear_value": 1.0,
            "depth_load_op": load,
            "depth_store_op": "store",
        }
        return [color], depth

    def clear(self, color=(0.0, 0.0, 0.0, 1.0)):
        enc = self.ctx.device.create_command_encoder(label="clear")
        colors, depth = self.attachments(color, clear=True)
        enc.begin_render_pass(color_attachments=colors,
                              depth_stencil_attachment=depth).end()
        self.ctx.queue.submit([enc.finish()])

    def read(self):
        """(H, W, 4) float32 in [0, 1], row 0 at the top: WebGPU's origin is
        top-left, matching tests/golden/metrics.py::from_blender."""
        raw = self.ctx.queue.read_texture(
            {"texture": self.color, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": self.width * 4,
             "rows_per_image": self.height},
            (self.width, self.height, 1))
        img = np.frombuffer(raw, dtype=np.uint8)
        img = img.reshape(self.height, self.width, 4)
        return (img.astype(np.float32) / 255.0)

    def save_png(self, path):
        from PIL import Image

        raw = self.ctx.queue.read_texture(
            {"texture": self.color, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": self.width * 4,
             "rows_per_image": self.height},
            (self.width, self.height, 1))
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.height, self.width, 4)
        Image.fromarray(arr, mode="RGBA").save(str(path))
        return path

    def destroy(self):
        for tex in (self.color, self.depth):
            try:
                tex.destroy()
            except Exception:   # noqa: BLE001 - best-effort
                pass
