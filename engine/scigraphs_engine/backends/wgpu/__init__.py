# MeshSpec in, pixels out. Importing this pulls in wgpu and a native library,
# so nothing else in scigraphs_engine imports it. README 1 is the shader JSON
# schema, README 3 the DrawState fields WebGPU cannot express.
#
#     ctx = acquire()
#     target = Target(ctx, 960, 540)
#     renderer = Renderer(ctx, target)
#     spec, _weight = mesh.point_specs(coords, colors)[0]
#     renderer.render_specs([spec], camera.fit_view(coords, target.size))
#     target.save_png("out.png")

from . import camera  # noqa: F401
from .buffers import DrawCall, UploadedMesh, UploadError, plan, upload  # noqa: F401
from .device import (  # noqa: F401
    Context, DeviceError, DeviceLostError, NoAdapterError, NoDeviceError,
    acquire,
)
from .pipelines import BLEND, PipelineCache, PipelineError, state_key  # noqa: F401
from .renderer import Renderer, draw_block  # noqa: F401
from .shaders import (  # noqa: F401
    Shader, ShaderError, ShaderLibrary, variant_name,
)
from .target import COLOR_FORMAT, DEPTH_FORMAT, Target  # noqa: F401

__all__ = [
    "BLEND", "COLOR_FORMAT", "Context", "DEPTH_FORMAT", "DeviceError",
    "DeviceLostError", "DrawCall", "NoAdapterError", "NoDeviceError",
    "PipelineCache", "PipelineError", "Renderer", "Shader", "ShaderError",
    "ShaderLibrary", "Target", "UploadError", "UploadedMesh", "acquire",
    "camera", "draw_block", "plan", "state_key", "upload", "variant_name",
]
