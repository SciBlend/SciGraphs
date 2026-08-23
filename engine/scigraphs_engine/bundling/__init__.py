# Edge bundling in numpy, one module per algorithm. Each takes node coordinates
# and an edge list and returns control polygons: `ctrl` (E, K, 3) float32 padded
# to a common K, and `counts` (E,) int32 saying how many of those K are real.
# Both the Blender add-on and the wgpu backend draw from that pair.
#
# Submodules import eagerly because the add-on resolves names through them at
# import time.

from . import fdeb, mingle, routed, sbeb  # noqa: F401

__all__ = ["fdeb", "mingle", "routed", "sbeb"]
