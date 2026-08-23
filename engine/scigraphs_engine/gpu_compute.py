# GPU compute culling and indirect draw: the interface only. Blender's `gpu` has
# compute dispatch, but no storage buffers with atomic append and no
# draw-indirect, so the CPU still reads the visible set back every frame.

GPU_COMPUTE_AVAILABLE = False


def is_available():
    """Always False; there is no backend yet."""
    return GPU_COMPUTE_AVAILABLE


def cull_and_compact(blocks, view_proj, region_size, budget):
    raise NotImplementedError(
        "GPU compute culling requires a native backend; use the CPU blocks path "
        "(see lod.py / draw._draw_blocks)."
    )
