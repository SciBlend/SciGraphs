# Device acquisition, and that its three failure paths stay three.
#
# One check asserts the module it imported is the working-tree copy. Testing an
# installed copy while you edit the tree is one `pip install .` away.

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from harness import Report, finish   # noqa: E402

import scigraphs_engine                                   # noqa: E402
from scigraphs_engine.backends import wgpu as backend     # noqa: E402
from scigraphs_engine.backends.wgpu import device         # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(HERE)))))


def check_import_is_the_working_tree(r):
    got = os.path.abspath(backend.__file__)
    want = os.path.join(HERE, "..", "__init__.py")
    r.check("imported from the working tree",
            os.path.realpath(got) == os.path.realpath(want), got)
    r.check("engine imported alongside it",
            os.path.realpath(os.path.dirname(scigraphs_engine.__file__))
            == os.path.realpath(os.path.join(HERE, "..", "..", "..")),
            scigraphs_engine.__file__)


def check_acquire(r):
    ctx = device.acquire()
    r.check("acquire returns a Context", isinstance(ctx, device.Context))
    r.check("device present", ctx.device is not None)
    r.check("queue present", ctx.queue is not None)
    info = ctx.info
    for key in ("vendor", "device", "backend", "type", "driver"):
        r.check(f"info has {key}", key in info and info[key] != "",
                info.get(key))
    r.note(f"adapter: {ctx.describe()}")
    return ctx


def check_limits(r, ctx):
    align = ctx.limit("min-uniform-buffer-offset-alignment", 0)
    r.check("min-uniform-buffer-offset-alignment is a power of two",
            align > 0 and (align & (align - 1)) == 0, align)
    r.check("hyphenated limit names are the right spelling",
            ctx.limit("max-texture-dimension-2d", 0) >= 2048,
            ctx.limit("max-texture-dimension-2d", 0))
    r.check("a camelCase name does NOT resolve (it would silently default)",
            ctx.limit("maxTextureDimension2D", -1) == -1)
    r.note(f"max vertex buffers {ctx.limit('max-vertex-buffers', 0)}, "
           f"max attributes {ctx.limit('max-vertex-attributes', 0)}")


def check_failure_paths(r, ctx):
    # Impossible requests; a real adapter will not vanish mid-process.
    r.raises("a nonsense power preference is NoAdapterError",
             device.NoAdapterError, device.acquire,
             power_preference="teleportation")

    r.raises("an impossible limit is NoDeviceError", device.NoDeviceError,
             device.acquire,
             required_limits={"max-texture-dimension-2d": 1 << 30})

    lost = device.acquire()
    r.check("a fresh context passes check()", lost.check() is None)
    lost.mark_lost("test")
    r.raises("a marked-lost context raises from check()",
             device.DeviceLostError, lost.check)
    r.check("all three are DeviceError",
            all(issubclass(t, device.DeviceError)
                for t in (device.NoAdapterError, device.NoDeviceError,
                          device.DeviceLostError)))
    r.check("and are distinct from each other",
            len({device.NoAdapterError, device.NoDeviceError,
                 device.DeviceLostError}) == 3)
    lost.destroy()


def check_fallback_adapter(r):
    try:
        ctx = device.acquire(fallback=True)
    except device.DeviceError as exc:
        r.note(f"no fallback adapter here ({type(exc).__name__})")
        return
    r.check("fallback adapter yields a usable Context",
            ctx.device is not None, ctx.describe())
    ctx.destroy()


def main():
    r = Report("test_device", minimum=18)
    r.section("Import")
    check_import_is_the_working_tree(r)
    r.section("Acquisition")
    ctx = check_acquire(r)
    r.section("Limits")
    check_limits(r, ctx)
    r.section("Failure paths")
    check_failure_paths(r, ctx)
    r.section("Fallback adapter")
    check_fallback_adapter(r)
    finish(r)


if __name__ == "__main__":
    main()
