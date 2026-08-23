# Render a graph to a PNG.
#     python -m scigraphs_engine.backends.wgpu.render --demo grid --out g.png
#     python -m scigraphs_engine.backends.wgpu.render --npz layout.npz --size 1920x1080
#
# Draws whatever wgsl/ has ported; `--list-shaders` says which.

import argparse
import json
import os
import sys
import time

import numpy as np

from ... import mesh
from ...mesh import DrawState
from ...source import ArraySource
from . import camera as cam_mod
from .device import DeviceError, acquire
from .renderer import Renderer
from .shaders import ShaderLibrary
from .target import Target


def _demo_source(kind, n, seed=0x5C16):
    """(ArraySource, colors) for a generated graph; 0x5C16 is the seed."""
    rng = np.random.default_rng(seed)
    if kind == "grid":
        # A flat grid makes a wrong projection obvious: rows stay rows.
        side = max(int(round(np.sqrt(n))), 2)
        xs, ys = np.meshgrid(np.linspace(-1, 1, side), np.linspace(-1, 1, side))
        coords = np.stack([xs.ravel(), np.zeros(xs.size), ys.ravel()],
                          axis=1).astype(np.float32)
        colors = np.ones((coords.shape[0], 4), np.float32)
        colors[:, 0] = (coords[:, 0] * 0.5 + 0.5)
        colors[:, 2] = (coords[:, 2] * 0.5 + 0.5)
        colors[:, 1] = 0.35
    elif kind == "cloud":
        coords = rng.normal(0.0, 1.0, size=(n, 3)).astype(np.float32)
        colors = np.ones((n, 4), np.float32)
        norm = np.linalg.norm(coords, axis=1)
        norm = norm / max(float(norm.max()), 1e-6)
        colors[:, 0] = norm
        colors[:, 1] = 1.0 - norm
        colors[:, 2] = 0.6
    else:
        raise SystemExit(f"unknown --demo {kind!r}; try 'grid' or 'cloud'")

    m = coords.shape[0]
    edges = None
    if m > 1:
        a = rng.integers(0, m, size=min(m * 2, 200_000))
        b = rng.integers(0, m, size=a.size)
        keep = a != b
        edges = np.stack([a[keep], b[keep]], axis=1).astype(np.int32)
    return ArraySource(coords, edges=edges, colors=colors), colors


def _npz_source(path):
    """(ArraySource, colors) from an .npz with coords[, edges][, colors]."""
    data = np.load(path)
    if "coords" not in data:
        raise SystemExit(f"{path}: no 'coords' array")
    coords = np.asarray(data["coords"], np.float32)
    edges = np.asarray(data["edges"], np.int32) if "edges" in data else None
    if "colors" in data:
        colors = np.asarray(data["colors"], np.float32)
    else:
        colors = np.ones((coords.shape[0], 4), np.float32) * 0.85
        colors[:, 3] = 1.0
    return ArraySource(coords, edges=edges, colors=colors), colors


# The 80-byte rig the impostor shaders declare, packed byte for byte like
# SciGraphs/ui/gpu_render/draw.py::_lighting_ubo. Plain three-point values, not
# the add-on's lamp-derived ones, so shading here won't match the viewport.
def default_light_rig():
    rig = np.zeros(20, np.float32)
    rig[0:3] = (0.32, 0.42, 0.85)      # key, from upper right, toward camera
    rig[4:8] = (1.0, 0.98, 0.94, 0.35)  # key color, w = rim
    rig[8:11] = (-0.55, -0.25, 0.45)   # fill, from lower left
    rig[12:16] = (0.32, 0.35, 0.42, 0.0)
    rig[16:20] = (0.16, 0.17, 0.20, 0.0)
    for off in (0, 8):
        v = rig[off:off + 3]
        rig[off:off + 3] = v / max(float(np.linalg.norm(v)), 1e-9)
    return rig


def _parse_size(text):
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise SystemExit(f"--size wants WIDTHxHEIGHT, got {text!r}")


def _parse_vec(text, name):
    try:
        parts = [float(v) for v in text.split(",")]
    except ValueError:
        raise SystemExit(f"--{name} wants comma-separated numbers, got {text!r}")
    return parts


def build_specs(source, colors, point_size, sphere_radius, use_spheres,
                library):
    """MeshSpecs for a source, given what ``library`` has. Asking for spheres
    where only round_point is ported draws points and warns."""
    coords = source.coords()
    specs = []
    if use_spheres and library.get(mesh.ShaderRef("sphere")) is not None:
        radii = np.full(coords.shape[0], float(sphere_radius), np.float32)
        specs.append(mesh.sphere_spec(coords, colors, radii))
    else:
        if use_spheres:
            print("sphere shader not ported yet; drawing round points",
                  file=sys.stderr)
        for spec, _weight in mesh.point_specs(coords, colors):
            specs.append(mesh.MeshSpec(
                topology=spec.topology, shader=spec.shader, attrs=spec.attrs,
                indices=spec.indices,
                state=DrawState(point_size=float(point_size))))
    return [s for s in specs if s is not None]


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m scigraphs_engine.backends.wgpu.render",
        description="Render a graph to a PNG with wgpu.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--npz", help="an .npz with coords[, edges][, colors]")
    src.add_argument("--demo", default="grid", choices=("grid", "cloud"),
                     help="a generated graph (default: grid)")
    p.add_argument("--nodes", type=int, default=2500,
                   help="node count for --demo (default: 2500)")
    p.add_argument("--out", default="render.png", help="output PNG path")
    p.add_argument("--size", default="960x540", help="WIDTHxHEIGHT")
    p.add_argument("--point-size", type=float, default=6.0,
                   help="point diameter in pixels")
    p.add_argument("--spheres", action="store_true",
                   help="draw sphere impostors instead of flat points")
    p.add_argument("--radius", type=float, default=0.02,
                   help="sphere radius in world units, with --spheres")
    p.add_argument("--background", default="0.05,0.05,0.07,1.0")
    p.add_argument("--eye", help="camera position x,y,z (default: framed)")
    p.add_argument("--target", help="camera aim point x,y,z")
    p.add_argument("--fov", type=float, default=45.0)
    p.add_argument("--fallback-adapter", action="store_true",
                   help="use a software adapter (CI without a GPU)")
    p.add_argument("--list-shaders", action="store_true",
                   help="print which shaders this build actually has, and exit")
    p.add_argument("--stats", help="write a JSON report next to the PNG")
    args = p.parse_args(argv)

    width, height = _parse_size(args.size)

    try:
        ctx = acquire(fallback=args.fallback_adapter)
    except DeviceError as exc:
        print(f"no GPU: {exc}", file=sys.stderr)
        return 2

    library = ShaderLibrary(ctx)
    if args.list_shaders:
        for name in library.available():
            print(name)
        return 0

    t0 = time.perf_counter()
    source, colors = (_npz_source(args.npz) if args.npz
                      else _demo_source(args.demo, args.nodes))
    coords = source.coords()
    t_build = time.perf_counter()

    target = Target(ctx, width, height)
    renderer = Renderer(ctx, target, library=library)

    if args.eye:
        eye = _parse_vec(args.eye, "eye")
        aim = _parse_vec(args.target, "target") if args.target \
            else list(coords.mean(axis=0))
        span = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
        dist = float(np.linalg.norm(np.asarray(eye) - np.asarray(aim)))
        cam = cam_mod.Camera(
            cam_mod.look_at(eye, aim),
            cam_mod.perspective(args.fov, width / height,
                                max(dist - span, dist * 1e-3), dist + span * 2),
            near=max(dist - span, dist * 1e-3), far=dist + span * 2)
    else:
        cam = cam_mod.fit_view(coords, (width, height), fov_y_deg=args.fov)

    specs = build_specs(source, colors, args.point_size, args.radius,
                        args.spheres, library)
    meshes = [renderer.upload(s) for s in specs]
    meshes = [m for m in meshes if m is not None]
    renderer.sync()
    t_upload = time.perf_counter()

    if not meshes:
        print("nothing drawable: no ported shader matched any spec",
              file=sys.stderr)
        target.clear(tuple(_parse_vec(args.background, "background")))

    # Named blocks past the two the renderer owns, read off the shader
    # declarations so a newly ported shader needs no edit here.
    wanted = {b.name for m in meshes for b in m.shader.bindings} - {"camera",
                                                                    "draw"}
    blocks = {}
    for name in sorted(wanted):
        if name == "lights":
            blocks[name] = renderer.block_buffer(name, default_light_rig())
        else:
            print(f"shader wants an unknown uniform block {name!r}; "
                  f"supplying zeros", file=sys.stderr)
            blocks[name] = renderer.block_buffer(name, np.zeros(64, np.float32))

    stats = renderer.render(
        meshes, cam, blocks=blocks,
        clear_color=tuple(_parse_vec(args.background, "background")))
    renderer.sync()
    t_draw = time.perf_counter()

    target.save_png(args.out)
    t_end = time.perf_counter()

    report = {
        "output": os.path.abspath(args.out),
        "size": [width, height],
        "nodes": int(coords.shape[0]),
        "adapter": ctx.info,
        "shaders_available": library.available(),
        "draws": stats.get("draws", 0),
        "pipelines": stats.get("pipelines_cached", 0),
        "ms": {
            "build": round((t_build - t0) * 1e3, 3),
            "upload": round((t_upload - t_build) * 1e3, 3),
            "draw": round((t_draw - t_upload) * 1e3, 3),
            "encode_png": round((t_end - t_draw) * 1e3, 3),
        },
    }
    if args.stats:
        with open(args.stats, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
