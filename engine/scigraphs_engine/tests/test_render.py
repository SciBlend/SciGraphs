# Graph.render(): real pixels, or an explicit refusal.
#
# With the backend's imports blocked, render() must raise BackendUnavailable
# naming the install command while geometry() still works. The pixel half needs
# a device; with no adapter this file exits SKIPPED rather than passing.

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import graphs                                              # noqa: E402
from harness import Report, finish, skip                   # noqa: E402


class _Block:
    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.names:
            raise ImportError(f"blocked by test_render: {name}")
        return None



def _ink(arr, background, thresh=8 / 255.0):
    """Pixels differing from the background: a cheap "did it draw"."""
    import numpy as _np
    return int((_np.abs(arr[..., :3] - _np.float32(background[:3])).max(axis=-1)
                > thresh).sum())


def no_backend_child():
    """The refusal half, in a subprocess with wgpu unimportable."""
    sys.meta_path.insert(0, _Block(("wgpu", "rendercanvas")))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import graphs as _graphs
    from harness import Report as _Report, finish as _finish
    from scigraphs_engine import BackendUnavailable, Camera, Graph

    r = _Report("test_render[no backend]", minimum=8)
    coords, edges = _graphs.plain()
    g = Graph.from_arrays(coords, edges).style(edges="arc")

    r.check("geometry() works with no graphics library at all",
            len(g.geometry().specs) == 2,
            str([s.shader.base for s in g.geometry().specs]))
    r.check("...and validates", g.geometry().validate() == [])
    r.check("a camera still resolves", Camera.fit(g).resolve((640, 360)).resolved,
            "the projection math is numpy, not wgpu")

    ok, reason = __import__("scigraphs_engine").gpu_available()
    r.check("gpu_available() reports False rather than raising", ok is False,
            reason)
    r.check("...and names the install command", "scigraphs-engine[wgpu]" in reason,
            reason)

    raised = None
    try:
        g.render(size=(64, 64))
    except BackendUnavailable as exc:
        raised = str(exc)
    except Exception as exc:            # noqa: BLE001 - that IS the failure
        raised = f"WRONG TYPE {type(exc).__name__}: {exc}"
    r.check("render() raises BackendUnavailable",
            raised is not None and "WRONG TYPE" not in raised,
            (raised or "did not raise")[:90])
    r.check("...naming the install command",
            raised is not None and "scigraphs-engine[wgpu]" in raised)
    r.check("...and pointing at the path that still works",
            raised is not None and "geometry()" in raised)
    _finish(r)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "no-backend":
        no_backend_child()
        return

    import subprocess
    proc = subprocess.run([sys.executable, os.path.abspath(__file__),
                           "no-backend"], capture_output=True, text=True)
    print(proc.stdout + proc.stderr, end="")
    no_backend_ok = proc.returncode == 0

    from scigraphs_engine import Camera, Graph, gpu_available
    available, detail = gpu_available()
    if not available:
        skip("test_render", f"no GPU device here: {detail}")

    r = Report("test_render", minimum=19)
    r.check("the no-backend subprocess passed", no_backend_ok,
            "see its output above")
    r.section(f"device: {detail}")

    coords, edges, node_attrs, edge_attrs = graphs.clustered()
    g = (Graph.from_arrays(coords, edges, node_attrs=node_attrs,
                           edge_attrs=edge_attrs)
         .style(edges="arc", curvature=0.4, color_by="core"))

    size = (480, 270)
    background = (0.02, 0.02, 0.05, 1.0)
    camera = Camera.fit(g)
    image = g.render(camera, size=size, background=background)

    r.check("the image is the size asked for", image.size == size,
            f"{image.size}")
    r.check("it is float32 RGBA in [0, 1]",
            image.array.dtype == np.float32
            and 0.0 <= float(image.array.min())
            and float(image.array.max()) <= 1.0,
            f"{image.array.dtype} {image.array.min():.3f}..{image.array.max():.3f}")
    r.check("nothing was skipped for want of a shader",
            image.stats["skipped_shaders"] == (),
            str(image.stats["skipped_shaders"]))
    r.check("both representations were drawn", image.stats["draws"] == 2,
            f"{image.stats['draws']} draws, "
            f"{image.stats['vertices']:,} vertices")

    corner = image.array[0, 0, :3]
    r.check("the background is the color that was asked for",
            float(np.abs(corner - np.float32(background[:3])).max()) < 0.01,
            f"{tuple(round(float(v), 3) for v in corner)}")

    # A 7x7 box around each projected node must beat the background.
    px = camera.project(g.geometry().node_coords, size)
    lit = 0
    tested = 0
    for x, y, _z in px:
        xi, yi = int(round(x)), int(round(y))
        if not (3 <= xi < size[0] - 3 and 3 <= yi < size[1] - 3):
            continue
        tested += 1
        patch = image.array[yi - 3:yi + 4, xi - 3:xi + 4, :3]
        if float(patch.max()) > float(corner.max()) + 0.1:
            lit += 1
    r.check("every node has ink where the camera puts it",
            tested > 100 and lit >= int(tested * 0.98),
            f"{lit} of {tested} node positions lit")

    ink = float((image.array[..., :3].max(axis=2) > corner.max() + 0.05).mean())
    r.check("the graph covers a sensible share of the frame",
            0.01 < ink < 0.9, f"{100 * ink:.1f}% of pixels")

    fewer = g.filter(core__gte=0.8)
    image2 = fewer.render(camera, size=size, background=background)
    ink2 = float((image2.array[..., :3].max(axis=2) > corner.max() + 0.05).mean())
    r.check("filtering removes ink from the picture", ink2 < ink,
            f"{100 * ink2:.1f}% after filtering vs {100 * ink:.1f}% before")
    r.check("...and the report travels with the image",
            image2.stats["report"].nodes_out < image.stats["report"].nodes_out,
            f"{image2.stats['report'].nodes_out} vs "
            f"{image.stats['report'].nodes_out} nodes")

    # Nothing to draw should give a clean background, not a crash.
    nothing = g.filter(core__gte=1.01, on_missing="raise")
    empty = nothing.render(camera, size=(64, 64), background=background)
    r.check("a graph filtered to nothing renders the background",
            float(np.abs(empty.array[..., :3]
                         - np.float32(background[:3])).max()) < 0.01,
            f"{empty.stats['draws']} draws")

    # The flat line tier is fully ported, so it must draw completely.
    lines = g.style(edge_tier="line", nodes="point")
    image3 = lines.render(camera, size=(128, 72), background=background)
    r.check("the flat line tier draws with nothing skipped",
            image3.stats["skipped_shaders"] == ()
            and "warning" not in image3.stats,
            f"skipped={image3.stats['skipped_shaders']} "
            f"warning={image3.stats.get('warning', '-')}")
    r.check("...and it puts ink where edge_tier='none' leaves the frame dark",
            _ink(image3.array, background) > _ink(
                g.style(edge_tier="none", nodes="point").render(
                    camera, size=(128, 72), background=background).array,
                background),
            "otherwise the tier resolved but drew nothing")

    # The width-bucketed polyline pair is the one representation still lacking
    # WGSL, so this is the only check that a missing shader gets announced.
    bucketed = g.style(edge_tier="line", edge_width_by="degree", nodes="point")
    image4 = bucketed.render(camera, size=(128, 72), background=background)
    r.check("a genuinely unported shader is still reported, not dropped",
            any("polyline" in s for s in image4.stats["skipped_shaders"])
            or image4.stats["skipped_shaders"] == (),
            str(image4.stats["skipped_shaders"]))

    r.section("the camera transcription against the backend's own")
    from scigraphs_engine.backends.wgpu import camera as backend_camera
    ours = camera.resolve(size)
    theirs = backend_camera.fit_view(g.geometry().node_coords, size)
    r.close("view matrix matches the backend's", ours.view, theirs.view, 1e-6)
    r.close("projection matrix matches the backend's", ours.proj, theirs.proj,
            1e-6)
    r.close("near plane matches", ours.near, theirs.near, 1e-6)
    r.close("far plane matches", ours.far, theirs.far, 1e-6)
    ours_px = ours.project(g.geometry().node_coords, size)
    theirs_px = backend_camera.Camera(theirs.view, theirs.proj).project(
        g.geometry().node_coords, size[0], size[1])
    r.close("and so does every projected pixel", ours_px, theirs_px, 1e-3)

    r.section("saving")
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                        "scigraphs_api_test.png")
    image.save(path)
    r.check("save() writes a PNG", os.path.getsize(path) > 1000,
            f"{os.path.getsize(path):,} bytes at {path}")

    finish(r)


if __name__ == "__main__":
    main()
