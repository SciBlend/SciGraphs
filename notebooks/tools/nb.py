"""Helpers for the published notebooks: output paths, path scrubbing, render
legibility checks, the environment report. Importing puts the repo on sys.path.
"""

import os
import pathlib
import sys

# momepy's joblib/loky resource-tracker subprocess does not inherit the add-on's
# `sys.path` and dies with `ModuleNotFoundError: No module named 'joblib'`, and
# the relaunches bury the output. `n_jobs=1` does not help; set before joblib.
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

# Blender's cwd is wherever it was launched from, so paths hang off this file.
TOOLS_DIR = pathlib.Path(__file__).resolve().parent
NOTEBOOKS_DIR = TOOLS_DIR.parent
REPO_ROOT = NOTEBOOKS_DIR.parent

OUT_DIR = NOTEBOOKS_DIR / "out"
DATA_DIR = NOTEBOOKS_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
for _d in (OUT_DIR, DATA_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Blender auto-loads the *installed* extension, a snapshot that goes stale, and
# registration does not follow `sys.path`: hence `SciGraphs.api` via operators.
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))


def out(*parts):
    """Path under `notebooks/out/`, with parent directories created."""
    path = OUT_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _png(name):
    name = str(name)
    return name if name.lower().endswith(".png") else name + ".png"


def repo(*parts):
    return REPO_ROOT.joinpath(*parts)


def rel(path):
    """A path string relative to the repository root, or with `~` for $HOME."""
    path = pathlib.Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        text = str(path)
        home = str(pathlib.Path.home())
        return "~" + text[len(home):] if text.startswith(home) else text


class _PathScrubbingStream:
    """stdout proxy for the absolute paths operators print past `rel()`."""

    def __init__(self, wrapped, replacements):
        self._wrapped = wrapped
        self._replacements = replacements

    def write(self, text):
        if text:
            for old, new in self._replacements:
                text = text.replace(old, new)
        return self._wrapped.write(text)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def _scrub_paths():
    replacements = [(str(REPO_ROOT).rstrip("/") + "/", ""),
                    (str(REPO_ROOT), "."),
                    (str(pathlib.Path.home()).rstrip("/") + "/", "~/")]
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if isinstance(stream, _PathScrubbingStream):
            continue
        setattr(sys, name, _PathScrubbingStream(stream, replacements))
    return True


_scrub_paths()


class quiet:
    """Redirect file descriptor 1 for a block. `bpy.ops.render.render()` writes
    `Saved: '/path.png'` from C, below where `redirect_stdout` can see it.
    """

    def __enter__(self):
        self._saved = None
        try:
            sys.stdout.flush()
            self._null = os.open(os.devnull, os.O_WRONLY)
            self._saved = os.dup(1)
            os.dup2(self._null, 1)
        except Exception:  # noqa: BLE001 - never fail a render over logging
            self._saved = None
        return self

    def __exit__(self, *exc):
        if self._saved is None:
            return False
        try:
            os.dup2(self._saved, 1)
            os.close(self._saved)
            os.close(self._null)
        except Exception:  # noqa: BLE001
            pass
        return False


def osmnx():
    """Point OSMnx at a cache in `notebooks/data/` so re-runs work offline."""
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(CACHE_DIR / "osmnx")
    return ox


_CORE_MODULES = [
    ("city2graph", "the library this whole suite is about"),
    ("geopandas", "GeoDataFrames, city2graph's canonical data structure"),
    ("shapely", "geometry"),
    ("osmnx", "OpenStreetMap street networks and POIs"),
    ("networkx", "graph algorithms"),
    ("libpysal", "spatial weights, used by contiguity_graph"),
    ("momepy", "urban morphometrics, used by the tessellation"),
    ("pandas", ""),
    ("numpy", ""),
]

_OPTIONAL_MODULES = [
    ("torch", "only needed for gdf_to_pyg and GNN work"),
    ("torch_geometric", "same"),
    ("matplotlib", "static plots; the notebooks work without it"),
    ("duckdb", "GTFS loading"),
]


def env_report(verbose=True):
    """Check the notebooks' imports; returns a dict a cell can assert on."""
    import importlib

    report = {"blender": None, "python": sys.version.split()[0],
              "required": {}, "optional": {}, "scigraphs": {}}

    try:
        import bpy
        report["blender"] = bpy.app.version_string
    except ImportError:
        report["blender"] = None

    def _probe(name):
        try:
            mod = importlib.import_module(name)
            return getattr(mod, "__version__", "?")
        except Exception as exc:  # noqa: BLE001 - report anything that goes wrong
            return f"MISSING ({type(exc).__name__})"

    for name, _why in _CORE_MODULES:
        report["required"][name] = _probe(name)
    for name, _why in _OPTIONAL_MODULES:
        report["optional"][name] = _probe(name)

    if report["blender"]:
        import bpy

        scene = bpy.context.scene
        report["scigraphs"] = {
            "scene.scigraphs": hasattr(scene, "scigraphs"),
            "scene.city2graph": hasattr(scene, "city2graph"),
            "operators": len(dir(bpy.ops.scigraphs)) if hasattr(bpy.ops, "scigraphs") else 0,
        }

    if verbose:
        _print_report(report)
    return report


def _print_report(report):
    line = "-" * 62
    print(line)
    print(f"Blender          {report['blender'] or 'NOT RUNNING INSIDE BLENDER'}")
    print(f"Python           {report['python']}")
    print(line)
    print("Required:")
    for name, why in _CORE_MODULES:
        version = report["required"][name]
        mark = "!!" if str(version).startswith("MISSING") else "ok"
        print(f"  [{mark}] {name:<16} {version:<12} {why}")
    print("Optional:")
    for name, why in _OPTIONAL_MODULES:
        version = report["optional"][name]
        mark = "--" if str(version).startswith("MISSING") else "ok"
        print(f"  [{mark}] {name:<16} {version:<12} {why}")
    if report["scigraphs"]:
        print(line)
        sg = report["scigraphs"]
        print(f"SciGraphs        {sg['operators']} operators registered")
        print(f"  scene.scigraphs   {sg['scene.scigraphs']}")
        print(f"  scene.city2graph  {sg['scene.city2graph']}")
    print(line)


def missing(report=None):
    """Names of required modules that failed to import."""
    report = report or env_report(verbose=False)
    return [n for n, v in report["required"].items() if str(v).startswith("MISSING")]

def ink(path):
    """Non-background pixel fraction: above ~60% the nodes have merged into a
    solid mass, below ~1% there is nothing to see.
    """
    import numpy as np

    try:
        from PIL import Image
    except ImportError:
        return None
    image = np.asarray(Image.open(str(path)).convert("RGB"), dtype=np.float32)
    background = image.reshape(-1, 3)[0]
    return float((np.abs(image - background).sum(axis=2) > 24).mean())


def show(path, width=900):
    """Display a PNG inline, or print its path when there is no kernel."""
    try:
        from IPython import get_ipython
        from IPython.display import Image, display

        if get_ipython() is None:
            raise RuntimeError("no kernel")
        display(Image(filename=str(path), width=width))
        return True
    except Exception:
        print(f"  [image] {rel(path)}")
        return False


# Every render below runs inside `quiet()`, or `Saved:` leaks an absolute path.

def figure_preview(obj, filename, resolution=(1000, 750), width=900,
                   legible=True, **kwargs):
    """Render with the SciGraphs engine and show inline. Returns nothing: a cell
    ending in a call would echo the absolute path into the stored output.
    """
    from SciGraphs.api import preview

    with quiet():
        path = preview.render(obj, out(_png(filename)), resolution=resolution,
                              verbose=False, **kwargs)
    show(path, width=width)
    if legible:
        check_render(path)
    return None


def gallery_preview(objects, folder, resolution=(1280, 960), **kwargs):
    """Render each object in `{stem: object}` alone and check it is legible."""
    from SciGraphs.api import preview

    objects = {k: v for k, v in objects.items() if v is not None}

    results = {}
    for name, obj in objects.items():
        with quiet():
            path = preview.render(obj, out(_png(f"{folder}/{name}")),
                                  resolution=resolution, **kwargs)
        coverage = ink(path)
        results[name] = (path, coverage)
        show(path)
        check_render(path)
    return results


def check_render(path, minimum=0.005, maximum=0.60):
    """Assert a render is legible: neither empty nor a solid block of nodes."""
    coverage = ink(path)
    if coverage is None:
        print(f"[----] {pathlib.Path(path).name}: no Pillow, cannot measure")
        return True
    return check(f"{pathlib.Path(path).name} legible",
                 minimum <= coverage <= maximum,
                 f"{coverage * 100:.1f}% ink (healthy range "
                 f"{minimum * 100:.1f}–{maximum * 100:.0f}%)")


def describe_hetero(nodes_dict, edges_dict):
    """Print the shape of a heterogeneous graph: layers, then relations."""
    print("nodes")
    for layer, gdf in nodes_dict.items():
        print(f"  {layer:<22} {len(gdf):>8} {gdf.geometry.geom_type.iloc[0] if len(gdf) else '-'}")
    print("edges")
    for key, gdf in edges_dict.items():
        label = " -> ".join(str(k) for k in key) if isinstance(key, tuple) else str(key)
        print(f"  {label:<50} {len(gdf):>8}")
    return {"layers": {k: len(v) for k, v in nodes_dict.items()},
            "relations": {str(k): len(v) for k, v in edges_dict.items()}}


def check(label, condition, detail=""):
    """One line of a notebook's self-check, and the boolean for an assert."""
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))
    return bool(condition)


def render(obj, filename, **kwargs):
    """`figure()` without display or legibility check; returns the path."""
    from SciGraphs.api import render as _render

    with quiet():
        return _render.eevee(obj, out(_png(filename)), **kwargs)


def render_preview(obj, filename, **kwargs):
    """The same, through the add-on's own GPU engine rather than EEVEE."""
    from SciGraphs.api import preview

    with quiet():
        return preview.render(obj, out(_png(filename)), **kwargs)


def figure(obj, filename, resolution=(1000, 750), width=900, legible=True,
           **kwargs):
    """Render with EEVEE and show inline; returns nothing, unlike `render`."""
    from SciGraphs.api import render

    with quiet():
        path = render.eevee(obj, out(_png(filename)), resolution=resolution,
                            verbose=False, **kwargs)
    show(path, width=width)
    if legible:
        check_render(path)
    return None


def gallery(objects, folder, resolution=(1000, 750), **kwargs):
    """`gallery_preview` with EEVEE, returning the same {stem: (path, ink)}."""
    from SciGraphs.api import render

    objects = {k: v for k, v in objects.items() if v is not None}

    results = {}
    for name, obj in objects.items():
        with quiet():
            path = render.eevee(obj, out(_png(f"{folder}/{name}")),
                                resolution=resolution, **kwargs)
        coverage = ink(path)
        results[name] = (path, coverage)
        show(path)
        check_render(path)
    return results

