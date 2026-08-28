# A ``frame_change_post`` handler drives the layout, not a modal timer: the
# handler also fires while scrubbing and during F12, where a timer leaves the
# render showing whatever the mesh held. First visit to a frame advances and
# records; later visits replay, since a force-directed layout has no inverse.

import numpy as np

import bpy
from bpy.app.handlers import persistent

from . import draw, dynamic, geometry
from .state import is_graph_object
def _settings(scene, st=None):
    if st is not None:
        return st
    from .host import settings_from_scene
    return settings_from_scene(scene)


# Keyed by object name: custom properties are too slow for tens of MB per frame.
_PLAYBACKS = {}

# Past this, recording holds the last frame; a trajectory with a hole scrubs wrong.
_TRAJECTORY_MAX_BYTES = 512 * 1024 * 1024


class Source:
    """``advance(n)`` gives (N, 3) float32 positions, None if already on GPU."""

    def advance(self, iterations):
        raise NotImplementedError

    @property
    def positions(self):
        raise NotImplementedError

    def set_params(self, params):
        """Retune mid-run; return the refused names, or None if this source has
        no parameters to retune at all."""
        return None

    def rewind(self, coords):
        """Restart at *coords*, keeping nothing from the steps after them."""
        raise NotImplementedError


class SimSource(Source):
    def __init__(self, coords, edges, **params):
        from scigraphs_core.mesh.layouts.simulation import ForceSim
        self.sim = ForceSim(coords, edges, **params)
        self.moved = 0.0

    def advance(self, iterations):
        self.moved = self.sim.step(iterations)
        return self.sim.positions

    @property
    def positions(self):
        return self.sim.positions

    def set_params(self, params):
        return self.sim.update_params(**params)

    def rewind(self, coords):
        self.sim.reset_state(coords)


def _refresh_structures(sim_gpu, gpu_sim, mass, near_k):
    """The two spatial structures the force kernel reads, rebuilt together.

    Near field on the GPU (bins, then a neighbor list); far field here, as one
    monopole per occupied cell of the same coarse grid the numpy reference uses.
    Returns the ``(cells, cell_of)`` textures and writes the cell count and the
    softening into the sim's params, which is where the kernel reads them.
    """
    pos = gpu_sim.positions()
    lo = pos.min(axis=0)
    span = np.maximum(np.ptp(pos, axis=0), 1e-9)
    gpu_sim.refresh_structures(lo, span, k=near_k)
    from scigraphs_core.mesh.layouts import simulation as ref
    res = ref.FAR_RES
    cell = np.clip(((pos - lo) / span * res).astype(np.int32), 0, res - 1)
    flat = (cell[:, 0] * res + cell[:, 1]) * res + cell[:, 2]
    occupied, inverse = np.unique(flat, return_inverse=True)
    cmass = np.bincount(inverse, weights=mass,
                        minlength=occupied.size).astype(np.float32)
    centroid = np.stack([
        np.bincount(inverse, weights=mass * pos[:, ax],
                    minlength=occupied.size).astype(np.float32) / cmass
        for ax in range(3)
    ], axis=1)
    buf = np.zeros((occupied.size, 4), dtype=np.float32)
    buf[:, :3] = centroid
    buf[:, 3] = cmass
    cells = sim_gpu._texture(buf)
    cell_of = sim_gpu._texture(np.stack([inverse.astype(np.float32)] * 4, axis=1))
    gpu_sim.params["cells"] = int(occupied.size)
    gpu_sim.params["far_soften"] = float(
        (ref.FAR_SOFTEN * float(span.max()) / res) ** 2)
    counts = np.bincount(inverse, minlength=occupied.size)
    return cells, cell_of, (lo, span, occupied, counts)


class ComputeSource(Source):
    """Compute-shader layout, matching numpy to float32. Readback is the cap.

    ``ref`` is the numpy simulator the parameters came from, kept but never
    stepped: it owns the normalization, so a live change goes through it and
    the two backends cannot drift apart."""

    def __init__(self, coords, edges, mass, k, params, gravity, dims=3,
                 strong_gravity=False, refresh_every=8, near_k=16, ref=None):
        from . import sim_gpu
        self._gpu = sim_gpu
        self.sim = sim_gpu.GpuSim(coords, edges, mass, k, params)
        self.ref = ref
        self.gravity = float(gravity)
        self.dims = int(dims)
        self.strong_gravity = bool(strong_gravity)
        self.refresh_every = max(1, int(refresh_every))
        self.near_k = int(near_k)
        self.moved = 0.0
        self._cells = None
        self._cell_of = None
        self.grid_snapshot = None
        self._age = self.refresh_every
        self._pos = np.ascontiguousarray(coords, dtype=np.float32)
        self.mass = np.ascontiguousarray(mass, dtype=np.float32)
        self._center0 = self._pos.mean(axis=0)

    def usable(self):
        return self.sim.usable()

    def _refresh(self):
        self._cells, self._cell_of, self.grid_snapshot = _refresh_structures(
            self._gpu, self.sim, self.mass, self.near_k)
        self._age = 0

    def advance(self, iterations):
        for _ in range(max(1, int(iterations))):
            if self._age >= self.refresh_every:
                self._refresh()
            self._age += 1
            if not self.sim.advance(self._cells, self._cell_of, self.gravity,
                                    strong_gravity=self.strong_gravity,
                                    dims=self.dims):
                return None
        self.moved = float(self.sim.moved)
        self._pos = self.sim.positions()
        self._pos -= self._pos.mean(axis=0) - self._center0
        return self._pos

    @property
    def positions(self):
        return self._pos.copy()

    def set_params(self, params):
        if self.ref is None:
            return tuple(params)
        ref = self.ref
        refused = ref.update_params(**params)
        remass = not np.array_equal(self.mass, ref.mass)
        self.mass = np.ascontiguousarray(ref.mass, dtype=np.float32)
        self.sim.set_params({
            "model": ref.model,
            "attraction": ref.attraction,
            "repulsion": ref.repulsion,
            "k": ref.k,
            "near_soften": float((0.01 * ref.k) ** 2),
            "jitter_tolerance": ref.jitter_tolerance,
        })
        self.gravity = float(ref.gravity)
        self.strong_gravity = bool(ref.strong_gravity)
        self.dims = int(ref.dimensions)
        if remass:
            self.sim.set_state(self.sim.positions(), self.mass, reset=False)
            self._age = self.refresh_every
        return refused

    def rewind(self, coords):
        pos = np.ascontiguousarray(coords, dtype=np.float32)
        if self.ref is not None:
            self.ref.reset_state(pos)
        self.sim.set_state(pos, self.mass)
        self._pos = pos.copy()
        self._center0 = self._pos.mean(axis=0)
        self._age = self.refresh_every


class YifanHuComputeSource(Source):
    """Yifan Hu with each level's arithmetic in compute shaders.

    The schedule does not come along: ForceSim keeps the hierarchy, the cooling
    rule, the convergence test and the prolongations, and calls in here through
    ``step_external`` for the one thing a GPU is better at. So the levels, the
    matching and the step rule stay in the module that has tests for them, and
    a prolongation is the only event that rebuilds GPU state, four or five times
    over a whole run rather than once a frame.
    """

    def __init__(self, sim, near_k=16, refresh_every=8):
        from . import sim_gpu
        self._gpu = sim_gpu
        self.sim = sim
        self.near_k = int(near_k)
        self.refresh_every = max(1, int(refresh_every))
        self.moved = 0.0
        self.gpu = None
        self.level = None
        self.mass = None
        self._cells = self._cell_of = None
        self._age = 0
        self._pos = np.ascontiguousarray(sim.pos, dtype=np.float32)
        self._ok = (sim_gpu.available()
                    and sim_gpu._integrate_shader() is not None)

    def _build(self, level):
        """Upload a level. Its own k sets the near-field softening, so this has
        to happen per level rather than once.

        Returns False for a level the kernel cannot take, which is not a
        failure: contraction can leave a coarse level with no edges, and there
        is no CSR to build for one.
        """
        params = {
            "cells": 0, "model": level.model,
            "attraction": level.attraction, "repulsion": level.repulsion,
            "k": level.k, "far_soften": 0.0,
            "near_soften": float((0.01 * level.k) ** 2),
            "jitter_tolerance": level.jitter_tolerance,
        }
        self.mass = np.ascontiguousarray(level.mass, dtype=np.float32)
        gpu_sim = self._gpu.GpuSim(level.pos, level.edges, level.mass,
                                   level.k, params)
        self.gpu = gpu_sim if gpu_sim.usable() else None
        self.level = level
        self._cells = self._cell_of = None
        self.grid_snapshot = None
        self._age = self.refresh_every
        return self.gpu is not None

    def usable(self):
        return bool(self._ok)

    def _advance_level(self, level, distance):
        """``step_external``'s callback: move *level* by *distance* along each
        node's own force, and report back what the step rule needs. None hands
        the level back, which the schedule then runs in numpy."""
        if level.n < _GPU_MIN_NODES:
            return None
        if self.level is not level:
            self._build(level)
        if self.gpu is None:
            return None
        if self._age >= self.refresh_every:
            self._cells, self._cell_of, self.grid_snapshot = _refresh_structures(
                self._gpu, self.gpu, self.mass, self.near_k)
            self._age = 0
        self._age += 1
        if not self.gpu.advance(self._cells, self._cell_of, level.gravity,
                                strong_gravity=level.strong_gravity,
                                dims=level.dimensions,
                                integrator=self._gpu.INT_YIFAN_HU,
                                yh_step=distance):
            self.gpu = None
            return None
        self.gpu.reduce()
        level.pos = self.gpu.positions()
        return self.gpu.moved, self.gpu.fnorm

    def advance(self, iterations):
        for _ in range(max(1, int(iterations))):
            self.moved = self.sim.step_external(self._advance_level)
        self._pos = np.ascontiguousarray(self.sim.pos, dtype=np.float32).copy()
        return self._pos

    @property
    def positions(self):
        return self._pos.copy()

    def set_params(self, params):
        refused = self.sim.update_params(**params)
        self.level = None
        return refused

    def rewind(self, coords):
        self.sim.reset_state(np.ascontiguousarray(coords, dtype=np.float32))
        self.level = None
        self._pos = np.ascontiguousarray(self.sim.pos, dtype=np.float32).copy()


class TargetSource(Source):
    """Ease towards a one-shot layout, so every menu entry animates."""

    def __init__(self, coords, target, frames=60):
        self.start = np.ascontiguousarray(coords, dtype=np.float32)
        self.target = np.ascontiguousarray(target, dtype=np.float32)
        self.frames = max(1, int(frames))
        self.t = 0.0
        self.moved = 0.0
        self._pos = self.start.copy()

    def advance(self, iterations):
        prev = self._pos
        self.t = min(1.0, self.t + float(iterations) / self.frames)
        # Smoothstep: leaves and arrives at rest, unlike a linear ramp.
        e = self.t * self.t * (3.0 - 2.0 * self.t)
        self._pos = self.start + (self.target - self.start) * np.float32(e)
        self.moved = float(np.linalg.norm(self._pos - prev, axis=1).sum())
        return self._pos

    @property
    def positions(self):
        return self._pos.copy()


class Playback:
    """A recorded head and a live tail.

    Frames from ``start_frame`` up to the recording limit are stored, and a
    stored frame is always exactly what was drawn there. Past the limit the
    simulation keeps running but nothing is kept, so those frames are not
    scrubbable and not reproducible; ``live`` says which side of the line the
    playhead is on and the panel has to show it.

    A live parameter change drops the recorded frames after the playhead and
    rewinds the source to it, so the trajectory stays the trajectory of one
    parameter schedule rather than a splice of two.
    """

    def __init__(self, obj, source, start_frame, steps_per_frame=1,
                 max_frames=500, params=None):
        self.name = obj.name
        self.source = source
        self.start_frame = int(start_frame)
        self.steps_per_frame = max(1, int(steps_per_frame))
        self.max_frames = max(1, int(max_frames))
        self.frames = [source.positions]
        self.settled = False
        self._bytes = self.frames[0].nbytes
        self.model = None
        self.algorithm = ""
        self.interpolated = False
        self.rebuilds = 0
        self.stalled = False
        self.params = dict(params or {})
        self.param_note = ""
        self.changes = 0
        self.live = False
        self._live_idx = 0
        self._live_pos = None

    def _capacity_left(self):
        return (len(self.frames) < self.max_frames
                and self._bytes < _TRAJECTORY_MAX_BYTES)

    def positions_at(self, frame):
        idx = int(frame) - self.start_frame
        if idx <= 0:
            self.live = False
            return self.frames[0]

        while len(self.frames) <= idx and self._capacity_left():
            pos = self.source.advance(self.steps_per_frame)
            if pos is None:
                pos = self.source.positions
            self.frames.append(pos)
            self._bytes += pos.nbytes
        if idx < len(self.frames):
            self.live = False
            return self.frames[idx]

        self.live = True
        frontier = len(self.frames) - 1
        if self._live_idx < frontier:
            self._live_idx, self._live_pos = frontier, self.frames[-1]
        if idx > self._live_idx:
            pos = self.source.advance(self.steps_per_frame)
            if pos is None:
                pos = self.source.positions
            self._live_idx, self._live_pos = idx, pos
        return self._live_pos

    def final_positions(self):
        """What the mesh should keep: the newest state, recorded or live."""
        return self._live_pos if self._live_pos is not None else self.frames[-1]

    def apply_params(self, params, frame):
        """Retune the running source from a scene snapshot; True if anything
        changed. Frames after *frame* were simulated under the old values, so
        they are dropped and the source rewound, which leaves every frame that
        survives still exactly what was drawn at it."""
        if params == self.params:
            return False
        forces = {name: value for name, value in params.items()
                  if name not in _PANEL_ONLY and value is not None}
        refused = self.source.set_params(forces)
        self.params = dict(params)
        if refused is None:
            return False

        self.param_note = ", ".join(sorted(refused))
        if 'model' not in refused:
            self.model = forces.get("model", self.model)
        self.algorithm = params.get("algorithm") or self.algorithm

        idx = max(0, int(frame) - self.start_frame)
        if idx < len(self.frames) - 1:
            for dropped in self.frames[idx + 1:]:
                self._bytes -= dropped.nbytes
            del self.frames[idx + 1:]
            self.source.rewind(self.frames[-1])
            self.live = False
            self._live_idx, self._live_pos = 0, None
        self.changes += 1
        return True

    def stats(self):
        return {
            "frames": len(self.frames),
            "mb": self._bytes / (1024 * 1024),
            "full": not self._capacity_left(),
            "live": self.live,
            "changes": self.changes,
            "note": self.param_note,
            "moved": getattr(self.source, "moved", 0.0),
            "rebuilds": self.rebuilds,
            "stalled": self.stalled,
        }


ITERATIVE_MODELS = {
    'FORCEATLAS2': 'FA2',
    'SPRING': 'FR',
    'SPRING_3D': 'FR',
    'IGRAPH_FR': 'FR',
}

_ANIMATED_APPROXIMATION = {}


def animated_note(algo):
    """What the preview is really simulating, when that differs from the name."""
    return _ANIMATED_APPROXIMATION.get(algo)


def algorithm_model(scene):
    """``(model, algorithm)``: the force law to run, or None to interpolate.

    The animation has its own algorithm setting, listing only the models that
    can be stepped. It falls back to the layout panel's choice so an older
    scene, or a build without the preview properties, still resolves."""
    props = getattr(scene, "scigraphs", None)
    algo = getattr(scene, "scigraphs_preview_animate_algorithm", None)
    if not algo:
        algo = str(getattr(props, "layout_algorithm", 'FORCEATLAS2')) if props \
            else 'FORCEATLAS2'
    algo = str(algo)
    model = ITERATIVE_MODELS.get(algo)
    if model == 'FA2' and props is not None \
            and bool(getattr(props, "fa2_lin_log_mode", False)):
        model = 'LINLOG'
    return model, algo


# Below this the GPU setup costs more than it saves: 9 ms numpy vs 2.4 ms at 10k.
_GPU_MIN_NODES = 20_000


def _compute_source(sim_source, scene):
    try:
        from . import sim_gpu
        if not sim_gpu.available():
            return None
        ref = sim_source.sim
        if ref.model not in sim_gpu.MODEL_CODES:
            return None
        if ref.model == 'YIFAN_HU':
            src = YifanHuComputeSource(
                ref, near_k=int(getattr(ref, "near_k", 16)))
            return src if src.usable() else None
        params = {
            "cells": 0, "model": ref.model,
            "attraction": ref.attraction, "repulsion": ref.repulsion,
            "k": ref.k, "far_soften": 0.0,
            "near_soften": float((0.01 * ref.k) ** 2),
            "jitter_tolerance": ref.jitter_tolerance,
        }
        src = ComputeSource(
            ref.pos, ref.edges, ref.mass, ref.k, params,
            gravity=ref.gravity, dims=ref.dimensions,
            strong_gravity=ref.strong_gravity,
            near_k=int(getattr(ref, "near_k", 16)),
            ref=ref,
        )
        return src if src.usable() else None
    except Exception:  # noqa: BLE001 - a backend that refuses is not a crash
        return None


def _sim_params(scene):
    props = getattr(scene, "scigraphs", None)
    if props is None:
        return {}
    return {
        "repulsion": float(getattr(props, "repulsion_strength", 1.0)),
        "attraction": float(getattr(props, "attraction_strength", 1.0)),
        "gravity": float(getattr(props, "gravity_strength", 0.1)),
        "scale": float(getattr(props, "layout_scale", 5.0)),
        "strong_gravity": bool(getattr(props, "fa2_strong_gravity", False)),
        "edge_weight_influence": float(
            getattr(props, "fa2_edge_weight_influence", 1.0)),
        "jitter_tolerance": float(getattr(props, "fa2_jitter_tolerance", 1.0)),
        "repulsion_mode": str(getattr(
            scene, "scigraphs_preview_repulsion", 'GRID')),
        "theta": float(getattr(scene, "scigraphs_preview_theta", 0.6)),
    }


def quadtree_available():
    """Whether the wheel in this Blender carries Graphviz's quadtree."""
    from scigraphs_core.mesh.layouts import simulation as ref
    return ref.quadtree_repulsion() is not None


def _yh_params(scene):
    """Yifan Hu's own four, kept out of the snapshot for every other model:
    ``update_params`` refuses a name it does not know, and a refusal shows in
    the panel, so ForceAtlas2 would carry a permanent complaint about them.

    Unused while YIFAN_HU is out of ITERATIVE_MODELS; putting the key back is
    what makes this reachable again.
    """
    props = getattr(scene, "scigraphs", None)
    if props is None:
        return {}
    return {
        "yh_step": float(getattr(props, "yh_initial_step", 0.15)),
        "yh_cool": float(getattr(props, "yh_step_ratio", 0.9)),
        "yh_tol": float(getattr(props, "yh_convergence", 0.0015)),
        "yh_adaptive": bool(getattr(props, "yh_adaptive_cooling", True)),
    }


def _sim_dimensions(scene, algo):
    """How many axes the simulation moves, matching what the one-shot run of the
    same algorithm produces. Not part of the live snapshot: ForceSim refuses a
    dimension change mid-run, since flattening is not invertible."""
    if algo == 'SPRING':
        return 2
    if algo == 'YIFAN_HU':
        props = getattr(scene, "scigraphs", None)
        return 3 if str(getattr(props, "sfdp_dim", '2Z')) == '3' else 2
    return 3


_PANEL_ONLY = frozenset({"algorithm"})


def _live_params(scene):
    """The scene snapshot a running playback is diffed against each frame.

    ``model`` is left out when the chosen algorithm has no iterative form:
    swapping one in would mean a different kind of source, not a retune, so
    the running force model is kept instead.
    """
    params = _sim_params(scene)
    model, algo = algorithm_model(scene)
    if model is not None:
        params["model"] = model
    if model == 'YIFAN_HU':
        params.update(_yh_params(scene))
    params["algorithm"] = algo
    return params


# ``_yifan_hu_layout`` routes to Graphviz sfdp, which aborts Blender on an
# 800-node random graph, and a C-level ``abort()`` cannot be caught. The rest
# only block long enough to look like a crash (Circle Packing: tens of seconds).
_UNSAFE_TARGETS = frozenset({
    'YIFAN_HU',
    'CIRCLE_PACKING',
    'GRAPHVIZ_DOT', 'GRAPHVIZ_NEATO', 'GRAPHVIZ_FDP', 'GRAPHVIZ_SFDP',
    'GRAPHVIZ_TWOPI', 'GRAPHVIZ_CIRCO', 'GRAPHVIZ_OSAGE',
    'GRAPHVIZ_PATCHWORK',
})


def target_is_safe(algo):
    return algo not in _UNSAFE_TARGETS


def _target_positions(obj, scene, algo, coords):
    if not target_is_safe(algo):
        return None
    try:
        from scigraphs_core.mesh.layouts.interactive import (
            _build_networkx_graph, _compute_layout_for_algorithm,
        )
        from scigraphs_core.mesh.mesh_utils import layout_edge_pairs
        # Edges read here, since the layout package never opens a datablock.
        graph, num_nodes = _build_networkx_graph(obj, layout_edge_pairs(obj))
        if graph is None:
            return None
        props = getattr(scene, "scigraphs", None)
        target = _compute_layout_for_algorithm(
            graph, num_nodes, algo,
            float(getattr(props, "layout_scale", 5.0)) if props else 5.0,
            props)
    except Exception:  # noqa: BLE001 - a missing backend is not a crash
        return None
    target = np.asarray(target, dtype=np.float32)
    return target if target.shape == coords.shape else None


def start(obj, scene, st=None):
    st = _settings(scene, st)
    coords = geometry.extract_node_coords(obj.data)
    edges = geometry.extract_edges(obj.data)
    seed = int(obj.get("scigraphs_layout_seed", 0))
    model, algo = algorithm_model(scene)

    source = None
    if model is None:
        target = _target_positions(obj, scene, algo, coords)
        if target is not None:
            source = TargetSource(
                coords, target,
                frames=max(2, int(st.animate_max_frames)
                           // 2))
    if source is None:
        sim_params = _sim_params(scene)
        if model == 'YIFAN_HU':
            sim_params["multilevel"] = False
            sim_params.update(_yh_params(scene))
        source = SimSource(coords, edges, seed=seed, model=model or 'FA2',
                           dimensions=_sim_dimensions(scene, algo),
                           **sim_params)
        # Built on the numpy simulator so the constants cannot drift.
        if bool(st.animate_gpu) \
                and coords.shape[0] >= _GPU_MIN_NODES:
            gpu_source = _compute_source(source, scene)
            if gpu_source is not None:
                source = gpu_source

    pb = Playback(
        obj, source, scene.frame_start,
        steps_per_frame=int(st.animate_steps),
        max_frames=int(st.animate_max_frames),
        params=_live_params(scene),
    )
    pb.model = model
    pb.algorithm = algo
    pb.interpolated = isinstance(source, TargetSource)
    _PLAYBACKS[obj.name] = pb
    return pb


def stop(obj=None, write_back=True):
    """Stop recording. With ``write_back`` the mesh keeps the final layout;
    that write fires the depsgraph, so it happens once here, not per frame."""
    names = [obj.name] if obj is not None else list(_PLAYBACKS)
    for name in names:
        pb = _PLAYBACKS.pop(name, None)
        if pb is None or not write_back:
            continue
        target = bpy.data.objects.get(name)
        if target is None or not is_graph_object(target):
            continue
        final = pb.final_positions()
        if len(target.data.vertices) == final.shape[0]:
            target.data.vertices.foreach_set("co", final.ravel())
            target.data.update()


def get(obj):
    return _PLAYBACKS.get(obj.name) if obj is not None else None


def apply_scene_params(scene):
    """Push the scene's force settings into every running playback.

    The frame handler does this too, so this only matters while the timeline is
    paused: it is what a property ``update`` callback should call so a slider
    dragged at a standstill still retunes the simulation.
    """
    if not _PLAYBACKS:
        return False
    params = _live_params(scene)
    frame = scene.frame_current
    return any([pb.apply_params(params, frame) for pb in _PLAYBACKS.values()])


def is_running(obj=None):
    return bool(_PLAYBACKS) if obj is None else obj.name in _PLAYBACKS


@persistent
def _on_frame_change(scene, _depsgraph=None):
    """A failed refresh rebuilds and retries; dropping it froze playback.

    The scene is also where a live parameter change is noticed: the handler
    already runs on every frame, and a slider has no other moment at which it
    could take effect, since only frames not yet simulated can answer to it.
    """
    if not _PLAYBACKS:
        return
    frame = scene.frame_current
    params = _live_params(scene)
    for name in list(_PLAYBACKS):
        obj = bpy.data.objects.get(name)
        if obj is None or not is_graph_object(obj):
            _PLAYBACKS.pop(name, None)
            continue
        pb = _PLAYBACKS[name]
        pb.apply_params(params, frame)
        coords = pb.positions_at(frame)
        if draw.refresh_positions(obj, coords):
            continue
        try:
            draw.get_cache_entry(obj, scene)
        except Exception:  # noqa: BLE001 - never let a handler break playback
            pass
        pb.rebuilds += 1
        if not draw.refresh_positions(obj, coords):
            pb.stalled = True


def register_handler():
    handlers = bpy.app.handlers.frame_change_post
    if _on_frame_change not in handlers:
        handlers.append(_on_frame_change)


def unregister_handler():
    handlers = bpy.app.handlers.frame_change_post
    if _on_frame_change in handlers:
        handlers.remove(_on_frame_change)
    _PLAYBACKS.clear()


def drop_cache(obj=None):
    stop(obj, write_back=False)
    dynamic.drop_cache(obj)


__all__ = [
    "Playback", "Source", "SimSource",
    "start", "stop", "get", "is_running", "apply_scene_params",
    "register_handler", "unregister_handler", "drop_cache",
]


def trajectory_bytes(num_nodes, frames):
    return int(num_nodes) * 12 * int(frames)


def max_frames_for(num_nodes):
    per = max(1, int(num_nodes) * 12)
    return max(1, min(500, _TRAJECTORY_MAX_BYTES // per))
