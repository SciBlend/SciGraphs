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


class ComputeSource(Source):
    """Compute-shader layout, matching numpy to float32. Readback is the cap."""

    def __init__(self, coords, edges, mass, k, params, gravity, dims=3,
                 strong_gravity=False, refresh_every=8, near_k=16):
        from . import sim_gpu
        self._gpu = sim_gpu
        self.sim = sim_gpu.GpuSim(coords, edges, mass, k, params)
        self.gravity = float(gravity)
        self.dims = int(dims)
        self.strong_gravity = bool(strong_gravity)
        self.refresh_every = max(1, int(refresh_every))
        self.near_k = int(near_k)
        self.moved = 0.0
        self._cells = None
        self._cell_of = None
        self._age = self.refresh_every
        self._pos = np.ascontiguousarray(coords, dtype=np.float32)
        self.mass = np.ascontiguousarray(mass, dtype=np.float32)

    def usable(self):
        return self.sim.usable()

    def _refresh(self):
        pos = self.sim.positions()
        lo = pos.min(axis=0)
        span = np.maximum(np.ptp(pos, axis=0), 1e-9)
        self.sim.refresh_structures(lo, span, k=self.near_k)
        from scigraphs_core.mesh.layouts import simulation as ref
        res = ref.FAR_RES
        cell = np.clip(((pos - lo) / span * res).astype(np.int32), 0, res - 1)
        flat = (cell[:, 0] * res + cell[:, 1]) * res + cell[:, 2]
        occupied, inverse = np.unique(flat, return_inverse=True)
        mass = self.mass
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
        self._cells = self._gpu._texture(buf)
        self._cell_of = self._gpu._texture(
            np.stack([inverse.astype(np.float32)] * 4, axis=1))
        self.sim.params["cells"] = int(occupied.size)
        self.sim.params["far_soften"] = float(
            (ref.FAR_SOFTEN * float(span.max()) / res) ** 2)
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
        return self._pos

    @property
    def positions(self):
        return self._pos.copy()


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
    def __init__(self, obj, source, start_frame, steps_per_frame=1,
                 max_frames=500):
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

    def _capacity_left(self):
        return (len(self.frames) < self.max_frames
                and self._bytes < _TRAJECTORY_MAX_BYTES)

    def positions_at(self, frame):
        idx = int(frame) - self.start_frame
        if idx <= 0:
            return self.frames[0]
        if idx < len(self.frames):
            return self.frames[idx]

        while len(self.frames) <= idx and self._capacity_left():
            pos = self.source.advance(self.steps_per_frame)
            if pos is None:
                pos = self.source.positions
            self.frames.append(pos)
            self._bytes += pos.nbytes
        return self.frames[-1]

    def stats(self):
        return {
            "frames": len(self.frames),
            "mb": self._bytes / (1024 * 1024),
            "full": not self._capacity_left(),
            "moved": getattr(self.source, "moved", 0.0),
            "rebuilds": self.rebuilds,
            "stalled": self.stalled,
        }


ITERATIVE_MODELS = {
    'FORCEATLAS2': 'FA2',
    'SPRING': 'FR',
    'SPRING_3D': 'FR',
    'IGRAPH_FR': 'FR',
    'LINLOG': 'LINLOG',
}


def algorithm_model(scene):
    """``(model, algorithm)``: the force law to run, or None to interpolate."""
    props = getattr(scene, "scigraphs", None)
    algo = str(getattr(props, "layout_algorithm", 'FORCEATLAS2')) if props \
        else 'FORCEATLAS2'
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
    }


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
        source = SimSource(coords, edges, seed=seed, model=model or 'FA2',
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
        final = pb.frames[-1]
        if len(target.data.vertices) == final.shape[0]:
            target.data.vertices.foreach_set("co", final.ravel())
            target.data.update()


def get(obj):
    return _PLAYBACKS.get(obj.name) if obj is not None else None


def is_running(obj=None):
    return bool(_PLAYBACKS) if obj is None else obj.name in _PLAYBACKS


@persistent
def _on_frame_change(scene, _depsgraph=None):
    """A failed refresh rebuilds and retries; dropping it froze playback."""
    if not _PLAYBACKS:
        return
    frame = scene.frame_current
    for name in list(_PLAYBACKS):
        obj = bpy.data.objects.get(name)
        if obj is None or not is_graph_object(obj):
            _PLAYBACKS.pop(name, None)
            continue
        pb = _PLAYBACKS[name]
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
    "start", "stop", "get", "is_running",
    "register_handler", "unregister_handler", "drop_cache",
]


def trajectory_bytes(num_nodes, frames):
    return int(num_nodes) * 12 * int(frames)


def max_frames_for(num_nodes):
    per = max(1, int(num_nodes) * 12)
    return max(1, min(500, _TRAJECTORY_MAX_BYTES // per))
