"""Helpers to enumerate scalar mesh attributes and read their numeric range.

Every Blender ``Mesh.attributes`` entry exposes a ``data_type`` and a
``domain``. For coloring we are interested in float-like scalar attributes
(``FLOAT``, ``INT``, ``INT8``) on POINT/EDGE/CORNER/FACE domains. These
functions return plain Python types so the UI layer can drive enums and
labels without holding references to live RNA structs.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


# Scalar data types we know how to coerce into floats.
SCALAR_DATA_TYPES = ("FLOAT", "INT", "INT8")

# Domains that make sense to color a graph mesh (vertex, edge, face, corner).
COLORABLE_DOMAINS = ("POINT", "EDGE", "CORNER", "FACE")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def list_scalar_attributes(mesh) -> List[Tuple[str, str, str]]:
    """List ``(name, data_type, domain)`` for every scalar attribute on the mesh.

    Internal Blender attributes (those starting with ``"."``) are filtered
    out: they are not user-facing data and are usually noise (``.position``,
    ``.corner_vert`` ...). The order of the result follows the iteration order
    of ``mesh.attributes`` so the UI can rely on it being stable across calls.
    """
    if mesh is None or not hasattr(mesh, "attributes"):
        return []

    found: List[Tuple[str, str, str]] = []
    for attr in mesh.attributes:
        name = getattr(attr, "name", "")
        if not name or name.startswith("."):
            continue
        data_type = getattr(attr, "data_type", "")
        domain = getattr(attr, "domain", "")
        if data_type not in SCALAR_DATA_TYPES:
            continue
        if domain not in COLORABLE_DOMAINS:
            continue
        found.append((name, data_type, domain))
    return found


def first_scalar_attribute(mesh) -> Optional[Tuple[str, str, str]]:
    """Return the first scalar attribute or ``None`` when there is none."""
    items = list_scalar_attributes(mesh)
    return items[0] if items else None


def find_attribute(mesh, name: str):
    """Return the RNA attribute with ``name`` or ``None`` if missing."""
    if mesh is None or not name:
        return None
    if not hasattr(mesh, "attributes"):
        return None
    return mesh.attributes.get(name)


# ---------------------------------------------------------------------------
# Value extraction
# ---------------------------------------------------------------------------

def read_attribute_values(mesh, name: str) -> np.ndarray:
    """Return the attribute values as a 1D float ``numpy`` array.

    Empty if the attribute is missing, has no data, or has an unsupported
    data type. Non-finite samples are kept (``values_to_rgba`` handles them).
    """
    attr = find_attribute(mesh, name)
    if attr is None:
        return np.zeros(0, dtype=float)

    data = getattr(attr, "data", None)
    if not data:
        return np.zeros(0, dtype=float)

    # ``data`` is a Blender RNA collection (always iterable) but Pylint can
    # only infer ``Optional[Any]`` from ``getattr`` so we guard manually.
    try:
        count = len(data)  # type: ignore[arg-type]
    except TypeError:
        return np.zeros(0, dtype=float)

    try:
        values = np.fromiter(
            (float(getattr(d, "value", 0.0)) for d in data),  # type: ignore[union-attr]
            dtype=float,
            count=count,
        )
    except (TypeError, ValueError):
        values = np.array(
            [float(getattr(d, "value", 0.0)) for d in data],  # type: ignore[union-attr]
            dtype=float,
        )

    return values


def attribute_value_range(
    mesh,
    name: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(vmin, vmax)`` for an attribute, or ``(None, None)`` if empty."""
    values = read_attribute_values(mesh, name)
    if values.size == 0:
        return (None, None)
    finite = np.isfinite(values)
    if not finite.any():
        return (None, None)
    return (float(values[finite].min()), float(values[finite].max()))


# ---------------------------------------------------------------------------
# Domain mapping
# ---------------------------------------------------------------------------

def color_domain_for(source_domain: str, mesh=None) -> str:
    """Pick a color-attribute domain compatible with a source attribute domain.

    Blender color attributes only live on ``POINT`` or ``CORNER``. ``EDGE``
    and ``FACE`` source attributes have to be promoted to ``CORNER`` colors.
    """
    domain = (source_domain or "POINT").upper()
    if domain == "POINT":
        return "POINT"
    if domain in ("EDGE", "FACE", "CORNER"):
        if mesh is not None and len(getattr(mesh, "loops", [])) == 0:
            # No loops -> CORNER would crash. Best-effort fall back to POINT.
            return "POINT"
        return "CORNER"
    return "POINT"


def expand_to_loops(
    mesh,
    edge_values: Iterable[float],
) -> np.ndarray:
    """Convert per-edge values into per-loop values for a Blender mesh.

    For meshes built with faces (most City2Graph polygon graphs) every loop
    knows the edge it belongs to via ``loop.edge_index``. We use that to
    spread edge values onto loop colors. Loops without a valid edge map fall
    back to ``0.0``.
    """
    edge_arr = np.asarray(list(edge_values), dtype=float)
    loops = getattr(mesh, "loops", None)
    if loops is None or len(loops) == 0 or edge_arr.size == 0:
        return np.zeros(0, dtype=float)

    out = np.zeros(len(loops), dtype=float)
    for i, loop in enumerate(loops):
        edge_index = getattr(loop, "edge_index", -1)
        if 0 <= edge_index < edge_arr.size:
            out[i] = edge_arr[edge_index]
    return out


def expand_to_corners(
    mesh,
    face_values: Iterable[float],
) -> np.ndarray:
    """Convert per-face values into per-loop values for a Blender mesh."""
    face_arr = np.asarray(list(face_values), dtype=float)
    loops = getattr(mesh, "loops", None)
    polygons = getattr(mesh, "polygons", None)
    if loops is None or polygons is None or face_arr.size == 0:
        return np.zeros(0, dtype=float)
    if len(loops) == 0 or len(polygons) == 0:
        return np.zeros(0, dtype=float)

    out = np.zeros(len(loops), dtype=float)
    for face_idx, poly in enumerate(polygons):
        if face_idx >= face_arr.size:
            break
        value = face_arr[face_idx]
        start = poly.loop_start
        end = start + poly.loop_total
        for li in range(start, end):
            if 0 <= li < out.size:
                out[li] = value
    return out


def values_for_color_domain(
    mesh,
    source_domain: str,
    values: Sequence[float],
    color_domain: str,
) -> np.ndarray:
    """Return values laid out for ``color_domain`` (POINT or CORNER)."""
    arr = np.asarray(list(values), dtype=float)
    src = (source_domain or "POINT").upper()
    target = (color_domain or "POINT").upper()

    if target == "POINT":
        if src == "POINT":
            return arr
        # No clean fallback for non-POINT -> POINT. Best effort: average
        # contributions to vertices via the mesh edge/face topology.
        if src == "EDGE":
            verts = getattr(mesh, "vertices", None)
            edges = getattr(mesh, "edges", None)
            if verts is None or edges is None or len(verts) == 0:
                return np.zeros(0, dtype=float)
            sums = np.zeros(len(verts), dtype=float)
            counts = np.zeros(len(verts), dtype=float)
            for ei, edge in enumerate(edges):
                if ei >= arr.size:
                    break
                v0, v1 = edge.vertices[0], edge.vertices[1]
                value = arr[ei]
                if 0 <= v0 < sums.size:
                    sums[v0] += value
                    counts[v0] += 1
                if 0 <= v1 < sums.size:
                    sums[v1] += value
                    counts[v1] += 1
            counts[counts == 0] = 1
            return sums / counts

        if src == "FACE":
            verts = getattr(mesh, "vertices", None)
            polygons = getattr(mesh, "polygons", None)
            if verts is None or polygons is None or len(verts) == 0:
                return np.zeros(0, dtype=float)
            sums = np.zeros(len(verts), dtype=float)
            counts = np.zeros(len(verts), dtype=float)
            for fi, poly in enumerate(polygons):
                if fi >= arr.size:
                    break
                value = arr[fi]
                for vi in poly.vertices:
                    if 0 <= vi < sums.size:
                        sums[vi] += value
                        counts[vi] += 1
            counts[counts == 0] = 1
            return sums / counts

        return arr  # CORNER -> POINT not commonly required.

    # target == CORNER
    if src == "EDGE":
        return expand_to_loops(mesh, arr)
    if src == "FACE":
        return expand_to_corners(mesh, arr)
    if src == "POINT":
        # Spread vertex values to each loop's vertex.
        loops = getattr(mesh, "loops", None)
        if loops is None or len(loops) == 0:
            return np.zeros(0, dtype=float)
        out = np.zeros(len(loops), dtype=float)
        for i, loop in enumerate(loops):
            vi = getattr(loop, "vertex_index", -1)
            if 0 <= vi < arr.size:
                out[i] = arr[vi]
        return out
    return arr  # CORNER source -> CORNER target.
