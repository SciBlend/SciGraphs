"""Materialize a top-k edge backbone as a second, thinner graph object.

GPU preview sparsification does not update the mesh, so EEVEE still draws every
edge. Thinning happens here on the GeoDataFrame instead.

`topk_mask` reimplements the engine's `_topk_mask` (no import) so
`verify_against_engine()` is a real comparison. Rank half-edges by weight at
each endpoint; an edge survives if top-k at either end. `sense='high'` (default)
keeps largest weights; negatives are clamped like `backbone_mask`.

`mesh_edge_rows()` mirrors mesh construction (drops self-loops, duplicate pairs,
missing endpoints). Unresolved weights become uniform — run `gpu_weights()`
before trusting backbone figures.
"""

import numpy as np

from . import graphs


def _gpu_simplify():
    """Lazy import of GPU simplify helpers (avoids pulling the full preview stack)."""
    from ..ui.gpu_render import simplify
    return simplify


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------

def topk_mask(edges, weights, k, sense='high'):
    """Boolean mask: keep edges in the top-`k` at either endpoint.

    Independent reimplementation of the engine's `_topk_mask`. `edges` is (E, 2),
    `weights` is (E,) or None (uniform). Ties break by position via `np.lexsort`.
    """
    edges = np.asarray(edges, dtype=np.int64)
    e = edges.shape[0]
    if e == 0:
        return np.zeros(0, dtype=bool)

    if weights is None:
        w = np.ones(e, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).copy()
        w[~np.isfinite(w)] = 0.0
    # Match backbone_mask: clamp negatives before ranking.
    w = np.maximum(w, 0.0)

    node = np.concatenate([edges[:, 0], edges[:, 1]])
    eid = np.concatenate([np.arange(e), np.arange(e)])
    ww = np.concatenate([w, w])

    # 'high': sort on -w so largest weight ranks 0.
    primary = -ww if str(sense).lower() == 'high' else ww
    order = np.lexsort((primary, node))

    grouped = node[order]
    starts = np.ones(grouped.size, dtype=bool)
    starts[1:] = grouped[1:] != grouped[:-1]
    group_start = np.maximum.accumulate(
        np.where(starts, np.arange(grouped.size), 0))
    rank = np.arange(grouped.size) - group_start

    mask = np.zeros(e, dtype=bool)
    mask[eid[order][rank < max(1, int(k))]] = True
    return mask


def verify_against_engine(edges, weights, k):
    """Compare `topk_mask` to the engine's `backbone_mask` on the same input.

    Returns agree/counts, or `available: False` if the engine cannot be imported.
    Compares source in the working tree, not necessarily the installed extension.
    """
    try:
        from ..core.render.simplify import backbone_mask
    except Exception as exc:  # noqa: BLE001 - the engine is optional here
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    theirs, _stats = backbone_mask(np.asarray(edges, dtype=np.int64), weights,
                                   'TOPK', k=k)
    mine = topk_mask(edges, weights, k)
    if theirs is None:
        return {"available": False, "reason": "backbone_mask declined the input"}
    differing = int(np.count_nonzero(theirs != mine))
    return {
        "available": True,
        "agree": differing == 0,
        "engine_kept": int(theirs.sum()),
        "ours_kept": int(mine.sum()),
        "differing_rows": differing,
    }


# --------------------------------------------------------------------------
# Reproducing the mesh the GPU filtered
# --------------------------------------------------------------------------

def mesh_edge_rows(nodes_gdf, edges_gdf):
    """Rows of `edges_gdf` that become mesh edges, plus endpoint indices.

    Mirrors `create_native_graph_from_gdfs`: skip missing endpoints, self-loops,
    and duplicate undirected pairs. Returns `(rows, pairs)`.
    """
    import pandas as pd

    position = {node_id: i for i, node_id in enumerate(nodes_gdf.index)}

    rows = []
    pairs = []
    seen = set()
    index = edges_gdf.index
    if not (isinstance(index, pd.MultiIndex) and index.nlevels >= 2):
        # Flat index → mesh with vertices but no edges.
        raise ValueError(
            "the edges frame needs the (source, target) MultiIndex city2graph "
            f"produces; got a {type(index).__name__} of {index.nlevels} level(s)")

    for row, key in enumerate(index):
        src = position.get(key[0])
        tgt = position.get(key[1])
        if src is None or tgt is None or src == tgt:
            continue
        pair = (min(src, tgt), max(src, tgt))
        if pair in seen:
            continue
        seen.add(pair)
        rows.append(row)
        pairs.append((src, tgt))

    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 2), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64), np.asarray(pairs, dtype=np.int64)


def gpu_weights(obj, attribute):
    """What `simplify.edge_weights_raw` would hand the backbone for this mesh.

    Returns `name`, `uniform`, `values`, and EDGE-domain scalar `candidates`.
    """
    if obj is None or obj.type != 'MESH':
        return {"name": None, "uniform": True, "values": None, "candidates": []}

    mesh = obj.data
    count = len(mesh.edges)
    candidates = [a.name for a in mesh.attributes
                  if a.domain == 'EDGE' and a.data_type in ('FLOAT', 'INT')]

    # Ask the same resolver the filter uses.
    gpu_simplify = _gpu_simplify()
    name = gpu_simplify.resolve_edge_weight_attr(mesh, attribute)
    if name is None:
        return {"name": None, "uniform": True, "values": None,
                "candidates": candidates}
    values = gpu_simplify.edge_weights_raw(mesh, attribute)
    return {"name": name, "uniform": values is None,
            "values": values, "candidates": candidates}


def mesh_edges(obj):
    """(E, 2) vertex-index array the GPU backbone ranks over."""
    count = len(obj.data.edges)
    flat = np.empty(count * 2, dtype=np.int32)
    obj.data.edges.foreach_get("vertices", flat)
    return flat.reshape(-1, 2).astype(np.int64)


def gpu_backbone(obj, attribute, k, sense='high'):
    """TOPK mask the GPU engine would produce for `obj`."""
    weights = gpu_weights(obj, attribute)
    edges = mesh_edges(obj)
    mask = topk_mask(edges, weights["values"], k, sense=sense)
    return {
        "mask": mask,
        "edges": edges,
        "kept": int(mask.sum()),
        "total": int(edges.shape[0]),
        "weight_name": weights["name"],
        "uniform": weights["uniform"],
        "candidates": weights["candidates"],
    }


# --------------------------------------------------------------------------
# Thinning the data
# --------------------------------------------------------------------------

def top_k(nodes_gdf, edges_gdf, weight, k=3, sense='high', verbose=True):
    """Top-k-per-node subset of `edges_gdf`, plus a report.

    `weight` is an edges column (not the mesh `edge_` prefix); None → uniform.
    `sense='high'` keeps largest weights. Returns `(kept_edges_gdf, report)`.
    """
    rows, pairs = mesh_edge_rows(nodes_gdf, edges_gdf)
    if rows.size == 0:
        return edges_gdf.iloc[0:0], {"kept": 0, "total": 0, "rows_total": len(edges_gdf),
                                     "weight_frac": 1.0, "weight": weight,
                                     "sense": sense}

    if weight is None:
        values = None
    else:
        if weight not in edges_gdf.columns:
            raise KeyError(
                f"'{weight}' is not a column of the edges frame; "
                f"have {[c for c in edges_gdf.columns if c != 'geometry']}")
        values = edges_gdf[weight].to_numpy(dtype=float)[rows]

    mask = topk_mask(pairs, values, k, sense=sense)
    kept_rows = rows[mask]
    kept = edges_gdf.iloc[kept_rows]

    if values is None:
        weight_frac = float(mask.sum()) / float(mask.size)
    else:
        clamped = np.maximum(np.nan_to_num(values, nan=0.0, posinf=0.0,
                                           neginf=0.0), 0.0)
        total = float(clamped.sum())
        weight_frac = float(clamped[mask].sum()) / total if total > 0 else 1.0

    report = {
        "kept": int(mask.sum()),
        "total": int(mask.size),
        "rows_total": int(len(edges_gdf)),
        "dropped_by_mesh": int(len(edges_gdf) - mask.size),
        "weight_frac": weight_frac,
        "weight": weight,
        "sense": sense,
        "k": int(k),
    }
    if verbose:
        print(f"  top-{k} per node on '{weight or 'uniform'}' "
              f"({'largest' if sense == 'high' else 'smallest'} first): "
              f"{report['kept']:,} of {report['total']:,} edges "
              f"({report['kept'] / max(report['total'], 1) * 100:.1f}%), "
              f"{weight_frac * 100:.1f}% of the total weight")
    return kept, report


def graph(nodes_gdf, edges_gdf, name, weight, k=3, sense='high',
          ref=None, coll=None, markers=None, verbose=True):
    """Build a Blender graph object with only the top-k backbone edges.

    Same node set (not pruned), subset of edges. Returns `(obj, report)`.
    """
    kept, report = top_k(nodes_gdf, edges_gdf, weight, k=k, sense=sense,
                         verbose=verbose)
    marks = {
        "backbone_mode": "TOPK",
        "backbone_k": int(k),
        "backbone_weight": str(weight or "uniform"),
        "backbone_sense": str(sense),
        "backbone_kept": int(report["kept"]),
        "backbone_total": int(report["total"]),
    }
    marks.update(markers or {})
    obj = graphs.from_gdf(nodes_gdf, kept, name, ref=ref, coll=coll,
                          markers=marks)
    if obj is not None and verbose:
        print(f"  {obj.name}: {obj.get('num_nodes'):,} nodes, "
              f"{obj.get('num_edges'):,} edges")
    return obj, report


# --------------------------------------------------------------------------
# Proving the two agree
# --------------------------------------------------------------------------

def verify(thin_obj, full_obj, nodes_gdf, edges_gdf, weight, k=3, sense='high',
           verbose=True):
    """Compare materialized backbone to the GPU filter, edge for edge.

    Uses `edge_<weight>` on the GPU side; reports `unprefixed_name_resolves`.
    Pairs compared unordered. Returns a report; `identical` is the headline.
    """
    report = {"identical": False}
    if thin_obj is None or full_obj is None:
        report["reason"] = "one of the objects is missing"
        return report

    prefixed = f"edge_{weight}" if weight else ""
    gpu = gpu_backbone(full_obj, prefixed, k, sense=sense)
    naive = gpu_weights(full_obj, weight or "")

    report["gpu_kept"] = gpu["kept"]
    report["gpu_total"] = gpu["total"]
    report["gpu_weight_attribute"] = gpu["weight_name"]
    report["gpu_uniform"] = gpu["uniform"]
    report["edge_candidates"] = gpu["candidates"]
    report["unprefixed_name_resolves"] = not naive["uniform"]

    kept_pairs = {tuple(sorted(p)) for p in gpu["edges"][gpu["mask"]]}

    thin_pairs = {tuple(sorted(p)) for p in mesh_edges(thin_obj)}
    report["materialized_kept"] = len(thin_pairs)
    report["only_in_gpu"] = len(kept_pairs - thin_pairs)
    report["only_in_materialized"] = len(thin_pairs - kept_pairs)
    report["identical"] = (kept_pairs == thin_pairs)

    # Spot-check ranking independently of set agreement.
    report["spot_check"] = _spot_check(nodes_gdf, edges_gdf, thin_pairs, weight,
                                       k, sense)

    if verbose:
        print(f"  GPU filter would keep     {report['gpu_kept']:,} of "
              f"{report['gpu_total']:,} mesh edges")
        print(f"  materialized backbone has {report['materialized_kept']:,}")
        print(f"  identical edge sets:      {report['identical']}")
        if report["gpu_uniform"]:
            print("  ! the GPU filter had NO weight and ranked uniformly; "
                  f"EDGE attributes present: {report['edge_candidates']}")
        if report["spot_check"]:
            spot = report["spot_check"]
            print(f"  spot check: {spot['nodes_checked']} nodes, "
                  f"{spot['violations']} missing an edge strictly in their "
                  f"top {k} ({spot['tied_at_cut']} tied at the cut, where the "
                  f"choice is the sort's and not the data's)")
    return report


def _spot_check(nodes_gdf, edges_gdf, kept_pairs, weight, k, sense, limit=40):
    """Naive per-node top-k check against `kept_pairs` (independent of `topk_mask`).

    Only strict-better-than-cut edges are required kept; ties at the cut are
    counted separately. An edge may be kept via the other endpoint.
    """
    if weight is None or weight not in edges_gdf.columns:
        return None

    rows, pairs = mesh_edge_rows(nodes_gdf, edges_gdf)
    if rows.size == 0:
        return None
    values = edges_gdf[weight].to_numpy(dtype=float)[rows]
    values = np.maximum(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0),
                        0.0)

    incident = {}
    for i, (a, b) in enumerate(pairs):
        incident.setdefault(int(a), []).append(i)
        incident.setdefault(int(b), []).append(i)

    nodes = sorted(incident)
    if len(nodes) > limit:
        step = max(1, len(nodes) // limit)
        nodes = nodes[::step][:limit]

    high = str(sense).lower() == 'high'
    kk = max(1, int(k))

    violations = 0
    tied = 0
    for node in nodes:
        ids = incident[node]
        ordered = sorted(ids, key=lambda i: values[i], reverse=high)
        cut = values[ordered[min(kk, len(ordered)) - 1]]
        # Strictly better than the k-th: ties cannot excuse dropping these.
        strict = [i for i in ordered[:kk]
                  if (values[i] > cut if high else values[i] < cut)]
        # More edges at cut than slots → tie (k-th always equals cut).
        at_cut = sum(1 for i in ids if values[i] == cut)
        if len(strict) + at_cut > min(kk, len(ordered)):
            tied += 1
        if any(tuple(sorted(pairs[i])) not in kept_pairs for i in strict):
            violations += 1
    return {"nodes_checked": len(nodes), "violations": violations,
            "tied_at_cut": tied, "k": int(k), "sense": sense}


__all__ = [
    "gpu_backbone",
    "gpu_weights",
    "mesh_edge_rows",
    "mesh_edges",
    "edges",
    "graph",
    "topk_mask",
    "verify",
    "verify_against_engine",
]
