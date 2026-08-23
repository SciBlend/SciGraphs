# The draw loop culls and budgets per block, so per-frame cost tracks a few
# thousand blocks instead of millions of nodes.

import numpy as np


def _grid_resolution(num_nodes, target_blocks):
    """Pick a per-axis cell count so total cells ~= ``target_blocks``."""
    res = int(round(target_blocks ** (1.0 / 3.0)))
    return max(1, min(res, 256))


def build_spatial_blocks(coords, edges, target_blocks=4096, seg_mid=None):
    """Partition nodes into a uniform grid. Per non-empty block: int32
    ``node_indices``, int32 ``edge_indices``, ``centers`` (B, 3), ``radii`` (B,)
    of the bounding spheres. An edge goes to the block of its midpoint, or of
    ``seg_mid`` (S, 3) when given, where ``edge_indices`` holds segment ids."""
    num_nodes = coords.shape[0]
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    extent = np.maximum(hi - lo, 1e-6)

    res = _grid_resolution(num_nodes, target_blocks)
    cell = extent / res

    ijk = np.clip(((coords - lo) / cell).astype(np.int64), 0, res - 1)
    cell_id = (ijk[:, 0] * res + ijk[:, 1]) * res + ijk[:, 2]

    order = np.argsort(cell_id, kind="stable")
    sorted_ids = cell_id[order]
    unique_ids, starts = np.unique(sorted_ids, return_index=True)
    ends = np.append(starts[1:], sorted_ids.size)

    node_indices = []
    centers = np.empty((unique_ids.size, 3), dtype=np.float32)
    radii = np.empty(unique_ids.size, dtype=np.float32)
    for b, (s, e) in enumerate(zip(starts, ends)):
        idx = order[s:e].astype(np.int32)
        node_indices.append(idx)
        pts = coords[idx]
        cmin = pts.min(axis=0)
        cmax = pts.max(axis=0)
        center = 0.5 * (cmin + cmax)
        centers[b] = center
        radii[b] = float(np.linalg.norm(pts - center, axis=1).max()) if idx.size else 0.0

    edge_indices = [[] for _ in range(unique_ids.size)]
    mid = None
    if seg_mid is not None and seg_mid.shape[0]:
        mid = seg_mid
    elif edges is not None and edges.shape[0]:
        mid = 0.5 * (coords[edges[:, 0]] + coords[edges[:, 1]])
    if mid is not None:
        mijk = np.clip(((mid - lo) / cell).astype(np.int64), 0, res - 1)
        mcell = (mijk[:, 0] * res + mijk[:, 1]) * res + mijk[:, 2]
        pos = np.searchsorted(unique_ids, mcell)
        pos[pos >= unique_ids.size] = 0
        valid = unique_ids[pos] == mcell
        eids = np.nonzero(valid)[0].astype(np.int32)
        blocks_of = pos[valid]
        eorder = np.argsort(blocks_of, kind="stable")
        sorted_blocks = blocks_of[eorder]
        ub, bstarts = np.unique(sorted_blocks, return_index=True)
        bends = np.append(bstarts[1:], sorted_blocks.size)
        edge_indices = [np.empty(0, dtype=np.int32)
                        for _ in range(unique_ids.size)]
        for b, s, e in zip(ub, bstarts, bends):
            edge_indices[b] = eids[eorder[s:e]]

    return {
        "node_indices": node_indices,
        "edge_indices": edge_indices,
        "centers": centers,
        "radii": radii,
        "count": int(unique_ids.size),
    }
