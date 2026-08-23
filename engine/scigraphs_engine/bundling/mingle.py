# Multilevel agglomerative bundling by ink minimization (Gansner et al. 2011).
# An edge is a chain a -> p -> q -> b, every edge in a bundle sharing p and q:
#     ink(u) = sum_{s in S} |s - p| + |p - q| + sum_{t in T} |t - q|
# A merge scores ink(u) + ink(v) - ink(u | v) and is refused unless positive.
# Weights stay out of the ink: the trunk is one drawn line whatever it carries,
# so weight 100 on both sides would send every gain negative. They enter the
# solve instead, as each point's pull on the meeting point.

import numpy as np

# The GPU kernel holds its k best in a fixed-size local array. Sixteen clears
# the default of ten without spilling registers; ``settings`` clips to it.
MAX_K = 16

# Points per side of a bundle, so also the most that can merge in one round.
# The ink kernel loops over them; unbounded, one invocation walks a thousand
# while its neighbors idle. It compounds: 32 a round is a thousand by round 2.
MAX_GROUP = 32

# Against a 64-iteration solve the meeting points move at most 0.13% of the
# bounding-box diagonal on the last step (0.03% mean), ink 0.12% short. Four
# iterations: 2.9% short, a visible 1.2% move. Twenty: 0.1% for 2/3 more time.
SOLVE_ITERS = 12

# Floor on a Weiszfeld distance, a fraction of the graph diagonal: the optimum
# sits on a source when a bundle's members share a node. Under a pixel here.
SOLVE_EPS_FRAC = 1e-4

# Each pass is a matching, so size at most doubles; eight passes hit MAX_GROUP.
PASSES_PER_ROUND = 8

DEFAULTS = {
    "mingle_neighbors": 10,
    "mingle_rounds": 4,
    "mingle_min_gain": 0.0,
}


def settings(params):
    """Return the mode's parameters; no ``mingle_*`` key is registered yet."""
    s = {k[7:]: params.get(k, v) for k, v in DEFAULTS.items()}
    s["neighbors"] = int(np.clip(int(s["neighbors"]), 1, MAX_K))
    s["min_gain"] = float(max(0.0, float(s["min_gain"])))

    # Capped by the curve's sample count: a chain of 2R+2 control points throws
    # resolution away past R = (segments - 1) / 2.
    cap = max(1, (max(1, int(params.get("segments", 24))) - 1) // 2)
    s["rounds"] = int(np.clip(int(s["rounds"]), 0, cap))
    return s


# Point groups are padded to MAX_GROUP with zero weight rather than stored
# ragged, so the numpy path vectorizes and the kernel indexes arithmetically.


def _seed_level(coords, edges):
    """Round 0: every edge is its own bundle, meeting at its own endpoints."""
    e = int(edges.shape[0])
    a = np.asarray(coords, dtype=np.float32)[edges[:, 0]]
    b = np.asarray(coords, dtype=np.float32)[edges[:, 1]]
    src = np.zeros((e, MAX_GROUP, 3), dtype=np.float32)
    dst = np.zeros((e, MAX_GROUP, 3), dtype=np.float32)
    sw = np.zeros((e, MAX_GROUP), dtype=np.float32)
    tw = np.zeros((e, MAX_GROUP), dtype=np.float32)
    src[:, 0] = a
    dst[:, 0] = b
    sw[:, 0] = 1.0
    tw[:, 0] = 1.0
    return {
        "src": src, "dst": dst, "sw": sw, "tw": tw,
        "p": a.copy(), "q": b.copy(),
        # A singleton's ink is |a - b|: all trunk.
        "ink": np.linalg.norm(b - a, axis=1).astype(np.float32),
    }


def _level_size(level):
    """(B,) live points per side, so how many edges merged into each bundle."""
    return (level["sw"] > 0.0).sum(axis=1).astype(np.int32)


def _proximity_points(level):
    """(B, 6) points in edge proximity space, lexicographically oriented: else
    two coincident edges stored opposite ways land at opposite corners."""
    p, q = level["p"], level["q"]
    diff = p - q
    nz = np.abs(diff) > 0.0
    first = np.argmax(nz, axis=1)
    any_nz = nz.any(axis=1)
    lead = np.where(any_nz, diff[np.arange(p.shape[0]), first], 0.0)
    swap = lead > 0.0
    out = np.empty((p.shape[0], 6), dtype=np.float32)
    out[:, :3] = np.where(swap[:, None], q, p)
    out[:, 3:] = np.where(swap[:, None], p, q)
    return out


def knn_numpy(x6, k):
    """(B, k) nearest bundles in proximity space, by (distance, index). Ties
    break by index as they do in the shader. Tiled: no full (B, B) block."""
    b = int(x6.shape[0])
    kk = int(min(k, max(b - 1, 0)))
    out = np.full((b, k), -1, dtype=np.int64)
    if kk == 0:
        return out
    tile = max(1, int(8_000_000 // max(b, 1)))
    for s in range(0, b, tile):
        e = min(s + tile, b)
        d = ((x6[s:e, None, :] - x6[None, :, :]) ** 2).sum(-1).astype(np.float32)
        d[np.arange(e - s), np.arange(s, e)] = np.float32(np.inf)
        # Stable, so equal distances come out in the kernel's index order.
        out[s:e, :kk] = np.argsort(d, axis=1, kind="stable")[:, :kk]
    return out


def candidates(near):
    """(C, 2) deduplicated undirected pairs from a neighbor list. Ink is
    symmetric, so u holding v and v holding u is one merge scored twice."""
    b, k = near.shape
    u = np.repeat(np.arange(b, dtype=np.int64), k)
    v = near.ravel()
    live = v >= 0
    u, v = u[live], v[live]
    lo = np.minimum(u, v)
    hi = np.maximum(u, v)
    key = lo * np.int64(b) + hi
    _, first = np.unique(key, return_index=True)
    return np.stack([lo[first], hi[first]], axis=1)


def _solve_numpy(ap, aw, bp, bw, eps, iters):
    """Weiszfeld on the two meeting points, from the weighted centroids. ``ap``
    and ``bp`` are (C, 2M, 3) padded groups, ``aw``/``bw`` their weights, zero
    where padding. ``ink`` is unweighted, so weights only move p and q."""
    tiny = np.float32(1e-20)
    p = ((ap * aw[..., None]).sum(1)
         / np.maximum(aw.sum(1), tiny)[:, None]).astype(np.float32)
    q = ((bp * bw[..., None]).sum(1)
         / np.maximum(bw.sum(1), tiny)[:, None]).astype(np.float32)

    move = np.zeros(ap.shape[0], dtype=np.float32)
    for it in range(iters):
        da = np.maximum(np.sqrt(((ap - p[:, None]) ** 2).sum(-1)), eps)
        db = np.maximum(np.sqrt(((bp - q[:, None]) ** 2).sum(-1)), eps)
        wa = (aw / da).astype(np.float32)
        wb = (bw / db).astype(np.float32)
        dpq = np.maximum(np.sqrt(((p - q) ** 2).sum(-1)), eps).astype(np.float32)
        inv = (np.float32(1.0) / dpq)[:, None]
        # Jacobi, so the kernel and this cannot disagree about what was read.
        pn = (((ap * wa[..., None]).sum(1) + q * inv)
              / (wa.sum(1)[:, None] + inv)).astype(np.float32)
        qn = (((bp * wb[..., None]).sum(1) + p * inv)
              / (wb.sum(1)[:, None] + inv)).astype(np.float32)
        if it == iters - 1:
            move = np.maximum(np.linalg.norm(pn - p, axis=1),
                              np.linalg.norm(qn - q, axis=1)).astype(np.float32)
        p, q = pn, qn

    live_a = aw > 0.0
    live_b = bw > 0.0
    da = np.sqrt(((ap - p[:, None]) ** 2).sum(-1))
    db = np.sqrt(((bp - q[:, None]) ** 2).sum(-1))
    ink = ((da * live_a).sum(1)
           + np.sqrt(((p - q) ** 2).sum(-1))
           + (db * live_b).sum(1)).astype(np.float32)
    return p, q, ink, move


def score_numpy(level, cand, eps, iters=SOLVE_ITERS):
    """(gain, flip, p, q, residual) per candidate merge. Both pairings are
    tried: pairing antiparallel edges head to tail opens an X."""
    src, dst, sw, tw = level["src"], level["dst"], level["sw"], level["tw"]
    ink = level["ink"]
    c = int(cand.shape[0])
    gain = np.full(c, -np.inf, dtype=np.float32)
    flip = np.zeros(c, dtype=np.int32)
    out_p = np.zeros((c, 3), dtype=np.float32)
    out_q = np.zeros((c, 3), dtype=np.float32)
    residual = np.zeros(c, dtype=np.float32)
    if c == 0:
        return gain, flip, out_p, out_q, residual

    # The working set is (tile, 2*MAX_GROUP, 3) four times over, so hundreds of
    # megabytes on a large graph if C is left unbounded.
    tile = max(1, int(2_000_000 // (2 * MAX_GROUP * 3)))
    for s0 in range(0, c, tile):
        s1 = min(s0 + tile, c)
        u = cand[s0:s1, 0]
        v = cand[s0:s1, 1]
        solo = (ink[u] + ink[v]).astype(np.float32)
        for f in (0, 1):
            v_head_p, v_head_w = (dst, tw) if f else (src, sw)
            v_tail_p, v_tail_w = (src, sw) if f else (dst, tw)
            ap = np.concatenate([src[u], v_head_p[v]], axis=1)
            aw = np.concatenate([sw[u], v_head_w[v]], axis=1)
            bp = np.concatenate([dst[u], v_tail_p[v]], axis=1)
            bw = np.concatenate([tw[u], v_tail_w[v]], axis=1)
            p, q, joint, move = _solve_numpy(ap, aw, bp, bw, eps, iters)
            g = (solo - joint).astype(np.float32)
            take = g > gain[s0:s1]
            idx = np.nonzero(take)[0]
            if idx.size:
                gain[s0 + idx] = g[idx]
                flip[s0 + idx] = f
                out_p[s0 + idx] = p[idx]
                out_q[s0 + idx] = q[idx]
                residual[s0 + idx] = move[idx]
    return gain, flip, out_p, out_q, residual


def matching(cand, gain, size, min_gain, solo):
    """The merges one pass accepts: best gain first, each bundle once. A
    matching, not a chain, since once two accepted merges share a bundle the
    second's score was measured against something gone. ``min_gain`` is a
    fraction of the pair's unbundled ink, so it holds at any size."""
    if cand.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    frac = gain / np.maximum(solo, np.float32(1e-20))
    live = (gain > 0.0) & (frac > np.float32(min_gain))
    keep = np.nonzero(live)[0]
    if keep.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    # Descending gain, ties by the candidate's position, itself sorted by
    # (u, v), so two runs of the same input accept the same merges.
    order = keep[np.argsort(-gain[keep], kind="stable")]

    # Python lists first: the loop cannot be vectorized, and a numpy scalar
    # extraction per iteration cost 5x a list lookup at 558k candidates.
    us = cand[order, 0].tolist()
    vs = cand[order, 1].tolist()
    sz = size.astype(np.int64).tolist()
    used = bytearray(size.shape[0])
    pair_rows = []
    cap = MAX_GROUP
    for n, r in enumerate(order.tolist()):
        u = us[n]
        v = vs[n]
        if used[u] or used[v] or sz[u] + sz[v] > cap:
            continue
        used[u] = used[v] = 1
        pair_rows.append(r)
    rows = np.asarray(pair_rows, dtype=np.int64)
    return cand[rows], rows


def merge(level, pairs, rows, flip, out_p, out_q):
    """Apply a pass's matching. Returns the new level, the id remap, and in
    ``rev`` whether each bundle's head and tail were swapped."""
    b = int(level["p"].shape[0])
    if pairs.shape[0] == 0:
        return level, np.arange(b, dtype=np.int64), np.zeros(b, dtype=bool)

    u = pairs[:, 0]
    v = pairs[:, 1]
    f = flip[rows].astype(bool)

    src, dst = level["src"], level["dst"]
    sw, tw = level["sw"], level["tw"]

    # v's head joins u's head unless flipped, when its tail joins instead.
    v_head_p = np.where(f[:, None, None], dst[v], src[v])
    v_head_w = np.where(f[:, None], tw[v], sw[v])
    v_tail_p = np.where(f[:, None, None], src[v], dst[v])
    v_tail_w = np.where(f[:, None], sw[v], tw[v])

    def _compact(pa, wa, pb, wb):
        """Concatenate two padded groups, live entries pushed to the front."""
        cat_p = np.concatenate([pa, pb], axis=1)
        cat_w = np.concatenate([wa, wb], axis=1)
        # Stable, so the merged group is deterministic in its two inputs.
        take = np.argsort(cat_w <= 0.0, axis=1, kind="stable")[:, :MAX_GROUP]
        return (np.take_along_axis(cat_p, take[:, :, None].repeat(3, axis=2),
                                   axis=1).astype(np.float32),
                np.take_along_axis(cat_w, take, axis=1).astype(np.float32))

    new_src, new_sw = _compact(src[u], sw[u], v_head_p, v_head_w)
    new_dst, new_tw = _compact(dst[u], tw[u], v_tail_p, v_tail_w)

    alive = np.ones(b, dtype=bool)
    alive[v] = False
    new_id = np.full(b, -1, dtype=np.int64)
    new_id[alive] = np.arange(int(alive.sum()), dtype=np.int64)
    new_id[v] = new_id[u]

    out = {
        "src": src[alive].copy(), "dst": dst[alive].copy(),
        "sw": sw[alive].copy(), "tw": tw[alive].copy(),
        "p": level["p"][alive].copy(), "q": level["q"][alive].copy(),
        "ink": level["ink"][alive].copy(),
    }
    slot = new_id[u]
    out["src"][slot] = new_src
    out["dst"][slot] = new_dst
    out["sw"][slot] = new_sw
    out["tw"][slot] = new_tw
    p_new = out_p[rows]
    q_new = out_q[rows]
    out["p"][slot] = p_new
    out["q"][slot] = q_new

    # Measured on the merged groups, not subtracted from the gain: every later
    # gain is compared against this, so an error here compounds.
    out["ink"][slot] = _group_ink(new_src, new_sw, new_dst, new_tw,
                                  p_new, q_new)

    rev = np.zeros(b, dtype=bool)
    rev[v] = f
    return out, new_id, rev


def _group_ink(src, sw, dst, tw, p, q):
    """(m,) unweighted drawn length of padded groups meeting at (p, q)."""
    da = np.sqrt(((src - p[:, None]) ** 2).sum(-1)) * (sw > 0.0)
    db = np.sqrt(((dst - q[:, None]) ** 2).sum(-1)) * (tw > 0.0)
    return (da.sum(axis=1) + np.linalg.norm(p - q, axis=1)
            + db.sum(axis=1)).astype(np.float32)


def bundle(coords, edges, params, report=None, knn=knn_numpy,
           score=score_numpy, backend='numpy'):
    """``knn`` and ``score`` are the two steps a host may replace with compute
    kernels; either returning None aborts the run and the caller falls back.
    ``backend`` only names the pair in ``report``. Everything else is CPU."""
    s = settings(params)
    e_count = int(edges.shape[0])
    diag = float(np.linalg.norm(np.ptp(np.asarray(coords, np.float64), axis=0)))
    eps = np.float32(max(SOLVE_EPS_FRAC * diag, 1e-12))

    level = _seed_level(coords, edges)
    # Which current edge each original edge rides, and which way round. ``ori``
    # is +1 while the head is still the original a side.
    cur = np.arange(e_count, dtype=np.int64)
    ori = np.ones(e_count, dtype=np.int64)
    rounds = int(s["rounds"])
    head = np.zeros((e_count, max(rounds, 1), 3), dtype=np.float32)
    tail = np.zeros((e_count, max(rounds, 1), 3), dtype=np.float32)
    depth = np.zeros(e_count, dtype=np.int64)

    log = []
    worst_move = 0.0
    for r in range(rounds):
        b0 = int(level["p"].shape[0])
        if b0 < 2:
            break
        bid = np.arange(b0, dtype=np.int64)
        rev = np.zeros(b0, dtype=bool)
        merged_here = 0
        passes = 0
        for _ in range(PASSES_PER_ROUND):
            b = int(level["p"].shape[0])
            if b < 2:
                break
            x6 = _proximity_points(level)
            near = knn(x6, s["neighbors"])
            if near is None:
                return None
            cand = candidates(near)
            if cand.shape[0] == 0:
                break
            scored = score(level, cand, eps)
            if scored is None:
                return None
            gain, flip, out_p, out_q, move = scored
            if move.size:
                worst_move = max(worst_move, float(move.max()))
            solo = (level["ink"][cand[:, 0]] + level["ink"][cand[:, 1]])
            pairs, rows = matching(cand, gain, _level_size(level),
                                   s["min_gain"], solo)
            passes += 1
            if pairs.shape[0] == 0:
                break
            level, remap, pass_rev = merge(level, pairs, rows, flip, out_p, out_q)
            rev ^= pass_rev[bid]
            bid = remap[bid]
            merged_here += int(pairs.shape[0])

        log.append({"round": r, "bundles_in": b0,
                    "bundles_out": int(level["p"].shape[0]),
                    "merges": merged_here, "passes": passes})
        if merged_here == 0:
            # The next round would be handed exactly this edge set again.
            break

        size = _level_size(level)
        flipped = rev[cur]
        ori = np.where(flipped, -ori, ori)
        target = bid[cur]
        grew = size[target] > 1
        if grew.any():
            rows_e = np.nonzero(grew)[0]
            forward = ori[rows_e] > 0
            pb = level["p"][target[rows_e]]
            qb = level["q"][target[rows_e]]
            head[rows_e, depth[rows_e]] = np.where(forward[:, None], pb, qb)
            tail[rows_e, depth[rows_e]] = np.where(forward[:, None], qb, pb)
            depth[rows_e] += 1
        cur = target

        level = _trunk_level(level)

    ctrl, counts = _assemble(coords, edges, head, tail, depth)
    if report is not None:
        report.update({"rounds": log, "backend": backend,
                       "solve_residual": worst_move,
                       "diag": diag, "eps": float(eps),
                       "kmax": int(ctrl.shape[1]),
                       "bundled_edges": int((depth > 0).sum())})
    return ctrl, counts


def _trunk_level(level):
    """The next round's edges: one trunk per bundle, weighted by its load. That
    stops the point count growing and holds a heavy bundle in place."""
    b = int(level["p"].shape[0])
    weight = level["sw"].sum(axis=1).astype(np.float32)
    src = np.zeros((b, MAX_GROUP, 3), dtype=np.float32)
    dst = np.zeros((b, MAX_GROUP, 3), dtype=np.float32)
    sw = np.zeros((b, MAX_GROUP), dtype=np.float32)
    tw = np.zeros((b, MAX_GROUP), dtype=np.float32)
    src[:, 0] = level["p"]
    dst[:, 0] = level["q"]
    sw[:, 0] = weight
    tw[:, 0] = weight
    return {
        "src": src, "dst": dst, "sw": sw, "tw": tw,
        "p": level["p"].copy(), "q": level["q"].copy(),
        "ink": np.linalg.norm(level["q"] - level["p"],
                              axis=1).astype(np.float32),
    }


def _assemble(coords, edges, head, tail, depth):
    """(E, K, 3) control polygons and (E,) their real lengths. Each is a, the
    head chain outward, the tail chain back inward, b, at most 2R+2 points.
    Padded by repeating the last point: a texture of zeros reads as a bug."""
    e_count = int(edges.shape[0])
    coords = np.asarray(coords, dtype=np.float32)
    a = coords[edges[:, 0]]
    b = coords[edges[:, 1]]
    dmax = int(depth.max()) if e_count else 0
    kmax = 2 * dmax + 2
    ctrl = np.zeros((e_count, kmax, 3), dtype=np.float32)
    counts = (2 * depth + 2).astype(np.int32)

    ctrl[:, 0] = a
    idx = np.arange(dmax)[None, :]
    live = idx < depth[:, None]
    if dmax:
        # Head i sits at control point 1 + i, tail i at count - 2 - i.
        rows = np.repeat(np.arange(e_count), dmax)
        col = np.tile(np.arange(dmax), e_count)
        keep = live.ravel()
        # Sliced to dmax, not the round count. An early stop leaves the tail
        # columns untouched, and reshaping those in misaligns every row.
        ctrl[rows[keep], (1 + col)[keep]] = head[:, :dmax].reshape(-1, 3)[keep]
        back = (counts[:, None] - 2 - idx).ravel()
        ctrl[rows[keep], back[keep]] = tail[:, :dmax].reshape(-1, 3)[keep]
    ctrl[np.arange(e_count), counts - 1] = b
    pad = np.arange(kmax)[None, :] >= counts[:, None]
    ctrl[pad] = np.broadcast_to(b[:, None, :], ctrl.shape)[pad]
    return np.ascontiguousarray(ctrl, dtype=np.float32), counts


def drawn_ink(ctrl, counts, tol=1e-4):
    """(ink, distinct segments) actually drawn by a set of control polygons.
    Deduplicated: every edge in a bundle carries the shared trunk in its own
    polygon, so summing per-edge lengths counts the trunk once per member and
    says bundling made the drawing longer. True of arc length, false of ink."""
    e, k, _ = ctrl.shape
    idx = np.arange(k - 1)[None, :]
    live = idx < (counts[:, None] - 1)
    a = ctrl[:, :-1][live]
    b = ctrl[:, 1:][live]
    if a.shape[0] == 0:
        return 0.0, 0
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    key = np.round(np.concatenate([lo, hi], axis=1) / tol).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return float(np.linalg.norm(b[first] - a[first], axis=1).sum()), int(first.size)
