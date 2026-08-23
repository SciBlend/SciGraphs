# Level of detail over simplify.build_hierarchy's community tree: one ancestor
# per node. Crowding opens a community, the caller's ratios merge it back.

import collections
import math

import numpy as np

Camera = collections.namedtuple(
    "Camera", "persp focal_y half_w half_h forward eye")

CROWDING_REFRESH_COS = 0.9994  # ~2 deg; crowding varies only with direction

# Expand over this, merge under COLLAPSE_MAX_MEASURED. The gap damps thrash.
EXPAND_MIN_PREDICTED = 0.55

EXPAND_MIN_PX = 2.5

COLLAPSE_MAX_MEASURED = 0.25

COLLAPSE_HIDDEN_FRACTION = 0.5

DIRECTION_BINS = 192  # ~15 degree bins over the sphere

BLAME_SHARE_DEG = 0.0  # also file a merge under directions within this angle

EXPAND_HYSTERESIS = 0.0

MAX_REGROUP_FRACTION = 1.0

MAX_MERGE_LOSS = 0.3  # 0 draws every node, unbounded merges them all

SIGMA_K = 2.0


class CutState:
    """Per-object persistent state for the adaptive cut."""

    def __init__(self, levels):
        self.sizes = [lvl["centers"].shape[0] for lvl in levels]
        self.hidden = [{} for _ in levels]
        self.futile = [{} for _ in levels]
        self.tried = [{} for _ in levels]
        self.expanded = [np.zeros(n, dtype=bool) for n in self.sizes]
        self.mean_child_r = [None] * len(levels)
        self.crowd = [None] * len(levels)
        self.crowd_dir = np.zeros(3)
        self.changes = 0
        self.frames = 0

    def _from(self, caches, level_index, bin_index, create=False):
        mask = caches[level_index].get(bin_index)
        if mask is None:
            mask = np.zeros(self.sizes[level_index], dtype=bool)
            if create:
                caches[level_index][bin_index] = mask
        return mask

    def hidden_from(self, level_index, bin_index, create=False):
        """Which entries of this level scored poorly from this direction."""
        return self._from(self.hidden, level_index, bin_index, create)

    def futile_from(self, level_index, bin_index, create=False):
        """Where merging was tried from this direction and did not pay."""
        return self._from(self.futile, level_index, bin_index, create)

    def tried_from(self, level_index, bin_index, create=False):
        """Whose merge has already been tried from this direction."""
        return self._from(self.tried, level_index, bin_index, create)

    def reset(self):
        for caches in (self.hidden, self.futile, self.tried):
            for cache in caches:
                cache.clear()
        for arr in self.expanded:
            arr[:] = False
        self.crowd = [None] * len(self.crowd)
        self.changes = 0


def camera_terms(persp_mat, window_mat, height):
    """Camera terms for the predictor. The focal term must come from
    ``window_mat``; taken off ``persp_mat`` it picks up orientation."""
    window = np.asarray(window_mat, dtype=np.float64)
    persp = np.asarray(persp_mat, dtype=np.float64)
    focal_y = float(window[1][1])
    aspect = focal_y / float(window[0][0]) if window[0][0] else 1.0
    # w = -z_view, so the bottom row is the viewing direction in source space.
    forward = persp[3, :3]
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    # Dividing the projection out of persp leaves the modelview.
    modelview = np.linalg.solve(window, persp)
    eye = -modelview[:3, :3].T @ modelview[:3, 3]
    return Camera(persp, focal_y, 0.5 * height * aspect, 0.5 * height,
                  forward, eye)


def depth_of(points, camera):
    """Perspective divide (distance along the view axis) for each point."""
    return np.abs(points @ camera.persp[3, :3] + camera.persp[3, 3]) + 1e-9


def project_px(points, camera):
    """Project points to pixel coordinates, plus pixels per unit at each depth."""
    homog = np.empty((points.shape[0], 4), dtype=np.float64)
    homog[:, :3] = points
    homog[:, 3] = 1.0
    clip = homog @ camera.persp.T
    w = np.abs(clip[:, 3]) + 1e-9
    px = np.empty((points.shape[0], 2), dtype=np.float64)
    px[:, 0] = clip[:, 0] / w * camera.half_w
    px[:, 1] = clip[:, 1] / w * camera.half_h
    return px, camera.focal_y / w * camera.half_h


def children_of(levels, level_index, leaf):
    """Centers, radii and parent map one step below ``level_index``. Below
    level 0 that is the original nodes, taken from ``leaf``."""
    if level_index > 0:
        child = levels[level_index - 1]
        return (child["centers"].astype(np.float64),
                child["radii"].astype(np.float64),
                child["parent"])
    radii = np.asarray(leaf["radii"], dtype=np.float64)
    coords = np.asarray(leaf["coords"], dtype=np.float64)
    if radii.ndim == 0:
        radii = np.full(coords.shape[0], float(radii))
    return coords, radii, levels[0]["member_of"]


def crowding(levels, leaf, level_index, camera):
    """Screen room the children have, in [0, 1]: footprint over the area their
    centers cover. The ellipse is dilated or it collapses on two children."""
    n_parent = levels[level_index]["centers"].shape[0]
    centers, radii, parent = children_of(levels, level_index, leaf)

    px, scale = project_px(centers, camera)
    child_px = radii * scale

    count = np.bincount(parent, minlength=n_parent).astype(np.float64)
    safe = np.maximum(count, 1.0)
    mean_x = np.bincount(parent, weights=px[:, 0], minlength=n_parent) / safe
    mean_y = np.bincount(parent, weights=px[:, 1], minlength=n_parent) / safe
    dx = px[:, 0] - mean_x[parent]
    dy = px[:, 1] - mean_y[parent]
    s_xx = np.bincount(parent, weights=dx * dx, minlength=n_parent) / safe
    s_yy = np.bincount(parent, weights=dy * dy, minlength=n_parent) / safe
    s_xy = np.bincount(parent, weights=dx * dy, minlength=n_parent) / safe

    trace = s_xx + s_yy
    gap = np.sqrt(np.maximum(trace * trace - 4.0 * (s_xx * s_yy - s_xy * s_xy),
                             0.0))
    mean_r = np.bincount(parent, weights=child_px, minlength=n_parent) / safe
    semi_major = SIGMA_K * np.sqrt(np.maximum(0.5 * (trace + gap), 0.0)) + mean_r
    semi_minor = SIGMA_K * np.sqrt(np.maximum(0.5 * (trace - gap), 0.0)) + mean_r
    available = np.pi * semi_major * semi_minor

    demand = np.bincount(parent, weights=np.pi * child_px * child_px,
                         minlength=n_parent)
    ratio = demand / np.maximum(available, 1e-12)
    return np.clip(1.0 / np.maximum(ratio, 1.0), 0.0, 1.0), count


def legibility(levels, leaf, level_index, camera, state=None):
    """Mean on-screen radius the children of each community would have."""
    level = levels[level_index]
    mean_r = _mean_child_radius(levels, leaf, level_index, state)
    return mean_r * camera.focal_y / depth_of(
        level["centers"], camera) * camera.half_h


def _mean_child_radius(levels, leaf, level_index, state=None):
    """Mean world radius of each community's children. Camera independent."""
    if state is not None and state.mean_child_r[level_index] is not None:
        return state.mean_child_r[level_index]
    n_parent = levels[level_index]["centers"].shape[0]
    _centers, radii, parent = children_of(levels, level_index, leaf)
    count = np.maximum(np.bincount(parent, minlength=n_parent), 1)
    mean_r = np.bincount(parent, weights=radii, minlength=n_parent) / count
    if state is not None:
        state.mean_child_r[level_index] = mean_r
    return mean_r


def predict_expand(levels, leaf, level_index, camera, state=None,
                   min_predicted=EXPAND_MIN_PREDICTED, min_px=EXPAND_MIN_PX,
                   hysteresis=0.0):
    """``(ok, predicted, child_px)``: whether communities here may be opened.
    ``hysteresis`` keeps a drifting camera from flipping one open and shut."""
    predicted, count = None, None
    if state is not None:
        cached = state.crowd[level_index]
        if cached is not None and \
                float(state.crowd_dir @ camera.forward) >= CROWDING_REFRESH_COS:
            predicted, count = cached
    if predicted is None:
        predicted, count = crowding(levels, leaf, level_index, camera)
        if state is not None:
            state.crowd[level_index] = (predicted, count)

    child_px = legibility(levels, leaf, level_index, camera, state)
    open_now = (state.expanded[level_index] if state is not None
                else np.zeros(count.shape, dtype=bool))
    slack = np.where(open_now, 1.0 - hysteresis, 1.0 + hysteresis)
    ok = ((count > 0) & (predicted >= min_predicted * slack)
          & (child_px >= min_px * slack))
    return ok, predicted, child_px


def _bin_directions(count=DIRECTION_BINS):
    """Fibonacci spiral, not a lat-long grid: at the poles a lat-long grid bins
    two identical directions apart on the sign of a zero."""
    i = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / count
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = math.pi * (1.0 + math.sqrt(5.0)) * i
    return np.column_stack((r * np.cos(theta), r * np.sin(theta), z))


BIN_DIRECTIONS = _bin_directions()


def _bin_neighbors(degrees=BLAME_SHARE_DEG):
    if degrees <= 0.0:
        return [np.array([i]) for i in range(DIRECTION_BINS)]
    close = BIN_DIRECTIONS @ BIN_DIRECTIONS.T >= math.cos(math.radians(degrees))
    return [np.flatnonzero(row) for row in close]


BIN_NEIGHBORS = _bin_neighbors()


def direction_bin(view_dir):
    d = np.asarray(view_dir, dtype=np.float64)
    return int(np.argmax(BIN_DIRECTIONS @ (d / (np.linalg.norm(d) + 1e-12))))


def apply_measurement(levels, state, drawn, measured, view_dir,
                      max_measured=COLLAPSE_MAX_MEASURED,
                      min_hidden_fraction=COLLAPSE_HIDDEN_FRACTION,
                      max_merge_loss=MAX_MERGE_LOSS, share=True):
    """Record a collapse wherever the caller's ratios came back below threshold.
    ``drawn`` is what :func:`select_cut` returns; ``measured`` holds ratios in
    [0, 1]. A parent is penalized when over ``min_hidden_fraction`` of its shown
    children fall below ``max_measured``. Returns the entries changed for
    ``view_dir``; callers loop until it is zero, so count every change."""
    bin_index = direction_bin(view_dir)
    bins = BIN_NEIGHBORS[bin_index] if share else (bin_index,)
    added = 0
    for child_level, (indices, ratios) in enumerate(zip(drawn, measured)):
        if indices is None or len(indices) == 0:
            continue
        indices = np.asarray(indices, dtype=np.int64)
        ratios = np.nan_to_num(np.asarray(ratios, dtype=np.float64), nan=1.0)

        # A supernode that measured poorly is opened once, to see if it helps.
        if child_level < len(levels):
            poor = indices[ratios < max_measured]
            for blame_bin in bins:
                tried = state.tried_from(child_level, int(blame_bin),
                                         create=True)
                fresh = poor[~tried[poor]]
                if not fresh.size:
                    continue
                if blame_bin == bin_index:
                    added += int(fresh.size)
                tried[fresh] = True
                state.futile_from(child_level, int(blame_bin),
                                  create=True)[fresh] = True

        # Ratios for level L-1 penalize level L; the top has no parent.
        if child_level == len(levels):
            parent_level, parent_of = 0, levels[0]["member_of"]
        elif child_level + 1 < len(levels):
            parent_level, parent_of = child_level + 1, levels[child_level]["parent"]
        else:
            continue
        parent = parent_of[indices]

        n_parent = state.sizes[parent_level]
        shown = np.bincount(parent, minlength=n_parent)
        poorly = (ratios < max_measured).astype(np.float64)
        hidden = np.bincount(parent, weights=poorly, minlength=n_parent)
        blame = (shown > 0) & (hidden >= min_hidden_fraction * shown)

        added += int((blame & ~state.hidden_from(
            parent_level, bin_index)).sum())
        for blame_bin in bins:
            state.hidden_from(parent_level, int(blame_bin),
                              create=True)[:] |= blame

        # Cost is counted in members, not drawn elements, and charged to every
        # ancestor, because the cut can drop several levels at once.
        weight = (np.ones(indices.size) if child_level == len(levels)
                  else levels[child_level]["counts"][indices].astype(np.float64))
        ancestor, level_index = parent, parent_level
        while True:
            size = state.sizes[level_index]
            members = np.bincount(ancestor, weights=weight, minlength=size)
            lost = np.bincount(ancestor, weights=weight * poorly,
                               minlength=size)
            seen = members > 0
            cost = np.divide(lost, members, out=np.ones(size), where=seen)
            for blame_bin in bins:
                judged = seen & state.tried_from(level_index, int(blame_bin))
                if not judged.any():
                    continue
                verdict = (1.0 - cost[judged]) > max_merge_loss
                settled = state.futile_from(level_index, int(blame_bin),
                                            create=True)
                if blame_bin == bin_index:
                    added += int((settled[judged] != verdict).sum())
                settled[judged] = verdict
            if level_index + 1 >= len(levels):
                break
            ancestor = levels[level_index]["parent"][ancestor]
            level_index += 1
    return added


def select_cut(levels, leaf, state, camera, max_regroup=MAX_REGROUP_FRACTION,
               hysteresis=EXPAND_HYSTERESIS, **predict_kwargs):
    """Which communities to draw, top level down. ``drawn[i]`` masks level ``i``
    (finest first) plus a final mask over the original nodes; a community is
    drawn when every ancestor was expanded and it was not."""
    top = len(levels) - 1
    want = [None] * len(levels)
    reached = np.ones(levels[top]["centers"].shape[0], dtype=bool)
    stale = float(state.crowd_dir @ camera.forward) < CROWDING_REFRESH_COS

    bin_index = direction_bin(camera.forward)

    for level_index in range(top, -1, -1):
        may_open, _predicted, _px = predict_expand(
            levels, leaf, level_index, camera, state,
            hysteresis=hysteresis, **predict_kwargs)
        # Recorded poor from here stays whole, unless the merge did not pay.
        may_open &= ~(state.hidden_from(level_index, bin_index)
                      & ~state.futile_from(level_index, bin_index))
        may_open &= reached
        want[level_index] = may_open
        if level_index > 0:
            reached = may_open[levels[level_index - 1]["parent"]]
    if stale:
        state.crowd_dir = np.asarray(camera.forward, dtype=np.float64).copy()

    # Budget in original nodes: one collapse near the root rewrites half the
    # screen. Collapses go first, since they fix a misleading fragment.
    reach = reach_masks(levels, state.expanded)
    n_nodes = levels[0]["member_of"].shape[0]
    budget = (None if max_regroup >= 1.0
              else max(1, int(round(max_regroup * n_nodes))))
    changes = 0
    for collapsing in (True, False):
        for level_index in range(top, -1, -1):
            # Reachability follows the level above, recomputed in the descent.
            if level_index < top:
                reach[level_index] = (
                    state.expanded[level_index + 1] & reach[level_index + 1]
                )[levels[level_index]["parent"]]
            if budget is not None and budget <= 0:
                continue
            current = state.expanded[level_index]
            # Flipping an entry buried under a collapsed ancestor is free.
            differing = np.flatnonzero((current != want[level_index])
                                       & reach[level_index]
                                       & (current if collapsing else ~current))
            if differing.size == 0:
                continue
            if budget is not None:
                # Cheapest first; one change is always allowed, per frame.
                cost = levels[level_index]["counts"][differing]
                differing = differing[np.argsort(cost, kind='stable')]
                affordable = int(np.searchsorted(
                    np.cumsum(np.sort(cost)), budget, 'right'))
                differing = differing[:max(affordable,
                                           1 if changes == 0 else 0)]
                if differing.size == 0:
                    continue
                budget -= int(levels[level_index]["counts"][differing].sum())
            current[differing] = want[level_index][differing]
            changes += int(differing.size)

    # Derive the cut from the applied flags: a rate-limited frame stays valid.
    drawn = [r & ~(e & r) for r, e in zip(reach, state.expanded)]
    expanded_leaf = state.expanded[0] & reach[0]
    drawn.append(expanded_leaf[levels[0]["member_of"]])

    state.changes = changes
    state.frames += 1
    return drawn, changes


def regrouped(before, after):
    """Which nodes are drawn in another group; compares partitions, not labels."""
    before = np.asarray(before, dtype=np.int64)
    after = np.asarray(after, dtype=np.int64)
    joint = before * (after.max() + 1) + after
    return ((np.bincount(joint)[joint] != np.bincount(before)[before])
            | (np.bincount(joint)[joint] != np.bincount(after)[after]))


def reach_masks(levels, expanded):
    """Which entries of each level are on screen: every ancestor was expanded."""
    top = len(levels) - 1
    reach = [None] * len(levels)
    mask = np.ones(levels[top]["centers"].shape[0], dtype=bool)
    for level_index in range(top, -1, -1):
        reach[level_index] = mask
        if level_index > 0:
            mask = (expanded[level_index] & mask)[
                levels[level_index - 1]["parent"]]
    return reach


def cut_labels(levels, drawn, n_nodes):
    """One label per node, coarsest first: what ``build_coarse_level`` consumes."""
    labels = np.full(n_nodes, -1, dtype=np.int64)
    offset = 0
    for level_index in range(len(levels) - 1, -1, -1):
        mask = drawn[level_index]
        taken = int(mask.sum())
        if taken == 0:
            continue
        gid = np.full(mask.size, -1, dtype=np.int64)
        gid[mask] = np.arange(taken, dtype=np.int64) + offset
        offset += taken
        mine = gid[levels[level_index]["member_of"]]
        labels = np.where(mine >= 0, mine, labels)

    leaf = drawn[len(levels)]
    taken = int(leaf.sum())
    if taken:
        labels[leaf] = np.arange(taken, dtype=np.int64) + offset
    return labels


def split_by_level(levels, drawn, values):
    """Undo :func:`cut_labels`: split per-element values back onto the levels."""
    values = np.asarray(values)
    indices, out = [None] * (len(levels) + 1), [None] * (len(levels) + 1)
    offset = 0
    order = list(range(len(levels) - 1, -1, -1)) + [len(levels)]
    for slot in order:
        idx = np.flatnonzero(drawn[slot])
        indices[slot] = idx
        out[slot] = values[offset:offset + idx.size]
        offset += idx.size
    return indices, out


def cut_summary(levels, drawn):
    """One line describing what a cut draws."""
    parts = []
    total_nodes = 0
    for level_index in range(len(levels) - 1, -1, -1):
        count = int(drawn[level_index].sum())
        if count:
            members = int(levels[level_index]["counts"][drawn[level_index]].sum())
            total_nodes += members
            parts.append(f"L{level_index + 1}: {count} blobs ({members} nodes)")
    leaf = int(drawn[len(levels)].sum())
    if leaf:
        total_nodes += leaf
        parts.append(f"L0: {leaf} nodes")
    drawn_count = sum(int(d.sum()) for d in drawn)
    return (f"{drawn_count} drawn elements covering {total_nodes} nodes | "
            + ", ".join(parts))
