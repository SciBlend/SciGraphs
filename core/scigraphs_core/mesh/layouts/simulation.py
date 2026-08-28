"""Incremental force-directed layout in numpy, steppable without touching bpy.

``ForceSim.step(n)`` advances the state and ``positions`` returns the array to
upload. Repulsion is exact near-field over a bounded neighbor list plus a coarse
monopole grid far-field; the integrator is ForceAtlas2's, except under the
``YIFAN_HU`` model, which runs the multilevel schedule described on
:class:`ForceSim` instead.
"""

import numpy as np

DTYPE = np.float32

DIRECT_MAX = 400  # above this, the near/far split replaces all-pairs

NEAR_K = 16
NEAR_REFRESH = 8

# Strip size for O(n²) work so temporaries stay bounded.
CHUNK = 2048

FAR_RES = 8
FAR_SOFTEN = 0.5  # as a fraction of cell size

# Corrects the monopole gravity underestimate; fitted across n and scale.
_GRAVITY_FIT = 0.38

_NEAR_WARNED = False

YH_STEP = 0.15
YH_TOL = 0.0015
YH_COOL = 0.90

YH_MAXITER = 500
YH_COARSE_ITER = 120

YH_MATCH_ROUNDS = 12
YH_COARSEN = 0.6
YH_STALL = 0.8
YH_MINSIZE = 4
YH_MAX_LEVELS = 24
YH_JITTER = 0.001


_QUADTREE = None


def quadtree_repulsion():
    """``scigraphs_utils.quadtree_repulsion``, or None. Resolved once."""
    global _QUADTREE
    if _QUADTREE is None:
        try:
            from scigraphs_utils import quadtree_repulsion as fn
            _QUADTREE = fn
        except Exception:  # noqa: BLE001 - an old wheel is not an error
            _QUADTREE = False
    return _QUADTREE or None


REPULSION_MODES = ('GRID', 'TREE')

DEFAULT_THETA = 0.6


def _as_coords(coords):
    out = np.array(coords, dtype=DTYPE, order="C", copy=True)
    if out.ndim != 2 or out.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3), got {out.shape}")
    return out


def _pair_force(targets, sources, t_mass, s_mass, coeff_scale, soften,
                skip_self=False):
    out = np.empty((targets.shape[0], 3), dtype=DTYPE)
    s2 = np.einsum("ij,ij->i", sources, sources)
    for lo in range(0, targets.shape[0], CHUNK):
        hi = min(lo + CHUNK, targets.shape[0])
        t = targets[lo:hi]
        cross = t @ sources.T
        d2 = np.einsum("ij,ij->i", t, t)[:, None] + s2[None, :]
        d2 -= 2.0 * cross
        np.maximum(d2, 0.0, out=d2)
        d2 += soften

        coeff = (coeff_scale * t_mass[lo:hi, None]) * s_mass[None, :]
        coeff /= d2
        if skip_self:
            rows = np.arange(lo, hi)
            coeff[rows - lo, rows] = 0.0

        out[lo:hi] = t * coeff.sum(axis=1)[:, None] - coeff @ sources
    return out


def _as_edges(edges, num_nodes):
    """``(edges, keep)``: the (E, 2) array and the mask that built it, so
    per-edge data can be filtered the same way instead of length-checked
    against a shorter array."""
    if edges is None:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=bool)
    out = np.ascontiguousarray(edges, dtype=np.int64)
    if out.ndim != 2 or out.shape[1] != 2:
        raise ValueError(f"edges must be (E, 2), got {out.shape}")
    if out.size and (out.min() < 0 or out.max() >= num_nodes):
        raise ValueError("edge endpoints out of range")
    # Self-loops contribute nothing and would divide by zero.
    keep = out[:, 0] != out[:, 1]
    return out[keep], keep


def _yh_hash(x):
    """splitmix64, the deterministic stand-in for sfdp's use of rand()."""
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _yh_priority(n):
    """A fixed pseudo-random rank per node index.

    Only a tie-break, but not an optional one: with edge weights equal and ties
    going to the lowest index, every node of a grid picks the neighbor above it
    and almost nothing is mutual, so a 20x20 grid coarsens 400 nodes to 399.
    This is what sfdp's random permutation buys, made reproducible. Shifted down
    a bit so the negation in the lexsort cannot overflow.
    """
    return _yh_hash(np.arange(n, dtype=np.uint64)).view(np.int64) >> np.int64(1)


def _yh_dither(index, amount):
    """(m, 3) offsets in [-amount/2, amount/2), from the node index alone.

    sfdp draws these from ``drand``. Hashing the index instead keeps the whole
    model free of the generator, so two runs agree whatever the seed was and
    :meth:`ForceSim.reset_state` really does replay the schedule; drawing from
    ``self.rng`` left a reset run 4.9e-02 from a fresh one.
    """
    key = (index.astype(np.uint64)[:, None] * np.uint64(3)
           + np.arange(3, dtype=np.uint64)[None, :])
    bits = (_yh_hash(key) >> np.uint64(40)).astype(DTYPE)
    return ((bits * DTYPE(1.0 / 16777216.0)) - DTYPE(0.5)) * DTYPE(amount)


def _yh_match(n, edges, weights):
    """``(parent, nc)``: each node's cluster, clusters numbered from 0.

    Rounds of mutual-heaviest-neighbor over the still free nodes. Each round is
    two lexsorts and a gather, so the cost is in the first one; the later rounds
    run on what is left and are what turns a 0.73 shrink into 0.55.
    """
    parent = np.full(n, -1, dtype=np.int64)
    if not edges.size:
        return np.arange(n, dtype=np.int64), n
    u = np.concatenate([edges[:, 0], edges[:, 1]])
    v = np.concatenate([edges[:, 1], edges[:, 0]])
    w = np.concatenate([weights, weights])
    rank = _yh_priority(n)
    ar = np.arange(n, dtype=np.int64)
    free = np.ones(n, dtype=bool)
    pairs = 0
    for _ in range(YH_MATCH_ROUNDS):
        live = free[u] & free[v]
        if not live.any():
            break
        uu, vv, ww = u[live], v[live], w[live]
        order = np.lexsort((vv, -rank[vv], -ww, uu))
        uu, vv = uu[order], vv[order]
        head = np.ones(uu.size, dtype=bool)
        head[1:] = uu[1:] != uu[:-1]
        best = np.full(n, -1, dtype=np.int64)
        best[uu[head]] = vv[head]
        has = best >= 0
        mate = np.where(has, best, ar)
        take = has & (best[mate] == ar) & (ar < mate)
        if not take.any():
            break
        lo = np.flatnonzero(take)
        hi = best[lo]
        ids = np.arange(pairs, pairs + lo.size, dtype=np.int64)
        parent[lo] = ids
        parent[hi] = ids
        pairs += lo.size
        free[lo] = False
        free[hi] = False
    rest = np.flatnonzero(parent < 0)
    parent[rest] = np.arange(pairs, pairs + rest.size, dtype=np.int64)
    return parent, pairs + rest.size


def _yh_contract(edges, weights, parent, nc):
    """The coarse graph: sfdp's R*A*P with the diagonal removed, which is edges
    between distinct clusters with their weights summed."""
    cu = parent[edges[:, 0]]
    cv = parent[edges[:, 1]]
    keep = cu != cv
    cu, cv, cw = cu[keep], cv[keep], weights[keep]
    if not cu.size:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=DTYPE)
    key = np.minimum(cu, cv) * np.int64(nc) + np.maximum(cu, cv)
    uniq, inverse = np.unique(key, return_inverse=True)
    acc = np.bincount(inverse, weights=cw, minlength=uniq.size)
    return (np.stack([uniq // nc, uniq % nc], axis=1),
            acc.astype(DTYPE))


def _yh_smooth(pos, edges, n):
    """sfdp's interpolate_coord: half a node's own place, half its neighbors'.

    Run once per prolongation. Without it the two members of a cluster land on
    the same point and only the softened repulsion tells them apart.
    """
    if not edges.size:
        return pos
    src, dst = edges[:, 0], edges[:, 1]
    deg = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
    has = deg > 0
    if not has.any():
        return pos
    acc = np.empty((n, 3), dtype=DTYPE)
    for ax in range(3):
        col = np.bincount(src, weights=pos[dst, ax], minlength=n)
        col += np.bincount(dst, weights=pos[src, ax], minlength=n)
        acc[:, ax] = col.astype(DTYPE)
    acc[has] /= deg[has, None].astype(DTYPE)
    pos[has] = DTYPE(0.5) * pos[has] + DTYPE(0.5) * acc[has]
    return pos


class ForceSim:
    """Steppable force-directed layout; parameter names match ForceAtlas2
    and the scene properties.

    ``repulsion`` and ``gravity`` are rewritten in place by the constructor,
    normalized by n and *scale* so that 1.0 means the same thing on any graph.
    Read them back afterwards to drive another backend with the same numbers,
    as ``gpu_render.playback`` does; do not read back what you passed in. The
    values as passed stay on ``_raw_repulsion`` and ``_raw_gravity``, which is
    what lets :meth:`update_params` renormalize instead of compounding.

    Every force term reads its parameters fresh each step, so
    :meth:`update_params` changes a running simulation without restarting it.

    ``YIFAN_HU`` is the exception that proves that rule, because it is a
    schedule and not only a force law. It borrows FR's forces, which are Hu's,
    swaps ForceAtlas2's integrator for sfdp's cooling one, and runs the whole
    thing over a coarsened hierarchy: the graph is contracted until it is a
    handful of nodes, that level is solved, and each finer one is seeded from
    the level above and refined. The hierarchy is state on the object, so a
    caller that keeps calling ``step(1)`` and drawing sees the schedule play
    out; ``yh_level`` counts down to 0 and ``yh_done`` says when the finest
    level has cooled. While a coarse level is active, ``pos`` stays (N, 3) and
    every fine node sits on its representative, so a viewer watches a few dozen
    clumps find their places and then split, repeatedly, into the real graph.
    """

    #: FA2: mass = deg+1, repulsion k_r m_i m_j / d. FR: no mass, repulsion
    #: k²/d, attraction d²/k. LINLOG: FA2 masses with log attraction.
    MODELS = ('FA2', 'FR', 'LINLOG', 'YIFAN_HU')

    _FR_FORCES = ('FR', 'YIFAN_HU')

    def __init__(self, coords, edges, weights=None, *, seed=None,
                 model='FA2',
                 repulsion=1.0, attraction=1.0, gravity=0.1,
                 strong_gravity=False, lin_log=False,
                 edge_weight_influence=1.0, jitter_tolerance=1.0,
                 scale=5.0, dimensions=3, fixed_speed=None,
                 multilevel=True,
                 yh_step=YH_STEP, yh_cool=YH_COOL, yh_tol=YH_TOL,
                 yh_adaptive=True,
                 repulsion_mode='GRID', theta=DEFAULT_THETA):
        self.pos = _as_coords(coords)
        self.n = self.pos.shape[0]
        self.edges, kept = _as_edges(edges, self.n)
        self.dimensions = int(dimensions)

        if weights is None or not self.edges.size:
            self.weights = None
        else:
            w = np.ascontiguousarray(weights, dtype=DTYPE).ravel()
            if w.size == kept.size:
                w = w[kept]
            self.weights = w if w.size == self.edges.shape[0] else None

        self._raw_repulsion = float(repulsion)
        self._raw_gravity = float(gravity)
        self.attraction = float(attraction)
        self.strong_gravity = bool(strong_gravity)
        self.lin_log = bool(lin_log)
        self.edge_weight_influence = float(edge_weight_influence)
        self.jitter_tolerance = float(jitter_tolerance)
        self.scale = float(scale)
        self.yh_start_step = max(float(yh_step), 1e-9)
        self.yh_cool_ratio = float(np.clip(yh_cool, 0.05, 0.999))
        self.yh_tolerance = max(float(yh_tol), 0.0)
        self.yh_want_adaptive = bool(yh_adaptive)
        self.repulsion_mode = str(repulsion_mode).upper()
        if self.repulsion_mode not in REPULSION_MODES:
            self.repulsion_mode = 'GRID'
        self.theta = float(np.clip(theta, 0.0, 4.0))
        # Pin step size for GPU diffs; unused interactively.
        self.fixed_speed = None if fixed_speed is None else float(fixed_speed)

        self.rng = np.random.default_rng(seed)

        self.model = str(model).upper()
        if self.model not in self.MODELS:
            self.model = 'FA2'
        if lin_log and self.model == 'FA2':
            self.model = 'LINLOG'

        deg = np.bincount(self.edges.ravel(), minlength=self.n) \
            if self.edges.size else np.zeros(self.n, dtype=np.int64)
        self._deg = deg
        self.degree = deg.astype(DTYPE)
        self._build_mass()
        self._build_edge_w()

        self.iteration = 0
        self.speed = 1.0
        self._prev_force = np.zeros_like(self.pos)
        self.near_k = NEAR_K
        self.near_reason = None
        self._near_idx = None
        self._near_age = 0

        self._seed = seed
        self.want_multilevel = bool(multilevel)
        self._levels = ()
        self._maps = ()
        self._sub = None
        self.yh_level = 0
        self._yh_unit_n = self.n
        self._yh_restart(adaptive=True)

        self._renormalize()

        # A planar start stays planar forever (forces along p_i - p_j).
        if self.dimensions >= 3 and self.n > 2:
            span = float(self.pos[:, 2].max() - self.pos[:, 2].min())
            if span < 1e-6 * max(self.scale, 1e-9):
                self.pos[:, 2] += (self.rng.standard_normal(self.n)
                                   * DTYPE(0.05 * self.k)).astype(DTYPE)

        if self.model == 'YIFAN_HU' and self.want_multilevel:
            self._yh_enable()

    def _build_mass(self):
        self.mass = (self._deg + 1).astype(DTYPE) \
            if self.model not in self._FR_FORCES \
            else np.ones(self.n, dtype=DTYPE)

    def _build_edge_w(self):
        if self.weights is not None and self.edge_weight_influence != 1.0:
            self.edge_w = np.power(np.abs(self.weights),
                                   self.edge_weight_influence).astype(DTYPE)
        elif self.weights is not None:
            self.edge_w = np.abs(self.weights)
        else:
            self.edge_w = None

    def _renormalize(self):
        """Derive ``k`` and the two norms, then reapply them to the raw values.

        Normalized so that 1.0 means the same thing at any n and scale. The
        equilibrium it lands on is a fit, not a derivation: measured after
        2000 steps, edges settle at 2.8k (karate) to 5.0k (grid), and after
        600 steps the mean radius is 1.63 x scale with a max of 3.3 x, linear
        in scale from 1.0 to 50.0.

        Everything here reads ``scale``, ``model`` and ``mass``, so any live
        change to those has to come back through this rather than scaling what
        is already normalized.
        """
        self.k = self.scale / max(np.cbrt(max(self.n, 1)), 1.0)
        # Edges settle near k: repulsion balances attraction at d = k, gravity
        # balances the whole-graph monopole at R = scale/2.
        mean_mass = float(self.mass.mean()) if self.n else 1.0
        m2 = max(mean_mass ** 2, 1e-9)
        m3 = max(mean_mass ** 3, 1e-9)
        if self.model == 'LINLOG':
            self._repulsion_norm = (self.k * float(np.log1p(self.k))) / m3
        else:
            self._repulsion_norm = (self.k ** 2) / m3
        self._gravity_norm = (_GRAVITY_FIT * 2.0 * (self.scale ** 2)
                              / (max(self.k, 1e-9) * m2))
        self.repulsion = self._raw_repulsion * self._repulsion_norm
        self.gravity = self._raw_gravity * self._gravity_norm

    def update_params(self, **params):
        """Retune a running simulation; return the names that were refused.

        Nothing here touches ``pos``, so the layout carries on from where it
        is under the new forces. ``scale`` and ``model`` rebuild what they
        feed (``k``, the norms, and for ``model`` the masses too) before any
        of it is read again, so a change is never half applied.

        ``dimensions`` is refused: dropping to 2D would be live, but going
        back to 3D needs the constructor's planar nudge, and consuming the rng
        mid-run would make the trajectory depend on how often the control was
        touched. ``lin_log`` is refused as well; pass ``model='LINLOG'``.
        ``multilevel`` is refused too: it decides whether the hierarchy exists
        at all, so turning it on needs a rebuild.

        Under ``YIFAN_HU`` the live parameters reach the active coarse level as
        well, and any change to a force term un-cools the step and clears
        ``yh_done``, since the level it had converged to is no longer the one
        being solved. Two things stay behind: selecting the model rebuilds the
        hierarchy, which is the one expensive change here at 55 ms for 5000
        nodes and 1.9 s for 34722, and ``edge_weight_influence`` retunes the
        forces but not the matching, which the hierarchy already committed to.
        """
        refused = []
        renorm = remass = reheat = False
        was_yh = self.model == 'YIFAN_HU'
        for name, value in params.items():
            if name == 'repulsion':
                self._raw_repulsion = float(value)
                renorm = reheat = True
            elif name == 'gravity':
                self._raw_gravity = float(value)
                renorm = reheat = True
            elif name == 'attraction':
                self.attraction = float(value)
                reheat = True
            elif name == 'jitter_tolerance':
                self.jitter_tolerance = float(value)
            elif name == 'strong_gravity':
                self.strong_gravity = bool(value)
                reheat = True
            elif name == 'scale':
                self.scale = float(value)
                renorm = reheat = True
            elif name == 'model':
                model = str(value).upper()
                if model not in self.MODELS:
                    refused.append(name)
                elif model != self.model:
                    self.model = model
                    self.lin_log = model == 'LINLOG'
                    renorm = remass = True
            elif name == 'edge_weight_influence':
                self.edge_weight_influence = float(value)
                self._build_edge_w()
            elif name == 'yh_step':
                self.yh_start_step = max(float(value), 1e-9)
            elif name == 'yh_cool':
                self.yh_cool_ratio = float(np.clip(value, 0.05, 0.999))
                reheat = True
            elif name == 'yh_tol':
                self.yh_tolerance = max(float(value), 0.0)
                reheat = True
            elif name == 'yh_adaptive':
                self.yh_want_adaptive = bool(value)
                reheat = True
            elif name == 'repulsion_mode':
                mode = str(value).upper()
                if mode in REPULSION_MODES:
                    self.repulsion_mode = mode
                else:
                    refused.append(name)
            elif name == 'theta':
                self.theta = float(np.clip(value, 0.0, 4.0))
            else:
                refused.append(name)
        if remass:
            self._build_mass()
        if renorm:
            self._renormalize()

        is_yh = self.model == 'YIFAN_HU'
        if was_yh and not is_yh:
            self._yh_disable()
        elif is_yh and not was_yh:
            if self.want_multilevel:
                self._yh_enable()
            else:
                self._yh_restart(adaptive=True)
        elif is_yh and reheat:
            self._yh_restart(adaptive=self._yh_level_adaptive,
                             budget=self._yh_budget)
        if self._sub is not None:
            self._sub.update_params(**{n: v for n, v in params.items()
                                       if n not in ('model', 'multilevel')})
        return tuple(refused)

    def reset_state(self, coords=None):
        """Restart the integrator at *coords*, or where the layout already is.

        Equivalent to constructing a fresh simulation on those positions: the
        previous force, the adaptive speed and the neighbor list all go, so
        nothing from the steps before leaks into the ones after. ``iteration``
        keeps counting, since it reports work done rather than progress.

        Under ``YIFAN_HU`` that includes the schedule: the run drops back to
        the coarsest level and plays out again from these positions. The
        hierarchy itself survives, so this costs no rebuild.
        """
        if coords is not None:
            pos = _as_coords(coords)
            if pos.shape[0] != self.n:
                raise ValueError(
                    f"expected {self.n} positions, got {pos.shape[0]}")
            self.pos = pos
        self._flatten_z(self.pos)
        self._prev_force = np.zeros_like(self.pos)
        self.speed = 1.0
        self._near_idx = None
        self._near_age = 0
        self._sub = None
        self.yh_level = 0
        self._yh_restart(adaptive=True)
        if self.model == 'YIFAN_HU' and self._levels:
            self._yh_descend()

    @property
    def positions(self):
        """(N, 3) float32 copy, ready to upload."""
        return self.pos.copy()

    def _flatten_z(self, arr):
        if self.dimensions < 3:
            arr[:, 2] = 0.0
        return arr

    def _repulsion_direct(self):
        return _pair_force(self.pos, self.pos, self.mass, self.mass,
                           self.repulsion, DTYPE((0.01 * self.k) ** 2),
                           skip_self=True)

    def _repulsion_tree(self):
        """Barnes-Hut repulsion, or None when the backend cannot supply it.

        Graphviz's law with ``p = -1`` is ``w_i w_j K^2 (x_i - x_j) / d^2``,
        which is :meth:`_repulsion_direct` under ``w = mass`` and
        ``K = sqrt(repulsion)``; measured, the two agree to 1.3e-15 once theta
        goes to zero. They part only at short range, where this crops the
        distance at Graphviz's MINDIST rather than adding our softening, so a
        pair sitting on top of each other pulls a far larger force. Both
        integrators bound the step that comes out of it, one by capping the
        displacement and one by normalizing the direction.
        """
        fn = quadtree_repulsion()
        if fn is None:
            return None
        force = fn(np.ascontiguousarray(self.pos, dtype=np.float64),
                   weights=np.ascontiguousarray(self.mass, dtype=np.float64),
                   theta=float(self.theta), p=-1.0,
                   K=float(np.sqrt(max(self.repulsion, 0.0))))
        out = np.ascontiguousarray(force, dtype=DTYPE)
        return out if np.isfinite(out).all() else None

    def _refresh_near(self):
        global _NEAR_WARNED
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            self._near_idx = None
            self.near_reason = (
                "scipy is not installed; the near field is off, so repulsion "
                "is the coarse grid alone and understates it at short range")
            if not _NEAR_WARNED:
                _NEAR_WARNED = True
                print("Warning: %s" % self.near_reason)
            return
        k = min(int(self.near_k) + 1, self.n)
        if k < 2:
            self._near_idx = None
            self.near_reason = "fewer than two nodes"
            return
        tree = cKDTree(self.pos)
        _dist, idx = tree.query(self.pos, k=k, workers=-1)
        self._near_idx = np.ascontiguousarray(idx[:, 1:])
        self.near_reason = None

    def _repulsion_near(self):
        if self._near_idx is None:
            return np.zeros((self.n, 3), dtype=DTYPE)
        nbr = self._near_idx
        kk = nbr.shape[1]
        soften = DTYPE((0.01 * self.k) ** 2)
        out = np.empty((self.n, 3), dtype=DTYPE)

        diff = np.empty((min(CHUNK, self.n), kk, 3), dtype=DTYPE)
        d2 = np.empty((min(CHUNK, self.n), kk), dtype=DTYPE)
        for lo in range(0, self.n, CHUNK):
            hi = min(lo + CHUNK, self.n)
            c = hi - lo
            idx = nbr[lo:hi]
            d, dd = diff[:c], d2[:c]

            np.take(self.pos, idx, axis=0, out=d)
            np.subtract(self.pos[lo:hi, None, :], d, out=d)

            np.square(d[:, :, 0], out=dd)
            dd += np.square(d[:, :, 1])
            dd += np.square(d[:, :, 2])
            dd += soften

            coeff = np.take(self.mass, idx)
            coeff *= self.mass[lo:hi, None]
            coeff *= self.repulsion
            coeff /= dd

            d *= coeff[:, :, None]
            np.sum(d, axis=1, out=out[lo:hi])
        return out

    def _repulsion_far(self):
        """Monopole repulsion from a coarse grid, own-cell pairs left to the
        near field.

        The error is not flat in n. FAR_RES gives 512 cells at every size, so
        occupancy grows linearly and the own cell this skips grows with it.
        Scoring what ``step`` applies, ``_repulsion_near()`` plus this, against
        exact all-pairs repulsion at the same softening and coefficient, on
        Barabasi-Albert graphs (m=3) settled for 200 iterations, sampling 128
        targets against all n sources. Median relative error of the force
        vector, median direction error, and the fullest cell's occupancy:

                          FA2                    YIFAN_HU
            n       cell    err    dir      cell    err    dir
              400      5    7.3%   3.0         6   12.9%   5.5
             2000     47   13.7%   6.6       111   23.6%  11.6
             5000    170   13.5%   6.9       704   33.9%  18.0
            10000    415   15.1%   7.7      1686   38.8%  20.2
            30000   1844   20.2%  10.0      5335   31.5%  17.2

        p95 runs 2 to 5x the median. YIFAN_HU is worse at equal n because it
        packs tighter, and real sparse graphs are worse still: SuiteSparse
        rajat23 at 110355 nodes puts 24293 of them, 22% of the graph, in one
        cell, for 39.2% and 21.3 degrees. A length-only metric,
        abs(norm(approx) - norm(exact)) / norm(exact), reads 3 to 5% across
        this whole range because it drops the transverse error, which is most
        of it. Downstream the settled mean radius lands 5% short at 2000 nodes
        and 9% at 5000.

        Handing the own cell to the near field only covers it while the cell
        holds fewer than NEAR_K nodes. At 5000 nodes under FA2 the grid has 242
        occupied cells, median occupancy 11 but up to 118, giving 108000
        same-cell pairs against at most 80000 near-field slots; every pair past
        a node's K nearest gets repulsion from neither field, which is where
        most of the error above comes from.

        A Barnes-Hut octree bounds that error by construction, and was built and
        measured as a replacement rather than argued about. Scoring the near
        field and this one together against ``_repulsion_direct`` on the same
        settled 2000-node graph: 13.8% median relative error and 7.32 degrees
        here, against 5.13% and 0.46 for the tree at an opening angle of 1.2.
        It cost 4 to 6x the time at every size from 1000 to 100000 nodes, and no
        opening angle got that back: on 300 frames of a 34722-node graph, 58 ms
        a frame here against 199 for the tree at 1.2 and 137 at 2.0, where 3.0
        is no better than 2.0 because the floor is the tree build rather than
        the walk.

        The gap is structural, not an implementation detail to tune away. A grid
        evaluates O(C^2) cell-cell interactions once and broadcasts each result
        to every node in the cell, while any angle-bounded scheme has to
        evaluate O(n*C) node-cell ones, and numpy cannot do those for less than
        about 15 ns each. So this belongs in the compute shader, which pays
        O(n*C) already and does the same 300 frames in 0.82 s, and not here.
        Note that ``gpu_render.playback`` picks the shader above 20000 nodes, so
        the sizes where this error matters most are not running this code.

        Below about 1300 nodes ``_repulsion_direct`` is cheaper than such a tree
        as well as exact, so a tree would only ever serve the sizes above.
        """
        lo = self.pos.min(axis=0)
        span = np.maximum(self.pos.max(axis=0) - lo, DTYPE(1e-9))
        cell = np.clip(((self.pos - lo) / span * FAR_RES).astype(np.int32),
                       0, FAR_RES - 1)
        flat = (cell[:, 0] * FAR_RES + cell[:, 1]) * FAR_RES + cell[:, 2]

        occupied, inverse = np.unique(flat, return_inverse=True)
        c = occupied.size
        if c < 2:
            return np.zeros((self.n, 3), dtype=DTYPE)

        cmass = np.bincount(inverse, weights=self.mass,
                            minlength=c).astype(DTYPE)
        centroid = np.stack([
            np.bincount(inverse, weights=self.mass * self.pos[:, ax],
                        minlength=c).astype(DTYPE) / cmass
            for ax in range(3)
        ], axis=1)

        soften = DTYPE((FAR_SOFTEN * float(span.max()) / FAR_RES) ** 2)
        ones = np.ones(c, dtype=DTYPE)
        cell_force = _pair_force(centroid, centroid, ones, cmass,
                                 self.repulsion, soften, skip_self=True)
        return cell_force[inverse] * self.mass[:, None]

    def _attraction(self):
        force = np.zeros((self.n, 3), dtype=DTYPE)
        if not self.edges.size:
            return force

        src, dst = self.edges[:, 0], self.edges[:, 1]
        diff = self.pos[dst] - self.pos[src]

        if self.model == 'LINLOG':
            dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            factor = np.log1p(dist) / np.maximum(dist, DTYPE(1e-9))
        elif self.model in self._FR_FORCES:
            dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            factor = dist / DTYPE(max(self.k, 1e-9))
        else:
            factor = np.ones(src.size, dtype=DTYPE)

        factor = factor * self.attraction
        if self.edge_w is not None:
            factor = factor * self.edge_w

        if self.model not in self._FR_FORCES:
            factor = factor / (DTYPE(0.5) * (self.mass[src] + self.mass[dst]))

        pull = diff * factor[:, None]

        for ax in range(3):
            acc = np.bincount(src, weights=pull[:, ax], minlength=self.n)
            acc -= np.bincount(dst, weights=pull[:, ax], minlength=self.n)
            force[:, ax] = acc.astype(DTYPE)
        return force

    def _gravity(self):
        center = self.pos.mean(axis=0)
        delta = self.pos - center
        if self.strong_gravity:
            return (-self.gravity * self.mass[:, None] * delta).astype(DTYPE)
        dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        scale = -(self.gravity * self.mass / np.maximum(dist, DTYPE(1e-9)))
        return (scale[:, None] * delta).astype(DTYPE)

    def _integrate(self, force):
        swing_v = force - self._prev_force
        swing = np.sqrt(np.einsum("ij,ij->i", swing_v, swing_v))
        trac_v = 0.5 * (force + self._prev_force)
        traction = np.sqrt(np.einsum("ij,ij->i", trac_v, trac_v))

        total_swing = float(np.dot(self.mass, swing))
        total_traction = float(np.dot(self.mass, traction))

        if self.fixed_speed is not None:
            self.speed = self.fixed_speed
        elif total_swing > 0.0:
            target = self.jitter_tolerance * total_traction / total_swing
            # Cap speed jumps at 50% so one noisy step cannot oscillate.
            self.speed = float(np.clip(target, self.speed * 0.5,
                                       self.speed * 1.5))
        self.speed = float(np.clip(self.speed, 1e-4, 10.0))

        factor = self.speed / (1.0 + self.speed * np.sqrt(swing))
        disp = force * factor[:, None]

        norm = np.sqrt(np.einsum("ij,ij->i", disp, disp))
        cap = self.k
        over = norm > cap
        if np.any(over):
            disp[over] *= (cap / norm[over])[:, None]

        self.pos += disp
        self._flatten_z(self.pos)
        self._prev_force = force
        return float(np.sqrt(np.einsum("ij,ij->i", disp, disp)).sum())

    def _yh_restart(self, adaptive, budget=YH_MAXITER):
        """Put the step rule back at the top of its cooling curve."""
        self._yh_step = self.yh_start_step
        self._yh_fnorm = 0.0
        self._yh_iter = 0
        self._yh_budget = int(budget)
        self._yh_level_adaptive = bool(adaptive)
        self._yh_adaptive = bool(adaptive) and self.yh_want_adaptive
        self.yh_done = False

    @property
    def _yh_unit(self):
        """What one unit of step is worth, in scene units.

        The coarsest level's k, the same on every level, because sfdp resets
        ``ctrl->step`` to 0.1 after each prolongation while its K keeps
        shrinking: the step measured against the level's own K therefore grows
        as the run refines, and by the finest level it is ten times what it was
        at the coarsest. Anchoring here instead of to ``self.k`` is what gives
        each level enough travel to expand into the finer layout, and it is not
        cosmetic: against the level's own k a 20x20 grid settles at edge length
        1.15 where a long FR run finds 2.36, and its residual force per node is
        4.73 rather than 2.03. On a 2000-node small world the same swap costs
        0.53 of graph-distance correlation against 0.36.
        """
        return self.scale / max(np.cbrt(max(self._yh_unit_n, 1)), 1.0)

    def _integrate_yh(self, force):
        """Yifan Hu's move: every node travels the same distance, along its own
        force, and only the distance cools.

        ``update_step`` in sfdp's spring_electrical.c. Note that it can *grow*
        the step, by 0.99/0.90 = 1.1x, whenever the total force norm drops more
        than 5% in one iteration; ForceAtlas2's swing rule never does, and that
        growth is what lets a level recover from cooling that ran ahead of the
        layout.

        Only the coarsest level gets the adaptive rule; sfdp turns it off for
        every level it prolongs into, leaving plain 0.90 cooling, which is what
        makes a refinement level exactly 44 iterations long.
        """
        fnorm = np.sqrt(np.einsum("ij,ij->i", force, force))
        total = float(fnorm.sum())

        disp = force / np.maximum(fnorm, DTYPE(1e-12))[:, None]
        disp *= DTYPE(self._yh_step * self._yh_unit)
        self.pos += disp
        self._flatten_z(self.pos)

        self._yh_cool(total)
        return float(np.sqrt(np.einsum("ij,ij->i", disp, disp)).sum())

    def _yh_cool(self, total):
        """``update_step`` alone, so a backend that moved the nodes elsewhere
        cools by the same rule rather than by a copy of it."""
        cool = self.yh_cool_ratio
        if not self._yh_adaptive or total >= self._yh_fnorm:
            self._yh_step *= cool
        elif total <= 0.95 * self._yh_fnorm:
            self._yh_step = min(0.99 * self._yh_step / cool, self.yh_start_step)
        self._yh_fnorm = total
        self._yh_iter += 1
        if self._yh_step <= self.yh_tolerance \
                or self._yh_iter >= self._yh_budget:
            self.yh_done = True

    def _yh_enable(self):
        """Build the hierarchy and drop to its coarsest level.

        Levels are held as parent maps rather than as a chain of graphs plus a
        chain of matrices, because everything the schedule needs from sfdp's P
        and R is ``parent`` and a gather by it.
        """
        self._levels = ()
        self._maps = ()
        levels = []
        maps = []
        n, edges = self.n, self.edges
        weights = self.edge_w if self.edge_w is not None \
            else np.ones(edges.shape[0], dtype=DTYPE)
        while n > YH_MINSIZE and len(levels) < YH_MAX_LEVELS:
            step = self._yh_one_level(n, edges, weights)
            if step is None:
                break
            parent, n, edges, weights = step
            levels.append((parent, n, edges))
            maps.append(parent if not maps else parent[maps[-1]])
        self._levels = tuple(levels)
        self._maps = tuple(maps)
        self._sub = None
        self.yh_level = 0
        self._yh_unit_n = levels[-1][1] if levels else self.n
        self._yh_restart(adaptive=not self._levels)
        if self._levels:
            self._yh_descend()

    @staticmethod
    def _yh_one_level(n, edges, weights):
        """One rung: match and contract, repeating while the level is still
        more than YH_COARSEN of what it was, as sfdp's Multilevel_coarsen does.
        Returns None when there is nothing worth solving separately."""
        parent = np.arange(n, dtype=np.int64)
        cn, ce, cw = n, edges, weights
        for _ in range(4):
            p, nc = _yh_match(cn, ce, cw)
            if nc == cn or nc < YH_MINSIZE:
                break
            ce, cw = _yh_contract(ce, cw, p, nc)
            parent = p[parent]
            cn = nc
            if nc <= YH_COARSEN * n:
                break
        if cn == n or cn > YH_STALL * n:
            return None
        return parent, cn, ce, cw

    def _yh_sub(self, pos, edges, adaptive, budget=YH_MAXITER):
        """The active level as a plain single-level simulation.

        Its own n gives it its own k, so a level of 800 nodes lays itself out
        at 5.0/cbrt(800) and the next finer one contracts to fit; sfdp gets the
        same effect by multiplying K by 0.75 per level, which for a matching
        that halves the graph is cbrt(2) = 1.26 against 1/0.75 = 1.33.

        No weights: sfdp's contraction sums them, but none of its three force
        loops ever reads a matrix value, so the coarse weights exist only to
        pick heavy edges to match on. The weights a caller passed in still
        drive the finest level, which is this object.
        """
        sub = ForceSim(pos, edges, seed=self._seed, model='YIFAN_HU',
                       repulsion=self._raw_repulsion,
                       attraction=self.attraction,
                       gravity=self._raw_gravity,
                       strong_gravity=self.strong_gravity,
                       edge_weight_influence=self.edge_weight_influence,
                       jitter_tolerance=self.jitter_tolerance,
                       scale=self.scale, dimensions=self.dimensions,
                       fixed_speed=self.fixed_speed, multilevel=False,
                       yh_step=self.yh_start_step, yh_cool=self.yh_cool_ratio,
                       yh_tol=self.yh_tolerance,
                       yh_adaptive=self.yh_want_adaptive,
                       repulsion_mode=self.repulsion_mode, theta=self.theta)
        sub._yh_unit_n = self._yh_unit_n
        sub._yh_restart(adaptive=adaptive, budget=budget)
        return sub

    @property
    def active_level(self):
        """The simulation a step actually moves: the coarse level while the
        schedule is still descending, this object once it reaches the finest."""
        return self._sub if self._sub is not None else self

    def _yh_sync(self):
        """Fine positions follow their representative, so the draw path keeps
        getting (N, 3) whatever level is running."""
        self.pos = self._sub.pos[self._maps[self.yh_level - 1]]

    def _yh_descend(self):
        """Project the current layout onto the coarsest level and start there.

        Each cluster takes the mean of its members, which is sfdp's R once its
        rows are divided by their degree.
        """
        top = len(self._levels)
        _parent, nc, edges = self._levels[top - 1]
        m = self._maps[top - 1]
        count = np.maximum(np.bincount(m, minlength=nc), 1).astype(DTYPE)
        cpos = np.empty((nc, 3), dtype=DTYPE)
        for ax in range(3):
            cpos[:, ax] = np.bincount(m, weights=self.pos[:, ax],
                                      minlength=nc).astype(DTYPE)
        cpos /= count[:, None]
        self.yh_level = top
        self._sub = self._yh_sub(cpos, edges, adaptive=True,
                                 budget=YH_COARSE_ITER)
        self._yh_sync()

    def _yh_prolong(self):
        """Move one level finer: seed it from the level above, then refine."""
        level = self.yh_level
        parent = self._levels[level - 1][0]
        if level == 1:
            fine_n, fine_edges = self.n, self.edges
        else:
            _p, fine_n, fine_edges = self._levels[level - 2]
        pos = self._sub.pos[parent]
        _yh_smooth(pos, fine_edges, fine_n)
        self._yh_split(pos, parent,
                       self.scale / max(np.cbrt(max(fine_n, 1)), 1.0))
        self.yh_level = level - 1
        if level == 1:
            self._sub = None
            self.pos = pos
            self._yh_restart(adaptive=False)
        else:
            self._sub = self._yh_sub(pos, fine_edges, adaptive=False)
            self._yh_sync()

    def _yh_split(self, pos, parent, k):
        """sfdp's prolongation jitter: every cluster member but the first gets
        a nudge of K/1000, so a pair that landed on one point has somewhere to
        go even where the smoothing left it symmetric."""
        _uniq, first = np.unique(parent, return_index=True)
        mask = np.ones(parent.size, dtype=bool)
        mask[first] = False
        rest = np.flatnonzero(mask)
        if not rest.size:
            return
        pos[rest] += _yh_dither(rest, YH_JITTER * k)
        self._flatten_z(pos)

    def _yh_disable(self):
        """Leave the schedule for another model, carrying the layout up to the
        finest level first so the next model does not inherit N/2^L points
        stacked on top of each other."""
        while self._sub is not None:
            self._yh_prolong()
        self._levels = ()
        self._maps = ()
        self.yh_level = 0
        self._yh_unit_n = self.n
        self._yh_restart(adaptive=True)

    def _yh_step_coarse(self):
        sub = self._sub
        spread = self.n / max(sub.n, 1)
        moved = sub.step(1)
        self._yh_sync()
        self.iteration += 1
        if sub.yh_done or sub.n < 2:
            self._yh_prolong()
        return moved * spread

    def _force_field(self):
        """Everything acting on a node this iteration, in one array."""
        force = None
        if self.n <= DIRECT_MAX:
            force = self._repulsion_direct()
        elif self.repulsion_mode == 'TREE':
            force = self._repulsion_tree()
        if force is None:
            if self._near_idx is None or self._near_age >= NEAR_REFRESH:
                self._refresh_near()
                self._near_age = 0
            self._near_age += 1
            force = self._repulsion_near() + self._repulsion_far()
        force += self._attraction()
        if self.gravity:
            force += self._gravity()
        self._flatten_z(force)
        return force

    def step_external(self, advance):
        """One iteration with *advance* doing the level's force work.

        ``advance(sim, distance)`` moves ``sim.pos`` by ``distance`` along each
        node's own force and returns ``(moved, total_force_norm)``. Everything
        that makes this Yifan Hu rather than Fruchterman-Reingold stays here:
        the hierarchy, the cooling rule, the convergence test and the
        prolongation. Only the arithmetic leaves, which is what lets a level run
        in a compute shader without the schedule following it out of this
        module. Grouping is preserved the same way ``step`` preserves it.
        """
        if self.model != 'YIFAN_HU':
            raise ValueError("step_external is the Yifan Hu schedule's seam")
        if self._sub is not None:
            sub = self._sub
            spread = self.n / max(sub.n, 1)
            moved = sub.step_external(advance)
            self._yh_sync()
            self.iteration += 1
            if sub.yh_done or sub.n < 2:
                self._yh_prolong()
            return moved * spread
        if self.n < 2 or self.yh_done:
            return 0.0
        center_was = self.pos.mean(axis=0)
        result = advance(self, float(self._yh_step * self._yh_unit))
        if result is None:
            moved = self._integrate_yh(self._force_field())
        else:
            moved, total = result
            self._flatten_z(self.pos)
            self._yh_cool(float(total))
        self.pos -= (self.pos.mean(axis=0) - center_was).astype(DTYPE)
        self._flatten_z(self.pos)
        self.iteration += 1
        return float(moved)

    def step(self, iterations=1):
        """Advance the layout; return total distance moved.

        Under ``YIFAN_HU`` this is also where the schedule advances: a step
        goes to whichever level is current, and meeting that level's
        convergence test prolongs to the next one before the call returns.
        Nothing in it depends on how the iterations were grouped, so
        ``step(120)`` lands exactly where 120 calls of ``step(1)`` do. Once the
        finest level has cooled, ``yh_done`` is set and further steps are free
        and do nothing, until :meth:`update_params` or :meth:`reset_state`
        gives the run something new to solve.
        """
        moved = 0.0
        for _ in range(max(1, int(iterations))):
            if self._sub is not None:
                moved = self._yh_step_coarse()
                continue
            if self.n < 2 or self.yh_done:
                break
            center_was = self.pos.mean(axis=0)
            force = self._force_field()

            moved = self._integrate_yh(force) if self.model == 'YIFAN_HU' \
                else self._integrate(force)

            self.pos -= (self.pos.mean(axis=0) - center_was).astype(DTYPE)
            self._flatten_z(self.pos)

            self.iteration += 1
        return moved


def random_positions(num_nodes, scale=5.0, seed=None, dimensions=3):
    """Seeded start positions, centered on the origin."""
    rng = np.random.default_rng(seed)
    pos = ((rng.random((num_nodes, 3)) - 0.5) * scale).astype(DTYPE)
    if dimensions < 3:
        pos[:, 2] = 0.0
    return pos
