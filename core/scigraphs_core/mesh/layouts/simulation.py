"""Incremental force-directed layout in numpy, steppable without touching bpy.

``ForceSim.step(n)`` advances the state and ``positions`` returns the array to
upload. Repulsion is exact near-field over a bounded neighbor list plus a coarse
monopole grid far-field; the integrator is ForceAtlas2's.
"""

import numpy as np

DTYPE = np.float32

DIRECT_MAX = 1000  # above this, the near/far split replaces all-pairs

NEAR_K = 16
NEAR_REFRESH = 8

# Strip size for O(n²) work so temporaries stay bounded.
CHUNK = 2048

FAR_RES = 8
FAR_SOFTEN = 0.5  # as a fraction of cell size

# Corrects the monopole gravity underestimate; fitted across n and scale.
_GRAVITY_FIT = 0.38


def _as_coords(coords):
    out = np.ascontiguousarray(coords, dtype=DTYPE)
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
    if edges is None:
        return np.zeros((0, 2), dtype=np.int64)
    out = np.ascontiguousarray(edges, dtype=np.int64)
    if out.ndim != 2 or out.shape[1] != 2:
        raise ValueError(f"edges must be (E, 2), got {out.shape}")
    if out.size and (out.min() < 0 or out.max() >= num_nodes):
        raise ValueError("edge endpoints out of range")
    # Self-loops contribute nothing and would divide by zero.
    return out[out[:, 0] != out[:, 1]]


class ForceSim:
    """Steppable force-directed layout; parameter names match ForceAtlas2
    and the scene properties."""

    #: FA2: mass = deg+1, repulsion k_r m_i m_j / d. FR: no mass, repulsion
    #: k²/d, attraction d²/k. LINLOG: FA2 masses with log attraction.
    MODELS = ('FA2', 'FR', 'LINLOG')

    def __init__(self, coords, edges, weights=None, *, seed=None,
                 model='FA2',
                 repulsion=1.0, attraction=1.0, gravity=0.1,
                 strong_gravity=False, lin_log=False,
                 edge_weight_influence=1.0, jitter_tolerance=1.0,
                 scale=5.0, dimensions=3, fixed_speed=None):
        self.pos = _as_coords(coords)
        self.n = self.pos.shape[0]
        self.edges = _as_edges(edges, self.n)
        self.dimensions = int(dimensions)

        if weights is None or not self.edges.size:
            self.weights = None
        else:
            w = np.ascontiguousarray(weights, dtype=DTYPE).ravel()
            self.weights = w if w.size == self.edges.shape[0] else None

        self.repulsion = float(repulsion)
        self.attraction = float(attraction)
        self.gravity = float(gravity)
        self.strong_gravity = bool(strong_gravity)
        self.lin_log = bool(lin_log)
        self.edge_weight_influence = float(edge_weight_influence)
        self.jitter_tolerance = float(jitter_tolerance)
        self.scale = float(scale)
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
        self.degree = deg.astype(DTYPE)
        self.mass = (deg + 1).astype(DTYPE) if self.model != 'FR' \
            else np.ones(self.n, dtype=DTYPE)

        if self.weights is not None and self.edge_weight_influence != 1.0:
            self.edge_w = np.power(np.abs(self.weights),
                                   self.edge_weight_influence).astype(DTYPE)
        elif self.weights is not None:
            self.edge_w = np.abs(self.weights)
        else:
            self.edge_w = None

        self.iteration = 0
        self.speed = 1.0
        self._prev_force = np.zeros_like(self.pos)
        self._near_idx = None
        self._near_age = 0

        self.k = self.scale / max(np.cbrt(max(self.n, 1)), 1.0)

        # A planar start stays planar forever (forces along p_i - p_j).
        if self.dimensions >= 3 and self.n > 2:
            span = float(self.pos[:, 2].max() - self.pos[:, 2].min())
            if span < 1e-6 * max(self.scale, 1e-9):
                self.pos[:, 2] += (self.rng.standard_normal(self.n)
                                   * DTYPE(0.05 * self.k)).astype(DTYPE)

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
        self.repulsion *= self._repulsion_norm
        self.gravity *= self._gravity_norm

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

    def _refresh_near(self):
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            self._near_idx = None
            return
        k = min(NEAR_K + 1, self.n)
        if k < 2:
            self._near_idx = None
            return
        tree = cKDTree(self.pos)
        _dist, idx = tree.query(self.pos, k=k, workers=-1)
        self._near_idx = np.ascontiguousarray(idx[:, 1:])

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
        near field. Overlapping neighbor cells cost ~6% magnitude error against
        exact, direction under half a degree."""
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
        elif self.model == 'FR':
            dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            factor = dist / DTYPE(max(self.k, 1e-9))
        else:
            factor = np.ones(src.size, dtype=DTYPE)

        factor = factor * self.attraction
        if self.edge_w is not None:
            factor = factor * self.edge_w

        pull = diff * factor[:, None]

        for ax in range(3):
            acc = np.bincount(src, weights=pull[:, ax], minlength=self.n)
            acc -= np.bincount(dst, weights=pull[:, ax], minlength=self.n)
            force[:, ax] = acc.astype(DTYPE)

        # FA2 divides by receiving mass after the scatter, not per edge.
        if self.model != 'FR':
            force /= self.mass[:, None]
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

    def step(self, iterations=1):
        """Advance the layout; return total distance moved."""
        moved = 0.0
        for _ in range(max(1, int(iterations))):
            if self.n < 2:
                break
            if self.n <= DIRECT_MAX:
                force = self._repulsion_direct()
            else:
                if self._near_idx is None or self._near_age >= NEAR_REFRESH:
                    self._refresh_near()
                    self._near_age = 0
                self._near_age += 1
                force = self._repulsion_near() + self._repulsion_far()

            force += self._attraction()
            if self.gravity:
                force += self._gravity()
            self._flatten_z(force)

            moved = self._integrate(force)
            self.iteration += 1
        return moved


def random_positions(num_nodes, scale=5.0, seed=None, dimensions=3):
    """Seeded start positions, centered on the origin."""
    rng = np.random.default_rng(seed)
    pos = ((rng.random((num_nodes, 3)) - 0.5) * scale).astype(DTYPE)
    if dimensions < 3:
        pos[:, 2] = 0.0
    return pos
