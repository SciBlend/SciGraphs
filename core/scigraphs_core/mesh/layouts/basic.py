"""Basic geometric layout algorithms."""

from .common import *

def _random_layout(num_nodes, scale, seed=None):
    if seed is None:
        seed = get_layout_seed()
    rng = np.random.RandomState(seed)
    return rng.rand(num_nodes, 3) * scale

def _grid_layout(num_nodes, scale):
    grid_size = int(np.ceil(np.sqrt(num_nodes)))
    positions = np.zeros((num_nodes, 3))

    for i in range(num_nodes):
        x = (i % grid_size) * scale / grid_size
        y = (i // grid_size) * scale / grid_size
        positions[i] = [x, y, 0]

    return positions

def _sphere_layout(num_nodes, scale):
    """Even distribution over a sphere, by the Fibonacci sphere construction.
    Latitudes sit at the midpoints ``(i+0.5)/n`` of equal-area bands rather than
    at the band edges, which is what keeps the two poles from being over-packed
    (Gonzalez 2010; Marques et al. 2013)."""
    i = np.arange(num_nodes)

    y = 1.0 - 2.0 * (i + 0.5) / num_nodes
    radius = np.sqrt(1.0 - y * y)

    theta = np.pi * (3.0 - np.sqrt(5.0)) * i

    return np.column_stack((np.cos(theta) * radius, y, np.sin(theta) * radius)) * scale

def _spiral_layout_3d(num_nodes, scale, turns=None):
    """A conical spiral climbing from radius ``scale*0.5`` to ``scale``, evenly
    spaced along its own arc. *turns* defaults to a count that keeps the gap
    between successive turns close to the spacing along the curve."""
    if turns is None:
        turns = max(2, int(round(np.sqrt(num_nodes / (0.75 * np.pi)))))

    omega = 2.0 * np.pi * turns

    grid = np.linspace(0.0, 1.0, 1 << 16)
    speed = np.sqrt((0.5 * scale) ** 2
                    + (0.5 * scale * (1.0 + grid) * omega) ** 2
                    + (2.0 * scale) ** 2)
    step = grid[1] - grid[0]
    length = np.concatenate(([0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * step)))

    if num_nodes > 1:
        wanted = np.linspace(0.0, length[-1], num_nodes)
    else:
        wanted = np.array([0.5 * length[-1]])
    t = np.interp(wanted, length, grid)

    angle = t * omega
    radius = scale * 0.5 * (1.0 + t)

    return np.column_stack((radius * np.cos(angle),
                            radius * np.sin(angle),
                            scale * (2.0 * t - 1.0)))

def _helix_layout(num_nodes, scale):
    """A double helix, alternating nodes between the two strands. An odd node
    count leaves the extra node on strand 0."""
    levels = (num_nodes + 1) // 2
    i = np.arange(num_nodes)

    if levels > 1:
        t = (i // 2) / (levels - 1)
    else:
        t = np.full(num_nodes, 0.5)

    angle = t * 4 * np.pi + (i % 2) * np.pi
    radius = scale * 0.3

    return np.column_stack((radius * np.cos(angle),
                            radius * np.sin(angle),
                            t * scale * 2 - scale))

def _cube_layout(num_nodes, scale):
    """The cube corners first, then the rest scattered strictly inside. The cube
    is centred on the origin with half-side ``scale``, so its side is ``2*scale``."""
    positions = np.zeros((num_nodes, 3))

    if num_nodes == 1:
        return positions

    corners = np.array([
        [1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0],
    ])

    idx = min(num_nodes, len(corners))
    positions[:idx] = corners[:idx] * scale

    if idx < num_nodes:
        rng = _get_layout_rng()
        positions[idx:] = rng.uniform(-1.0, 1.0, (num_nodes - idx, 3)) * (scale * 0.8)

    return positions

__all__ = [name for name in globals() if not name.startswith('__')]
