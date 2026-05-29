"""Basic geometric layout algorithms."""

from .common import *

def _random_layout(num_nodes, scale, seed=None):
    """Random 3D positions with reproducible seed."""
    if seed is None:
        seed = get_layout_seed()
    rng = np.random.RandomState(seed)
    return rng.rand(num_nodes, 3) * scale

def _grid_layout(num_nodes, scale):
    """2D grid layout."""
    grid_size = int(np.ceil(np.sqrt(num_nodes)))
    positions = np.zeros((num_nodes, 3))

    for i in range(num_nodes):
        x = (i % grid_size) * scale / grid_size
        y = (i // grid_size) * scale / grid_size
        positions[i] = [x, y, 0]

    return positions

def _sphere_layout(num_nodes, scale):
    """
    Distribute nodes evenly on the surface of a sphere.
    Uses Fibonacci sphere algorithm for optimal distribution.
    """
    positions = np.zeros((num_nodes, 3))

    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

    for i in range(num_nodes):
        y = 1 - (i / float(num_nodes - 1 if num_nodes > 1 else 1)) * 2
        radius = np.sqrt(1 - y * y)

        theta = phi * i

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        positions[i] = [x * scale, y * scale, z * scale]

    return positions

def _spiral_layout_3d(num_nodes, scale):
    """
    Arrange nodes in a 3D spiral pattern.
    Creates an upward spiral with increasing radius.
    """
    positions = np.zeros((num_nodes, 3))

    for i in range(num_nodes):
        t = i / max(1, num_nodes - 1)

        # Spiral parameters
        angle = t * 4 * np.pi  # Multiple rotations
        height = t * scale * 2 - scale  # Vertical spread
        radius = scale * 0.5 * (1 + t)  # Increasing radius

        x = radius * np.cos(angle)
        y = height
        z = radius * np.sin(angle)

        positions[i] = [x, y, z]

    return positions

def _helix_layout(num_nodes, scale):
    """
    Arrange nodes in a double helix pattern.
    Similar to DNA structure visualization.
    """
    positions = np.zeros((num_nodes, 3))

    nodes_per_strand = num_nodes // 2

    for i in range(num_nodes):
        strand = i % 2
        t = (i // 2) / max(1, nodes_per_strand - 1)

        # Helix parameters
        angle = t * 4 * np.pi + strand * np.pi  # Offset second strand
        height = t * scale * 2 - scale
        radius = scale * 0.3

        x = radius * np.cos(angle)
        y = height
        z = radius * np.sin(angle)

        positions[i] = [x, y, z]

    return positions

def _cube_layout(num_nodes, scale):
    """
    Distribute nodes on the surface and interior of a cube.
    First places nodes on cube vertices, edges, faces, then fills interior.
    """
    positions = np.zeros((num_nodes, 3))

    cube_side = scale
    idx = 0

    # 8 vertices of the cube
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                if idx >= num_nodes:
                    break
                positions[idx] = [x * cube_side, y * cube_side, z * cube_side]
                idx += 1
            if idx >= num_nodes:
                break
        if idx >= num_nodes:
            break

    # Fill remaining nodes randomly within the cube
    if idx < num_nodes:
        remaining = num_nodes - idx
        positions[idx:] = (np.random.rand(remaining, 3) * 2 - 1) * cube_side

    return positions

__all__ = [name for name in globals() if not name.startswith('__')]
