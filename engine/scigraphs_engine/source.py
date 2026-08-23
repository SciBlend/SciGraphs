# Where graph data comes from. I/O belongs here, computation in channels.py.
# num_vertices counts mesh vertices, not nodes: a baked edge style interleaves
# nodes with curve points, and node_mask is what tells them apart.

from typing import Optional, Protocol, runtime_checkable

import numpy as np

# Attribute domains, spelled as Blender spells them.
POINT = 'POINT'
EDGE = 'EDGE'


@runtime_checkable
class GraphSource(Protocol):
    """Read-only access to one graph. Never cache here: the picture goes stale."""

    @property
    def key(self):
        """Stable identity for cache keying. Distinct graphs must not collide."""
        ...

    @property
    def epoch(self) -> int:
        """Bump this when the geometry moves; a key alone cannot see that."""
        ...

    @property
    def num_vertices(self) -> int:
        ...

    def coords(self) -> np.ndarray:
        """(N, 3) float32 vertex positions, object space."""
        ...

    def edges(self) -> Optional[np.ndarray]:
        """(E, 2) int32 endpoint indices, or None when there are no edges."""
        ...

    def node_mask(self) -> Optional[np.ndarray]:
        """(N,) bool of which vertices are real nodes, or None when all are."""
        ...

    def point_flags(self, name: str) -> Optional[np.ndarray]:
        """A named POINT attribute as a boolean mask; ``scalar_names`` omits BOOLEAN."""
        ...

    def point_scalar(self, name: str) -> Optional[np.ndarray]:
        """A POINT scalar attribute as float32, or None if absent or unusable."""
        ...

    def edge_scalar(self, name: str) -> Optional[np.ndarray]:
        """An EDGE scalar attribute as float32, or None if absent or unusable."""
        ...

    def point_colors(self) -> Optional[np.ndarray]:
        """(N, 4) float32 of the active POINT color attribute, or None."""
        ...

    def scalar_names(self, domain: str = POINT):
        """Scalar attribute names on ``domain``, for the UI menus."""
        ...


class ArraySource:
    __slots__ = ("_coords", "_edges", "_mask", "_point", "_edge",
                 "_colors", "_key", "_epoch")

    def __init__(self, coords, edges=None, node_mask=None, point_attrs=None,
                 edge_attrs=None, colors=None, key=None, epoch=0):
        self._coords = np.ascontiguousarray(coords, dtype=np.float32)
        if self._coords.ndim != 2 or self._coords.shape[1] != 3:
            raise ValueError(f"coords must be (N, 3), got {self._coords.shape}")

        if edges is None:
            self._edges = None
        else:
            e = np.ascontiguousarray(edges, dtype=np.int32)
            if e.ndim != 2 or e.shape[1] != 2:
                raise ValueError(f"edges must be (E, 2), got {e.shape}")
            self._edges = e if e.shape[0] else None

        self._mask = None if node_mask is None else \
            np.ascontiguousarray(node_mask, dtype=bool)
        self._point = dict(point_attrs or {})
        self._edge = dict(edge_attrs or {})
        self._colors = None if colors is None else \
            np.ascontiguousarray(colors, dtype=np.float32)
        self._key = key if key is not None else id(self)
        self._epoch = int(epoch)

    @property
    def key(self):
        return self._key

    @property
    def epoch(self):
        return self._epoch

    @property
    def num_vertices(self):
        return int(self._coords.shape[0])

    def coords(self):
        return self._coords

    def edges(self):
        return self._edges

    def node_mask(self):
        return self._mask

    def point_flags(self, name):
        values = self._point.get(name)
        if values is None:
            return None
        arr = np.asarray(values)
        if arr.size != self.num_vertices:
            return None
        return arr.astype(bool)

    def point_scalar(self, name):
        return _validated(self._point.get(name), self.num_vertices)

    def edge_scalar(self, name):
        n = 0 if self._edges is None else int(self._edges.shape[0])
        return _validated(self._edge.get(name), n)

    def point_colors(self):
        return self._colors

    def scalar_names(self, domain=POINT):
        return tuple(sorted(self._edge if domain == EDGE else self._point))


def _validated(values, expected):
    """Drop an attribute whose length does not match its domain: wrong topology."""
    if values is None or not expected:
        return None
    arr = np.asarray(values)
    if arr.size != expected:
        return None
    return arr.astype(np.float32, copy=False)
