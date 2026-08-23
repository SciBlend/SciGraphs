# Only module here that may import bpy; defaults come from RNA, not literals.

import numpy as np

import bpy

from scigraphs_core.coloring.attributes import find_attribute, read_attribute_values
from ...core.render.settings import PREFIX, Clause, Settings
from ...core.render.source import EDGE, POINT

# POINTER and COLLECTION are structure; the filter stack comes from clauses_from_scene.
_VALUE_TYPES = frozenset({'BOOLEAN', 'INT', 'FLOAT', 'STRING', 'ENUM'})


# Memoized: walking ``Scene.bl_rna.properties`` was 82 of a snapshot's 103 us.
_PROPS = None


def _rna_properties():
    global _PROPS
    if _PROPS is not None:
        return _PROPS
    out = {}
    for prop in bpy.types.Scene.bl_rna.properties:
        ident = prop.identifier
        if not ident.startswith(PREFIX):
            continue
        if prop.type not in _VALUE_TYPES:
            continue
        out[ident[len(PREFIX):]] = prop
    # Empty means this ran before register_properties; caching it poisons the rest.
    if out:
        _PROPS = out
    return out


def invalidate_properties():
    global _PROPS
    _PROPS = None


def _rna_default(prop):
    if getattr(prop, "is_array", False):
        return tuple(prop.default_array)
    return prop.default


def _plain(value):
    """Blender's array wrappers are not hashable and do not survive a session."""
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        return tuple(value)
    return value


def defaults():
    return {name: _plain(_rna_default(prop))
            for name, prop in _rna_properties().items()}


def clauses_from_scene(scene):
    slots = getattr(scene, "scigraphs_filters", None) or ()
    return tuple(
        Clause(
            channel=slot.channel,
            attr_name=slot.attr_name,
            range_min=float(slot.range_min),
            range_max=float(slot.range_max),
            invert=bool(slot.invert),
            mute=bool(slot.mute),
        )
        for slot in slots
    )


_SCALAR_TYPES = ('FLOAT', 'INT')


class BlenderGraphSource:
    """The only mesh reader; uncached, since a cache outlives a mesh edit."""

    __slots__ = ("obj", "mesh")

    def __init__(self, obj):
        self.obj = obj
        self.mesh = obj.data

    @property
    def key(self):
        return self.obj.as_pointer()

    @property
    def epoch(self):
        return int(self.obj.get("scigraphs_preview_epoch", 0))

    @property
    def name(self):
        return self.obj.name

    @property
    def num_vertices(self):
        return len(self.mesh.vertices)

    def coords(self):
        n = len(self.mesh.vertices)
        out = np.empty(n * 3, dtype=np.float32)
        self.mesh.vertices.foreach_get("co", out)
        return out.reshape(n, 3)

    def edges(self):
        n = len(self.mesh.edges)
        if not n:
            return None
        out = np.empty(n * 2, dtype=np.int32)
        self.mesh.edges.foreach_get("vertices", out)
        return out.reshape(n, 2)

    def node_mask(self):
        """None when every vertex is a node, so callers can skip the mask."""
        attr = self.mesh.attributes.get("is_intersection")
        if attr is None or attr.domain != POINT:
            return None
        raw = np.empty(len(self.mesh.vertices), dtype=np.int32)
        try:
            attr.data.foreach_get("value", raw)
        except (RuntimeError, TypeError):
            return None
        return raw == 1

    # BOOLEAN is what the seed operator writes; INT and FLOAT are imported masks.
    _FLAG_TYPES = {'BOOLEAN': bool, 'INT': np.int32, 'FLOAT': np.float32}

    def point_flags(self, name):
        attr = self.mesh.attributes.get(name)
        if attr is None or attr.domain != POINT:
            return None
        dtype = self._FLAG_TYPES.get(attr.data_type)
        if dtype is None:
            return None
        n = len(self.mesh.vertices)
        if not n or len(attr.data) != n:
            return None
        raw = np.empty(n, dtype=dtype)
        try:
            attr.data.foreach_get("value", raw)
        except (RuntimeError, TypeError):
            return None
        return raw.astype(bool)

    def point_scalar(self, name):
        """Raw POINT scalar. The finite check and min/max live in render.channels."""
        if not name:
            return None
        attr = find_attribute(self.mesh, name)
        if attr is None or getattr(attr, "domain", "") != POINT:
            return None
        values = read_attribute_values(self.mesh, name)
        if values.size != len(self.mesh.vertices):
            return None
        return values

    def edge_scalar(self, name):
        if not name:
            return None
        attr = self.mesh.attributes.get(name)
        if attr is None or attr.domain != EDGE \
                or attr.data_type not in _SCALAR_TYPES:
            return None
        n = len(self.mesh.edges)
        if not n or len(attr.data) != n:
            return None
        out = np.empty(
            n, dtype=np.float32 if attr.data_type == 'FLOAT' else np.int32)
        try:
            attr.data.foreach_get("value", out)
        except (RuntimeError, TypeError):
            return None
        return out.astype(np.float32)

    def point_colors(self):
        attrs = getattr(self.mesh, "color_attributes", None)
        if not attrs:
            return None
        active = attrs.active_color
        n = len(self.mesh.vertices)
        if active is None or active.domain != POINT or len(active.data) != n:
            return None
        out = np.empty(n * 4, dtype=np.float32)
        active.data.foreach_get("color", out)
        return out.reshape(n, 4)

    def scalar_names(self, domain=POINT):
        return tuple(a.name for a in self.mesh.attributes
                     if a.domain == domain and a.data_type in _SCALAR_TYPES
                     and not a.name.startswith("."))


def source_from_object(obj):
    from .state import is_graph_object
    return BlenderGraphSource(obj) if is_graph_object(obj) else None


def settings_from_scene(scene):
    """Complete by construction, so a reader never needs a fallback."""
    values = {}
    for name, prop in _rna_properties().items():
        full = PREFIX + name
        try:
            values[name] = _plain(getattr(scene, full))
        except AttributeError:
            values[name] = _plain(_rna_default(prop))
    return Settings(values, clauses_from_scene(scene))
