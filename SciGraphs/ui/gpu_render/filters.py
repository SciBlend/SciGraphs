# The filter stack: invertible (channel, range) slots ANDed together. A channel
# is a named scalar per node or edge, normalized to [0, 1] so every slider means
# the same thing. Cached: coreness is 23 ms at 21k nodes, 768 ms at 2M.

import numpy as np

from .attributes import (normalized_values, read_edge_scalar,  # noqa: F401
                         read_edge_value_channel, read_value_channel)

# CHANNEL_ITEMS is built at module level, so a bottom import would be a NameError.
from ...core.render.filters import (  # noqa: F401
    BETWEEN_BUDGET, BETWEEN_SAMPLES, CHANNEL_GROUPS, CHANNEL_LABELS,
    CROWDING_K, EDGE_CHANNELS, FLAG_CHANNELS, IGRAPH_AVAILABLE, NEEDS_GROUPS,
    NEEDS_SEEDS, NEEDS_WEIGHT, NODE_CHANNELS, PRODUCERS, SCIPY_AVAILABLE,
    SEED_ATTR, ChannelContext,
    _adjacency, _endpoint_intersections, _igraph_of, _peel_core,
    _propagate_labels, articulation_flags, betweenness_sample_size,
    betweenness_sampled, bridge_flags, channel_values,
    clustering_coefficients, component_sizes, core_numbers, crowding,
    disparity_significance, group_crossings, hop_distances,
    module_roles_from_groups, neighborhood_overlap, pagerank_scores,
    rich_club_channel, slot_mask, strength_channel, thinning_ranks,
    triangle_support,
)


def _settings(scene, st=None):
    # Keep this import inside the function; at module level it pulls in Blender.
    if st is not None:
        return st
    from .host import settings_from_scene
    return settings_from_scene(scene)


# Blender draws a heading from an item whose identifier is empty.
CHANNEL_ITEMS = []
for _group, _entries in CHANNEL_GROUPS:
    if CHANNEL_ITEMS:
        CHANNEL_ITEMS.append(None)
    CHANNEL_ITEMS.append(('', _group, ""))
    CHANNEL_ITEMS.extend((e[0], e[1], e[3]) for e in _entries)

# Seed count for the panel; SEED_ATTR holds the truth.
SEED_COUNT = "scigraphs_seed_count"

_CHANNEL_CACHE = {}
_CHANNEL_CACHE_MAX = 12


def edge_spans_groups(mesh, edges, num_nodes, group_attr):
    if not group_attr or edges is None or edges.size == 0:
        return None
    values, _, _ = read_value_channel(mesh, group_attr, num_nodes)
    return group_crossings(values, edges)


def module_roles(mesh, edges, num_nodes, group_attr):
    if not group_attr or edges is None or edges.size == 0:
        return None, None
    values, _, _ = read_value_channel(mesh, group_attr, num_nodes)
    return module_roles_from_groups(values, edges, num_nodes)


def read_seeds(mesh, name=SEED_ATTR):
    attr = mesh.attributes.get(name)
    if attr is None or attr.domain != 'POINT' \
            or attr.data_type not in ('BOOLEAN', 'INT', 'FLOAT'):
        return None
    count = len(mesh.vertices)
    if len(attr.data) != count or count == 0:
        return None
    if attr.data_type == 'BOOLEAN':
        raw = np.empty(count, dtype=bool)
    elif attr.data_type == 'INT':
        raw = np.empty(count, dtype=np.int32)
    else:
        raw = np.empty(count, dtype=np.float32)
    try:
        attr.data.foreach_get("value", raw)
    except (RuntimeError, TypeError):
        return None
    mask = raw.astype(bool)
    return mask if mask.any() else None


def _topology_key(obj):
    """Not ``batches.content_signature``, which moves with color and display."""
    mesh = obj.data
    return (obj.name, len(mesh.vertices), len(mesh.edges),
            int(obj.get("scigraphs_preview_epoch", 0)))


def channel(obj, scene, name, attr_name=""):
    """``(values, native_min, native_max, domain)`` for one channel, cached."""
    key = (_topology_key(obj), name, attr_name)
    hit = _CHANNEL_CACHE.get(key)
    if hit is not None:
        return hit

    from .host import BlenderGraphSource
    mesh = obj.data
    st = _settings(scene)
    result = channel_values(
        BlenderGraphSource(obj), st, name, attr_name,
        group_attr=_default_group_attr(mesh),
        weight_attr=_default_weight_attr(mesh),
    )

    if len(_CHANNEL_CACHE) >= _CHANNEL_CACHE_MAX:
        _CHANNEL_CACHE.pop(next(iter(_CHANNEL_CACHE)))
    _CHANNEL_CACHE[key] = result
    return result


def _normalize(values):
    """Min-max to [0, 1]. The engine test checks ``channel_normalized`` against
    this. A constant channel becomes ones; zeros emptied the viewport."""
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.ones(values.size, dtype=np.float32)
    lo = float(values[finite].min())
    hi = float(values[finite].max())
    if hi <= lo:
        return np.ones(values.size, dtype=np.float32)
    out = (np.where(finite, values, lo) - lo) / (hi - lo)
    return out.astype(np.float32)


def _default_group_attr(mesh):
    """First plausible community attribute, so SPAN needs no setup."""
    for name in ("community", "cluster_id", "group", "conference"):
        if name in mesh.attributes:
            return name
    return ""


def _default_weight_attr(mesh):
    """First plausible EDGE weight, so weighted channels need no setup."""
    for name in ("weight", "weights", "value", "strength", "count",
                 "passengers", "flights"):
        attr = mesh.attributes.get(name)
        if attr is not None and attr.domain == 'EDGE' \
                and attr.data_type in ('FLOAT', 'INT'):
            return name
    return ""


def drop_channel_cache(obj=None):
    if obj is None:
        _CHANNEL_CACHE.clear()
        return
    name = obj.name
    for key in [k for k in _CHANNEL_CACHE if k[0][0] == name]:
        del _CHANNEL_CACHE[key]


def active_slots(scene):
    slots = getattr(scene, "scigraphs_filters", None)
    if not slots:
        return []
    return [s for s in slots if not s.mute
            and (s.invert or s.range_min > 0.0 or s.range_max < 1.0)]


def slot_signature(scene):
    """What the batch cache keys on to notice a stack change."""
    return tuple(
        (s.channel, s.attr_name, round(float(s.range_min), 5),
         round(float(s.range_max), 5), bool(s.invert))
        for s in active_slots(scene))


_slot_mask = slot_mask


def masks(obj, scene):
    """``(node_mask, edge_mask, active)``; a removed node takes its edges."""
    mesh = obj.data
    num_nodes = len(mesh.vertices)
    num_edges = len(mesh.edges)
    node_mask = np.ones(num_nodes, dtype=bool)
    edge_mask = np.ones(num_edges, dtype=bool)

    slots = active_slots(scene)
    legacy = legacy_slot_values(scene)
    if not slots and legacy is None:
        return node_mask, edge_mask, False

    active = False
    if legacy is not None:
        values = channel(obj, scene, 'ATTR', legacy[0])[0]
        if values is not None:
            node_mask &= (values >= legacy[1]) & (values <= legacy[2])
            active = True

    for slot in slots:
        values, _, _, domain = channel(obj, scene, slot.channel, slot.attr_name)
        if values is None:
            continue
        keep = _slot_mask(values, slot)
        if domain == 'EDGE':
            if keep.size == num_edges:
                edge_mask &= keep
                active = True
        elif keep.size == num_nodes:
            node_mask &= keep
            active = True

    # Dangling edges drop once at the end, after every node slot has voted.
    if num_edges and not node_mask.all():
        from . import geometry
        edges = geometry.extract_edges(mesh)
        if edges is not None:
            edge_mask &= node_mask[edges[:, 0]] & node_mask[edges[:, 1]]

    return node_mask, edge_mask, active


def legacy_slot_values(scene, st=None):
    """``(attr_name, min, max)`` for the pre-stack filter, or None if off. The
    panel dropped it, but the showcase catalog still writes it."""
    st = _settings(scene, st)
    if not bool(st.filter_enabled):
        return None
    return (st.attr_name, float(st.filter_min), float(st.filter_max))


def describe(obj, scene):
    rows = []
    legacy = legacy_slot_values(scene)
    if legacy is not None:
        rows.append(f"{legacy[0] or 'attribute'}: "
                    f"{legacy[1]:.2f} - {legacy[2]:.2f} (from Filter by "
                    f"Attribute)")
    return rows
