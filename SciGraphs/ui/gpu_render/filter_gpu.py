# The filter stack as draw-time state. In numpy the predicate decides which
# vertices exist, so nudging a threshold rebuilds every batch: 33 ms at 31k
# nodes, 1.2 s at 2M. Here it is a texture and a UBO; a move costs one update.

import gpu
import numpy as np

from . import edge_styles_gpu, filters, shaders
def _settings(scene, st=None):
    if st is not None:
        return st
    from .host import settings_from_scene
    return settings_from_scene(scene)


SLOTS = 4

_CACHE = {}
_CACHE_MAX = 4

# The sampler and block must point somewhere valid even when unread.
_IDLE_TEX = None
_IDLE_UBO = None


def eligible(obj, scene, st=None):
    """``(ok, why)``: whether the GPU path agrees with numpy. ``is_intersection``
    is tested for presence, never read: 2M elements every draw is not free."""
    st = _settings(scene, st)
    if not filters.active_slots(scene) and filters.legacy_slot_values(scene) is None:
        return False, "no filter"
    if shaders.get_round_point_shader(filtered=True) is None:
        return False, "the backend rejected the filtered shaders"
    if st.volume_mode != 'OFF':
        return False, "the density field is built from the surviving nodes"
    if bool(st.lod_enabled):
        return False, "LOD subsamples the set it is given"
    if bool(st.use_blocks):
        return False, "the block budget counts the set it is given"
    if bool(st.coarsen):
        return False, "a supernode has no channel of its own"
    if bool(st.adaptive):
        return False, "the adaptive cut chooses its own level"
    if bool(st.render_id_pass):
        return False, "the ID pass is read back on the CPU"
    if edge_styles_gpu.enabled(scene):
        return False, "styled edges are tessellated from the surviving ones"
    if "is_intersection" in obj.data.attributes:
        return False, "a curve point is not a node and has no channel"

    node_slots, edge_slots = split_slots(scene)
    if len(node_slots) > SLOTS or len(edge_slots) > SLOTS:
        return False, f"more than {SLOTS} clauses in one domain"
    return True, ""


def split_slots(scene):
    node_slots, edge_slots = [], []
    legacy = filters.legacy_slot_values(scene)
    if legacy is not None:
        node_slots.append(('ATTR', legacy[0], legacy[1], legacy[2], False))
    for slot in filters.active_slots(scene):
        entry = (slot.channel, slot.attr_name, float(slot.range_min),
                 float(slot.range_max), bool(slot.invert))
        if slot.channel in filters.EDGE_CHANNELS:
            edge_slots.append(entry)
        else:
            node_slots.append(entry)
    return node_slots, edge_slots


def channel_signature(scene):
    """Channel identities, not thresholds, so moving one keeps the batch."""
    node_slots, edge_slots = split_slots(scene)
    return (tuple((s[0], s[1]) for s in node_slots),
            tuple((s[0], s[1]) for s in edge_slots))


def _pack(values_list, count):
    """A clause the graph cannot answer is ignored, not treated as rejecting."""
    out = np.zeros((count, 4), dtype=np.float32)
    avail = []
    for i, values in enumerate(values_list[:SLOTS]):
        ok = values is not None and values.size == count
        if ok:
            out[:, i] = values
        avail.append(ok)
    return out, tuple(avail)


def channel_rows(obj, scene):
    """Node rows first, then edge rows, so one sampler serves both halves."""
    mesh = obj.data
    num_nodes = len(mesh.vertices)
    num_edges = len(mesh.edges)
    node_slots, edge_slots = split_slots(scene)
    rows, avail_node = _pack(
        [filters.channel(obj, scene, s[0], s[1])[0] for s in node_slots],
        num_nodes)
    avail_edge = ()
    if num_edges:
        erows, avail_edge = _pack(
            [filters.channel(obj, scene, s[0], s[1])[0] for s in edge_slots],
            num_edges)
        rows = np.concatenate([rows, erows], axis=0)
    return rows, avail_node, avail_edge


def build(obj, scene):
    """``num_nodes`` is where the edge half of the texture starts."""
    mesh = obj.data
    num_nodes = len(mesh.vertices)
    num_edges = len(mesh.edges)

    key = (obj.name, num_nodes, num_edges,
           int(obj.get("scigraphs_preview_epoch", 0)), channel_signature(scene))
    if key in _CACHE:
        return _CACHE[key]

    rows, avail_node, avail_edge = channel_rows(obj, scene)
    try:
        tex = _texture(rows)
    except Exception:  # noqa: BLE001 - texture limits, out of VRAM
        return None

    result = {"tex": tex, "num_nodes": num_nodes,
              "avail_node": avail_node, "avail_edge": avail_edge}
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = result
    return result


def _texture(values):
    row = shaders.FILTER_TEX_ROW
    n = values.shape[0]
    height = max(1, int(np.ceil(n / row)))
    padded = np.zeros((height * row, 4), dtype=np.float32)
    padded[:n] = values
    return gpu.types.GPUTexture(
        (row, height), format='RGBA32F',
        data=gpu.types.Buffer('FLOAT', padded.size, padded.ravel()))


def drop_cache(obj=None):
    if obj is None:
        _CACHE.clear()
        return
    for key in [k for k in _CACHE if k[0] == obj.name]:
        del _CACHE[key]


def uniform_rows(scene, avail_node=(), avail_edge=()):
    """The 8 vec4 of the FilterStack UBO; float32 to match the numpy path's cast."""
    data = np.zeros((8, 4), dtype=np.float32)
    node_slots, edge_slots = split_slots(scene)
    for base, slots, avail in ((0, node_slots, avail_node),
                               (4, edge_slots, avail_edge)):
        for i, (_channel, _attr, low, high, invert) in enumerate(slots[:SLOTS]):
            if i >= len(avail) or not avail[i]:
                continue
            data[base + 0, i] = np.float32(low)
            data[base + 1, i] = np.float32(high)
            data[base + 2, i] = 1.0 if invert else 0.0
            data[base + 3, i] = 1.0
    return data.ravel()


def _uniform_buf(values):
    return gpu.types.GPUUniformBuf(
        gpu.types.Buffer('FLOAT', values.size, values))


def idle():
    global _IDLE_TEX, _IDLE_UBO
    if _IDLE_TEX is None:
        _IDLE_TEX = _texture(np.zeros((1, 4), dtype=np.float32))
    if _IDLE_UBO is None:
        _IDLE_UBO = _uniform_buf(np.zeros(32, dtype=np.float32))
    return _IDLE_TEX, _IDLE_UBO


def bind(shader, scene, packed):
    """The caller must keep the returned UBO referenced until the draw call."""
    if packed is None:
        tex, block = idle()
    else:
        tex = packed["tex"]
        block = _uniform_buf(uniform_rows(
            scene, packed["avail_node"], packed["avail_edge"]))
    shader.uniform_sampler("u_fchan", tex)
    shader.uniform_block("u_filter", block)
    return block


def node_frows(node_idx):
    """The node's own row twice: one ``scig_keep`` serves nodes and edges."""
    rows = np.asarray(node_idx, dtype=np.float32)
    out = np.empty((rows.size, 3), dtype=np.float32)
    out[:, 0] = rows
    out[:, 1] = rows
    out[:, 2] = -1.0
    return out


def edge_frows(edges, edge_idx, num_nodes, swap=False):
    """``swap`` flips endpoint order for a segment's second vertex; same verdict."""
    a, b = (1, 0) if swap else (0, 1)
    out = np.empty((edges.shape[0], 3), dtype=np.float32)
    out[:, 0] = edges[:, a]
    out[:, 1] = edges[:, b]
    out[:, 2] = np.asarray(edge_idx, dtype=np.float32) + num_nodes
    return out
