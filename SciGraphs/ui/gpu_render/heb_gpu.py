# Hierarchical bundling on the GPU. Control polygons go up instead of a
# tessellated segment soup: both endpoints, one center per level on each side
# and the ancestor, so at most 2*depth+3 points per edge against `segments + 1`
# samples. The numpy tessellator in edge_styles_gpu made 8.5M vertices and
# 239 MB of vertex buffer on a 213k-edge graph. Ribbons, self-loops and
# parallel-edge offsets are not covered; ``build`` returns None for them.

import numpy as np

from . import bundle_gpu

TEX_ROW = bundle_gpu.TEX_ROW


def _has_parallel(edges):
    return bundle_gpu.has_parallel(edges)


def check(params, edges, ctx):
    """Return (ok, reason); the one requirement is a cluster tree in ctx."""
    if params["style_type"] != 'HIERARCHICAL' or not ctx.get("hierarchy"):
        return False, "not hierarchical bundling"
    return True, ""


def usable(params, hierarchy, edges):
    """This mode's extra input is a positional argument, not a ``ctx``."""
    return bundle_gpu.usable('HIERARCHICAL', params, edges,
                             {"hierarchy": hierarchy})


def _control_polygons(coords, edges, levels, remove_lca):
    """(E, K, 3) padded control polygons and (E,) their real lengths."""
    depth = len(levels)
    a_idx, b_idx = edges[:, 0], edges[:, 1]
    anc_a = np.stack([lv["member_of"][a_idx] for lv in levels], axis=1)
    anc_b = np.stack([lv["member_of"][b_idx] for lv in levels], axis=1)
    same = anc_a == anc_b
    lca = np.where(same.any(axis=1), same.argmax(axis=1), depth)

    kmax = 2 * depth + 3
    ctrl = np.zeros((edges.shape[0], kmax, 3), dtype=np.float32)
    counts = np.zeros(edges.shape[0], dtype=np.int32)

    a = coords[a_idx].astype(np.float32)
    b = coords[b_idx].astype(np.float32)
    for level in np.unique(lca):
        rows = np.nonzero(lca == level)[0]
        up = [levels[l]["centers"][anc_a[rows, l]].astype(np.float32)
              for l in range(level)]
        down = [levels[l]["centers"][anc_b[rows, l]].astype(np.float32)
                for l in reversed(range(level))]
        shared = ([levels[level]["centers"][anc_a[rows, level]].astype(np.float32)]
                  if level < depth else [])
        chain = [a[rows]] + up + shared + down + [b[rows]]
        if remove_lca and shared and len(chain) > 3:
            del chain[len(up) + 1]
        stacked = np.stack(chain, axis=1)
        n = stacked.shape[1]
        ctrl[rows, :n] = stacked
        # Zeros would drag the bounding box bundle_gpu takes over the whole
        # of ``ctrl`` down to the origin, so repeat the last real point.
        ctrl[rows, n:] = stacked[:, -1:, :]
        counts[rows] = n
    return ctrl, counts, kmax


def produce(coords, edges, params, ctx):
    ctrl, counts, _kmax = _control_polygons(
        coords, edges, ctx["hierarchy"], bool(params["heb_remove_lca"]))
    return ctrl, counts


def build(coords, edges, params, hierarchy=None, node_colors=None,
          edge_color=None, edge_widths=None):
    return bundle_gpu.build(
        'HIERARCHICAL', coords, edges, params, ctx={"hierarchy": hierarchy},
        node_colors=node_colors, edge_color=edge_color, edge_widths=edge_widths)


def draw(data, mvp, beta, shader=None, width_range=None, viewport=None):
    return bundle_gpu.draw(data, mvp, beta, shader=shader,
                           width_range=width_range, viewport=viewport)
