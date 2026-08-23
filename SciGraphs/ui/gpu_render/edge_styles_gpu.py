# Edge styles tessellated at batch build time, not baked into the mesh, so a
# style change costs one batch rebuild. Wherever the mesh holds baked curve
# points (is_intersection == 0), ``recover_logical_edges`` runs first.

from functools import lru_cache

import numpy as np


def enabled(scene, st=None):
    from .host import settings_from_scene
    st = st if st is not None else settings_from_scene(scene)
    return (
        bool(st.edge_styles_gpu)
        and getattr(scene, "scigraphs", None) is not None
    )


def style_params(scene):
    p = scene.scigraphs
    return {
        "style_type": p.edge_style_type,
        "curvature": float(p.edge_curvature),
        "segments": int(p.edge_segments),
        "direction": p.edge_curve_direction,
        "orthogonal_style": p.edge_orthogonal_style,
        "self_loop_radius": float(p.edge_self_loop_radius),
        "parallel_offset": float(p.edge_parallel_offset),
        "auto_offset_parallel": bool(p.edge_auto_offset_parallel),
        "bundle_strength": float(p.edge_bundle_strength),
        "bundle_iterations": int(p.edge_bundle_iterations),
        "bundle_threshold": float(p.edge_bundle_compatibility_threshold),
        "heb_beta": float(p.edge_heb_beta),
        "heb_remove_lca": bool(p.edge_heb_remove_lca),
        "heb_fade_long": float(p.edge_heb_fade_long),
        "heb_gradient": p.edge_heb_gradient,
        "taper_start": float(p.edge_taper_start),
        "taper_end": float(p.edge_taper_end),
    }


def bundle_params(scene):
    """Kept apart from ``style_params``, whose keys the engine's tessellator
    checks against a fixed list and raises on."""
    p = scene.scigraphs
    out = style_params(scene)
    out.update({
        "bundle_turn_limit": float(p.edge_bundle_turn_limit),
        "bundle_adaptive_beta": float(p.edge_bundle_adaptive_beta),
        "bundle_density_opacity": float(p.edge_bundle_density_opacity),
        "bundle_company": float(p.edge_bundle_company),
        "fdeb_cycles": int(p.edge_fdeb_cycles),
        "fdeb_radius": float(p.edge_fdeb_radius),
        "fdeb_visibility": bool(p.edge_fdeb_visibility),
        "fdeb_threshold": float(p.edge_fdeb_threshold),
        # Spelling changes on purpose: the right side is the registered
        # Blender property, an RNA name in .blend files. It cannot change.
        "mingle_neighbors": int(p.edge_mingle_neighbours),
        "mingle_rounds": int(p.edge_mingle_rounds),
        "mingle_min_gain": float(p.edge_mingle_min_gain),
        "sbeb_resolution": int(p.edge_sbeb_resolution),
        "sbeb_iterations": int(p.edge_sbeb_iterations),
        "sbeb_clusters": int(p.edge_sbeb_clusters),
        "sbeb_threshold": float(p.edge_sbeb_threshold),
        "sbeb_attraction": float(p.edge_sbeb_attraction),
        "sbeb_smooth": int(p.edge_sbeb_smooth),
        "sbeb_recluster": bool(p.edge_sbeb_recluster),
        "routed_resolution": int(p.edge_routed_resolution),
        "routed_iterations": int(p.edge_routed_iterations),
        "routed_reinforce": float(p.edge_routed_reinforce),
        "routed_avoid_nodes": float(p.edge_routed_avoid_nodes),
    })
    # FDEB scores a pair on four terms where the CPU path scores on three.
    if out["style_type"] == 'FDEB':
        out["bundle_threshold"] = out["fdeb_threshold"]
    return out


def needs_hierarchy(scene):
    return (enabled(scene)
            and scene.scigraphs.edge_style_type == 'HIERARCHICAL')


def params_signature(scene, without=()):
    """``without`` names parameters the bundling path reads at draw time, not
    at bake time. Covers the full bundling set, not the engine's subset."""
    p = bundle_params(scene)
    return tuple(round(v, 5) if isinstance(v, float) else v
                 for k, v in p.items() if k not in without)


# Above BUNDLE_MAX_EDGES edges, FDEB's (E, E) compatibility matrix costs too
# much and BUNDLED falls back to CURVED. HEB_DEGREE is Holten's piecewise
# cubic, dropped lower when a hierarchy path is too short.
from ...core.render.edge_styles import (  # noqa: E402,F401
    BUNDLE_MAX_EDGES, HEB_DEGREE, HEB_SOURCE_RGB, HEB_TARGET_RGB,
    recover_logical_edges, tessellate,
)
