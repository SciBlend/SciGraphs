"""Edge styling: curved, bundled, tapered and geometric edges. Styles add
intermediate vertices along each edge; `is_intersection=1` marks the real graph
nodes in the resulting mesh and 0 marks the control points shaping the curve."""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict

from scigraphs_core.logger import log


EDGE_STYLE_PRESETS = {
    'GEPHI_DEFAULT': {
        'edge_style_type': 'CURVED',
        'edge_curvature': 0.3,
        'edge_segments': 10,
        'edge_curve_direction': 'AUTO',
        'edge_parallel_offset': 0.05,
        'edge_auto_offset_parallel': True,
    },
    'CYTOSCAPE_BEZIER': {
        'edge_style_type': 'QUADRATIC',
        'edge_curvature': 0.5,
        'edge_segments': 8,
        'edge_curve_direction': 'AUTO',
        'edge_parallel_offset': 0.08,
        'edge_auto_offset_parallel': True,
    },
    'SCHEMATIC': {
        'edge_style_type': 'ORTHOGONAL',
        'edge_curvature': 0.0,
        'edge_segments': 3,
        'edge_curve_direction': 'AUTO',
        'edge_parallel_offset': 0.1,
        'edge_auto_offset_parallel': True,
        'edge_orthogonal_style': 'CENTERED',
    },
    'BUNDLED_DENSE': {
        'edge_style_type': 'BUNDLED',
        'edge_curvature': 0.7,
        'edge_segments': 12,
        'edge_bundle_strength': 0.8,
        'edge_bundle_iterations': 8,
        'edge_bundle_compatibility_threshold': 0.5,
    },
    'FLOW_DIAGRAM': {
        'edge_style_type': 'TAPERED',
        'edge_curvature': 0.2,
        'edge_segments': 6,
        'edge_taper_start': 1.0,
        'edge_taper_end': 0.3,
        'edge_parallel_offset': 0.05,
    },
    'MINIMAL': {
        'edge_style_type': 'STRAIGHT',
        'edge_curvature': 0.0,
        'edge_segments': 1,
        'edge_parallel_offset': 0.0,
        'edge_auto_offset_parallel': False,
    },
}


def apply_preset(props, preset_name: str) -> bool:
    """Copy a preset's values onto the scene properties. 'CUSTOM' is a no-op."""
    if preset_name not in EDGE_STYLE_PRESETS and preset_name != 'CUSTOM':
        log(f"Unknown preset: {preset_name}")
        return False
    
    if preset_name == 'CUSTOM':
        return True
    
    preset = EDGE_STYLE_PRESETS[preset_name]
    
    for key, value in preset.items():
        if hasattr(props, key):
            setattr(props, key, value)
    
    log(f"Applied edge style preset: {preset_name}")
    return True


def quadratic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, 
                     num_points: int) -> List[np.ndarray]:
    """Sample a quadratic Bezier with control point p1, endpoints included."""
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        point = (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
        points.append(point)
    return points


def cubic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, 
                 p3: np.ndarray, num_points: int) -> List[np.ndarray]:
    """Sample a cubic Bezier with control points p1 and p2, endpoints included."""
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        point = ((1 - t)**3 * p0 + 
                 3 * (1 - t)**2 * t * p1 + 
                 3 * (1 - t) * t**2 * p2 + 
                 t**3 * p3)
        points.append(point)
    return points


def compute_perpendicular_offset(p0: np.ndarray, p1: np.ndarray, 
                                  offset: float, direction: str = 'AUTO',
                                  edge_index: int = 0) -> np.ndarray:
    """Offset vector perpendicular to the edge p0 -> p1, for a control point.
    ``direction`` is AUTO, CLOCKWISE, COUNTER_CLOCKWISE or ALTERNATING; only
    ALTERNATING reads ``edge_index``, and AUTO bends an edge the same way twice."""
    d = p1 - p0
    length = np.linalg.norm(d)
    
    if length < 1e-10:
        return np.zeros(3)
    
    d = d / length
    
    # Perpendicular in the XY plane, which is what 2D-ish graphs want.
    up = np.array([0.0, 0.0, 1.0])
    perp = np.cross(d, up)
    perp_len = np.linalg.norm(perp)
    
    if perp_len < 1e-10:
        perp = np.cross(d, np.array([1.0, 0.0, 0.0]))
        perp_len = np.linalg.norm(perp)
    
    if perp_len > 1e-10:
        perp = perp / perp_len
    else:
        perp = np.array([1.0, 0.0, 0.0])
    
    if direction == 'CLOCKWISE':
        sign = 1.0
    elif direction == 'COUNTER_CLOCKWISE':
        sign = -1.0
    elif direction == 'ALTERNATING':
        sign = 1.0 if edge_index % 2 == 0 else -1.0
    else:  # AUTO
        sign = 1.0 if (p0[0] + p0[1]) > (p1[0] + p1[1]) else -1.0
    
    return perp * offset * sign


def generate_straight_edge(p0: np.ndarray, p1: np.ndarray, 
                           segments: int = 1) -> List[np.ndarray]:
    """Intermediate points along a straight edge; empty when segments <= 1."""
    if segments <= 1:
        return []
    
    points = []
    for i in range(1, segments):
        t = i / segments
        point = p0 + (p1 - p0) * t
        points.append(point)
    return points


def generate_curved_edge(p0: np.ndarray, p1: np.ndarray,
                         curvature: float, segments: int,
                         direction: str = 'AUTO',
                         edge_index: int = 0,
                         use_cubic: bool = True) -> List[np.ndarray]:
    """Bezier-curved edge, intermediate points only. ``curvature`` runs 0 to 1
    and scales the bend by edge length; below 0.001 the edge comes back
    straight. ``use_cubic`` picks a cubic over a quadratic bulging at midpoint."""
    if curvature < 0.001 or segments < 2:
        return generate_straight_edge(p0, p1, segments)

    edge_length = np.linalg.norm(p1 - p0)
    offset = edge_length * curvature * 0.5

    mid = (p0 + p1) / 2

    perp_offset = compute_perpendicular_offset(p0, p1, offset, direction, edge_index)

    if use_cubic:
        ctrl1 = p0 + (p1 - p0) * 0.25 + perp_offset * 0.5
        ctrl2 = p0 + (p1 - p0) * 0.75 + perp_offset * 0.5
        all_points = cubic_bezier(p0, ctrl1, ctrl2, p1, segments + 1)
    else:
        ctrl = mid + perp_offset
        all_points = quadratic_bezier(p0, ctrl, p1, segments + 1)

    return all_points[1:-1]


def generate_arc_edge(p0: np.ndarray, p1: np.ndarray,
                      curvature: float, segments: int,
                      direction: str = 'AUTO',
                      edge_index: int = 0) -> List[np.ndarray]:
    """Circular arc edge, intermediate points only. ``curvature`` sets the
    sagitta as a fraction of the chord; too flat an arc degrades to straight."""
    if curvature < 0.001 or segments < 2:
        return generate_straight_edge(p0, p1, segments)

    chord = p1 - p0
    chord_length = np.linalg.norm(chord)

    if chord_length < 1e-10:
        return []

    sagitta = chord_length * curvature * 0.5

    # r = (c**2 / 8h) + h/2, with c the chord length and h the sagitta.
    if sagitta > 0.001:
        radius = (chord_length**2 / (8 * sagitta)) + (sagitta / 2)
    else:
        return generate_straight_edge(p0, p1, segments)
    
    mid = (p0 + p1) / 2
    perp = compute_perpendicular_offset(p0, p1, 1.0, direction, edge_index)

    dist_to_center = radius - sagitta
    center = mid - perp * dist_to_center

    v0 = p0 - center
    v1 = p1 - center

    angle = np.arccos(np.clip(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)), -1, 1))

    points = []
    for i in range(1, segments):
        t = i / segments
        theta = angle * t

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Lerp and renormalize rather than slerp: arc spacing only roughly uniform.
        v0_norm = v0 / np.linalg.norm(v0)
        v1_norm = v1 / np.linalg.norm(v1)

        v_interp = v0_norm * (1 - t) + v1_norm * t
        v_interp = v_interp / np.linalg.norm(v_interp) * radius

        point = center + v_interp
        points.append(point)

    return points


def generate_orthogonal_edge(p0: np.ndarray, p1: np.ndarray,
                             style: str = 'CENTERED',
                             segments: int = 3) -> List[np.ndarray]:
    """Right-angle edge returning the bend points. ``style`` is CENTERED,
    HORIZONTAL_FIRST, VERTICAL_FIRST or SHORTEST; two bends need segments >= 3."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dz = p1[2] - p0[2]
    
    points = []
    
    if style == 'CENTERED':
        mid_x = (p0[0] + p1[0]) / 2
        mid_y = (p0[1] + p1[1]) / 2
        mid_z = (p0[2] + p1[2]) / 2

        points.append(np.array([mid_x, p0[1], p0[2]]))
        points.append(np.array([mid_x, p1[1], mid_z]))

    elif style == 'HORIZONTAL_FIRST':
        points.append(np.array([p1[0], p0[1], p0[2]]))
        if abs(dz) > 1e-6:
            points.append(np.array([p1[0], p1[1], p0[2]]))

    elif style == 'VERTICAL_FIRST':
        points.append(np.array([p0[0], p1[1], p0[2]]))
        if abs(dz) > 1e-6:
            points.append(np.array([p0[0], p1[1], p1[2]]))

    elif style == 'SHORTEST':
        # Both orders give the same total length, so pick by dominant axis.
        horiz_first_len = abs(dx) + abs(dy) + abs(dz)
        if abs(dx) > abs(dy):
            points.append(np.array([p1[0], p0[1], p0[2]]))
        else:
            points.append(np.array([p0[0], p1[1], p0[2]]))
    
    return points


def generate_self_loop(center: np.ndarray, radius: float,
                       segments: int = 12,
                       normal: np.ndarray = None) -> List[np.ndarray]:
    """Points forming a loop from a node back to itself; ``normal`` is the loop
    plane's normal and defaults to Z-up."""
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(normal, [1, 0, 0])) < 0.9:
        tangent = np.cross(normal, [1, 0, 0])
    else:
        tangent = np.cross(normal, [0, 1, 0])
    tangent = tangent / np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    
    points = []
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        offset = radius * (np.cos(angle) * tangent + np.sin(angle) * bitangent)
        # Push the loop off center so it does not sit on top of the node sphere.
        loop_center = center + normal * radius * 0.5 + tangent * radius * 0.5
        points.append(loop_center + offset)
    
    return points


def edge_compatibility(e1_start: np.ndarray, e1_end: np.ndarray,
                       e2_start: np.ndarray, e2_end: np.ndarray) -> float:
    """Bundling compatibility of two edges, 0 to 1, following Holten and van
    Wijk's force-directed edge bundling minus the visibility term."""
    d1 = e1_end - e1_start
    d2 = e2_end - e2_start
    
    len1 = np.linalg.norm(d1)
    len2 = np.linalg.norm(d2)
    
    if len1 < 1e-10 or len2 < 1e-10:
        return 0.0
    
    d1 = d1 / len1
    d2 = d2 / len2
    
    angle_compat = abs(np.dot(d1, d2))

    l_avg = (len1 + len2) / 2
    scale_compat = 2 / (l_avg / min(len1, len2) + max(len1, len2) / l_avg)

    mid1 = (e1_start + e1_end) / 2
    mid2 = (e2_start + e2_end) / 2
    mid_dist = np.linalg.norm(mid1 - mid2)
    pos_compat = l_avg / (l_avg + mid_dist)

    # Visibility compatibility pinned to 1.0; the crossing test is not worth it.
    vis_compat = 1.0

    return angle_compat * scale_compat * pos_compat * vis_compat


def bundle_edges_fdeb(edges: List[Tuple[np.ndarray, np.ndarray]],
                      strength: float = 0.6,
                      iterations: int = 6,
                      segments: int = 10,
                      compatibility_threshold: float = 0.6) -> List[List[np.ndarray]]:
    """Bundle (start, end) edges, returning intermediate points per edge. Pairs
    scoring below ``compatibility_threshold`` never attract. Quadratic in the
    edge count: the matrix is dense and the inner loop walks every other edge."""
    if not edges:
        return []

    num_edges = len(edges)

    edge_points = []
    for start, end in edges:
        points = [start.copy()]
        for i in range(1, segments):
            t = i / segments
            points.append(start + (end - start) * t)
        points.append(end.copy())
        edge_points.append(points)
    
    compat_matrix = np.zeros((num_edges, num_edges))
    for i in range(num_edges):
        for j in range(i + 1, num_edges):
            compat = edge_compatibility(
                edges[i][0], edges[i][1],
                edges[j][0], edges[j][1]
            )
            compat_matrix[i, j] = compat
            compat_matrix[j, i] = compat
    
    step_size = 0.1 * strength

    for iteration in range(iterations):
        current_step = step_size * (1 - iteration / iterations)

        for i in range(num_edges):
            for p in range(1, segments):
                force = np.zeros(3)

                prev_point = edge_points[i][p - 1]
                next_point = edge_points[i][p + 1]
                spring_force = (prev_point + next_point) / 2 - edge_points[i][p]
                force += spring_force * 0.5
                
                # Weighted mean, not sum: summing grows the gain with the
                # compatible-edge count until the update overshoots.
                attraction = np.zeros(3)
                total = 0.0
                for j in range(num_edges):
                    if i == j:
                        continue
                    
                    compat = compat_matrix[i, j]
                    if compat < compatibility_threshold:
                        continue
                    
                    other_point = edge_points[j][p]
                    attraction += (other_point - edge_points[i][p]) * compat
                    total += compat
                
                if total > 0.0:
                    force += attraction * (strength / total)

                edge_points[i][p] += force * current_step

    result = []
    for points in edge_points:
        result.append(points[1:-1])
    
    return result


def identify_parallel_edges(edges: List[Tuple[int, int]]) -> Dict[Tuple[int, int], List[int]]:
    """Group edge indices by node pair, keyed (lower, higher) so direction is ignored."""
    parallel_groups = defaultdict(list)
    
    for idx, (src, tgt) in enumerate(edges):
        key = (min(src, tgt), max(src, tgt))
        parallel_groups[key].append(idx)
    
    return parallel_groups


def offset_parallel_edge(p0: np.ndarray, p1: np.ndarray,
                         edge_num: int, total_parallel: int,
                         base_offset: float) -> Tuple[np.ndarray, np.ndarray]:
    """Shift one of several parallel edges sideways, returning new endpoints.
    ``edge_num`` is 0-based in the group, ``base_offset`` the neighbor spacing,
    and the fan comes out centered on the original edge."""
    if total_parallel <= 1:
        return p0, p1

    offset_range = base_offset * (total_parallel - 1)
    offset_amount = -offset_range / 2 + edge_num * base_offset

    perp = compute_perpendicular_offset(p0, p1, offset_amount, 'CLOCKWISE')
    
    return p0 + perp, p1 + perp


def compute_styled_edge_points(p0: np.ndarray, p1: np.ndarray,
                               style_type: str,
                               curvature: float = 0.3,
                               segments: int = 8,
                               direction: str = 'AUTO',
                               edge_index: int = 0,
                               orthogonal_style: str = 'CENTERED',
                               self_loop_radius: float = 0.2) -> List[np.ndarray]:
    """Intermediate points for one styled edge, all marked is_intersection=0.
    ``style_type`` is STRAIGHT, CURVED, QUADRATIC, ARC, ORTHOGONAL, TAPERED,
    BUNDLED or HIERARCHICAL; anything else logs and falls back to straight.
    Coincident endpoints become a self-loop."""
    if np.allclose(p0, p1, atol=1e-6):
        return generate_self_loop(p0, self_loop_radius, segments)
    
    if style_type == 'STRAIGHT':
        return generate_straight_edge(p0, p1, segments)
    
    elif style_type == 'CURVED':
        return generate_curved_edge(p0, p1, curvature, segments, 
                                    direction, edge_index, use_cubic=True)
    
    elif style_type == 'QUADRATIC':
        return generate_curved_edge(p0, p1, curvature, segments,
                                    direction, edge_index, use_cubic=False)
    
    elif style_type == 'ARC':
        return generate_arc_edge(p0, p1, curvature, segments,
                                 direction, edge_index)
    
    elif style_type == 'ORTHOGONAL':
        return generate_orthogonal_edge(p0, p1, orthogonal_style, segments)
    
    elif style_type == 'TAPERED':
        # Same geometry as curved; the thickness ramp happens in geometry nodes.
        return generate_curved_edge(p0, p1, curvature, segments,
                                    direction, edge_index, use_cubic=False)

    elif style_type == 'BUNDLED':
        # Real bundling needs every edge at once, so it runs elsewhere.
        return generate_curved_edge(p0, p1, curvature, segments,
                                    direction, edge_index, use_cubic=True)

    elif style_type == 'HIERARCHICAL':
        # Routing through the cluster tree needs the communities; GPU only.
        log("Hierarchical bundling is GPU-only, baking straight edges")
        return generate_straight_edge(p0, p1, segments)

    else:
        log(f"Unknown edge style: {style_type}, using straight")
        return generate_straight_edge(p0, p1, segments)


def get_style_params_from_props(props) -> Dict[str, Any]:
    """Pull the edge style settings off a props object into a plain dict."""
    return {
        'style_type': props.edge_style_type,
        'curvature': props.edge_curvature,
        'segments': props.edge_segments,
        'direction': props.edge_curve_direction,
        'orthogonal_style': props.edge_orthogonal_style,
        'self_loop_radius': props.edge_self_loop_radius,
        'parallel_offset': props.edge_parallel_offset,
        'auto_offset_parallel': props.edge_auto_offset_parallel,
        'bundle_strength': props.edge_bundle_strength,
        'bundle_iterations': props.edge_bundle_iterations,
        'bundle_compatibility_threshold': props.edge_bundle_compatibility_threshold,
        'taper_start': props.edge_taper_start,
        'taper_end': props.edge_taper_end,
        'preserve_osmnx': props.edge_style_preserve_osmnx,
    }
