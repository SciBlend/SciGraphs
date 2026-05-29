"""
Edge Styles Module for SciGraphs

This module provides professional edge styling capabilities similar to Gephi and Cytoscape,
including curved edges, edge bundling, tapered edges, and various geometric styles.

Key concepts:
- Nodes with is_intersection=1 are the actual graph nodes (shown as spheres)
- Nodes with is_intersection=0 are intermediate control points for curved edges
- Edge styles modify the mesh by adding intermediate vertices along edges
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict

from ...utils.logger import log


# =============================================================================
# PRESETS CONFIGURATION
# =============================================================================

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
    """
    Apply a preset configuration to the scene properties.
    
    Args:
        props: SciGraphsProperties instance
        preset_name: Name of the preset to apply
        
    Returns:
        True if preset was applied successfully
    """
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


# =============================================================================
# BEZIER CURVE UTILITIES
# =============================================================================

def quadratic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, 
                     num_points: int) -> List[np.ndarray]:
    """
    Generate points along a quadratic Bezier curve.
    
    Args:
        p0: Start point
        p1: Control point
        p2: End point
        num_points: Number of points to generate (including endpoints)
        
    Returns:
        List of points along the curve
    """
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        # B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
        point = (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
        points.append(point)
    return points


def cubic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, 
                 p3: np.ndarray, num_points: int) -> List[np.ndarray]:
    """
    Generate points along a cubic Bezier curve.
    
    Args:
        p0: Start point
        p1: First control point
        p2: Second control point
        p3: End point
        num_points: Number of points to generate (including endpoints)
        
    Returns:
        List of points along the curve
    """
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        # B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
        point = ((1 - t)**3 * p0 + 
                 3 * (1 - t)**2 * t * p1 + 
                 3 * (1 - t) * t**2 * p2 + 
                 t**3 * p3)
        points.append(point)
    return points


def compute_perpendicular_offset(p0: np.ndarray, p1: np.ndarray, 
                                  offset: float, direction: str = 'AUTO',
                                  edge_index: int = 0) -> np.ndarray:
    """
    Compute a perpendicular offset vector for curve control points.
    
    Args:
        p0: Start point
        p1: End point
        offset: Magnitude of offset
        direction: 'AUTO', 'CLOCKWISE', 'COUNTER_CLOCKWISE', 'ALTERNATING'
        edge_index: Edge index (used for ALTERNATING mode)
        
    Returns:
        Offset vector perpendicular to the edge
    """
    # Direction vector
    d = p1 - p0
    length = np.linalg.norm(d)
    
    if length < 1e-10:
        return np.zeros(3)
    
    d = d / length
    
    # Perpendicular in XY plane (for 2D-like graphs)
    # Use cross product with Z-axis
    up = np.array([0.0, 0.0, 1.0])
    perp = np.cross(d, up)
    perp_len = np.linalg.norm(perp)
    
    if perp_len < 1e-10:
        # Edge is vertical, use X-axis as reference
        perp = np.cross(d, np.array([1.0, 0.0, 0.0]))
        perp_len = np.linalg.norm(perp)
    
    if perp_len > 1e-10:
        perp = perp / perp_len
    else:
        perp = np.array([1.0, 0.0, 0.0])
    
    # Determine sign based on direction mode
    if direction == 'CLOCKWISE':
        sign = 1.0
    elif direction == 'COUNTER_CLOCKWISE':
        sign = -1.0
    elif direction == 'ALTERNATING':
        sign = 1.0 if edge_index % 2 == 0 else -1.0
    else:  # AUTO
        # Use deterministic sign based on node positions
        sign = 1.0 if (p0[0] + p0[1]) > (p1[0] + p1[1]) else -1.0
    
    return perp * offset * sign


# =============================================================================
# EDGE STYLE GENERATORS
# =============================================================================

def generate_straight_edge(p0: np.ndarray, p1: np.ndarray, 
                           segments: int = 1) -> List[np.ndarray]:
    """
    Generate a straight edge (with optional intermediate points for consistency).
    
    Args:
        p0: Start point
        p1: End point
        segments: Number of segments
        
    Returns:
        List of points along the edge (excluding endpoints)
    """
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
    """
    Generate a curved edge using Bezier curves.
    
    Args:
        p0: Start point
        p1: End point
        curvature: Curvature intensity (0-1)
        segments: Number of segments
        direction: Curve direction
        edge_index: Edge index for alternating mode
        use_cubic: Use cubic Bezier (True) or quadratic (False)
        
    Returns:
        List of intermediate points (excluding endpoints)
    """
    if curvature < 0.001 or segments < 2:
        return generate_straight_edge(p0, p1, segments)
    
    # Calculate edge length and offset
    edge_length = np.linalg.norm(p1 - p0)
    offset = edge_length * curvature * 0.5
    
    # Midpoint
    mid = (p0 + p1) / 2
    
    # Perpendicular offset for control point
    perp_offset = compute_perpendicular_offset(p0, p1, offset, direction, edge_index)
    
    if use_cubic:
        # Cubic Bezier with two control points
        ctrl1 = p0 + (p1 - p0) * 0.25 + perp_offset * 0.5
        ctrl2 = p0 + (p1 - p0) * 0.75 + perp_offset * 0.5
        all_points = cubic_bezier(p0, ctrl1, ctrl2, p1, segments + 1)
    else:
        # Quadratic Bezier with one control point at midpoint
        ctrl = mid + perp_offset
        all_points = quadratic_bezier(p0, ctrl, p1, segments + 1)
    
    # Return only intermediate points (exclude first and last)
    return all_points[1:-1]


def generate_arc_edge(p0: np.ndarray, p1: np.ndarray,
                      curvature: float, segments: int,
                      direction: str = 'AUTO',
                      edge_index: int = 0) -> List[np.ndarray]:
    """
    Generate a circular arc edge.
    
    Args:
        p0: Start point
        p1: End point
        curvature: Arc curvature (determines radius)
        segments: Number of segments
        direction: Arc direction
        edge_index: Edge index for alternating mode
        
    Returns:
        List of intermediate points
    """
    if curvature < 0.001 or segments < 2:
        return generate_straight_edge(p0, p1, segments)
    
    # Calculate chord length and arc properties
    chord = p1 - p0
    chord_length = np.linalg.norm(chord)
    
    if chord_length < 1e-10:
        return []
    
    # Height of arc (sagitta)
    sagitta = chord_length * curvature * 0.5
    
    # Calculate radius from chord and sagitta
    # r = (c²/8h) + h/2 where c = chord length, h = sagitta
    if sagitta > 0.001:
        radius = (chord_length**2 / (8 * sagitta)) + (sagitta / 2)
    else:
        return generate_straight_edge(p0, p1, segments)
    
    # Find center of arc
    mid = (p0 + p1) / 2
    perp = compute_perpendicular_offset(p0, p1, 1.0, direction, edge_index)
    
    # Distance from midpoint to center
    dist_to_center = radius - sagitta
    center = mid - perp * dist_to_center
    
    # Generate arc points
    v0 = p0 - center
    v1 = p1 - center
    
    # Calculate angle
    angle = np.arccos(np.clip(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)), -1, 1))
    
    # Generate points along arc
    points = []
    for i in range(1, segments):
        t = i / segments
        # Slerp-like interpolation
        theta = angle * t
        
        # Rotate v0 around the perpendicular axis
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Simplified rotation in the plane defined by v0 and v1
        v0_norm = v0 / np.linalg.norm(v0)
        v1_norm = v1 / np.linalg.norm(v1)
        
        # Interpolate direction
        v_interp = v0_norm * (1 - t) + v1_norm * t
        v_interp = v_interp / np.linalg.norm(v_interp) * radius
        
        point = center + v_interp
        points.append(point)
    
    return points


def generate_orthogonal_edge(p0: np.ndarray, p1: np.ndarray,
                             style: str = 'CENTERED',
                             segments: int = 3) -> List[np.ndarray]:
    """
    Generate an orthogonal (right-angle) edge.
    
    Args:
        p0: Start point
        p1: End point
        style: 'HORIZONTAL_FIRST', 'VERTICAL_FIRST', 'SHORTEST', 'CENTERED'
        segments: Minimum segments (at least 3 for two bends)
        
    Returns:
        List of intermediate points
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dz = p1[2] - p0[2]
    
    points = []
    
    if style == 'CENTERED':
        # Two bends meeting at midpoint
        mid_x = (p0[0] + p1[0]) / 2
        mid_y = (p0[1] + p1[1]) / 2
        mid_z = (p0[2] + p1[2]) / 2
        
        # First bend: go to mid_x, stay at p0's y and z
        points.append(np.array([mid_x, p0[1], p0[2]]))
        
        # Second bend: at mid_x, go to p1's y
        points.append(np.array([mid_x, p1[1], mid_z]))
        
    elif style == 'HORIZONTAL_FIRST':
        # Go horizontal (X) first, then vertical (Y), then Z
        points.append(np.array([p1[0], p0[1], p0[2]]))
        if abs(dz) > 1e-6:
            points.append(np.array([p1[0], p1[1], p0[2]]))
            
    elif style == 'VERTICAL_FIRST':
        # Go vertical (Y) first, then horizontal (X), then Z
        points.append(np.array([p0[0], p1[1], p0[2]]))
        if abs(dz) > 1e-6:
            points.append(np.array([p0[0], p1[1], p1[2]]))
            
    elif style == 'SHORTEST':
        # Choose based on which results in shorter total path
        horiz_first_len = abs(dx) + abs(dy) + abs(dz)  # Same either way
        # Just use the simpler path based on dominant direction
        if abs(dx) > abs(dy):
            points.append(np.array([p1[0], p0[1], p0[2]]))
        else:
            points.append(np.array([p0[0], p1[1], p0[2]]))
    
    return points


def generate_self_loop(center: np.ndarray, radius: float,
                       segments: int = 12,
                       normal: np.ndarray = None) -> List[np.ndarray]:
    """
    Generate a self-loop (edge from node to itself).
    
    Args:
        center: Node position
        radius: Loop radius
        segments: Number of segments
        normal: Normal direction for loop plane (default: Z-up)
        
    Returns:
        List of points forming the loop (excluding connection point)
    """
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0])
    
    # Create orthonormal basis
    if abs(np.dot(normal, [1, 0, 0])) < 0.9:
        tangent = np.cross(normal, [1, 0, 0])
    else:
        tangent = np.cross(normal, [0, 1, 0])
    tangent = tangent / np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    
    # Generate circular loop
    points = []
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        offset = radius * (np.cos(angle) * tangent + np.sin(angle) * bitangent)
        # Offset the center slightly so loop doesn't overlap node
        loop_center = center + normal * radius * 0.5 + tangent * radius * 0.5
        points.append(loop_center + offset)
    
    return points


# =============================================================================
# EDGE BUNDLING (Force-Directed Edge Bundling - FDEB)
# =============================================================================

def edge_compatibility(e1_start: np.ndarray, e1_end: np.ndarray,
                       e2_start: np.ndarray, e2_end: np.ndarray) -> float:
    """
    Calculate compatibility score between two edges for bundling.
    Based on Holten & van Wijk's force-directed edge bundling.
    
    Returns value between 0 (incompatible) and 1 (very compatible).
    """
    # Angle compatibility
    d1 = e1_end - e1_start
    d2 = e2_end - e2_start
    
    len1 = np.linalg.norm(d1)
    len2 = np.linalg.norm(d2)
    
    if len1 < 1e-10 or len2 < 1e-10:
        return 0.0
    
    d1 = d1 / len1
    d2 = d2 / len2
    
    angle_compat = abs(np.dot(d1, d2))
    
    # Scale compatibility (similar length edges)
    l_avg = (len1 + len2) / 2
    scale_compat = 2 / (l_avg / min(len1, len2) + max(len1, len2) / l_avg)
    
    # Position compatibility (proximity of midpoints)
    mid1 = (e1_start + e1_end) / 2
    mid2 = (e2_start + e2_end) / 2
    mid_dist = np.linalg.norm(mid1 - mid2)
    pos_compat = l_avg / (l_avg + mid_dist)
    
    # Visibility compatibility (would edges cross if bundled?)
    # Simplified: check if one edge's midpoint is "visible" from the other
    vis_compat = 1.0  # Simplified for performance
    
    # Combined compatibility
    return angle_compat * scale_compat * pos_compat * vis_compat


def bundle_edges_fdeb(edges: List[Tuple[np.ndarray, np.ndarray]],
                      strength: float = 0.6,
                      iterations: int = 6,
                      segments: int = 10,
                      compatibility_threshold: float = 0.6) -> List[List[np.ndarray]]:
    """
    Apply force-directed edge bundling to a set of edges.
    
    Args:
        edges: List of (start, end) point tuples
        strength: Bundling strength (0-1)
        iterations: Number of iterations
        segments: Points per edge
        compatibility_threshold: Minimum compatibility to bundle
        
    Returns:
        List of point lists for each bundled edge
    """
    if not edges:
        return []
    
    num_edges = len(edges)
    
    # Initialize control points for each edge
    edge_points = []
    for start, end in edges:
        points = [start.copy()]
        for i in range(1, segments):
            t = i / segments
            points.append(start + (end - start) * t)
        points.append(end.copy())
        edge_points.append(points)
    
    # Precompute compatibility matrix
    compat_matrix = np.zeros((num_edges, num_edges))
    for i in range(num_edges):
        for j in range(i + 1, num_edges):
            compat = edge_compatibility(
                edges[i][0], edges[i][1],
                edges[j][0], edges[j][1]
            )
            compat_matrix[i, j] = compat
            compat_matrix[j, i] = compat
    
    # Iterative bundling
    step_size = 0.1 * strength
    
    for iteration in range(iterations):
        # Decrease step size over iterations
        current_step = step_size * (1 - iteration / iterations)
        
        # For each edge
        for i in range(num_edges):
            # For each internal point (not endpoints)
            for p in range(1, segments):
                force = np.zeros(3)
                
                # Spring force (keep points evenly spaced along edge)
                prev_point = edge_points[i][p - 1]
                next_point = edge_points[i][p + 1]
                spring_force = (prev_point + next_point) / 2 - edge_points[i][p]
                force += spring_force * 0.5
                
                # Attraction to compatible edges
                for j in range(num_edges):
                    if i == j:
                        continue
                    
                    compat = compat_matrix[i, j]
                    if compat < compatibility_threshold:
                        continue
                    
                    # Attract to corresponding point on edge j
                    other_point = edge_points[j][p]
                    attraction = (other_point - edge_points[i][p]) * compat
                    force += attraction * strength
                
                # Apply force
                edge_points[i][p] += force * current_step
    
    # Return intermediate points only (excluding endpoints)
    result = []
    for points in edge_points:
        result.append(points[1:-1])
    
    return result


# =============================================================================
# PARALLEL EDGE HANDLING
# =============================================================================

def identify_parallel_edges(edges: List[Tuple[int, int]]) -> Dict[Tuple[int, int], List[int]]:
    """
    Identify edges between the same pair of nodes.
    
    Args:
        edges: List of (source, target) node index pairs
        
    Returns:
        Dictionary mapping (min_node, max_node) -> list of edge indices
    """
    parallel_groups = defaultdict(list)
    
    for idx, (src, tgt) in enumerate(edges):
        key = (min(src, tgt), max(src, tgt))
        parallel_groups[key].append(idx)
    
    return parallel_groups


def offset_parallel_edge(p0: np.ndarray, p1: np.ndarray,
                         edge_num: int, total_parallel: int,
                         base_offset: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate offset for a parallel edge.
    
    Args:
        p0: Start point
        p1: End point
        edge_num: Index of this edge among parallel edges (0-based)
        total_parallel: Total number of parallel edges
        base_offset: Base offset distance
        
    Returns:
        Tuple of (offset_p0, offset_p1) - the offset endpoints
    """
    if total_parallel <= 1:
        return p0, p1
    
    # Calculate offset: center the group of parallel edges
    offset_range = base_offset * (total_parallel - 1)
    offset_amount = -offset_range / 2 + edge_num * base_offset
    
    # Get perpendicular direction
    perp = compute_perpendicular_offset(p0, p1, offset_amount, 'CLOCKWISE')
    
    return p0 + perp, p1 + perp


# =============================================================================
# MAIN EDGE STYLE APPLICATION
# =============================================================================

def compute_styled_edge_points(p0: np.ndarray, p1: np.ndarray,
                               style_type: str,
                               curvature: float = 0.3,
                               segments: int = 8,
                               direction: str = 'AUTO',
                               edge_index: int = 0,
                               orthogonal_style: str = 'CENTERED',
                               self_loop_radius: float = 0.2) -> List[np.ndarray]:
    """
    Compute intermediate points for a styled edge.
    
    Args:
        p0: Start point (numpy array)
        p1: End point (numpy array)
        style_type: 'STRAIGHT', 'CURVED', 'QUADRATIC', 'ARC', 'ORTHOGONAL'
        curvature: Curvature intensity
        segments: Number of segments
        direction: Curve direction
        edge_index: Edge index for alternating styles
        orthogonal_style: Style for orthogonal edges
        self_loop_radius: Radius for self-loops
        
    Returns:
        List of intermediate points (is_intersection=0)
    """
    # Check for self-loop
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
        # Tapered uses same geometry as curved, thickness varies in geometry nodes
        return generate_curved_edge(p0, p1, curvature, segments,
                                    direction, edge_index, use_cubic=False)
    
    elif style_type == 'BUNDLED':
        # Bundling requires all edges at once, handled separately
        return generate_curved_edge(p0, p1, curvature, segments,
                                    direction, edge_index, use_cubic=True)
    
    else:
        log(f"Unknown edge style: {style_type}, using straight")
        return generate_straight_edge(p0, p1, segments)


def get_style_params_from_props(props) -> Dict[str, Any]:
    """
    Extract edge style parameters from Blender properties.
    
    Args:
        props: SciGraphsProperties instance
        
    Returns:
        Dictionary of style parameters
    """
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
