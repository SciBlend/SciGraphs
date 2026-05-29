from ...utils.logger import log
from .get_osmnx import get_osmnx


def get_basic_stats(G, area_km2=None):
    """
    Calculate basic network statistics.
    
    Args:
        G: OSMnx MultiDiGraph
        area_km2: Optional area in square kilometers for density calculations
    
    Returns:
        Dictionary with network statistics, or None on error
    """
    ox = get_osmnx()
    if ox is None or G is None:
        return None
    
    try:
        area_m2 = area_km2 * 1e6 if area_km2 else None
        
        stats = ox.basic_stats(G, area=area_m2)
        
        formatted_stats = {
            "n_nodes": stats.get("n", 0),
            "n_edges": stats.get("m", 0),
            "avg_degree": round(stats.get("k_avg", 0), 2),
            "total_length_km": round(stats.get("edge_length_total", 0) / 1000, 2),
            "avg_edge_length_m": round(stats.get("edge_length_avg", 0), 1),
            "circuity_avg": round(stats.get("circuity_avg", 0), 3) if stats.get("circuity_avg") else None,
            "street_segments_per_node": round(stats.get("streets_per_node_avg", 0), 2),
            "intersection_count": stats.get("intersection_count", 0),
            "dead_end_count": stats.get("dead_end_count", 0),
        }
        
        if area_km2:
            formatted_stats["node_density_per_km2"] = round(stats.get("node_density_km", 0), 2)
            formatted_stats["edge_density_per_km2"] = round(stats.get("edge_density_km", 0), 2)
            formatted_stats["street_density_km_per_km2"] = round(stats.get("street_density_km", 0), 2)
        
        log(f"Basic stats calculated: {formatted_stats['n_nodes']} nodes, {formatted_stats['n_edges']} edges")
        return formatted_stats
        
    except Exception as e:
        log(f"Error calculating basic stats: {e}")
        return None


def get_bearing_distribution(G, num_bins=36):
    """
    Get the distribution of edge bearings for orientation analysis.
    
    Args:
        G: OSMnx MultiDiGraph with bearings added
        num_bins: Number of bins for the histogram (default 36 = 10 degree bins)
    
    Returns:
        Dictionary with bearing distribution data, or None on error
    """
    ox = get_osmnx()
    if ox is None or G is None:
        return None
    
    try:
        import numpy as np
        
        bearings = []
        for u, v, data in G.edges(data=True):
            if "bearing" in data:
                bearings.append(data["bearing"])
        
        if not bearings:
            return None
        
        bearings = np.array(bearings)
        
        bin_edges = np.linspace(0, 360, num_bins + 1)
        hist, _ = np.histogram(bearings, bins=bin_edges)
        
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(num_bins)
        normalized_entropy = entropy / max_entropy
        
        mean_count = hist.mean()
        dominant_bins = np.where(hist > mean_count * 1.5)[0]
        dominant_bearings = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in dominant_bins]
        
        return {
            "histogram": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "entropy": round(entropy, 3),
            "normalized_entropy": round(normalized_entropy, 3),
            "dominant_bearings": [round(b, 1) for b in dominant_bearings],
            "num_edges_with_bearing": len(bearings),
        }
        
    except Exception as e:
        log(f"Error calculating bearing distribution: {e}")
        return None


def get_elevation_stats(G):
    """
    Get statistics about node elevations in the graph.
    
    Args:
        G: OSMnx MultiDiGraph with elevation on nodes
    
    Returns:
        Dictionary with elevation statistics, or None if no elevations
    """
    if G is None:
        return None
    
    try:
        import numpy as np
        
        elevations = []
        for n, data in G.nodes(data=True):
            if "elevation" in data:
                elevations.append(data["elevation"])
        
        if not elevations:
            return None
        
        elevations = np.array(elevations)
        
        stats = {
            "min_elevation": float(np.min(elevations)),
            "max_elevation": float(np.max(elevations)),
            "mean_elevation": float(np.mean(elevations)),
            "std_elevation": float(np.std(elevations)),
            "elevation_range": float(np.max(elevations) - np.min(elevations)),
            "nodes_with_elevation": len(elevations),
        }
        
        return stats
        
    except Exception as e:
        log(f"Error calculating elevation stats: {e}")
        return None


def get_grade_stats(G):
    """
    Get statistics about edge grades (slopes) in the graph.
    
    Args:
        G: OSMnx MultiDiGraph with grade on edges
    
    Returns:
        Dictionary with grade statistics, or None if no grades
    """
    if G is None:
        return None
    
    try:
        import numpy as np
        
        grades = []
        grades_abs = []
        for u, v, data in G.edges(data=True):
            if "grade" in data:
                grades.append(data["grade"])
            if "grade_abs" in data:
                grades_abs.append(data["grade_abs"])
        
        if not grades:
            return None
        
        grades = np.array(grades)
        
        stats = {
            "min_grade": float(np.min(grades)),
            "max_grade": float(np.max(grades)),
            "mean_grade": float(np.mean(grades)),
            "std_grade": float(np.std(grades)),
            "edges_with_grade": len(grades),
        }
        
        if grades_abs:
            grades_abs = np.array(grades_abs)
            stats["mean_grade_abs"] = float(np.mean(grades_abs))
            stats["max_grade_abs"] = float(np.max(grades_abs))
            steep_count = np.sum(grades_abs > 0.05)
            stats["steep_edge_pct"] = float(steep_count / len(grades_abs) * 100)
            very_steep_count = np.sum(grades_abs > 0.10)
            stats["very_steep_edge_pct"] = float(very_steep_count / len(grades_abs) * 100)
        
        return stats
        
    except Exception as e:
        log(f"Error calculating grade stats: {e}")
        return None


def circuity_avg(Gu):
    """
    Calculate average circuity of street network.
    
    Circuity is the ratio of network distance to straight-line distance.
    Higher values indicate more circuitous routes.
    
    Args:
        Gu: Undirected networkx.MultiGraph
    
    Returns:
        float: Average circuity value, or None on error
    """
    ox = get_osmnx()
    if ox is None or Gu is None:
        log("OSMnx not available or graph is None")
        return None
    
    if hasattr(ox, "stats") and hasattr(ox.stats, "circuity_avg"):
        circuity = ox.stats.circuity_avg(Gu)
        log(f"Circuity calculated: {circuity:.3f}")
        return circuity
    
    log("circuity_avg function not found in OSMnx")
    return None

