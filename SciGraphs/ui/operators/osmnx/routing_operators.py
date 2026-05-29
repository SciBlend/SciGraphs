"""Advanced routing operators for the OSMnx panel.

Provides:
    - K-shortest alternative paths.
    - Many-to-many batch routing (with optional random OD sampling).
    - Route-to-GDF summary (length, time, grade, rise).
    - Route elevation profile → creates a Blender curve.
"""

import bpy

from ....core.osmnx import (
    batch_shortest_paths,
    k_shortest_paths,
    route_elevation_profile,
    sample_random_od_pairs,
    summarize_route,
)
from .utils import (
    _get_osmnx_graph,
    _get_unprojected_graph,
    _mark_shortest_path_attributes,
)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _pick_routing_graph(obj):
    """Return the best available NetworkX graph for routing (projected preferred)."""
    G = _get_osmnx_graph(obj)
    G_un = _get_unprojected_graph(obj)
    return G or G_un


def _parse_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# K-shortest alternative routes
# ---------------------------------------------------------------------------

class SCIGRAPHS_OT_OSMnxKShortest(bpy.types.Operator):
    """Compute K alternative shortest paths between source and target."""
    bl_idname = "scigraphs.osmnx_k_shortest"
    bl_label = "K Shortest Paths"
    bl_description = "Find K alternative shortest paths between source and target"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or not obj.get("is_osmnx", False):
            return False
        props = context.scene.scigraphs
        return bool(props.osmnx_shortest_path_source and props.osmnx_shortest_path_target)

    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object

        G = _pick_routing_graph(obj)
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}

        src = _parse_int(props.osmnx_shortest_path_source)
        dst = _parse_int(props.osmnx_shortest_path_target)
        if src is None or dst is None:
            self.report({'ERROR'}, "Source/target must be integer node IDs")
            return {'CANCELLED'}
        if src not in G.nodes or dst not in G.nodes:
            self.report({'ERROR'}, "Source or target not in graph")
            return {'CANCELLED'}

        results = k_shortest_paths(
            G, src, dst,
            k=props.osmnx_k_shortest,
            weight=props.osmnx_path_weight,
            impedance_alpha=props.osmnx_impedance_alpha,
        )
        if not results:
            self.report({'ERROR'}, "No paths found")
            return {'CANCELLED'}

        obj["osmnx_k_shortest_count"] = len(results)
        obj["osmnx_k_shortest_km"] = [r["distance_km"] for r in results]

        # Mark the first (best) route on the mesh as the "current" path.
        first_path = results[0]["path"]
        _mark_shortest_path_attributes(obj, first_path)
        obj["osmnx_last_path"] = str(first_path)

        lengths = ", ".join(f"{r['distance_km']:.2f}km" for r in results)
        self.report({'INFO'}, f"Found {len(results)} routes: {lengths}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Many-to-many batch routing
# ---------------------------------------------------------------------------

class SCIGRAPHS_OT_OSMnxBatchRoutes(bpy.types.Operator):
    """Compute many-to-many routes between random origin-destination pairs."""
    bl_idname = "scigraphs.osmnx_batch_routes"
    bl_label = "Batch Routes (N random OD pairs)"
    bl_description = "Compute shortest paths for N random (origin, destination) pairs"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)

    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object

        G = _pick_routing_graph(obj)
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        if G.number_of_nodes() < 2:
            self.report({'ERROR'}, "Batch routing requires at least two graph nodes")
            return {'CANCELLED'}

        n = props.osmnx_od_random_n
        pairs = sample_random_od_pairs(G, n=n)
        if not pairs:
            self.report({'ERROR'}, "Failed to sample OD pairs")
            return {'CANCELLED'}

        origs = [o for o, _ in pairs]
        dests = [d for _, d in pairs]
        cpus = props.osmnx_od_batch_cpus or None

        result = batch_shortest_paths(
            G, origs, dests,
            weight=props.osmnx_path_weight,
            cpus=cpus,
            impedance_alpha=props.osmnx_impedance_alpha,
        )
        if result is None:
            self.report({'ERROR'}, "Batch routing failed")
            return {'CANCELLED'}

        s = result["summary"]
        obj["osmnx_batch_total"] = s["total_pairs"]
        obj["osmnx_batch_reached"] = s["reached"]
        obj["osmnx_batch_mean_dist_m"] = s["mean_distance_m"] or 0
        if s.get("mean_travel_time_s"):
            obj["osmnx_batch_mean_time_s"] = s["mean_travel_time_s"]

        self.report(
            {'INFO'},
            f"Batch: {s['reached']}/{s['total_pairs']} reached, "
            f"mean={s['mean_distance_m']:.0f}m"
        )
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Route summary (route_to_gdf aggregates)
# ---------------------------------------------------------------------------

class SCIGRAPHS_OT_OSMnxRouteSummary(bpy.types.Operator):
    """Summarise the last computed shortest path (length, time, grade, rise)."""
    bl_idname = "scigraphs.osmnx_route_summary"
    bl_label = "Summarize Last Route"
    bl_description = "Compute aggregate metrics for the last shortest path"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False) and obj.get("osmnx_last_path")

    def execute(self, context):
        obj = context.active_object

        G = _pick_routing_graph(obj)
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}

        path_str = obj.get("osmnx_last_path", "")
        if not path_str:
            self.report({'ERROR'}, "No last path stored; compute a shortest path first")
            return {'CANCELLED'}

        try:
            path = [int(x) for x in path_str.strip("[]").split(",") if x.strip()]
        except Exception:
            self.report({'ERROR'}, "Could not parse stored path")
            return {'CANCELLED'}

        summary = summarize_route(G, path)
        if summary is None:
            self.report({'ERROR'}, "Failed to summarize route")
            return {'CANCELLED'}

        obj["osmnx_route_length_km"] = summary["length_km"]
        obj["osmnx_route_time_min"] = summary["travel_time_min"]
        obj["osmnx_route_mean_grade_abs"] = summary["mean_grade_abs"]
        if "rise_m" in summary:
            obj["osmnx_route_rise_m"] = summary["rise_m"]

        msg = f"Length {summary['length_km']:.2f}km"
        if summary["travel_time_min"] > 0:
            msg += f", time {summary['travel_time_min']:.1f}min"
        if "rise_m" in summary:
            msg += f", rise {summary['rise_m']:.0f}m"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Route elevation profile (creates a Blender curve)
# ---------------------------------------------------------------------------

class SCIGRAPHS_OT_OSMnxRouteElevationProfile(bpy.types.Operator):
    """Generate a 2D curve showing elevation along the last route."""
    bl_idname = "scigraphs.osmnx_route_elev_profile"
    bl_label = "Generate Elevation Profile"
    bl_description = "Create a Blender curve with cumulative distance (X) vs. elevation (Y)"
    bl_options = {'REGISTER', 'UNDO'}

    profile_scale_x: bpy.props.FloatProperty(
        name="X Scale",
        description="Horizontal scale factor (m → Blender units)",
        default=0.001,
        min=1e-5,
        max=1.0,
        precision=5,
    )

    profile_scale_y: bpy.props.FloatProperty(
        name="Y Scale",
        description="Vertical exaggeration factor",
        default=0.01,
        min=1e-5,
        max=10.0,
        precision=5,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False) and obj.get("osmnx_last_path")

    def execute(self, context):
        obj = context.active_object

        G = _pick_routing_graph(obj)
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}

        path_str = obj.get("osmnx_last_path", "")
        try:
            path = [int(x) for x in path_str.strip("[]").split(",") if x.strip()]
        except Exception:
            self.report({'ERROR'}, "Could not parse stored path")
            return {'CANCELLED'}

        profile = route_elevation_profile(G, path)
        if not profile:
            self.report(
                {'ERROR'},
                "Route elevations unavailable — add node elevations first",
            )
            return {'CANCELLED'}

        curve_data = bpy.data.curves.new(name="OSMnx_elev_profile", type='CURVE')
        curve_data.dimensions = '3D'
        spline = curve_data.splines.new('POLY')
        spline.points.add(len(profile) - 1)

        min_elev = min(e for _, e in profile)
        for i, (d, e) in enumerate(profile):
            spline.points[i].co = (
                d * self.profile_scale_x,
                (e - min_elev) * self.profile_scale_y,
                0.0,
                1.0,
            )

        curve_obj = bpy.data.objects.new("OSMnx_elev_profile", curve_data)
        context.scene.collection.objects.link(curve_obj)
        curve_obj.data.bevel_depth = 0.005
        curve_obj.data.bevel_resolution = 2

        self.report({'INFO'}, f"Elevation profile: {len(profile)} samples")
        return {'FINISHED'}


classes = [
    SCIGRAPHS_OT_OSMnxKShortest,
    SCIGRAPHS_OT_OSMnxBatchRoutes,
    SCIGRAPHS_OT_OSMnxRouteSummary,
    SCIGRAPHS_OT_OSMnxRouteElevationProfile,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
