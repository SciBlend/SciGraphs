"""Accessibility operators: isochrones, ego subgraph, network-constrained DBSCAN."""

import bpy

from ....core.osmnx import (
    add_travel_time_from_speed,
    ego_subgraph,
    make_iso_polygons,
    network_dbscan,
)
from ....core.osmnx.metadata import get_graph_extent
from .utils import _get_osmnx_graph, _get_unprojected_graph, _store_osmnx_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_routing_graph(obj):
    G = _get_osmnx_graph(obj)
    G_un = _get_unprojected_graph(obj)
    return G or G_un


def _parse_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_trip_times_csv(text):
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except ValueError:
            continue
    return out


def _resolve_iso_center(props, obj, G):
    """Pick the center node for isochrones/ego graph."""
    candidates = (
        props.osmnx_iso_center_node.strip(),
        props.osmnx_selected_node_id.strip(),
        str(obj.get("osmnx_last_selected_node", "")).strip(),
    )
    has_candidate = any(candidates)
    for candidate in candidates:
        node_id = _parse_int(candidate)
        if node_id is not None and node_id in G.nodes:
            return node_id

    if has_candidate:
        return None

    # Fallback: first node.
    try:
        return next(iter(G.nodes))
    except StopIteration:
        return None


# ---------------------------------------------------------------------------
# Isochrone generation
# ---------------------------------------------------------------------------

class SCIGRAPHS_OT_OSMnxIsochrones(bpy.types.Operator):
    """Generate concentric isochrone polygons from a center node."""
    bl_idname = "scigraphs.osmnx_isochrones"
    bl_label = "Generate Isochrones"
    bl_description = "Create Blender meshes for isochrone polygons around the selected node"
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

        center = _resolve_iso_center(props, obj, G)
        if center is None:
            self.report({'ERROR'}, "No valid center node available. Pick a node from the current graph.")
            return {'CANCELLED'}

        times = _parse_trip_times_csv(props.osmnx_iso_trip_times)
        if not times:
            self.report({'ERROR'}, "Provide at least one trip time (minutes)")
            return {'CANCELLED'}

        added = add_travel_time_from_speed(G, props.osmnx_iso_travel_speed)
        if added:
            self.report({'INFO'}, f"Imputed travel_time on {added} edges @ {props.osmnx_iso_travel_speed} km/h")

        isos = make_iso_polygons(
            G,
            center_node=center,
            trip_times_minutes=times,
            travel_speed_kph=props.osmnx_iso_travel_speed,
            mode=props.osmnx_iso_mode,
            buffer_m=props.osmnx_iso_buffer,
        )
        if not isos:
            self.report({'ERROR'}, "Failed to build isochrone polygons")
            return {'CANCELLED'}

        extent = get_graph_extent(G) or {"center_lat": 0.0, "center_lon": 0.0}
        center_lat = extent["center_lat"]
        center_lon = extent["center_lon"]
        scale = obj.get("osmnx_scale", 0.001)

        EARTH_RADIUS = 6371000.0
        import math
        cos_lat = math.cos(math.radians(center_lat))
        mpd = math.pi / 180.0 * EARTH_RADIUS

        def lonlat_to_blender(lon, lat):
            x = (lon - center_lon) * mpd * cos_lat
            y = (lat - center_lat) * mpd
            return (x * scale, y * scale, 0.0)

        import bmesh

        collection = bpy.data.collections.new(f"OSMnx_isochrones_{center}")
        context.scene.collection.children.link(collection)

        created = 0
        for entry in isos:
            t = entry["time"]
            poly = entry["polygon"]
            if poly is None or poly.is_empty:
                continue

            # Handle MultiPolygon by iterating geoms.
            polys = getattr(poly, "geoms", None) or [poly]

            mesh_data = bpy.data.meshes.new(f"iso_{t}min")
            bm = bmesh.new()
            for p in polys:
                try:
                    exterior = list(p.exterior.coords)
                except AttributeError:
                    continue
                verts = [bm.verts.new(lonlat_to_blender(x, y)) for x, y in exterior]
                if len(verts) >= 3:
                    try:
                        bm.faces.new(verts)
                    except ValueError:
                        pass
            bm.normal_update()
            bm.to_mesh(mesh_data)
            bm.free()

            iso_obj = bpy.data.objects.new(f"iso_{t}min", mesh_data)
            iso_obj["isochrone_minutes"] = t
            iso_obj["isochrone_center_node"] = str(center)
            collection.objects.link(iso_obj)
            created += 1

        if created == 0:
            self.report({'WARNING'}, "No isochrone polygons created")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Created {created} isochrone polygon(s) around node {center}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Ego subgraph extraction
# ---------------------------------------------------------------------------

class SCIGRAPHS_OT_OSMnxEgoSubgraph(bpy.types.Operator):
    """Keep only nodes reachable within a network distance from a center node."""
    bl_idname = "scigraphs.osmnx_ego_subgraph"
    bl_label = "Extract Ego Subgraph"
    bl_description = "Truncate the graph to nodes reachable within a distance from a center"
    bl_options = {'REGISTER', 'UNDO'}

    radius: bpy.props.FloatProperty(
        name="Radius (m)",
        description="Maximum network distance in meters",
        default=1500.0,
        min=50.0,
        max=100000.0,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object

        G = _pick_routing_graph(obj)
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}

        center = _resolve_iso_center(props, obj, G)
        if center is None:
            self.report({'ERROR'}, "No center node available")
            return {'CANCELLED'}

        sub = ego_subgraph(G, center, self.radius, distance_attr="length")
        if sub is None or sub.number_of_nodes() == 0:
            self.report({'ERROR'}, "Ego subgraph is empty")
            return {'CANCELLED'}

        _store_osmnx_graph(obj, sub)
        obj["num_nodes"] = sub.number_of_nodes()
        obj["num_edges"] = sub.number_of_edges()
        self.report(
            {'INFO'},
            f"Ego subgraph: {sub.number_of_nodes()} nodes within {self.radius:.0f}m of {center}",
        )
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Network-constrained DBSCAN
# ---------------------------------------------------------------------------

class SCIGRAPHS_OT_OSMnxNetworkDBSCAN(bpy.types.Operator):
    """Cluster graph nodes by network distance using DBSCAN."""
    bl_idname = "scigraphs.osmnx_network_dbscan"
    bl_label = "Network-Constrained DBSCAN"
    bl_description = "Cluster nodes by shortest-path distance (not Euclidean)"
    bl_options = {'REGISTER', 'UNDO'}

    eps: bpy.props.FloatProperty(
        name="Eps (m)",
        description="Neighborhood radius in network meters",
        default=300.0,
        min=10.0,
        max=100000.0,
    )

    min_samples: bpy.props.IntProperty(
        name="Min Samples",
        description="DBSCAN min_samples",
        default=5,
        min=2,
        max=1000,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        obj = context.active_object
        G = _pick_routing_graph(obj)
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}

        if G.number_of_nodes() > 2000:
            self.report(
                {'WARNING'},
                f"Graph has {G.number_of_nodes()} nodes; DBSCAN is O(n^2). May be slow.",
            )

        labels = network_dbscan(
            G, eps_meters=self.eps, min_samples=self.min_samples, weight="length",
        )
        if labels is None:
            self.report({'ERROR'}, "DBSCAN failed (missing scikit-learn?)")
            return {'CANCELLED'}

        # Write labels into the graph as node attribute and into mesh as vertex attribute.
        import networkx as nx
        for n, lbl in labels.items():
            G.nodes[n]["dbscan_cluster"] = int(lbl)

        nodes_str = obj.get("nodes_data", "")
        node_ids = nodes_str.split(",") if nodes_str else []
        try:
            mesh = obj.data
            attr_name = "dbscan_cluster"
            if attr_name in mesh.attributes:
                mesh.attributes.remove(mesh.attributes[attr_name])
            attr = mesh.attributes.new(name=attr_name, type='INT', domain='POINT')
            for i, nid in enumerate(node_ids[: len(mesh.vertices)]):
                try:
                    lbl = labels.get(int(nid), -1)
                except ValueError:
                    lbl = -1
                attr.data[i].value = int(lbl)
        except Exception:
            pass

        n_clusters = len(set(labels.values()) - {-1})
        n_noise = sum(1 for v in labels.values() if v == -1)
        obj["osmnx_dbscan_clusters"] = n_clusters
        obj["osmnx_dbscan_noise"] = n_noise
        self.report(
            {'INFO'},
            f"DBSCAN: {n_clusters} clusters, {n_noise} noise points",
        )
        return {'FINISHED'}


classes = [
    SCIGRAPHS_OT_OSMnxIsochrones,
    SCIGRAPHS_OT_OSMnxEgoSubgraph,
    SCIGRAPHS_OT_OSMnxNetworkDBSCAN,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
