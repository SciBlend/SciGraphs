"""
Proximity graph operators for City2Graph.

Implements operators for generating single-layer and multi-layer proximity graphs
from OSM feature objects, with visualization support.
"""

import bpy
import bmesh
import time

from ....utils.blender_helpers import get_or_create_collection as _get_or_create_collection
from ....core.mesh.geo_mesh import (
    create_curves_from_gdf as _create_curves_from_gdf,
    create_nodes_mesh_from_gdf as _create_nodes_mesh_from_gdf,
    create_native_graph_from_gdfs as _create_native_graph_from_gdfs,
    create_native_heterograph_from_dicts as _create_native_heterograph_from_dicts,
    _resolve_projection_metadata,
)


def _effective_feature_count(obj):
    """Count the real features of a feature object from its mesh geometry.

    Mirrors how the geometry is interpreted downstream: one feature per face
    (polygons), per connected edge chain (lines), or per isolated vertex
    (points). Used as a fallback when the ``feature_count`` custom property is
    missing or zero.
    """
    if obj is None or obj.type != 'MESH' or not obj.data:
        return 0

    mesh = obj.data
    if len(mesh.polygons) > 0:
        return len(mesh.polygons)
    if len(mesh.edges) > 0:
        import bmesh

        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            bm.verts.ensure_lookup_table()
            parent = list(range(len(bm.verts)))

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            for edge in bm.edges:
                ra, rb = find(edge.verts[0].index), find(edge.verts[1].index)
                if ra != rb:
                    parent[ra] = rb

            connected = {find(v.index) for v in bm.verts if any(True for _ in v.link_edges)}
            isolated = sum(1 for v in bm.verts if not any(True for _ in v.link_edges))
            return len(connected) + isolated
        finally:
            bm.free()
    return len(mesh.vertices)


class SCIGRAPHS_OT_GenerateProximityGraph(bpy.types.Operator):
    """Generate single-layer proximity graph from OSM features."""
    bl_idname = "scigraphs.generate_proximity_graph"
    bl_label = "Generate Proximity Graph"
    bl_description = "Generate proximity graph from selected feature object"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from ....core.city2graph import proximity
        from ....utils.logger import log
        
        props = context.scene.city2graph
        feature_obj = props.prox_feature_object
        
        if not feature_obj:
            self.report({'ERROR'}, "No feature object selected")
            return {'CANCELLED'}
        
        if not feature_obj.get("is_osm_features") and not feature_obj.get("is_city2graph"):
            self.report({'ERROR'}, "Selected object is not a feature object (OSM or city2graph)")
            return {'CANCELLED'}
        
        # Validate feature count. ``feature_count`` may be missing or stale on
        # derived objects (e.g. centroids), so fall back to the real geometry.
        feature_count = feature_obj.get("feature_count", 0) or _effective_feature_count(feature_obj)
        if feature_count < 2:
            self.report({'ERROR'}, f"Need at least 2 features, found {feature_count}")
            return {'CANCELLED'}
        
        graph_type = props.prox_graph_type
        distance_metric = props.prox_distance_metric.lower()
        
        # Validate K parameter for KNN
        if graph_type == 'KNN' and props.prox_knn_k >= feature_count:
            self.report({'ERROR'}, f"K ({props.prox_knn_k}) must be less than feature count ({feature_count})")
            return {'CANCELLED'}
        
        network_obj = None
        if distance_metric == "network":
            network_obj = props.prox_network_object
            if not network_obj or not network_obj.get("is_osmnx"):
                self.report({'ERROR'}, "Network distance requires OSMnx street network")
                return {'CANCELLED'}
        
        log(f"Generating {graph_type} graph with metric={distance_metric}")
        log(f"  Deduplication: {props.prox_deduplicate}, tolerance={props.prox_dedup_tolerance}m")
        
        self.report({'INFO'}, f"Generating {graph_type} proximity graph...")
        start_time = time.time()
        
        try:
            # Common parameters for all graph types
            common_params = {
                'deduplicate': props.prox_deduplicate,
                'tolerance': props.prox_dedup_tolerance,
                'distance_metric': distance_metric,
                'network_obj': network_obj,
                'as_nx': False
            }
            
            if graph_type == 'KNN':
                nodes_gdf, edges_gdf = proximity.generate_knn_graph_from_features(
                    feature_obj,
                    k=props.prox_knn_k,
                    **common_params
                )
            elif graph_type == 'DELAUNAY':
                nodes_gdf, edges_gdf = proximity.generate_delaunay_graph_from_features(
                    feature_obj,
                    **common_params
                )
            elif graph_type == 'FIXED_RADIUS':
                nodes_gdf, edges_gdf = proximity.generate_fixed_radius_graph_from_features(
                    feature_obj,
                    radius=props.prox_radius,
                    **common_params
                )
            elif graph_type == 'WAXMAN':
                nodes_gdf, edges_gdf = proximity.generate_waxman_graph_from_features(
                    feature_obj,
                    beta=props.prox_waxman_beta,
                    r0=props.prox_waxman_r0,
                    seed=props.prox_waxman_seed if props.prox_waxman_seed > 0 else None,
                    **common_params
                )
            elif graph_type == 'GABRIEL':
                nodes_gdf, edges_gdf = proximity.generate_gabriel_graph_from_features(
                    feature_obj,
                    **common_params
                )
            elif graph_type == 'RNG':
                nodes_gdf, edges_gdf = proximity.generate_rng_graph_from_features(
                    feature_obj,
                    **common_params
                )
            elif graph_type == 'EMST':
                nodes_gdf, edges_gdf = proximity.generate_emst_graph_from_features(
                    feature_obj,
                    **common_params
                )
            elif graph_type == 'CONTIGUITY':
                nodes_gdf, edges_gdf = proximity.generate_contiguity_graph_from_features(
                    feature_obj,
                    contiguity=props.prox_contiguity_type.lower(),
                    **common_params
                )
            else:
                self.report({'ERROR'}, f"Unknown graph type: {graph_type}")
                return {'CANCELLED'}
            
            self.report({'INFO'}, f"Graph generated: {len(nodes_gdf)} nodes, {len(edges_gdf)} edges")
            
            collection_name = f"ProximityGraph_{graph_type}"
            collection = _get_or_create_collection(collection_name)
            
            graph_obj = _create_native_graph_from_gdfs(
                nodes_gdf,
                edges_gdf,
                f"{graph_type}_Graph",
                feature_obj,
                markers={
                    "is_proximity_graph": True,
                    "graph_type": graph_type,
                    "distance_metric": distance_metric,
                },
            )
            
            if graph_obj is None:
                self.report({'ERROR'}, "Failed to materialise graph mesh")
                return {'CANCELLED'}
            
            collection.objects.link(graph_obj)
            context.view_layer.objects.active = graph_obj
            graph_obj.select_set(True)
            
            elapsed = time.time() - start_time
            log(f"{graph_type} graph created successfully in {elapsed:.2f}s")
            self.report({'INFO'}, f"{graph_type} graph created successfully ({elapsed:.2f}s)")
            
            return {'FINISHED'}
            
        except ValueError as e:
            self.report({'ERROR'}, f"Invalid parameters: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
        except ImportError as e:
            self.report({'ERROR'}, f"Missing dependency: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to generate graph: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class SCIGRAPHS_OT_GenerateMultilayerGraph(bpy.types.Operator):
    """Generate multi-layer proximity graph connecting different feature types."""
    bl_idname = "scigraphs.generate_multilayer_graph"
    bl_label = "Generate Multi-Layer Graph"
    bl_description = "Connect multiple feature layers with proximity edges"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from ....core.city2graph import proximity
        
        props = context.scene.city2graph
        
        layer_objects = {}
        if props.prox_layer1_object:
            layer_objects["layer1"] = props.prox_layer1_object
        if props.prox_layer2_object:
            layer_objects["layer2"] = props.prox_layer2_object
        if props.prox_layer3_object:
            layer_objects["layer3"] = props.prox_layer3_object
        
        if len(layer_objects) < 2:
            self.report({'ERROR'}, "At least 2 feature layers required")
            return {'CANCELLED'}
        
        for name, obj in layer_objects.items():
            if not obj.get("is_osm_features") and not obj.get("is_city2graph"):
                self.report({'ERROR'}, f"{name} is not a valid feature object (OSM or city2graph)")
                return {'CANCELLED'}
        
        method = props.prox_multilayer_method.lower()
        distance_metric = props.prox_distance_metric.lower()
        
        network_obj = None
        if distance_metric == "network":
            network_obj = props.prox_network_object
            if not network_obj or not network_obj.get("is_osmnx"):
                self.report({'ERROR'}, "Network distance requires OSMnx street network")
                return {'CANCELLED'}
        
        self.report({'INFO'}, f"Generating multi-layer graph with {len(layer_objects)} layers...")
        start_time = time.time()
        
        try:
            nodes_dict, edges_dict = proximity.generate_bridge_nodes_from_features(
                layer_objects,
                proximity_method=method,
                k=props.prox_multilayer_k,
                radius=props.prox_multilayer_radius,
                distance_metric=distance_metric,
                network_obj=network_obj,
                as_nx=False
            )
            
            total_edges = sum(len(gdf) for gdf in edges_dict.values())
            self.report({'INFO'}, f"Generated {len(edges_dict)} edge types, {total_edges} total edges")
            
            collection_name = "ProximityGraph_MultiLayer"
            main_collection = _get_or_create_collection(collection_name)
            
            ref_obj = list(layer_objects.values())[0]
            
            graph_obj = _create_native_heterograph_from_dicts(
                nodes_dict,
                edges_dict,
                "MultiLayer_Graph",
                ref_obj,
                markers={
                    "is_multilayer_graph": True,
                    "num_layers": len(layer_objects),
                    "num_edge_types": len(edges_dict),
                },
            )
            
            if graph_obj is None:
                self.report({'ERROR'}, "Failed to materialise multi-layer graph mesh")
                return {'CANCELLED'}
            
            main_collection.objects.link(graph_obj)
            context.view_layer.objects.active = graph_obj
            graph_obj.select_set(True)
            
            elapsed = time.time() - start_time
            self.report({'INFO'}, f"Multi-layer graph created successfully ({elapsed:.2f}s)")
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to generate multi-layer graph: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class SCIGRAPHS_OT_GenerateGroupNodesGraph(bpy.types.Operator):
    """Generate graph connecting polygon zones to contained points."""
    bl_idname = "scigraphs.generate_group_nodes_graph"
    bl_label = "Generate Group Nodes Graph"
    bl_description = "Connect polygons to points they contain"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from ....core.city2graph import proximity
        from ....utils.logger import log
        
        props = context.scene.city2graph
        
        polygons_obj = props.prox_polygons_object
        points_obj = props.prox_points_object
        
        if not polygons_obj or not points_obj:
            self.report({'ERROR'}, "Both polygons and points objects required")
            return {'CANCELLED'}
        
        if not polygons_obj.get("is_osm_features") and not polygons_obj.get("is_city2graph"):
            self.report({'ERROR'}, "Polygons object is not a feature object (OSM or city2graph)")
            return {'CANCELLED'}
        
        if not points_obj.get("is_osm_features") and not points_obj.get("is_city2graph"):
            self.report({'ERROR'}, "Points object is not a feature object (OSM or city2graph)")
            return {'CANCELLED'}
        
        distance_metric = props.prox_distance_metric.lower()
        predicate = props.prox_group_predicate.lower()
        
        network_obj = None
        if distance_metric == "network":
            network_obj = props.prox_network_object
            if not network_obj or not network_obj.get("is_osmnx"):
                self.report({'ERROR'}, "Network distance requires OSMnx street network")
                return {'CANCELLED'}
        
        log(f"Generating group nodes graph with predicate={predicate}, metric={distance_metric}")
        
        self.report({'INFO'}, f"Generating group nodes graph...")
        start_time = time.time()
        
        try:
            nodes_dict, edges_dict = proximity.generate_group_nodes_from_features(
                polygons_obj=polygons_obj,
                points_obj=points_obj,
                predicate=predicate,
                distance_metric=distance_metric,
                network_obj=network_obj,
                as_nx=False
            )
            
            total_edges = sum(len(gdf) for gdf in edges_dict.values())
            log(f"Generated {len(edges_dict)} edge types, {total_edges} total edges")
            self.report({'INFO'}, f"Generated {len(edges_dict)} edge types, {total_edges} total edges")
            
            collection_name = "ProximityGraph_GroupNodes"
            main_collection = _get_or_create_collection(collection_name)
            
            ref_obj = polygons_obj
            
            graph_obj = _create_native_heterograph_from_dicts(
                nodes_dict,
                edges_dict,
                "GroupNodes_Graph",
                ref_obj,
                markers={
                    "is_group_nodes_graph": True,
                    "num_node_types": len(nodes_dict),
                    "num_edge_types": len(edges_dict),
                },
            )
            
            if graph_obj is None:
                self.report({'ERROR'}, "Failed to materialise group nodes graph mesh")
                return {'CANCELLED'}
            
            main_collection.objects.link(graph_obj)
            context.view_layer.objects.active = graph_obj
            graph_obj.select_set(True)
            
            elapsed = time.time() - start_time
            log(f"Group nodes graph created successfully in {elapsed:.2f}s")
            self.report({'INFO'}, f"Group nodes graph created successfully ({elapsed:.2f}s)")
            
            return {'FINISHED'}
            
        except ValueError as e:
            self.report({'ERROR'}, f"Invalid parameters: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
        except ImportError as e:
            self.report({'ERROR'}, f"Missing dependency: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to generate group nodes graph: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


classes = [
    SCIGRAPHS_OT_GenerateProximityGraph,
    SCIGRAPHS_OT_GenerateMultilayerGraph,
    SCIGRAPHS_OT_GenerateGroupNodesGraph,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

