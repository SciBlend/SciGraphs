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
)


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
        
        if not feature_obj.get("is_osm_features"):
            self.report({'ERROR'}, "Selected object is not OSM features")
            return {'CANCELLED'}
        
        # Validate feature count
        feature_count = feature_obj.get("feature_count", 0)
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
            
            curve_obj = _create_curves_from_gdf(
                edges_gdf,
                f"{graph_type}_Edges",
                feature_obj,
                thickness=props.prox_curve_thickness,
                limit=props.prox_visualize_limit
            )
            
            if curve_obj:
                collection.objects.link(curve_obj)
                
                curve_obj["is_proximity_graph"] = True
                curve_obj["graph_type"] = graph_type
                curve_obj["num_nodes"] = len(nodes_gdf)
                curve_obj["num_edges"] = len(edges_gdf)
                curve_obj["distance_metric"] = distance_metric
                
                mat = bpy.data.materials.get("ProximityGraph_Material")
                if not mat:
                    mat = bpy.data.materials.new(name="ProximityGraph_Material")
                    mat.use_nodes = True
                    bsdf = mat.node_tree.nodes.get("Principled BSDF")
                    if bsdf:
                        bsdf.inputs['Base Color'].default_value = (0.2, 0.6, 1.0, 1.0)
                        if 'Emission Color' in bsdf.inputs:
                            bsdf.inputs['Emission Color'].default_value = (0.2, 0.6, 1.0, 1.0)
                            bsdf.inputs['Emission Strength'].default_value = 1.0
                
                if curve_obj.data.materials:
                    curve_obj.data.materials[0] = mat
                else:
                    curve_obj.data.materials.append(mat)
                
                context.view_layer.objects.active = curve_obj
                curve_obj.select_set(True)
            
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
            if not obj.get("is_osm_features"):
                self.report({'ERROR'}, f"{name} is not a valid OSM features object")
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
            
            all_nodes_bm = bmesh.new()
            center_lat = ref_obj.get("osmnx_center_lat")
            center_lon = ref_obj.get("osmnx_center_lon")
            scale = ref_obj.get("osmnx_scale", 0.001)
            
            from ....core.mesh.geometry import _latlon_to_local_3d
            from shapely.geometry import Point
            
            for layer_name, layer_gdf in nodes_dict.items():
                if layer_gdf.crs and str(layer_gdf.crs).upper() != "EPSG:4326":
                    nodes_dict[layer_name] = layer_gdf.to_crs("EPSG:4326")
            
            node_attributes = {}
            
            for layer_name, layer_gdf in nodes_dict.items():
                for idx, row in layer_gdf.iterrows():
                    geom = row.geometry
                    if isinstance(geom, Point):
                        x, y, z = _latlon_to_local_3d(geom.y, geom.x, center_lat, center_lon, scale)
                        vert = all_nodes_bm.verts.new((x, y, z))
                        
                        for edge_key in edges_dict.keys():
                            src_layer = edge_key[0]
                            tgt_layer = edge_key[2]
                            attr_name = f"{src_layer}_{tgt_layer}"
                            if attr_name not in node_attributes:
                                node_attributes[attr_name] = []
                            
                            participates = 0
                            if layer_name == src_layer or layer_name == tgt_layer:
                                edges_gdf = edges_dict[edge_key]
                                if not edges_gdf.empty:
                                    participates = 1
                            
                            node_attributes[attr_name].append(participates)
            
            if len(all_nodes_bm.verts) > 0:
                mesh = bpy.data.meshes.new("MultiLayer_Nodes")
                all_nodes_bm.to_mesh(mesh)
                all_nodes_bm.free()
                
                for attr_name, values in node_attributes.items():
                    attr = mesh.attributes.new(name=attr_name, type='INT', domain='POINT')
                    attr.data.foreach_set('value', values)
                
                nodes_obj = bpy.data.objects.new("MultiLayer_Nodes", mesh)
                main_collection.objects.link(nodes_obj)
                
                nodes_obj["is_multilayer_graph"] = True
                nodes_obj["num_layers"] = len(layer_objects)
                nodes_obj["num_edge_types"] = len(edges_dict)
                
                context.view_layer.objects.active = nodes_obj
                nodes_obj.select_set(True)
                
                print(f"Node attributes created: {list(node_attributes.keys())}")
            else:
                all_nodes_bm.free()
            
            for edge_key, edges_gdf in edges_dict.items():
                src_layer, relation, tgt_layer = edge_key
                edge_name = f"{src_layer}_{relation}_{tgt_layer}"
                
                curve_obj = _create_curves_from_gdf(
                    edges_gdf,
                    edge_name,
                    ref_obj,
                    thickness=props.prox_curve_thickness,
                    limit=props.prox_visualize_limit
                )
                
                if curve_obj:
                    main_collection.objects.link(curve_obj)
                    
                    curve_obj["edge_type"] = edge_key
                    curve_obj["num_edges"] = len(edges_gdf)
            
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
        
        if not polygons_obj.get("is_osm_features"):
            self.report({'ERROR'}, "Polygons object is not OSM features")
            return {'CANCELLED'}
        
        if not points_obj.get("is_osm_features"):
            self.report({'ERROR'}, "Points object is not OSM features")
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
            
            # Create nodes visualization (combined from both layers)
            all_nodes_bm = bmesh.new()
            center_lat = ref_obj.get("osmnx_center_lat")
            center_lon = ref_obj.get("osmnx_center_lon")
            scale = ref_obj.get("osmnx_scale", 0.001)
            
            from ....core.mesh.geometry import _latlon_to_local_3d
            from shapely.geometry import Point
            
            # Convert all node GDFs to EPSG:4326
            for layer_name, layer_gdf in nodes_dict.items():
                if layer_gdf.crs and str(layer_gdf.crs).upper() != "EPSG:4326":
                    nodes_dict[layer_name] = layer_gdf.to_crs("EPSG:4326")
            
            # Create vertices for all nodes
            for layer_name, layer_gdf in nodes_dict.items():
                for idx, row in layer_gdf.iterrows():
                    geom = row.geometry
                    if isinstance(geom, Point):
                        x, y, z = _latlon_to_local_3d(geom.y, geom.x, center_lat, center_lon, scale)
                        all_nodes_bm.verts.new((x, y, z))
            
            if len(all_nodes_bm.verts) > 0:
                mesh = bpy.data.meshes.new("GroupNodes_Nodes")
                all_nodes_bm.to_mesh(mesh)
                all_nodes_bm.free()
                
                nodes_obj = bpy.data.objects.new("GroupNodes_Nodes", mesh)
                main_collection.objects.link(nodes_obj)
                
                nodes_obj["is_group_nodes_graph"] = True
                nodes_obj["num_node_types"] = len(nodes_dict)
                nodes_obj["num_edge_types"] = len(edges_dict)
                
                context.view_layer.objects.active = nodes_obj
                nodes_obj.select_set(True)
                
                log(f"Created {len(mesh.vertices)} node vertices")
            else:
                all_nodes_bm.free()
            
            # Create edge visualizations
            for edge_key, edges_gdf in edges_dict.items():
                src_layer, relation, tgt_layer = edge_key
                edge_name = f"{src_layer}_{relation}_{tgt_layer}"
                
                curve_obj = _create_curves_from_gdf(
                    edges_gdf,
                    edge_name,
                    ref_obj,
                    thickness=props.prox_curve_thickness,
                    limit=props.prox_visualize_limit
                )
                
                if curve_obj:
                    main_collection.objects.link(curve_obj)
                    
                    curve_obj["edge_type"] = edge_key
                    curve_obj["num_edges"] = len(edges_gdf)
            
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

