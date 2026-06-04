"""
Metapath analysis operators for heterogeneous urban graphs.

Provides operators to create street dual graphs, bridge amenities to streets,
and compute metapaths between amenities via the street network.
"""

import bpy
import bmesh
import time
from ....core.city2graph import metapaths
from ....core import importer
from ....core.osmnx.graph_cache import get_osmnx_graph as _get_osmnx_graph, get_osmnx_graph_diagnostic as _get_osmnx_graph_diagnostic
from ....utils.blender_helpers import get_or_create_collection as _get_or_create_collection


def _create_metapath_material():
    """Create or get the cyan emissive material for metapaths."""
    mat_name = "Metapath_Material"
    mat = bpy.data.materials.get(mat_name)
    
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (0.0, 1.0, 1.0, 1.0)
            
            if 'Emission Color' in bsdf.inputs:
                bsdf.inputs['Emission Color'].default_value = (0.0, 1.0, 1.0, 1.0)
                bsdf.inputs['Emission Strength'].default_value = 2.0
            else:
                bsdf.inputs['Emission'].default_value = (0.0, 1.0, 1.0, 1.0)
    
    return mat


def _reproject_to_geographic(gdf):
    """Return the GeoDataFrame in EPSG:4326 when a projected CRS is set.

    Aligning every overlay (dual nodes, bridges, metapaths) with the OSMnx
    mesh requires geographic coordinates, since the mesh is always built with
    an equirectangular projection centered on the network.
    """
    try:
        if gdf.crs is not None and not gdf.crs.is_geographic:
            return gdf.to_crs("EPSG:4326")
    except Exception:
        pass
    return gdf


def _resolve_geo_center(center_lat, center_lon, gdf):
    """Return a geographic center, falling back to the GeoDataFrame bounds."""
    if center_lat is None or center_lon is None:
        bounds = gdf.total_bounds
        center_lon = (bounds[0] + bounds[2]) / 2.0
        center_lat = (bounds[1] + bounds[3]) / 2.0
    return center_lat, center_lon


class SCIGRAPHS_OT_CreateStreetDualGraph(bpy.types.Operator):
    """Create dual graph where street segments become nodes (City2Graph method)"""
    bl_idname = "scigraphs.create_street_dual_graph"
    bl_label = "Create Street Dual Graph"
    bl_description = "Create dual graph using City2Graph (street segments become nodes)"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or not obj.get("is_osmnx"):
            self.report({'ERROR'}, "Select an OSMnx street network object")
            return {'CANCELLED'}
        
        # Get OSMnx graph from object
        osmnx_graph_data = _get_osmnx_graph(obj)
        
        if not osmnx_graph_data:
            diagnostic, suggestion = _get_osmnx_graph_diagnostic(obj)
            self.report({'ERROR'}, f"Could not retrieve OSMnx graph: {diagnostic}")
            self.report({'WARNING'}, f"Suggested fix: {suggestion}")
            return {'CANCELLED'}
        
        self.report({'INFO'}, "Creating street dual graph...")
        start_time = time.time()
        
        try:
            # Create dual graph using city2graph
            dual_nodes_gdf, dual_edges_gdf = metapaths.create_street_dual_graph_c2g(osmnx_graph_data)
            
            self.report({'INFO'}, f"Dual graph: {len(dual_nodes_gdf)} nodes, {len(dual_edges_gdf)} edges")
            
            # Use the same scale and center as the original OSMnx object
            # This ensures the dual graph is in the same coordinate space
            center_lat = obj.get("osmnx_center_lat")
            center_lon = obj.get("osmnx_center_lon")
            osmnx_scale = obj.get("osmnx_scale", 0.001)
            
            # Calculate centroid from all node coordinates
            import numpy as np
            all_coords = np.array([[geom.x, geom.y] for geom in dual_nodes_gdf.geometry])
            centroid_x, centroid_y = all_coords.mean(axis=0)
            
            # Check if GDF uses geographic or projected CRS
            crs_is_geographic = dual_nodes_gdf.crs and dual_nodes_gdf.crs.is_geographic
            
            # Use same scale as original OSMnx object
            dual_scale = osmnx_scale
            
            # Import coordinate conversion function
            from ....core.mesh.geometry import _latlon_to_local_3d
            
            # Create Blender mesh
            bm = bmesh.new()
            
            # CRITICAL: Add is_intersection layer for setup visual compatibility
            is_intersection_layer = bm.verts.layers.int.new("is_intersection")
            
            # Create vertices (dual nodes)
            placement_nodes_gdf = _reproject_to_geographic(dual_nodes_gdf)
            place_center_lat, place_center_lon = _resolve_geo_center(
                center_lat, center_lon, placement_nodes_gdf
            )
            vert_map = {}
            vert_index = {}
            node_positions = []
            for idx, row in placement_nodes_gdf.iterrows():
                geom = row.geometry
                x, y, z = _latlon_to_local_3d(
                    geom.y, geom.x, place_center_lat, place_center_lon, dual_scale
                )
                v = bm.verts.new((x, y, z))
                v[is_intersection_layer] = 1
                vert_map[idx] = v
                vert_index[idx] = len(node_positions)
                node_positions.append((float(x), float(y), float(z)))
            
            bm.verts.ensure_lookup_table()
            
            # Create edges (dual edges)
            created_edge_rows = []
            for row_pos, idx in enumerate(dual_edges_gdf.index):
                if isinstance(idx, tuple) and len(idx) == 2:
                    src, tgt = idx
                    if src in vert_map and tgt in vert_map:
                        try:
                            bm.edges.new([vert_map[src], vert_map[tgt]])
                            created_edge_rows.append(row_pos)
                        except ValueError:
                            pass
            
            # Create mesh and object
            mesh = bpy.data.meshes.new(name=f"{obj.name}_StreetDual_Mesh")
            bm.to_mesh(mesh)
            bm.free()
            
            dual_obj = bpy.data.objects.new(f"{obj.name}_StreetDual", mesh)
            
            # Add to collection
            collection_name = f"MetapathAnalysis_{obj.name}"
            main_collection = _get_or_create_collection(collection_name)
            dual_collection = _get_or_create_collection("StreetDual", main_collection)
            dual_collection.objects.link(dual_obj)
            
            # Store metadata
            dual_obj["is_street_dual"] = True
            dual_obj["is_city2graph"] = True
            dual_obj["num_nodes"] = len(dual_nodes_gdf)
            dual_obj["num_edges"] = len(dual_edges_gdf)
            dual_obj["original_graph"] = obj.name
            dual_obj["centroid_x"] = centroid_x
            dual_obj["centroid_y"] = centroid_y
            dual_obj["dual_scale"] = dual_scale
            dual_obj["crs"] = str(dual_nodes_gdf.crs)
            
            # CRITICAL: Store OSMnx transformation parameters for curve visualization
            # Curves must use the SAME transformation as the original OSMnx graph
            dual_obj["osmnx_center_lat"] = obj.get("osmnx_center_lat")
            dual_obj["osmnx_center_lon"] = obj.get("osmnx_center_lon")
            dual_obj["osmnx_scale"] = obj.get("osmnx_scale", 0.001)
            dual_obj["crs_is_geographic"] = crs_is_geographic
            
            # CRITICAL: Store nodes_data and edges_data for graph operations compatibility
            # This allows the dual graph to work with layout, analysis, and visualization operators
            nodes_list = [str(idx) for idx in dual_nodes_gdf.index]
            dual_obj["nodes_data"] = ",".join(nodes_list)
            
            edges_flat = []
            for idx in dual_edges_gdf.index:
                if isinstance(idx, tuple) and len(idx) == 2:
                    edges_flat.append(str(idx[0]))
                    edges_flat.append(str(idx[1]))
            dual_obj["edges_data"] = ",".join(edges_flat)
            
            dual_obj["node_positions"] = [c for p in node_positions for c in p]
            dual_obj["is_directed"] = False
            
            import numpy as _np
            for col in dual_nodes_gdf.columns:
                if col == "geometry":
                    continue
                try:
                    if not _np.issubdtype(dual_nodes_gdf[col].dtype, _np.number):
                        continue
                except TypeError:
                    continue
                values = [float(v) if v is not None else 0.0 for v in dual_nodes_gdf[col].tolist()]
                if len(values) == len(mesh.vertices):
                    attr = mesh.attributes.new(name=f"node_{col}", type='FLOAT', domain='POINT')
                    attr.data.foreach_set("value", values)
            
            for col in dual_edges_gdf.columns:
                if col == "geometry":
                    continue
                try:
                    if not _np.issubdtype(dual_edges_gdf[col].dtype, _np.number):
                        continue
                except TypeError:
                    continue
                series = dual_edges_gdf[col].tolist()
                values = [float(series[r]) if series[r] is not None else 0.0 for r in created_edge_rows]
                if len(values) == len(mesh.edges):
                    attr = mesh.attributes.new(name=f"edge_{col}", type='FLOAT', domain='EDGE')
                    attr.data.foreach_set("value", values)
            
            # Store dual graph data for next steps
            import pickle
            dual_obj["dual_nodes_gdf_pickle"] = pickle.dumps(dual_nodes_gdf)
            dual_obj["dual_edges_gdf_pickle"] = pickle.dumps(dual_edges_gdf)
            
            elapsed = time.time() - start_time
            
            # Verify graph attributes for debugging
            has_is_intersection = "is_intersection" in mesh.attributes
            self.report({'INFO'}, 
                f"Street dual graph created: {len(dual_nodes_gdf)} segments as nodes, "
                f"{len(dual_edges_gdf)} connections ({elapsed:.2f}s)")
            
            if has_is_intersection:
                print(f"✓ Dual graph has 'is_intersection' attribute (setup visual compatible)")
            else:
                print(f"⚠ Warning: 'is_intersection' attribute not found in mesh")
            
            print(f"Graph attributes: num_nodes={dual_obj['num_nodes']}, "
                  f"num_edges={dual_obj['num_edges']}, "
                  f"nodes_data={'present' if 'nodes_data' in dual_obj else 'missing'}, "
                  f"edges_data={'present' if 'edges_data' in dual_obj else 'missing'}")
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create dual graph: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class SCIGRAPHS_OT_BridgeAmenitiesToStreets(bpy.types.Operator):
    """Bridge amenity features to nearest street segments"""
    bl_idname = "scigraphs.bridge_amenities"
    bl_label = "Bridge Amenities to Streets"
    bl_description = "Connect amenities to their nearest street segments using KNN"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.city2graph
        obj = context.active_object
        
        # Validate dual graph object
        if not obj or not obj.get("is_street_dual"):
            self.report({'ERROR'}, "Select a street dual graph object")
            return {'CANCELLED'}
        
        # Get amenities object from scene properties
        props = context.scene.city2graph
        amenities_obj = props.metapath_amenities_object
        
        if not amenities_obj:
            self.report({'ERROR'}, "No amenities object selected. Select one in the Metapath Analysis panel.")
            return {'CANCELLED'}
        
        if not amenities_obj.get("is_osm_features") and not amenities_obj.get("is_city2graph"):
            self.report({'ERROR'}, "Selected object is not a valid features object")
            return {'CANCELLED'}
        
        self.report({'INFO'}, "Bridging amenities to street segments...")
        start_time = time.time()
        
        try:
            # Unpickle dual graph data
            import pickle
            dual_nodes_gdf = pickle.loads(obj["dual_nodes_gdf_pickle"])
            dual_edges_gdf = pickle.loads(obj["dual_edges_gdf_pickle"])
            
            # Prepare amenities
            target_crs = obj.get("crs", "EPSG:32630")
            amenity_limit = props.metapath_amenity_limit
            
            amenities_gdf = metapaths.prepare_amenities_from_features(
                amenities_obj, target_crs, limit=amenity_limit
            )
            
            self.report({'INFO'}, f"Using {len(amenities_gdf)} amenities")
            
            # Bridge amenities to segments
            k = props.metapath_k_neighbors
            nodes_dict, edges_dict = metapaths.bridge_amenities_to_segments(
                amenities_gdf, dual_nodes_gdf, k=k
            )
            
            # Add dual edges to edges_dict
            edges_dict[("segment", "connects_to", "segment")] = dual_edges_gdf
            
            # Count bridge connections
            bridge_count = len(edges_dict.get(('amenity', 'is_nearby', 'segment'), []))
            
            self.report({'INFO'}, f"Created {bridge_count} bridge connections")
            
            # Visualize bridges as curves
            self._visualize_bridges(context, obj, edges_dict, amenities_gdf)
            
            # Store for next step
            obj["nodes_dict_pickle"] = pickle.dumps(nodes_dict)
            obj["edges_dict_pickle"] = pickle.dumps(edges_dict)
            obj["has_metapath_bridges"] = True
            obj["num_bridges"] = bridge_count
            
            elapsed = time.time() - start_time
            self.report({'INFO'}, f"Bridging complete: {bridge_count} connections ({elapsed:.2f}s)")
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to bridge amenities: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
    
    def _visualize_bridges(self, context, dual_obj, edges_dict, amenities_gdf):
        """Create curve visualization of bridge connections."""
        bridge_key = ('amenity', 'is_nearby', 'segment')
        if bridge_key not in edges_dict:
            return
        
        bridge_edges = edges_dict[bridge_key]
        if len(bridge_edges) == 0:
            return
        
        scale = dual_obj.get("osmnx_scale", 0.001)
        
        from ....core.mesh.geometry import _latlon_to_local_3d
        
        bridge_edges = _reproject_to_geographic(bridge_edges)
        center_lat, center_lon = _resolve_geo_center(
            dual_obj.get("osmnx_center_lat"), dual_obj.get("osmnx_center_lon"), bridge_edges
        )
        
        # Create curve
        curve_data = bpy.data.curves.new(f'{dual_obj.name}_Bridges', type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.bevel_depth = 0.0001
        
        # Add splines for each bridge edge
        for idx, row in bridge_edges.head(200).iterrows():  # Limit visualization
            if hasattr(row, 'geometry') and row.geometry:
                coords = list(row.geometry.coords)
                if len(coords) >= 2:
                    polyline = curve_data.splines.new('POLY')
                    polyline.points.add(len(coords) - 1)
                    
                    for i, (lon, lat) in enumerate(coords):
                        blender_x, blender_y, blender_z = _latlon_to_local_3d(lat, lon, center_lat, center_lon, scale)
                        polyline.points[i].co = (blender_x, blender_y, blender_z + 0.01, 1.0)
        
        # Create object
        bridge_obj = bpy.data.objects.new(f'{dual_obj.name}_Bridges', curve_data)
        
        # Add to collection
        original_name = dual_obj.get("original_graph", "Graph")
        collection_name = f"MetapathAnalysis_{original_name}"
        main_collection = _get_or_create_collection(collection_name)
        bridge_collection = _get_or_create_collection("Bridges", main_collection)
        bridge_collection.objects.link(bridge_obj)
        
        # Apply material (orange for bridges)
        mat = bpy.data.materials.new(name="Bridge_Material")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (1.0, 0.6, 0.0, 1.0)
        bridge_obj.data.materials.append(mat)


class SCIGRAPHS_OT_ComputeMetapaths(bpy.types.Operator):
    """Compute metapaths between amenities via street segments"""
    bl_idname = "scigraphs.compute_metapaths"
    bl_label = "Compute Metapaths"
    bl_description = "Compute N-hop metapaths between amenities through the street network"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.city2graph
        obj = context.active_object
        
        # Validate object has bridges
        if not obj or not obj.get("has_metapath_bridges"):
            self.report({'ERROR'}, "Select a dual graph object with bridges computed")
            return {'CANCELLED'}
        
        hops = props.metapath_hops
        self.report({'INFO'}, f"Computing {hops}-hop metapaths...")
        start_time = time.time()
        
        try:
            # Unpickle graph data
            import pickle
            nodes_dict = pickle.loads(obj["nodes_dict_pickle"])
            edges_dict = pickle.loads(obj["edges_dict_pickle"])
            
            # Compute metapaths
            result_nodes, result_edges = metapaths.compute_metapaths(
                nodes_dict, edges_dict, hops=hops, directed=False
            )
            
            # Extract metapath connections (with multiplicity)
            metapath_gdf = metapaths.extract_metapath_connections(result_edges, add_multiplicity=True)
            
            if metapath_gdf is None or len(metapath_gdf) == 0:
                self.report({'WARNING'}, "No metapaths found with current parameters")
                return {'CANCELLED'}
            
            # Report counts
            unique_count = len(metapath_gdf)
            total_count = metapath_gdf['multiplicity'].sum() if 'multiplicity' in metapath_gdf.columns else unique_count
            
            self.report({'INFO'}, f"Found {int(total_count)} metapaths ({unique_count} unique connections)")
            
            # Store metapath GeoDataFrame for export
            import pickle
            obj["metapath_gdf_pickle"] = pickle.dumps(metapath_gdf)
            
            # Visualize metapaths
            self._visualize_metapaths(context, obj, metapath_gdf, hops)
            
            elapsed = time.time() - start_time
            self.report({'INFO'}, 
                f"Metapath computation complete: {int(total_count)} paths in {unique_count} connections ({elapsed:.2f}s)")
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to compute metapaths: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
    
    def _visualize_metapaths(self, context, dual_obj, metapath_gdf, hops):
        """Create curve visualization of metapaths with multiplicity from GeoDataFrame."""
        props = context.scene.city2graph
        vis_limit = props.metapath_visualize_limit
        curve_thickness = props.metapath_curve_thickness
        
        scale = dual_obj.get("osmnx_scale", 0.001)
        
        from ....core.mesh.geometry import _latlon_to_local_3d
        
        metapath_gdf = _reproject_to_geographic(metapath_gdf)
        center_lat, center_lon = _resolve_geo_center(
            dual_obj.get("osmnx_center_lat"), dual_obj.get("osmnx_center_lon"), metapath_gdf
        )
        
        # Metapaths already grouped with multiplicity column
        has_multiplicity = 'multiplicity' in metapath_gdf.columns
        
        if has_multiplicity:
            total_raw = int(metapath_gdf['multiplicity'].sum())
            unique_count = len(metapath_gdf)
            print(f"Visualizing {unique_count} unique metapath connections ({total_raw} total paths)")
        else:
            print(f"Visualizing {len(metapath_gdf)} metapaths")
        
        # Build a native MESH: each metapath connection becomes one edge
        # between its endpoint amenities, with multiplicity as an EDGE attribute.
        mesh = bpy.data.meshes.new(f'{dual_obj.name}_Metapaths_{hops}hop_mesh')
        bm = bmesh.new()
        
        vert_by_key = {}
        node_positions = []
        
        def _get_vert(lon, lat):
            key = (round(lon, 7), round(lat, 7))
            if key in vert_by_key:
                return vert_by_key[key]
            bx, by, bz = _latlon_to_local_3d(lat, lon, center_lat, center_lon, scale)
            v = bm.verts.new((bx, by, bz + 0.05))
            vert_by_key[key] = v
            node_positions.append((float(bx), float(by), float(bz + 0.05)))
            return v
        
        visualized = 0
        multiplicity_list = []
        edge_pairs = []
        
        for idx, row in metapath_gdf.head(vis_limit).iterrows():
            if not hasattr(row, 'geometry') or not row.geometry:
                continue
            
            geom = row.geometry
            multiplicity = int(row.get('multiplicity', 1)) if has_multiplicity else 1
            
            coords = list(geom.coords)
            if len(coords) >= 2:
                v_src = _get_vert(coords[0][0], coords[0][1])
                v_tgt = _get_vert(coords[-1][0], coords[-1][1])
                if v_src is v_tgt:
                    continue
                try:
                    bm.edges.new([v_src, v_tgt])
                except ValueError:
                    continue
                multiplicity_list.append(multiplicity)
                edge_pairs.append((v_src.index, v_tgt.index))
                visualized += 1
        
        bm.verts.ensure_lookup_table()
        bm.to_mesh(mesh)
        bm.free()
        
        metapath_obj = bpy.data.objects.new(
            f'{dual_obj.name}_Metapaths_{hops}hop',
            mesh
        )
        
        # Add to collection
        original_name = dual_obj.get("original_graph", "Graph")
        collection_name = f"MetapathAnalysis_{original_name}"
        main_collection = _get_or_create_collection(collection_name)
        metapath_collection = _get_or_create_collection("Metapaths", main_collection)
        metapath_collection.objects.link(metapath_obj)
        
        # Store metadata including multiplicity statistics
        import json
        import numpy as np
        
        metapath_obj["is_metapath_result"] = True
        metapath_obj["is_city2graph"] = True
        metapath_obj["num_metapaths_raw"] = int(metapath_gdf['multiplicity'].sum()) if has_multiplicity else len(metapath_gdf)
        metapath_obj["num_metapaths_unique"] = len(metapath_gdf)  # Unique connections
        metapath_obj["metapath_hops"] = hops
        metapath_obj["num_visualized"] = visualized
        metapath_obj["num_nodes"] = len(node_positions)
        metapath_obj["num_edges"] = visualized
        metapath_obj["is_directed"] = False
        metapath_obj["node_positions"] = [c for p in node_positions for c in p]
        metapath_obj["nodes_data"] = ",".join(str(i) for i in range(len(node_positions)))
        metapath_obj["edges_data"] = ",".join(str(i) for pair in edge_pairs for i in pair)
        metapath_obj["c2g_center_lat"] = center_lat
        metapath_obj["c2g_center_lon"] = center_lon
        metapath_obj["c2g_scale"] = scale
        
        # Multiplicity as an EDGE scalar attribute so the native coloring
        # pipeline can colour connections by how many raw paths they carry.
        if len(multiplicity_list) == len(mesh.edges) and multiplicity_list:
            mult_attr = mesh.attributes.new(name="edge_multiplicity", type='FLOAT', domain='EDGE')
            mult_attr.data.foreach_set("value", [float(m) for m in multiplicity_list])
        
        # Store multiplicity list (in order) for easy access
        metapath_obj["multiplicity_list"] = json.dumps(multiplicity_list)
        
        # Calculate stats
        if multiplicity_list:
            avg_multiplicity = np.mean(multiplicity_list)
            max_multiplicity = np.max(multiplicity_list)
            min_multiplicity = np.min(multiplicity_list)
            metapath_obj["avg_multiplicity"] = float(avg_multiplicity)
            metapath_obj["max_multiplicity"] = int(max_multiplicity)
            metapath_obj["min_multiplicity"] = int(min_multiplicity)
        
        # Add helper info
        metapath_obj["_info"] = (
            f"Metapath object with {metapath_obj['num_metapaths_raw']} raw paths in {len(metapath_gdf)} unique connections. "
            f"Access GeoDataFrame: import pickle; gdf = pickle.loads(obj['metapath_gdf_pickle'])"
        )
        
        print(f"\nMetapath visualization summary:")
        print(f"  Raw metapaths: {metapath_obj['num_metapaths_raw']}")
        print(f"  Unique connections: {len(metapath_gdf)}")
        print(f"  Visualized curves: {visualized}")
        if multiplicity_list:
            print(f"  Multiplicity range: {min_multiplicity} - {max_multiplicity}")
            print(f"  Avg multiplicity: {avg_multiplicity:.2f}")
        
        print(f"\nTo access metapath GeoDataFrame with multiplicity:")
        print(f"  import pickle")
        print(f"  obj = bpy.data.objects['{metapath_obj.name}']")
        print(f"  metapath_gdf = pickle.loads(obj['metapath_gdf_pickle'])")
        print(f"  print(metapath_gdf[['geometry', 'multiplicity']].head())")


class SCIGRAPHS_OT_ComputeMetapathsWizard(bpy.types.Operator):
    """Complete metapath analysis: dual graph + bridges + metapaths"""
    bl_idname = "scigraphs.compute_metapaths_wizard"
    bl_label = "Compute Metapaths (Complete)"
    bl_description = "Run complete metapath analysis pipeline from OSMnx graph to metapaths"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or not obj.get("is_osmnx"):
            self.report({'ERROR'}, "Select an OSMnx street network object")
            return {'CANCELLED'}
        
        # Get amenities object from scene properties
        props = context.scene.city2graph
        amenities_obj = props.metapath_amenities_object
        
        if not amenities_obj:
            self.report({'ERROR'}, "No amenities object selected. Select one in the Metapath Analysis panel.")
            return {'CANCELLED'}
        
        if not amenities_obj.get("is_osm_features") and not amenities_obj.get("is_city2graph"):
            self.report({'ERROR'}, "Selected object is not a valid features object")
            return {'CANCELLED'}
        
        self.report({'INFO'}, "Starting complete metapath analysis...")
        total_start = time.time()
        
        # Step 1: Create dual graph
        self.report({'INFO'}, "Step 1/3: Creating street dual graph...")
        bpy.ops.scigraphs.create_street_dual_graph()
        
        # Find the created dual graph object
        dual_obj = None
        for o in bpy.data.objects:
            if o.get("is_street_dual") and o.get("original_graph") == obj.name:
                dual_obj = o
                break
        
        if not dual_obj:
            self.report({'ERROR'}, "Failed to create dual graph")
            return {'CANCELLED'}
        
        # Select dual graph for next step
        bpy.ops.object.select_all(action='DESELECT')
        dual_obj.select_set(True)
        context.view_layer.objects.active = dual_obj
        
        # Step 2: Bridge amenities
        self.report({'INFO'}, "Step 2/3: Bridging amenities to streets...")
        bpy.ops.scigraphs.bridge_amenities()
        
        if not dual_obj.get("has_metapath_bridges"):
            self.report({'ERROR'}, "Failed to bridge amenities")
            return {'CANCELLED'}
        
        # Step 3: Compute metapaths
        self.report({'INFO'}, "Step 3/3: Computing metapaths...")
        result = bpy.ops.scigraphs.compute_metapaths()
        
        if result != {'FINISHED'}:
            self.report({'WARNING'}, "Metapath computation completed with warnings")
        
        total_elapsed = time.time() - total_start
        
        # Summary
        props = context.scene.city2graph
        self.report({'INFO'}, 
            f"Metapath analysis complete! "
            f"Analyzed {props.metapath_amenity_limit} amenities with {props.metapath_hops} hops "
            f"({total_elapsed:.2f}s total)")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_ConvertMetapathsToMesh(bpy.types.Operator):
    """Convert metapath curves to mesh with multiplicity attribute"""
    bl_idname = "scigraphs.convert_metapaths_to_mesh"
    bl_label = "Convert to Mesh (with Attributes)"
    bl_description = "Convert metapath curves to mesh to view multiplicity in Spreadsheet Editor"
    bl_options = {'REGISTER', 'UNDO'}
    
    min_multiplicity: bpy.props.IntProperty(
        name="Min Multiplicity",
        description="Remove splines with multiplicity below this value (1=keep all, 2=only multiple paths)",
        default=1,
        min=1,
        max=10,
    )
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or not obj.get("is_metapath_result"):
            self.report({'ERROR'}, "Select a metapath curve object")
            return {'CANCELLED'}
        
        if obj.type != 'CURVE':
            self.report({'ERROR'}, "Object is not a curve")
            return {'CANCELLED'}
        
        import json
        
        # Get multiplicity data
        if "spline_multiplicity" not in obj:
            self.report({'ERROR'}, "No multiplicity data found")
            return {'CANCELLED'}
        
        spline_mult = json.loads(obj["spline_multiplicity"])
        
        # Duplicate object
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.duplicate()
        
        curve_obj = context.active_object
        curve_obj.name = f"{obj.name}_Filtered"
        curve_data = curve_obj.data
        
        # FILTER SPLINES BEFORE CONVERSION
        if self.min_multiplicity > 1:
            splines_to_remove = []
            splines_removed = 0
            
            for spline_idx, spline in enumerate(curve_data.splines):
                mult = spline_mult.get(str(spline_idx), 1)
                if mult < self.min_multiplicity:
                    splines_to_remove.append(spline_idx)
            
            # Remove splines in reverse order
            for spline_idx in sorted(splines_to_remove, reverse=True):
                if spline_idx < len(curve_data.splines):
                    curve_data.splines.remove(curve_data.splines[spline_idx])
                    splines_removed += 1
            
            print(f"  Filtered: Removed {splines_removed} splines with multiplicity < {self.min_multiplicity}")
        
        # Now convert to mesh
        mesh_obj = curve_obj
        mesh_obj.name = f"{obj.name}_Mesh"
        bpy.ops.object.convert(target='MESH')
        
        # Add multiplicity attribute
        mesh = mesh_obj.data
        
        # Create attribute for edges
        if not mesh.attributes.get("multiplicity"):
            attr = mesh.attributes.new(name="multiplicity", type='INT', domain='EDGE')
        else:
            attr = mesh.attributes["multiplicity"]
        
        # Map remaining splines to edges
        # Get edge count per spline from FILTERED curve
        edge_idx = 0
        remaining_spline_idx = 0
        
        for orig_spline_idx in range(len(obj.data.splines)):
            mult = spline_mult.get(str(orig_spline_idx), 1)
            
            # Skip if this spline was filtered out
            if mult < self.min_multiplicity:
                continue
            
            # Get the spline from the converted mesh (before conversion)
            orig_spline = obj.data.splines[orig_spline_idx]
            num_points = len(orig_spline.points)
            
            if num_points >= 2:
                num_edges = num_points - 1
                
                # Assign multiplicity to all edges from this spline
                for i in range(edge_idx, edge_idx + num_edges):
                    if i < len(attr.data):
                        attr.data[i].value = mult
                
                edge_idx += num_edges
                remaining_spline_idx += 1
        
        # CRITICAL: Remove edges with multiplicity = 0 or below threshold
        # These are edges that weren't properly mapped or should be filtered
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.edges.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        
        # Add is_intersection layer for setup visual compatibility
        # All vertices in metapath mesh represent connection points
        is_intersection_layer = bm.verts.layers.int.new("is_intersection") if "is_intersection" not in bm.verts.layers.int else bm.verts.layers.int.get("is_intersection")
        
        # Mark all vertices as intersections
        for v in bm.verts:
            v[is_intersection_layer] = 1
        
        edges_to_delete = []
        for i, edge in enumerate(bm.edges):
            if i < len(attr.data):
                mult = attr.data[i].value
                if mult < self.min_multiplicity:
                    edges_to_delete.append(edge)
        
        # Remove edges
        for edge in edges_to_delete:
            bm.edges.remove(edge)
        
        # Remove loose vertices
        loose_verts = [v for v in bm.verts if not v.link_edges]
        for v in loose_verts:
            bm.verts.remove(v)
        
        bm.to_mesh(mesh)
        bm.free()
        
        if len(edges_to_delete) > 0:
            print(f"  Cleaned: Removed {len(edges_to_delete)} edges with multiplicity < {self.min_multiplicity}")
        
        # Copy custom properties
        for key in obj.keys():
            if key not in ['_RNA_UI']:
                mesh_obj[key] = obj[key]
        
        mesh_obj["is_metapath_mesh"] = True
        mesh_obj["min_multiplicity_filter"] = self.min_multiplicity
        
        # CRITICAL: Add graph metadata for full operator compatibility
        # This allows the mesh to work with layout, analysis, and other graph operators
        num_verts = len(mesh.vertices)
        num_edges_final = len(mesh.edges)
        mesh_obj["num_nodes"] = num_verts
        mesh_obj["num_edges"] = num_edges_final
        
        # Create simplified nodes_data (vertex indices as node IDs)
        mesh_obj["nodes_data"] = ",".join(str(i) for i in range(num_verts))
        
        # Store edges_data (for graph operations)
        edges_list = []
        for edge in mesh.edges:
            edges_list.append(str(edge.vertices[0]))
            edges_list.append(str(edge.vertices[1]))
        mesh_obj["edges_data"] = ",".join(edges_list)
        
        total_edges = len(mesh.edges)
        if self.min_multiplicity > 1:
            self.report({'INFO'}, 
                f"Converted to mesh: {total_edges} edges (filtered multiplicity ≥ {self.min_multiplicity})")
        else:
            self.report({'INFO'}, f"Converted to mesh: {total_edges} edges with multiplicity attribute")
        
        # Verify graph compatibility
        has_is_intersection = "is_intersection" in mesh.attributes
        has_num_nodes = "num_nodes" in mesh_obj
        
        print(f"\nMetapath curves converted to mesh:")
        print(f"  Object: {mesh_obj.name}")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Edges: {total_edges}")
        print(f"  Attribute 'multiplicity': ✓")
        print(f"  Attribute 'is_intersection': {'✓' if has_is_intersection else '✗'}")
        print(f"  Graph metadata (num_nodes, edges_data): {'✓' if has_num_nodes else '✗'}")
        
        if self.min_multiplicity > 1:
            print(f"  Filter: Only edges with multiplicity ≥ {self.min_multiplicity}")
        
        print(f"\nSpreadsheet Editor:")
        print(f"  1. Select {mesh_obj.name}")
        print(f"  2. Open Spreadsheet Editor")
        print(f"  3. Domain: 'Edge' → see 'multiplicity'")
        print(f"  4. Domain: 'Vertex' → see 'is_intersection'")
        
        if has_num_nodes:
            print(f"\n✓ Graph is compatible with:")
            print(f"  • Setup Visualization (Geometry Nodes)")
            print(f"  • Layout algorithms")
            print(f"  • Analysis operators")
            print(f"  • Export functions")
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class SCIGRAPHS_OT_ComputeMetapathsByWeight(bpy.types.Operator):
    """Compute weighted metapaths using Dijkstra cost threshold."""
    bl_idname = "scigraphs.compute_metapaths_by_weight"
    bl_label = "Weighted Metapaths"
    bl_description = "Connect nodes reachable within a travel cost threshold (Dijkstra)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.city2graph
        obj = context.active_object

        if not obj or not obj.get("has_metapath_bridges"):
            self.report({'ERROR'}, "Select a dual graph with bridges computed")
            return {'CANCELLED'}

        weight_attr = props.metapath_weight_attr.strip()
        threshold = props.metapath_weight_threshold
        min_threshold = props.metapath_weight_min_threshold
        endpoint_type = props.metapath_endpoint_type.strip()

        if not weight_attr:
            self.report({'ERROR'}, "Specify a weight attribute (e.g. 'length')")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Computing weighted metapaths (weight={weight_attr}, threshold={threshold})...")
        start_time = time.time()

        try:
            import pickle
            nodes_dict = pickle.loads(obj["nodes_dict_pickle"])
            edges_dict = pickle.loads(obj["edges_dict_pickle"])

            result_nodes, result_edges = metapaths.compute_metapaths_by_weight(
                nodes_dict, edges_dict,
                weight_attr=weight_attr,
                threshold=threshold,
                endpoint_type=endpoint_type,
                min_threshold=min_threshold,
                directed=False,
            )

            metapath_gdf = metapaths.extract_metapath_connections(result_edges, add_multiplicity=False)

            if metapath_gdf is None or len(metapath_gdf) == 0:
                self.report({'WARNING'}, "No weighted metapaths found with these parameters")
                return {'CANCELLED'}

            obj["metapath_gdf_pickle"] = pickle.dumps(metapath_gdf)

            self._visualize(context, obj, metapath_gdf, weight_attr)

            elapsed = time.time() - start_time
            self.report({'INFO'}, f"Found {len(metapath_gdf)} weighted metapaths ({elapsed:.2f}s)")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

    def _visualize(self, context, dual_obj, metapath_gdf, weight_attr):
        props = context.scene.city2graph
        vis_limit = props.metapath_visualize_limit
        curve_thickness = props.metapath_curve_thickness

        scale = dual_obj.get("osmnx_scale", 0.001)

        from ....core.mesh.geometry import _latlon_to_local_3d

        metapath_gdf = _reproject_to_geographic(metapath_gdf)
        center_lat, center_lon = _resolve_geo_center(
            dual_obj.get("osmnx_center_lat"), dual_obj.get("osmnx_center_lon"), metapath_gdf
        )

        mesh = bpy.data.meshes.new(f'{dual_obj.name}_WeightedMP_{weight_attr}_mesh')
        bm = bmesh.new()
        vert_by_key = {}
        node_positions = []
        edge_pairs = []

        def _get_vert(lon, lat):
            key = (round(lon, 7), round(lat, 7))
            if key in vert_by_key:
                return vert_by_key[key]
            bx, by, bz = _latlon_to_local_3d(lat, lon, center_lat, center_lon, scale)
            v = bm.verts.new((bx, by, bz + 0.05))
            vert_by_key[key] = v
            node_positions.append((float(bx), float(by), float(bz + 0.05)))
            return v

        visualized = 0
        for _, row in metapath_gdf.head(vis_limit).iterrows():
            if not hasattr(row, 'geometry') or not row.geometry:
                continue
            coords = list(row.geometry.coords)
            if len(coords) < 2:
                continue
            v_src = _get_vert(coords[0][0], coords[0][1])
            v_tgt = _get_vert(coords[-1][0], coords[-1][1])
            if v_src is v_tgt:
                continue
            try:
                bm.edges.new([v_src, v_tgt])
            except ValueError:
                continue
            edge_pairs.append((v_src.index, v_tgt.index))
            visualized += 1

        bm.verts.ensure_lookup_table()
        bm.to_mesh(mesh)
        bm.free()

        metapath_obj = bpy.data.objects.new(f'{dual_obj.name}_WeightedMP_{weight_attr}', mesh)

        original_name = dual_obj.get("original_graph", "Graph")
        collection_name = f"MetapathAnalysis_{original_name}"
        main_collection = _get_or_create_collection(collection_name)
        mp_collection = _get_or_create_collection("WeightedMetapaths", main_collection)
        mp_collection.objects.link(metapath_obj)

        metapath_obj["is_metapath_result"] = True
        metapath_obj["is_weighted_metapath"] = True
        metapath_obj["is_city2graph"] = True
        metapath_obj["num_metapaths_raw"] = len(metapath_gdf)
        metapath_obj["num_metapaths_unique"] = len(metapath_gdf)
        metapath_obj["weight_attr"] = weight_attr
        metapath_obj["num_visualized"] = visualized
        metapath_obj["num_nodes"] = len(node_positions)
        metapath_obj["num_edges"] = visualized
        metapath_obj["is_directed"] = False
        metapath_obj["node_positions"] = [c for p in node_positions for c in p]
        metapath_obj["nodes_data"] = ",".join(str(i) for i in range(len(node_positions)))
        metapath_obj["edges_data"] = ",".join(str(i) for pair in edge_pairs for i in pair)
        metapath_obj["c2g_center_lat"] = center_lat
        metapath_obj["c2g_center_lon"] = center_lon
        metapath_obj["c2g_scale"] = scale


classes = [
    SCIGRAPHS_OT_CreateStreetDualGraph,
    SCIGRAPHS_OT_BridgeAmenitiesToStreets,
    SCIGRAPHS_OT_ComputeMetapaths,
    SCIGRAPHS_OT_ComputeMetapathsWizard,
    SCIGRAPHS_OT_ConvertMetapathsToMesh,
    SCIGRAPHS_OT_ComputeMetapathsByWeight,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

