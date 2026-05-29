# Topological analysis operators for SciGraphs addon
# Handles planarity, genus, face detection, and surface embedding

import bpy
import bmesh
import numpy as np
from ....core import topology, geometry
from ....core.mesh.mesh_utils import (
    parse_graph_data_filtered as _parse_graph_data,
    get_vertex_positions as _get_vertex_positions,
    expand_node_values_to_mesh as _expand_node_values_to_mesh,
)


class SCIGRAPHS_OT_CheckPlanarity(bpy.types.Operator):
    """Check if the graph can be embedded in a plane without edge crossings."""
    bl_idname = "scigraphs.check_planarity"
    bl_label = "Check Planarity"
    bl_description = "Test if graph is planar (can be drawn without edge crossings)"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        graph_data = _parse_graph_data(obj)
        
        is_planar, embedding = topology.check_planarity_nx(graph_data)
        
        # Store results
        obj["topo_is_planar"] = is_planar
        
        if is_planar:
            # Compute Euler characteristic for planar graph
            euler_data = topology.get_euler_characteristic(graph_data, embedding)
            obj["topo_euler_V"] = euler_data['V']
            obj["topo_euler_E"] = euler_data['E']
            obj["topo_euler_F"] = euler_data['F']
            obj["topo_euler_chi"] = euler_data['chi']
            
            self.report({'INFO'}, 
                f"Graph is PLANAR: V={euler_data['V']}, E={euler_data['E']}, "
                f"F={euler_data['F']}, chi={euler_data['chi']}")
        else:
            # Find Kuratowski subgraph for non-planar
            kuratowski = topology.detect_kuratowski_subgraph(graph_data)
            obj["topo_kuratowski_type"] = kuratowski.get('kuratowski_type', 'unknown')
            
            self.report({'WARNING'}, 
                f"Graph is NON-PLANAR (contains {kuratowski.get('kuratowski_type', 'K5 or K3,3')} subdivision)")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_CalculateGenus(bpy.types.Operator):
    """Calculate the genus (number of handles) of the minimal embedding surface."""
    bl_idname = "scigraphs.calculate_genus"
    bl_label = "Calculate Genus"
    bl_description = "Compute the genus of the minimal surface for graph embedding"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        graph_data = _parse_graph_data(obj)
        
        genus_result = topology.calculate_genus(graph_data)
        
        # Store results
        obj["topo_is_planar"] = genus_result['is_planar']
        obj["topo_genus_lower_bound"] = genus_result['genus_lower_bound']
        
        if genus_result['genus_exact'] is not None:
            obj["topo_genus_exact"] = genus_result['genus_exact']
        
        euler_data = genus_result['euler_data']
        obj["topo_euler_V"] = euler_data['V']
        obj["topo_euler_E"] = euler_data['E']
        
        if genus_result['is_planar']:
            obj["topo_euler_F"] = euler_data['F']
            obj["topo_euler_chi"] = euler_data['chi']
            self.report({'INFO'}, 
                f"Genus = 0 (planar): V={euler_data['V']}, E={euler_data['E']}, F={euler_data['F']}")
        else:
            genus = genus_result['genus_lower_bound']
            self.report({'INFO'}, 
                f"Genus >= {genus} (non-planar): V={euler_data['V']}, E={euler_data['E']}")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_ComputeFaces(bpy.types.Operator):
    """Compute and visualize faces of a planar graph embedding."""
    bl_idname = "scigraphs.compute_faces"
    bl_label = "Compute Faces"
    bl_description = "Find all faces in the planar embedding and create face_id attribute"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        graph_data = _parse_graph_data(obj)
        
        # First check planarity
        is_planar, embedding = topology.check_planarity_nx(graph_data)
        
        if not is_planar:
            self.report({'ERROR'}, "Cannot compute faces: graph is not planar")
            return {'CANCELLED'}
        
        # Get face assignments
        face_result = topology.get_face_node_assignments(graph_data, embedding)
        
        if face_result.get('error'):
            self.report({'ERROR'}, face_result['error'])
            return {'CANCELLED'}
        
        # Store face data
        obj["topo_num_faces"] = face_result['num_faces']
        obj["topo_faces"] = str(face_result['faces'])
        
        # Create face_id attribute on mesh
        mesh = obj.data
        attr_name = "face_id"
        
        if attr_name in mesh.attributes:
            mesh.attributes.remove(mesh.attributes[attr_name])
        
        # Expand for OSMnx compatibility
        node_face_ids = face_result['node_face_ids']
        expanded_values = _expand_node_values_to_mesh(obj, node_face_ids, default_value=-1)
        
        attr = mesh.attributes.new(name=attr_name, type='INT', domain='POINT')
        attr.data.foreach_set("value", expanded_values)
        
        mesh.update()
        
        geometry._rebuild_visualization_if_present(obj)

        self.report({'INFO'},
            f"Found {face_result['num_faces']} faces. Attribute 'face_id' created.")

        return {'FINISHED'}


class SCIGRAPHS_OT_ValidateCrossings(bpy.types.Operator):
    """Validate if the current embedding has edge crossings."""
    bl_idname = "scigraphs.validate_crossings"
    bl_label = "Validate Crossings"
    bl_description = "Check if any edges cross in the current 3D embedding"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        mesh = obj.data
        
        # Get vertex positions
        vertices = []
        for v in mesh.vertices:
            vertices.append(list(v.co))
        
        # Get edges
        edges = []
        for e in mesh.edges:
            edges.append((e.vertices[0], e.vertices[1]))
        
        # Detect crossings
        result = topology.detect_edge_crossings_3d(vertices, edges, tolerance=0.01)
        
        obj["topo_has_crossings"] = result['has_crossings']
        obj["topo_num_crossings"] = result['num_crossings']
        
        if result['has_crossings']:
            self.report({'WARNING'}, 
                f"Embedding has {result['num_crossings']} edge crossing(s)!")
        else:
            self.report({'INFO'}, "Embedding is crossing-free!")
        
        return {'FINISHED'}



class SCIGRAPHS_OT_VisualizeSurface(bpy.types.Operator):
    """Compute a crossing-free planar embedding."""
    bl_idname = "scigraphs.visualize_surface"
    bl_label = "Compute Embedding"
    bl_description = "Compute planar embedding with surface mesh"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        graph_data = _parse_graph_data(obj)
        num_nodes = obj.get("num_nodes", 0)
        
        result = self._create_planar_embedding(context, obj, graph_data, num_nodes)
        
        if result['success']:
            self.report({'INFO'}, result['message'])
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, result['message'])
            return {'CANCELLED'}
    
    def _create_planar_embedding(self, context, obj, graph_data, num_nodes):
        """Create planar embedding with plane surface."""
        is_planar = obj.get("topo_is_planar", None)
        if is_planar is None:
            return {'success': False, 'message': "Run planarity check first"}
        if not is_planar:
            return {'success': False, 'message': "Graph is not planar"}
        
        # Check if graph has nodes
        if num_nodes == 0 or len(graph_data.nodes) == 0:
            return {'success': False, 'message': "Graph has no nodes"}
        
        layout_result = topology.compute_planar_layout(graph_data, scale=5.0)
        if not layout_result['success']:
            return {'success': False, 'message': f"Layout failed: {layout_result['error']}"}
        
        node_positions = layout_result['positions']
        
        # Check if positions are valid
        if node_positions is None or len(node_positions) == 0:
            return {'success': False, 'message': "Layout returned no positions"}
        
        # Create new mesh with is_intersection attribute
        self._rebuild_graph_mesh(obj, graph_data, node_positions)
        
        # Create plane surface mesh
        self._create_plane_surface(context, obj, node_positions)
        
        obj["topo_layout_type"] = "planar_crossing_free"
        return {'success': True, 'message': "Planar embedding - edges DO NOT cross"}
    
    def _rebuild_graph_mesh(self, obj, graph_data, node_positions):
        """
        Rebuild the graph mesh with proper is_intersection attribute.
        Creates straight edges for planar embedding.
        """
        import bmesh
        
        num_nodes = len(graph_data.nodes)
        
        # Create new bmesh
        bm = bmesh.new()
        
        # Track vertex indices
        vertex_list = []
        is_intersection_values = []
        
        # Add node vertices (is_intersection = 1)
        for i in range(num_nodes):
            v = bm.verts.new(node_positions[i])
            vertex_list.append(v)
            is_intersection_values.append(1)
        
        bm.verts.ensure_lookup_table()
        
        # Build node index mapping
        node_to_idx = graph_data.node_to_index
        
        # Add edges (straight)
        for src, tgt in graph_data.edges:
            if src not in node_to_idx or tgt not in node_to_idx:
                continue
            
            src_idx = node_to_idx[src]
            tgt_idx = node_to_idx[tgt]
            
            if src_idx >= num_nodes or tgt_idx >= num_nodes:
                continue
            
            src_vert = vertex_list[src_idx]
            tgt_vert = vertex_list[tgt_idx]
            
            try:
                bm.edges.new([src_vert, tgt_vert])
            except ValueError:
                pass
        
        # Update the mesh
        old_mesh = obj.data
        new_mesh = bpy.data.meshes.new(name="SciGraph_Embedded")
        bm.to_mesh(new_mesh)
        bm.free()
        
        obj.data = new_mesh
        
        # Remove old mesh if not used elsewhere
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
        
        # Create is_intersection attribute
        mesh = obj.data
        if "is_intersection" in mesh.attributes:
            mesh.attributes.remove(mesh.attributes["is_intersection"])
        
        attr = mesh.attributes.new(name="is_intersection", type='INT', domain='POINT')
        attr.data.foreach_set("value", is_intersection_values)
        
        # Update stored counts
        obj["num_nodes"] = num_nodes
        obj["num_mesh_verts"] = len(vertex_list)
        obj["num_curve_verts"] = 0
        
        mesh.update()
    def _create_plane_surface(self, context, obj, node_positions):
        """Create a plane mesh as child of the graph object."""
        # Remove old surface if exists
        old_surface_name = obj.get("topo_surface_child", "")
        if old_surface_name and old_surface_name in bpy.data.objects:
            old_obj = bpy.data.objects[old_surface_name]
            bpy.data.objects.remove(old_obj, do_unlink=True)
        
        # Handle empty positions
        if node_positions is None or len(node_positions) == 0:
            # Default plane at origin
            center = np.array([0.0, 0.0, 0.0])
            size = np.array([5.0, 5.0, 0.0])
        else:
            # Calculate bounds
            min_pos = np.min(node_positions, axis=0)
            max_pos = np.max(node_positions, axis=0)
            center = (min_pos + max_pos) / 2
            size = max_pos - min_pos
            # Ensure minimum size
            size = np.maximum(size, 1.0)
        
        padding = 0.5
        
        # Create plane
        bpy.ops.mesh.primitive_plane_add(
            size=1,
            location=(center[0], center[1], center[2] - 0.01)
        )
        plane = context.active_object
        plane.name = f"{obj.name}_Surface"
        plane.scale = (size[0] + padding, size[1] + padding, 1)
        
        # Apply scale
        bpy.ops.object.transform_apply(scale=True)
        
        # Semi-transparent material
        self._create_surface_material(plane, (0.3, 0.5, 0.8, 0.3), "TopoPlane_Mat")
        
        # Parent to graph object
        plane.parent = obj
        obj["topo_surface_child"] = plane.name
        
        # Restore active object
        context.view_layer.objects.active = obj
    def _create_surface_material(self, obj, color, mat_name):
        """Create a semi-transparent material for the surface."""
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            mat.blend_method = 'BLEND'
            
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            # Clear default nodes
            nodes.clear()
            
            # Create nodes
            output = nodes.new('ShaderNodeOutputMaterial')
            bsdf = nodes.new('ShaderNodeBsdfPrincipled')
            
            output.location = (300, 0)
            bsdf.location = (0, 0)
            
            # Set color and transparency
            bsdf.inputs['Base Color'].default_value = color[:3] + (1.0,)
            bsdf.inputs['Alpha'].default_value = color[3]
            bsdf.inputs['Roughness'].default_value = 0.8
            
            links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)



class SCIGRAPHS_OT_CreateDualGraph(bpy.types.Operator):
    """Create the geometric dual graph G* of a planar graph."""
    bl_idname = "scigraphs.create_dual_graph"
    bl_label = "Create Dual Graph"
    bl_description = "Generate the geometric dual graph (vertices at face centroids, edges between adjacent faces)"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        # Check if graph is planar
        is_planar = obj.get("topo_is_planar", None)
        if is_planar is None:
            self.report({'ERROR'}, "Run planarity check first")
            return {'CANCELLED'}
        
        if not is_planar:
            self.report({'ERROR'}, "Graph must be planar to compute dual")
            return {'CANCELLED'}
        
        graph_data = _parse_graph_data(obj)
        positions = _get_vertex_positions(obj)
        
        # Use only node positions (not curve vertices)
        num_nodes = obj.get("num_nodes", len(positions))
        node_positions = positions[:num_nodes]
        
        # Compute dual graph
        dual_result = topology.compute_geometric_dual_3d(graph_data, node_positions)
        
        if not dual_result.get('success', False):
            self.report({'ERROR'}, dual_result.get('error', 'Failed to compute dual'))
            return {'CANCELLED'}
        
        # Create the dual graph mesh object
        dual_positions = dual_result['positions']
        dual_edges = dual_result['edges']
        num_dual_nodes = len(dual_result['nodes'])
        
        # Create mesh
        bm = bmesh.new()
        
        # Add vertices
        dual_verts = []
        for pos in dual_positions:
            v = bm.verts.new(pos)
            dual_verts.append(v)
        
        bm.verts.ensure_lookup_table()
        
        # Add edges
        for u, v in dual_edges:
            if u < len(dual_verts) and v < len(dual_verts):
                try:
                    bm.edges.new([dual_verts[u], dual_verts[v]])
                except ValueError:
                    pass  # Edge already exists
        
        # Create Blender mesh and object
        mesh = bpy.data.meshes.new(name=f"{obj.name}_Dual_Mesh")
        bm.to_mesh(mesh)
        bm.free()
        
        dual_obj = bpy.data.objects.new(f"{obj.name}_Dual", mesh)
        context.collection.objects.link(dual_obj)
        
        # Create wireframe material for the dual
        self._create_dual_material(dual_obj)
        
        # Parent to original graph
        dual_obj.parent = obj
        
        # Store reference
        obj["topo_dual_child"] = dual_obj.name
        
        # Store dual graph data on the dual object (for further analysis)
        dual_obj["is_dual_graph"] = True
        dual_obj["num_nodes"] = num_dual_nodes
        dual_obj["num_edges"] = len(dual_edges)
        dual_obj["original_graph"] = obj.name
        
        # Store graph data in the same format as regular graphs
        # so that topology operators can work on the dual
        dual_nodes_str = ",".join([f"face_{i}" for i in range(num_dual_nodes)])
        dual_obj["nodes_data"] = dual_nodes_str
        
        # Store edges as flat list: "src1,tgt1,src2,tgt2,..."
        edges_flat = []
        for u, v in dual_edges:
            edges_flat.append(f"face_{u}")
            edges_flat.append(f"face_{v}")
        dual_obj["edges_data"] = ",".join(edges_flat)
        
        # Store positions for the dual (flattened)
        dual_obj["node_positions"] = dual_positions.flatten().tolist()
        
        self.report({'INFO'}, 
            f"Dual graph created: {num_dual_nodes} faces -> vertices, "
            f"{len(dual_edges)} shared edges -> dual edges")
        
        return {'FINISHED'}
    
    def _create_dual_material(self, obj):
        """Create a wireframe material for the dual graph."""
        mat_name = "DualGraph_Mat"
        mat = bpy.data.materials.get(mat_name)
        
        if mat is None:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            mat.blend_method = 'BLEND'
            
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            nodes.clear()
            
            output = nodes.new('ShaderNodeOutputMaterial')
            bsdf = nodes.new('ShaderNodeBsdfPrincipled')
            
            output.location = (300, 0)
            bsdf.location = (0, 0)
            
            # Orange-red color for dual (contrasts with typical graph blue)
            bsdf.inputs['Base Color'].default_value = (0.9, 0.4, 0.1, 1.0)
            bsdf.inputs['Alpha'].default_value = 0.8
            bsdf.inputs['Roughness'].default_value = 0.5
            bsdf.inputs['Emission Color'].default_value = (0.9, 0.4, 0.1, 1.0)
            bsdf.inputs['Emission Strength'].default_value = 0.2
            
            links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)



class SCIGRAPHS_OT_ToggleDualGraph(bpy.types.Operator):
    """Toggle visibility of the dual graph."""
    bl_idname = "scigraphs.toggle_dual_graph"
    bl_label = "Toggle Dual"
    bl_description = "Toggle visibility of the dual graph"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj:
            self.report({'ERROR'}, "No object selected")
            return {'CANCELLED'}
        
        dual_name = obj.get("topo_dual_child", "")
        if not dual_name or dual_name not in bpy.data.objects:
            self.report({'ERROR'}, "No dual graph found")
            return {'CANCELLED'}
        
        dual_obj = bpy.data.objects[dual_name]
        dual_obj.hide_viewport = not dual_obj.hide_viewport
        dual_obj.hide_render = dual_obj.hide_viewport
        
        status = "hidden" if dual_obj.hide_viewport else "visible"
        self.report({'INFO'}, f"Dual graph {status}")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_RemoveDualGraph(bpy.types.Operator):
    """Remove the dual graph object."""
    bl_idname = "scigraphs.remove_dual_graph"
    bl_label = "Remove Dual"
    bl_description = "Remove the dual graph object"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj:
            self.report({'ERROR'}, "No object selected")
            return {'CANCELLED'}
        
        dual_name = obj.get("topo_dual_child", "")
        if dual_name and dual_name in bpy.data.objects:
            dual_obj = bpy.data.objects[dual_name]
            bpy.data.objects.remove(dual_obj, do_unlink=True)
        
        if "topo_dual_child" in obj:
            del obj["topo_dual_child"]
        
        self.report({'INFO'}, "Dual graph removed")
        return {'FINISHED'}


class SCIGRAPHS_OT_ToggleTopoSurface(bpy.types.Operator):
    """Toggle visibility of the topology surface mesh."""
    bl_idname = "scigraphs.toggle_topo_surface"
    bl_label = "Toggle Surface"
    bl_description = "Toggle visibility of the embedding surface mesh"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj:
            self.report({'ERROR'}, "No object selected")
            return {'CANCELLED'}
        
        surface_name = obj.get("topo_surface_child", "")
        if not surface_name or surface_name not in bpy.data.objects:
            self.report({'ERROR'}, "No surface mesh found")
            return {'CANCELLED'}
        
        surface_obj = bpy.data.objects[surface_name]
        surface_obj.hide_viewport = not surface_obj.hide_viewport
        surface_obj.hide_render = surface_obj.hide_viewport
        
        status = "hidden" if surface_obj.hide_viewport else "visible"
        self.report({'INFO'}, f"Surface {status}")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_RemoveTopoSurface(bpy.types.Operator):
    """Remove the topology surface mesh."""
    bl_idname = "scigraphs.remove_topo_surface"
    bl_label = "Remove Surface"
    bl_description = "Remove the embedding surface mesh"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj:
            self.report({'ERROR'}, "No object selected")
            return {'CANCELLED'}
        
        surface_name = obj.get("topo_surface_child", "")
        if surface_name and surface_name in bpy.data.objects:
            surface_obj = bpy.data.objects[surface_name]
            bpy.data.objects.remove(surface_obj, do_unlink=True)
        
        if "topo_surface_child" in obj:
            del obj["topo_surface_child"]
        
        self.report({'INFO'}, "Surface removed")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(SCIGRAPHS_OT_CheckPlanarity)
    bpy.utils.register_class(SCIGRAPHS_OT_CalculateGenus)
    bpy.utils.register_class(SCIGRAPHS_OT_ComputeFaces)
    bpy.utils.register_class(SCIGRAPHS_OT_ValidateCrossings)
    bpy.utils.register_class(SCIGRAPHS_OT_VisualizeSurface)
    bpy.utils.register_class(SCIGRAPHS_OT_CreateDualGraph)
    bpy.utils.register_class(SCIGRAPHS_OT_ToggleDualGraph)
    bpy.utils.register_class(SCIGRAPHS_OT_RemoveDualGraph)
    bpy.utils.register_class(SCIGRAPHS_OT_ToggleTopoSurface)
    bpy.utils.register_class(SCIGRAPHS_OT_RemoveTopoSurface)


def unregister():
    bpy.utils.unregister_class(SCIGRAPHS_OT_RemoveTopoSurface)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ToggleTopoSurface)
    bpy.utils.unregister_class(SCIGRAPHS_OT_RemoveDualGraph)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ToggleDualGraph)
    bpy.utils.unregister_class(SCIGRAPHS_OT_CreateDualGraph)
    bpy.utils.unregister_class(SCIGRAPHS_OT_VisualizeSurface)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ValidateCrossings)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ComputeFaces)
    bpy.utils.unregister_class(SCIGRAPHS_OT_CalculateGenus)
    bpy.utils.unregister_class(SCIGRAPHS_OT_CheckPlanarity)
