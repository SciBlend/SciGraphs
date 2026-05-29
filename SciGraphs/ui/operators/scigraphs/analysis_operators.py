# Analysis operators (centrality, clustering, community detection, directed analysis)

import bpy
from ....core import analysis, geometry
from ....core.mesh.mesh_utils import parse_graph_data_filtered, expand_node_values_to_mesh
from ....core.visualization.animation import update_flow_activation


def _fit_attribute_values(values, target_length, default=0.0):
    """Return exactly target_length values for Blender foreach_set."""
    fitted = list(values)
    if len(fitted) > target_length:
        return fitted[:target_length]
    if len(fitted) < target_length:
        fitted.extend([default] * (target_length - len(fitted)))
    return fitted


class SCIGRAPHS_OT_CalculateCentrality(bpy.types.Operator):
    """Calculate node centrality metrics."""
    bl_idname = "scigraphs.calculate_centrality"
    bl_label = "Calculate Centrality"
    bl_description = "Calculate node centrality and store as attribute"
    bl_options = {'REGISTER', 'UNDO'}
    
    method: bpy.props.EnumProperty(
        name="Method",
        items=[
            ('degree', "Degree", "Number of connections"),
            ('betweenness', "Betweenness", "Shortest paths through node"),
            ('closeness', "Closeness", "Average distance to others"),
            ('eigenvector', "Eigenvector", "Influence based on connections"),
        ],
        default='degree',
    )
    
    def invoke(self, context, event):
        props = context.scene.scigraphs
        self.method = props.centrality_method
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data_filtered(obj)
        
        centrality_values = analysis.calculate_centrality(graph_data, method=self.method)
        
        expanded_values = expand_node_values_to_mesh(obj, centrality_values)
        
        mesh = obj.data
        attr_name = f"centrality_{self.method}"
        
        if attr_name in mesh.attributes:
            mesh.attributes.remove(mesh.attributes[attr_name])
        
        attr = mesh.attributes.new(name=attr_name, type='FLOAT', domain='POINT')
        attr.data.foreach_set("value", _fit_attribute_values(expanded_values, len(attr.data)))
        
        obj["centrality"] = ",".join(str(v) for v in centrality_values)

        geometry._rebuild_visualization_if_present(obj)

        self.report({'INFO'}, f"Centrality '{self.method}' calculated")
        return {'FINISHED'}


class SCIGRAPHS_OT_CalculateClustering(bpy.types.Operator):
    """Calculate clustering coefficient for nodes."""
    bl_idname = "scigraphs.calculate_clustering"
    bl_label = "Calculate Clustering"
    bl_description = "Calculate clustering coefficient for nodes"
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data_filtered(obj)
        clustering_values = analysis.calculate_clustering(graph_data)
        
        expanded_values = expand_node_values_to_mesh(obj, clustering_values)
        
        mesh = obj.data
        attr_name = "clustering"
        
        if attr_name in mesh.attributes:
            mesh.attributes.remove(mesh.attributes[attr_name])
        
        attr = mesh.attributes.new(name=attr_name, type='FLOAT', domain='POINT')
        attr.data.foreach_set("value", _fit_attribute_values(expanded_values, len(attr.data)))
        
        obj["clustering"] = ",".join(str(v) for v in clustering_values)

        geometry._rebuild_visualization_if_present(obj)

        self.report({'INFO'}, "Clustering attribute created")
        return {'FINISHED'}


class SCIGRAPHS_OT_ApplyClustering(bpy.types.Operator):
    """Apply community detection algorithm."""
    bl_idname = "scigraphs.apply_clustering"
    bl_label = "Apply Clustering"
    bl_description = "Apply clustering algorithm to detect communities"
    bl_options = {'REGISTER', 'UNDO'}
    
    algorithm: bpy.props.EnumProperty(
        name="Algorithm",
        items=[
            ('cpm', "CPM", "Clique Percolation Method"),
            ('infomap', "Infomap", "Map equation"),
            ('rb', "RB", "Reichardt-Bornholdt"),
            ('rn', "RN", "Ronhovde-Nussinov"),
            ('rnsc', "RNSC", "Restricted Neighbourhood Search"),
            ('scluster', "SCluster", "Hierarchical clustering"),
            ('uvcluster', "UVCluster", "Iterative cluster"),
        ],
        default='rn',
    )
    
    resolution: bpy.props.FloatProperty(
        name="Resolution",
        description="Higher = more communities",
        default=1.0,
        min=0.1,
        max=5.0,
    )
    
    def invoke(self, context, event):
        props = context.scene.scigraphs
        self.algorithm = props.clustering_algorithm
        self.resolution = props.clustering_resolution
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data_filtered(obj)
        mesh = obj.data
        
        result = analysis.apply_advanced_clustering(
            graph_data,
            algorithm=self.algorithm,
            resolution=self.resolution,
            seed=props.clustering_seed,
            threshold=props.clustering_threshold
        )
        
        if not result:
            self.report({'ERROR'}, f"{self.algorithm} clustering failed")
            return {'CANCELLED'}
        
        from ....core.mesh.mesh_utils import create_or_update_attribute

        nodes_list = graph_data.nodes
        create_or_update_attribute(mesh, 'cluster_id', 'INT', result['cluster_ids'], obj)
        create_or_update_attribute(mesh, 'cluster_size', 'INT', result['cluster_sizes'], obj)
        create_or_update_attribute(mesh, 'node_clustering', 'FLOAT', result['clustering_coefficients'], obj)
        create_or_update_attribute(mesh, 'modularity', 'FLOAT', [result['modularity']] * len(nodes_list), obj)
        create_or_update_attribute(mesh, 'surprise', 'FLOAT', [result['surprise']] * len(nodes_list), obj)
        
        mesh.update()
        obj.data.update_tag()
        
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        
        geometry._rebuild_visualization_if_present(obj)

        self.report({'INFO'},
                    f"{self.algorithm.upper()}: {result['num_clusters']} clusters, "
                    f"surprise={result['surprise']:.3f}, "
                    f"modularity={result['modularity']:.3f}")

        return {'FINISHED'}


class SCIGRAPHS_OT_CalculateDirectedCentrality(bpy.types.Operator):
    """Calculate centrality metrics for directed graphs."""
    bl_idname = "scigraphs.calculate_directed_centrality"
    bl_label = "Calculate Directed Centrality"
    bl_description = "Calculate centrality metrics for directed graphs"
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        if not obj.get("is_directed", False):
            self.report({'WARNING'}, "Graph is not directed. Use regular centrality.")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data_filtered(obj)
        centrality_values = analysis.calculate_directed_centrality(
            graph_data, method=props.directed_centrality_method
        )
        
        mesh = obj.data
        attr_name = f"directed_{props.directed_centrality_method}"
        
        if attr_name in mesh.attributes:
            mesh.attributes.remove(mesh.attributes[attr_name])
        
        expanded_values = expand_node_values_to_mesh(obj, centrality_values)
        
        attr = mesh.attributes.new(name=attr_name, type='FLOAT', domain='POINT')
        attr.data.foreach_set("value", _fit_attribute_values(expanded_values, len(attr.data)))
        
        geometry._rebuild_visualization_if_present(obj)

        self.report({'INFO'}, f"Directed centrality '{props.directed_centrality_method}' calculated")
        return {'FINISHED'}


class SCIGRAPHS_OT_DetectPatterns(bpy.types.Operator):
    """Detect structural patterns in directed graphs."""
    bl_idname = "scigraphs.detect_patterns"
    bl_label = "Detect Graph Patterns"
    bl_description = "Detect structural patterns (DAG, cycles, connectivity)"
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        if not obj.get("is_directed", False):
            self.report({'WARNING'}, "Graph is not directed")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data_filtered(obj)
        patterns = analysis.detect_graph_patterns(graph_data)
        
        for key, value in patterns.items():
            obj[f"pattern_{key}"] = value
        
        report_lines = []
        if patterns['is_dag']:
            report_lines.append("DAG (Acyclic)")
        if patterns['has_cycles']:
            report_lines.append(f"{patterns['num_cycles']} cycles")
        if patterns['is_strongly_connected']:
            report_lines.append("Strongly connected")
        
        report_msg = " | ".join(report_lines) if report_lines else "Patterns detected"
        self.report({'INFO'}, report_msg)
        
        return {'FINISHED'}


class SCIGRAPHS_OT_AnalyzeFlow(bpy.types.Operator):
    """Analyze flow structure in directed graphs."""
    bl_idname = "scigraphs.analyze_flow"
    bl_label = "Analyze Flow Structure"
    bl_description = "Identify sources, sinks, and bottlenecks"
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        if not obj.get("is_directed", False):
            self.report({'WARNING'}, "Graph is not directed")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data_filtered(obj)
        flow_info = analysis.analyze_flow_structure(graph_data)
        
        mesh = obj.data
        attr_name = "node_flow_type"
        
        if attr_name in mesh.attributes:
            mesh.attributes.remove(mesh.attributes[attr_name])
        
        expanded_node_types = expand_node_values_to_mesh(obj, flow_info['node_types'], default_value=-1)
        
        attr = mesh.attributes.new(name=attr_name, type='INT', domain='POINT')
        attr.data.foreach_set("value", _fit_attribute_values(expanded_node_types, len(attr.data), -1))
        
        attr_name2 = "flow_betweenness"
        if attr_name2 in mesh.attributes:
            mesh.attributes.remove(mesh.attributes[attr_name2])
        
        expanded_betweenness = expand_node_values_to_mesh(obj, flow_info['betweenness'])
        
        attr2 = mesh.attributes.new(name=attr_name2, type='FLOAT', domain='POINT')
        attr2.data.foreach_set("value", _fit_attribute_values(expanded_betweenness, len(attr2.data)))
        
        obj["flow_sources"] = flow_info['num_sources']
        obj["flow_sinks"] = flow_info['num_sinks']
        obj["flow_intermediaries"] = flow_info['num_intermediaries']
        
        geometry._rebuild_visualization_if_present(obj)

        self.report({'INFO'},
            f"Found {flow_info['num_sources']} sources, {flow_info['num_sinks']} sinks, "
            f"{len(flow_info['bottlenecks'])} bottlenecks")

        return {'FINISHED'}


class SCIGRAPHS_OT_FindSCCs(bpy.types.Operator):
    """Find strongly connected components."""
    bl_idname = "scigraphs.find_sccs"
    bl_label = "Find Strong Components"
    bl_description = "Find strongly connected components (cycles)"
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        if not obj.get("is_directed", False):
            self.report({'WARNING'}, "Graph is not directed")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data_filtered(obj)
        scc_info = analysis.find_strongly_connected_components(graph_data)
        
        mesh = obj.data
        attr_name = "scc_id"
        
        if attr_name in mesh.attributes:
            mesh.attributes.remove(mesh.attributes[attr_name])
        
        expanded_ids = expand_node_values_to_mesh(obj, scc_info['component_ids'], default_value=-1)
        
        attr = mesh.attributes.new(name=attr_name, type='INT', domain='POINT')
        attr.data.foreach_set("value", _fit_attribute_values(expanded_ids, len(attr.data), -1))
        
        obj["num_sccs"] = scc_info['num_components']
        obj["largest_scc"] = scc_info['largest_component_size']
        
        geometry._rebuild_visualization_if_present(obj)

        self.report({'INFO'},
            f"Found {scc_info['num_components']} strongly connected components, "
            f"largest: {scc_info['largest_component_size']} nodes")

        return {'FINISHED'}


class SCIGRAPHS_OT_AnimateFlow(bpy.types.Operator):
    """Create flow propagation animation."""
    bl_idname = "scigraphs.animate_flow"
    bl_label = "Animate Flow"
    bl_description = "Create flow animation showing propagation through the directed graph"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        if not obj.get("is_directed", False):
            self.report({'ERROR'}, "Flow animation only works with directed graphs")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data_filtered(obj)
        props = context.scene.scigraphs
        
        flow_data = analysis.calculate_flow_distances(graph_data)
        distances = flow_data['distances']
        max_distance = flow_data['max_distance']
        
        obj["flow_distances"] = distances
        obj["flow_max_distance"] = max_distance
        
        mode = props.flow_animation_mode
        loop = props.flow_animation_loop
        frames_per_cycle = props.flow_animation_speed
        smoothness = props.flow_animation_smoothness
        
        obj["flow_mode"] = mode
        obj["flow_smoothness"] = smoothness
        
        if mode == 'DISCRETE':
            max_flow_time = max_distance
        else:
            max_flow_time = max_distance + smoothness
        
        mesh = obj.data
        if "flow_distance" not in mesh.attributes:
            mesh.attributes.new(name="flow_distance", type='FLOAT', domain='POINT')
        
        expanded_distances = expand_node_values_to_mesh(obj, distances, default_value=-1)
        
        attr = mesh.attributes["flow_distance"]
        attr.data.foreach_set("value", _fit_attribute_values(expanded_distances, len(attr.data), -1))
        
        if "flow_activation" not in mesh.attributes:
            mesh.attributes.new(name="flow_activation", type='FLOAT', domain='POINT')
        
        obj["flow_time"] = 0.0
        
        if "_RNA_UI" not in obj:
            obj["_RNA_UI"] = {}
        
        obj["_RNA_UI"]["flow_time"] = {
            "min": 0.0,
            "max": float(max_flow_time),
            "soft_min": 0.0,
            "soft_max": float(max_flow_time),
            "description": "Current flow propagation time"
        }
        
        if loop:
            timeline_start = context.scene.frame_start
            timeline_end = context.scene.frame_end
            timeline_length = timeline_end - timeline_start
            
            num_cycles = max(1, int(timeline_length / frames_per_cycle))
            actual_cycle_frames = timeline_length / num_cycles
            
            for cycle in range(num_cycles + 1):
                frame_start = timeline_start + (cycle * actual_cycle_frames)
                obj["flow_time"] = 0.0
                obj.keyframe_insert(data_path='["flow_time"]', frame=int(frame_start))
                
                frame_end = timeline_start + ((cycle + 0.999) * actual_cycle_frames)
                obj["flow_time"] = float(max_flow_time)
                obj.keyframe_insert(data_path='["flow_time"]', frame=int(frame_end))
            
            self.report({'INFO'}, 
                f"Flow animation created: {num_cycles} cycles over {timeline_length} frames")
        else:
            frame_start = context.scene.frame_start
            frame_end = frame_start + frames_per_cycle
            
            obj["flow_time"] = 0.0
            obj.keyframe_insert(data_path='["flow_time"]', frame=frame_start)
            
            obj["flow_time"] = float(max_flow_time)
            obj.keyframe_insert(data_path='["flow_time"]', frame=frame_end)
            
            context.scene.frame_end = frame_end
            
            self.report({'INFO'}, f"Flow animation created: {frames_per_cycle} frames (single pass)")
        
        context.scene.frame_set(context.scene.frame_start)
        
        if update_flow_activation not in bpy.app.handlers.frame_change_post:
            bpy.app.handlers.frame_change_post.append(update_flow_activation)
        
        update_flow_activation(context.scene)
        
        return {'FINISHED'}


def register():
    bpy.utils.register_class(SCIGRAPHS_OT_CalculateCentrality)
    bpy.utils.register_class(SCIGRAPHS_OT_CalculateClustering)
    bpy.utils.register_class(SCIGRAPHS_OT_ApplyClustering)
    bpy.utils.register_class(SCIGRAPHS_OT_CalculateDirectedCentrality)
    bpy.utils.register_class(SCIGRAPHS_OT_DetectPatterns)
    bpy.utils.register_class(SCIGRAPHS_OT_AnalyzeFlow)
    bpy.utils.register_class(SCIGRAPHS_OT_FindSCCs)
    bpy.utils.register_class(SCIGRAPHS_OT_AnimateFlow)


def unregister():
    if update_flow_activation in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(update_flow_activation)
    
    bpy.utils.unregister_class(SCIGRAPHS_OT_AnimateFlow)
    bpy.utils.unregister_class(SCIGRAPHS_OT_FindSCCs)
    bpy.utils.unregister_class(SCIGRAPHS_OT_AnalyzeFlow)
    bpy.utils.unregister_class(SCIGRAPHS_OT_DetectPatterns)
    bpy.utils.unregister_class(SCIGRAPHS_OT_CalculateDirectedCentrality)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ApplyClustering)
    bpy.utils.unregister_class(SCIGRAPHS_OT_CalculateClustering)
    bpy.utils.unregister_class(SCIGRAPHS_OT_CalculateCentrality)
