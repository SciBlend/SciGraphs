# Graph algorithms operators (traversal, pathfinding, spanning trees, network flow)

import bpy
import numpy as np
from ....core import analysis, pathfinding, spanning, network_flow
from ....core.mesh.mesh_utils import parse_graph_data, expand_node_values_to_mesh
from ....core.visualization.animation import update_traversal_activation


class SCIGRAPHS_OT_AnimateTraversal(bpy.types.Operator):
    """Create animation showing graph traversal (BFS or DFS)."""
    bl_idname = "scigraphs.animate_traversal"
    bl_label = "Animate Traversal"
    bl_description = "Create animation showing graph traversal (BFS or DFS)"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data(obj)
        props = context.scene.scigraphs
        
        start_nodes = None
        if props.traversal_start_mode == 'MANUAL' and props.traversal_start_nodes.strip():
            try:
                start_nodes = [int(x.strip()) for x in props.traversal_start_nodes.split(',')]
                num_nodes = len(graph_data.nodes)
                invalid = [n for n in start_nodes if n < 0 or n >= num_nodes]
                if invalid:
                    self.report({'ERROR'}, f"Invalid node indices: {invalid}. Valid range: 0-{num_nodes-1}")
                    return {'CANCELLED'}
            except ValueError:
                self.report({'ERROR'}, "Invalid start nodes format. Use comma-separated integers")
                return {'CANCELLED'}
        
        is_directed = obj.get("is_directed", False)
        algorithm = props.traversal_algorithm
        
        if algorithm == 'BFS':
            traversal_data = analysis.calculate_bfs_traversal(graph_data, start_nodes, is_directed)
            algo_name = "BFS"
        else:
            traversal_data = analysis.calculate_dfs_traversal(graph_data, start_nodes, is_directed)
            algo_name = "DFS"
        
        order = traversal_data['order']
        max_order = traversal_data['max_order']
        visited_count = traversal_data['visited_count']
        
        obj["traversal_order"] = order
        obj["traversal_max_order"] = max_order
        obj["traversal_algorithm"] = algorithm
        obj["traversal_visited_count"] = visited_count
        
        mode = props.traversal_animation_mode
        loop = props.traversal_animation_loop
        frames_per_cycle = props.traversal_animation_speed
        smoothness = props.traversal_animation_smoothness
        
        obj["traversal_mode"] = mode
        obj["traversal_smoothness"] = smoothness
        
        max_traversal_time = max_order + smoothness if mode == 'CONTINUOUS' else max_order
        
        mesh = obj.data
        if "traversal_order" not in mesh.attributes:
            mesh.attributes.new(name="traversal_order", type='FLOAT', domain='POINT')
        
        expanded_order = expand_node_values_to_mesh(obj, order, default_value=-1)
        
        attr = mesh.attributes["traversal_order"]
        attr.data.foreach_set("value", expanded_order)
        
        if "traversal_activation" not in mesh.attributes:
            mesh.attributes.new(name="traversal_activation", type='FLOAT', domain='POINT')
        
        obj["traversal_time"] = 0.0
        if "_RNA_UI" not in obj:
            obj["_RNA_UI"] = {}
        obj["_RNA_UI"]["traversal_time"] = {
            "min": 0.0,
            "max": float(max_traversal_time),
            "description": "Current traversal progression time"
        }
        
        if loop:
            timeline_start = context.scene.frame_start
            timeline_end = context.scene.frame_end
            timeline_length = timeline_end - timeline_start
            num_cycles = max(1, int(timeline_length / frames_per_cycle))
            actual_cycle_frames = timeline_length / num_cycles
            
            for cycle in range(num_cycles + 1):
                frame_start = timeline_start + (cycle * actual_cycle_frames)
                obj["traversal_time"] = 0.0
                obj.keyframe_insert(data_path='["traversal_time"]', frame=int(frame_start))
                
                frame_end = timeline_start + ((cycle + 0.999) * actual_cycle_frames)
                obj["traversal_time"] = float(max_traversal_time)
                obj.keyframe_insert(data_path='["traversal_time"]', frame=int(frame_end))
            
            self.report({'INFO'}, f"{algo_name} animation: {num_cycles} cycles, {visited_count} nodes")
        else:
            frame_start = context.scene.frame_start
            frame_end = frame_start + frames_per_cycle
            
            obj["traversal_time"] = 0.0
            obj.keyframe_insert(data_path='["traversal_time"]', frame=frame_start)
            
            obj["traversal_time"] = float(max_traversal_time)
            obj.keyframe_insert(data_path='["traversal_time"]', frame=frame_end)
            
            context.scene.frame_end = frame_end
            self.report({'INFO'}, f"{algo_name} animation: {frames_per_cycle} frames, {visited_count} nodes")
        
        context.scene.frame_set(context.scene.frame_start)
        
        if update_traversal_activation not in bpy.app.handlers.frame_change_post:
            bpy.app.handlers.frame_change_post.append(update_traversal_activation)
        
        update_traversal_activation(context.scene)
        return {'FINISHED'}


class SCIGRAPHS_OT_FindShortestPath(bpy.types.Operator):
    """Find shortest path between two nodes using Dijkstra."""
    bl_idname = "scigraphs.find_shortest_path"
    bl_label = "Find Shortest Path"
    bl_description = "Find shortest path between source and target nodes"
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data(obj)
        
        try:
            source = int(props.pathfinding_source)
            target = int(props.pathfinding_target)
        except ValueError:
            self.report({'ERROR'}, "Invalid source or target node")
            return {'CANCELLED'}
        
        if props.pathfinding_algorithm == 'ASTAR' and "node_positions" in obj:
            positions_flat = obj["node_positions"]
            num_nodes = len(graph_data.nodes)
            positions = np.array(positions_flat).reshape((num_nodes, 3))
            result = pathfinding.a_star_path(graph_data, source, target, positions)
        else:
            result = pathfinding.dijkstra_shortest_path(graph_data, source, target)
        
        if result['exists']:
            mesh = obj.data
            path_attr = np.zeros(len(graph_data.nodes), dtype=np.float32)
            for node_idx in result['path']:
                path_attr[node_idx] = 1.0
            
            expanded_path = expand_node_values_to_mesh(obj, path_attr.tolist())
            
            if "shortest_path" not in mesh.attributes:
                mesh.attributes.new(name="shortest_path", type='FLOAT', domain='POINT')
            attr = mesh.attributes["shortest_path"]
            attr.data.foreach_set("value", expanded_path)

            obj["shortest_path_source"] = source
            obj["shortest_path_target"] = target
            obj["shortest_path_nodes"] = len(result['path'])
            obj["shortest_path_length"] = float(result['distance'])
            
            self.report({'INFO'}, f"Path found: {len(result['path'])} nodes, distance: {result['distance']:.2f}")
        else:
            self.report({'WARNING'}, "No path exists between source and target")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_ComputeMST(bpy.types.Operator):
    """Compute Minimum Spanning Tree."""
    bl_idname = "scigraphs.compute_mst"
    bl_label = "Compute MST"
    bl_description = "Compute Minimum Spanning Tree using selected algorithm"
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data(obj)
        
        if props.spanning_algorithm == 'KRUSKAL':
            result = spanning.minimum_spanning_tree_kruskal(graph_data)
            algo_name = "Kruskal"
        elif props.spanning_algorithm == 'PRIM':
            result = spanning.minimum_spanning_tree_prim(graph_data)
            algo_name = "Prim"
        else:
            result = spanning.maximum_spanning_tree(graph_data)
            algo_name = "Maximum ST"
        
        obj["mst_edges"] = result['num_edges']
        obj["mst_weight"] = result['total_weight']
        
        self.report({'INFO'}, f"{algo_name}: {result['num_edges']} edges, weight: {result['total_weight']:.2f}")
        return {'FINISHED'}


class SCIGRAPHS_OT_ComputeMaxFlow(bpy.types.Operator):
    """Compute maximum flow in network."""
    bl_idname = "scigraphs.compute_max_flow"
    bl_label = "Compute Max Flow"
    bl_description = "Compute maximum flow from source to sink"
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        if not obj.get("is_directed", False):
            self.report({'ERROR'}, "Max flow requires directed graph")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data(obj)
        
        try:
            source = int(props.flow_source)
            sink = int(props.flow_sink)
        except ValueError:
            self.report({'ERROR'}, "Invalid source or sink node")
            return {'CANCELLED'}
        
        result = network_flow.maximum_flow_ford_fulkerson(graph_data, source, sink)
        
        obj["max_flow_value"] = result['max_flow']
        
        self.report({'INFO'}, f"Max flow computed: {result['max_flow']:.2f}")
        return {'FINISHED'}


class SCIGRAPHS_OT_ComputeMinCut(bpy.types.Operator):
    """Compute minimum cut in network."""
    bl_idname = "scigraphs.compute_min_cut"
    bl_label = "Compute Min Cut"
    bl_description = "Find minimum cut separating source from sink"
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        if not obj.get("is_directed", False):
            self.report({'ERROR'}, "Min cut requires directed graph")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data(obj)
        
        try:
            source = int(props.flow_source)
            sink = int(props.flow_sink)
        except ValueError:
            self.report({'ERROR'}, "Invalid source or sink node")
            return {'CANCELLED'}
        
        result = network_flow.minimum_cut(graph_data, source, sink)
        
        mesh = obj.data
        partition_attr = np.zeros(len(graph_data.nodes), dtype=np.float32)
        for node_idx in result['reachable']:
            partition_attr[node_idx] = 1.0
        
        expanded_partition = expand_node_values_to_mesh(obj, partition_attr.tolist())
        
        if "min_cut_partition" not in mesh.attributes:
            mesh.attributes.new(name="min_cut_partition", type='FLOAT', domain='POINT')
        attr = mesh.attributes["min_cut_partition"]
        attr.data.foreach_set("value", expanded_partition)
        
        obj["min_cut_value"] = result['cut_value']
        
        self.report({'INFO'}, f"Min cut: {result['cut_value']:.2f}, {len(result['cut_edges'])} edges")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(SCIGRAPHS_OT_AnimateTraversal)
    bpy.utils.register_class(SCIGRAPHS_OT_FindShortestPath)
    bpy.utils.register_class(SCIGRAPHS_OT_ComputeMST)
    bpy.utils.register_class(SCIGRAPHS_OT_ComputeMaxFlow)
    bpy.utils.register_class(SCIGRAPHS_OT_ComputeMinCut)


def unregister():
    if update_traversal_activation in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(update_traversal_activation)
    
    bpy.utils.unregister_class(SCIGRAPHS_OT_ComputeMinCut)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ComputeMaxFlow)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ComputeMST)
    bpy.utils.unregister_class(SCIGRAPHS_OT_FindShortestPath)
    bpy.utils.unregister_class(SCIGRAPHS_OT_AnimateTraversal)
