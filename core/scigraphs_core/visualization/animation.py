import numpy as np
# ``bpy.app.handlers.frame_change_post`` handlers: they rewrite mesh attributes
# every frame so Geometry Nodes and shaders can follow the animation progress.


def update_flow_activation(scene):
    for obj in scene.objects:
        if "flow_time" not in obj or "num_nodes" not in obj:
            continue

        mesh = obj.data
        if (not mesh
                or "flow_distance" not in mesh.attributes
                or "flow_activation" not in mesh.attributes):
            continue

        flow_time = obj.get("flow_time", 0.0)
        mode = obj.get("flow_mode", "DISCRETE")
        smoothness = obj.get("flow_smoothness", 2.0)

        flow_distance_attr = mesh.attributes["flow_distance"]
        flow_activation_attr = mesh.attributes["flow_activation"]

        num_verts = len(mesh.vertices)
        activations = []

        if mode == "DISCRETE":
            for i in range(num_verts):
                distance = flow_distance_attr.data[i].value
                activation = 1.0 if distance <= flow_time else 0.0
                activations.append(activation)
        else:
            for i in range(num_verts):
                distance = flow_distance_attr.data[i].value
                diff = flow_time - distance

                if diff >= smoothness:
                    activation = 1.0
                elif diff <= 0:
                    activation = 0.0
                else:
                    activation = diff / smoothness

                activations.append(activation)

        flow_activation_attr.data.foreach_set("value", activations)
        mesh.update()


def update_traversal_activation(scene):
    for obj in scene.objects:
        if "traversal_time" not in obj or "num_nodes" not in obj:
            continue

        mesh = obj.data
        if (not mesh
                or "traversal_order" not in mesh.attributes
                or "traversal_activation" not in mesh.attributes):
            continue

        traversal_time = obj.get("traversal_time", 0.0)
        mode = obj.get("traversal_mode", "DISCRETE")
        smoothness = obj.get("traversal_smoothness", 2.0)

        traversal_order_attr = mesh.attributes["traversal_order"]
        traversal_activation_attr = mesh.attributes["traversal_activation"]

        num_verts = len(mesh.vertices)

        order = np.empty(num_verts, dtype=np.float32)
        traversal_order_attr.data.foreach_get("value", order)
        unvisited = order < 0

        if mode == "DISCRETE":
            activations = (order <= traversal_time).astype(np.float32)
        else:
            span = float(smoothness) if smoothness else 1.0
            activations = np.clip((traversal_time - order) / span,
                                  0.0, 1.0).astype(np.float32)
        activations[unvisited] = 0.0

        traversal_activation_attr.data.foreach_set("value", activations)
        mesh.update()
