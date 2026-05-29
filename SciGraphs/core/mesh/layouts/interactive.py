"""Interactive layout iteration helpers."""

from .common import *
from .basic import *
from .networkx_layouts import *
from .forceatlas import *
from .igraph_layouts import *
from .circle_packing import *
from .hierarchical import *
from .yifan_hu import *

def _compute_layout_for_algorithm(G, num_nodes, algorithm, scale, props=None):
    """
    Compute layout positions for a given algorithm with custom parameters.
    Returns numpy array of shape (num_nodes, 3).

    Args:
        G: NetworkX graph
        num_nodes: Number of nodes
        algorithm: Algorithm identifier string
        scale: Layout scale
        props: Scene properties (SciGraphsProperties) for algorithm-specific parameters
    """
    iterations = 50  # Default iterations

    if algorithm == 'FORCEATLAS2':
        if props:
            return _forceatlas2_layout(
                G, iterations, scale,
                scaling_ratio=props.fa2_scaling_ratio,
                gravity=props.fa2_gravity,
                strong_gravity=props.fa2_strong_gravity,
                lin_log_mode=props.fa2_lin_log_mode,
                barnes_hut_optimize=props.fa2_barnes_hut_optimize,
                barnes_hut_theta=props.fa2_barnes_hut_theta,
                jitter_tolerance=props.fa2_jitter_tolerance,
                edge_weight_influence=props.fa2_edge_weight_influence
            )
        return _forceatlas2_layout(G, iterations, scale)

    elif algorithm == 'IGRAPH_FR':
        # Note: igraph FR parameters (start_temp, coolexp, etc.) are only valid
        # in iterative mode with seed. In full-layout mode, only niter and dim work.
        # User-configured parameters are only used in interactive mode (_igraph_fr_iteration).
        return _igraph_fruchterman_reingold(G, iterations, scale)

    elif algorithm == 'IGRAPH_KK':
        if props:
            return _igraph_kamada_kawai(
                G, scale,
                maxiter=props.igraph_kk_maxiter if props.igraph_kk_maxiter > 0 else None,
                epsilon=props.igraph_kk_epsilon if props.igraph_kk_epsilon > 0 else None,
                kkconst=props.igraph_kk_kkconst if props.igraph_kk_kkconst > 0 else None
            )
        return _igraph_kamada_kawai(G, scale)

    elif algorithm == 'IGRAPH_DRL':
        if props:
            return _igraph_drl(G, iterations, scale,
                edge_cut=props.igraph_drl_edge_cut,
                init_iterations=props.igraph_drl_init_iterations,
                init_temperature=props.igraph_drl_init_temperature,
                init_attraction=props.igraph_drl_init_attraction,
                init_damping_mult=props.igraph_drl_init_damping_mult,
                liquid_iterations=props.igraph_drl_liquid_iterations,
                liquid_temperature=props.igraph_drl_liquid_temperature,
                liquid_attraction=props.igraph_drl_liquid_attraction,
                liquid_damping_mult=props.igraph_drl_liquid_damping_mult,
                expansion_iterations=props.igraph_drl_expansion_iterations,
                expansion_temperature=props.igraph_drl_expansion_temperature,
                expansion_attraction=props.igraph_drl_expansion_attraction,
                expansion_damping_mult=props.igraph_drl_expansion_damping_mult,
                cooldown_iterations=props.igraph_drl_cooldown_iterations,
                cooldown_temperature=props.igraph_drl_cooldown_temperature,
                cooldown_attraction=props.igraph_drl_cooldown_attraction,
                cooldown_damping_mult=props.igraph_drl_cooldown_damping_mult,
                crunch_iterations=props.igraph_drl_crunch_iterations,
                crunch_temperature=props.igraph_drl_crunch_temperature,
                crunch_attraction=props.igraph_drl_crunch_attraction,
                crunch_damping_mult=props.igraph_drl_crunch_damping_mult,
                simmer_iterations=props.igraph_drl_simmer_iterations,
                simmer_temperature=props.igraph_drl_simmer_temperature,
                simmer_attraction=props.igraph_drl_simmer_attraction,
                simmer_damping_mult=props.igraph_drl_simmer_damping_mult,
            )
        return _igraph_drl(G, iterations, scale)

    elif algorithm == 'IGRAPH_DRL_2D':
        if props:
            return _igraph_drl_2d(G, iterations, scale,
                edge_cut=props.igraph_drl_edge_cut,
                init_iterations=props.igraph_drl_init_iterations,
                init_temperature=props.igraph_drl_init_temperature,
                init_attraction=props.igraph_drl_init_attraction,
                init_damping_mult=props.igraph_drl_init_damping_mult,
                liquid_iterations=props.igraph_drl_liquid_iterations,
                liquid_temperature=props.igraph_drl_liquid_temperature,
                liquid_attraction=props.igraph_drl_liquid_attraction,
                liquid_damping_mult=props.igraph_drl_liquid_damping_mult,
                expansion_iterations=props.igraph_drl_expansion_iterations,
                expansion_temperature=props.igraph_drl_expansion_temperature,
                expansion_attraction=props.igraph_drl_expansion_attraction,
                expansion_damping_mult=props.igraph_drl_expansion_damping_mult,
                cooldown_iterations=props.igraph_drl_cooldown_iterations,
                cooldown_temperature=props.igraph_drl_cooldown_temperature,
                cooldown_attraction=props.igraph_drl_cooldown_attraction,
                cooldown_damping_mult=props.igraph_drl_cooldown_damping_mult,
                crunch_iterations=props.igraph_drl_crunch_iterations,
                crunch_temperature=props.igraph_drl_crunch_temperature,
                crunch_attraction=props.igraph_drl_crunch_attraction,
                crunch_damping_mult=props.igraph_drl_crunch_damping_mult,
                simmer_iterations=props.igraph_drl_simmer_iterations,
                simmer_temperature=props.igraph_drl_simmer_temperature,
                simmer_attraction=props.igraph_drl_simmer_attraction,
                simmer_damping_mult=props.igraph_drl_simmer_damping_mult,
            )
        return _igraph_drl_2d(G, iterations, scale)

    elif algorithm == 'IGRAPH_LGL':
        if props:
            return _igraph_lgl(
                G, scale,
                maxiter=props.igraph_lgl_maxiter,
                maxdelta=props.igraph_lgl_maxdelta if props.igraph_lgl_maxdelta > 0 else None,
                area=props.igraph_lgl_area if props.igraph_lgl_area > 0 else None,
                coolexp=props.igraph_lgl_coolexp,
                repulserad=props.igraph_lgl_repulserad if props.igraph_lgl_repulserad > 0 else None,
                cellsize=props.igraph_lgl_cellsize if props.igraph_lgl_cellsize > 0 else None
            )
        return _igraph_lgl(G, scale)

    elif algorithm == 'IGRAPH_DH':
        if props:
            return _igraph_davidson_harel(
                G, iterations, scale,
                maxiter=props.igraph_dh_maxiter,
                fineiter=props.igraph_dh_fineiter,
                cool_fact=props.igraph_dh_cool_fact,
                weight_node_dist=props.igraph_dh_weight_node_dist,
                weight_border=props.igraph_dh_weight_border,
                weight_edge_lengths=props.igraph_dh_weight_edge_lengths,
                weight_edge_crossings=props.igraph_dh_weight_edge_crossings,
                weight_node_edge_dist=props.igraph_dh_weight_node_edge_dist
            )
        return _igraph_davidson_harel(G, iterations, scale)

    elif algorithm == 'IGRAPH_GRAPHOPT':
        if props:
            return _igraph_graphopt(
                G, iterations, scale,
                niter=props.igraph_graphopt_niter,
                node_charge=props.igraph_graphopt_node_charge,
                node_mass=props.igraph_graphopt_node_mass,
                spring_length=props.igraph_graphopt_spring_length if props.igraph_graphopt_spring_length > 0 else 0,
                spring_constant=props.igraph_graphopt_spring_constant,
                max_sa_movement=props.igraph_graphopt_max_sa_movement
            )
        return _igraph_graphopt(G, iterations, scale)
    elif algorithm == 'YIFAN_HU':
        return _yifan_hu_layout(G, iterations, scale, props=props)
    elif algorithm in GRAPHVIZ_ENGINES:
        return _graphviz_engine_layout(G, algorithm, iterations, scale, props=props)
    elif algorithm == 'SPECTRAL_3D':
        return _spectral_layout_3d(G, scale)
    elif algorithm == 'MDS_3D':
        return _mds_layout_3d(G, scale)
    else:
        # Fallback to random
        return _random_layout(num_nodes, scale)

def execute_layout_iteration(obj, algorithm='SPRING_3D', scale=5.0, current_frame=1,
                            repulsion=1.0, attraction=1.0, gravity=0.1,
                            cooling=0.95, initial_temp=1.0, edge_dist=1.0,
                            auto_stop=0.0):
    """
    Executes a single iteration of the layout algorithm with advanced parameters.
    Returns tuple (success, energy) where success is True if iteration completed.

    This function is designed for interactive, step-by-step layout computation
    similar to Gephi's execution model.
    """
    start_time = time.time()

    G, num_nodes = _build_networkx_graph(obj)
    if G is None:
        return False, 0.0

    num_edges = G.number_of_edges()

    # Get current positions (or initialize if first iteration)
    if "node_positions" in obj:
        current_pos_flat = obj["node_positions"]
        current_pos = np.array(current_pos_flat).reshape(num_nodes, 3)
    else:
        # Initialize with random positions
        current_pos = np.random.rand(num_nodes, 3) * scale
        obj["layout_iteration"] = 0
        obj["layout_energy"] = 0.0

    # Get iteration counter
    iteration = obj.get("layout_iteration", 0)

    # Log only on first iteration or every 50 iterations (to avoid spam)
    should_log = (iteration == 0) or (iteration % 50 == 0)

    # Execute one iteration based on algorithm
    if algorithm in ['SPRING', 'SPRING_3D']:
        # True iterative spring layout with custom parameters
        new_pos, energy = _spring_iteration_advanced(
            G, current_pos, algorithm, scale, iteration,
            repulsion, attraction, gravity, cooling, initial_temp, edge_dist
        )
    elif algorithm == 'FORCEATLAS2':
        # ForceAtlas2: Execute small batches of iterations for real-time feel
        try:
            props = bpy.context.scene.scigraphs
        except:
            props = None

        new_pos, energy = _forceatlas2_iteration(G, current_pos, scale, props)

    elif algorithm == 'IGRAPH_FR':
        # Fruchterman-Reingold: Execute small batches of iterations
        try:
            props = bpy.context.scene.scigraphs
        except:
            props = None

        new_pos, energy = _igraph_fr_iteration(G, current_pos, scale, iteration, props)

    elif algorithm in ['IGRAPH_KK', 'IGRAPH_DRL', 'IGRAPH_LGL',
                       'IGRAPH_DH', 'IGRAPH_GRAPHOPT',
                       'YIFAN_HU', 'GRAPHVIZ_DOT', 'GRAPHVIZ_NEATO',
                       'GRAPHVIZ_FDP', 'GRAPHVIZ_SFDP', 'GRAPHVIZ_TWOPI',
                       'GRAPHVIZ_CIRCO', 'GRAPHVIZ_OSAGE',
                       'GRAPHVIZ_PATCHWORK', 'SPECTRAL_3D', 'MDS_3D']:
        # These algorithms don't support true iteration (compute full layout at once)
        # Use interpolation approach for smooth animation
        if iteration == 0:
            try:
                props = bpy.context.scene.scigraphs
            except:
                props = None
            final_pos = _compute_layout_for_algorithm(G, num_nodes, algorithm, scale, props)

            obj["layout_target_positions"] = final_pos.flatten().tolist()
            if "node_positions" not in obj or len(obj["node_positions"]) == 0:
                current_pos = np.random.rand(num_nodes, 3) * scale
            obj["layout_initial_positions"] = current_pos.flatten().tolist()
            obj["layout_total_iters"] = 50

        initial_pos = np.array(obj.get("layout_initial_positions", current_pos.flatten())).reshape(num_nodes, 3)
        target_pos = np.array(obj.get("layout_target_positions", current_pos.flatten())).reshape(num_nodes, 3)
        total_iters = obj.get("layout_total_iters", 50)

        t = min(1.0, iteration / total_iters)
        t_eased = 1 - (1 - t) ** 2
        new_pos = initial_pos + (target_pos - initial_pos) * t_eased

        energy = np.linalg.norm(new_pos - target_pos)

        if t >= 1.0 or energy < 0.001:
            return False, energy

    else:
        # For truly static algorithms (Grid, Random, Sphere, Spiral, Circle Packing, etc.)
        # compute the full layout on first iteration only
        if iteration == 0:
            new_pos = _compute_static_layout(G, algorithm, num_nodes, scale, obj=obj)
            energy = 0.0
        else:
            # Already computed, no more iterations needed
            return False, 0.0

    # Check auto-stop condition
    prev_energy = obj.get("layout_energy", float('inf'))
    energy_change = abs(energy - prev_energy)

    if auto_stop > 0 and energy_change < auto_stop and iteration > 10:
        print(f"Auto-stop: Energy change {energy_change:.6f} < threshold {auto_stop}")
        return False, energy

    # Update positions
    obj["node_positions"] = new_pos.flatten().tolist()
    obj["layout_iteration"] = iteration + 1
    obj["layout_energy"] = energy

    # Insert keyframe at current frame
    mesh = obj.data
    if mesh and mesh.vertices:
        for i, vert in enumerate(mesh.vertices):
            if i < len(new_pos):
                vert.co = new_pos[i]
        mesh.update()

        # Insert keyframe
        obj.keyframe_insert(data_path="location", frame=current_frame)
        for i, vert in enumerate(mesh.vertices):
            vert.keyframe_insert(data_path="co", frame=current_frame)

    # Log iteration progress
    if should_log:
        params = {
            'mode': 'INTERACTIVE',
            'iteration': iteration,
            'frame': current_frame,
            'scale': scale,
            'energy': energy,
            'energy_change': abs(energy - obj.get("prev_energy", energy))
        }

        if algorithm in ['SPRING', 'SPRING_3D']:
            params.update({
                'repulsion': repulsion,
                'attraction': attraction,
                'gravity': gravity,
                'cooling': cooling,
                'initial_temp': initial_temp,
                'edge_dist': edge_dist
            })

        _log_layout(algorithm, num_nodes, num_edges, params, start_time, True)

    # Store energy for next comparison
    obj["prev_energy"] = energy

    return True, energy

def _spring_iteration_advanced(G, current_pos, algorithm, scale, iteration,
                              repulsion_strength=1.0, attraction_strength=1.0,
                              gravity_strength=0.1, cooling_factor=0.95,
                              initial_temperature=1.0, edge_distance=1.0):
    """
    Performs one iteration of spring/force-directed layout with advanced parameters.
    Returns (new_positions, total_energy).
    """
    num_nodes = len(G.nodes())
    new_pos = current_pos.copy()

    # Force-directed parameters
    k = edge_distance * scale / np.sqrt(num_nodes)  # Optimal distance

    # Temperature with cooling schedule
    temperature = initial_temperature * (cooling_factor ** iteration)

    # Calculate forces
    forces = np.zeros_like(current_pos)
    total_energy = 0.0

    # Center of mass for gravity
    center = current_pos.mean(axis=0)

    # Repulsive forces (all pairs)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            delta = current_pos[i] - current_pos[j]
            distance = np.linalg.norm(delta)
            if distance > 0.01:  # Avoid division by zero
                # Coulomb repulsion with strength parameter
                force_mag = repulsion_strength * (k * k / distance)
                force = force_mag * (delta / distance)
                forces[i] += force
                forces[j] -= force
                total_energy += force_mag

    # Attractive forces (edges)
    for edge in G.edges():
        i, j = edge
        delta = current_pos[i] - current_pos[j]
        distance = np.linalg.norm(delta)
        if distance > 0.01:
            # Hooke's law with strength parameter
            force_mag = attraction_strength * (distance * distance / k)
            force = force_mag * (delta / distance)
            forces[i] -= force
            forces[j] += force
            total_energy += force_mag

    # Gravity (pull to center)
    for i in range(num_nodes):
        delta = current_pos[i] - center
        distance = np.linalg.norm(delta)
        if distance > 0:
            force = gravity_strength * (delta / distance) * distance
            forces[i] -= force
            total_energy += gravity_strength * distance

    # Apply forces with temperature (simulated annealing)
    for i in range(num_nodes):
        force_magnitude = np.linalg.norm(forces[i])
        if force_magnitude > 0:
            displacement = (forces[i] / force_magnitude) * min(force_magnitude, temperature)
            new_pos[i] += displacement

    return new_pos, total_energy

def _compute_static_layout(G, algorithm, num_nodes, scale, obj=None):
    """
    Computes layouts that don't have iterative versions.

    Args:
        G: NetworkX graph
        algorithm: Algorithm name
        num_nodes: Number of nodes
        scale: Layout scale
        obj: Optional Blender object for storing extra data (like radii)
    """
    if algorithm == 'RANDOM':
        return _random_layout(num_nodes, scale)
    elif algorithm == 'GRID':
        return _grid_layout(num_nodes, scale)
    elif algorithm == 'CIRCLE_PACKING':
        positions, radii = _circle_packing_layout(G, iterations=500, scale=scale)
        # Store radii as custom property and mesh attribute
        if obj is not None:
            obj["circle_packing_radii"] = radii.tolist()
            _store_radii_as_mesh_attribute(obj, radii)
        return positions
    elif algorithm == 'IGRAPH_KK':
        return _igraph_kamada_kawai(G, scale)
    elif algorithm == 'IGRAPH_DRL':
        return _igraph_drl(G, 50, scale)
    elif algorithm == 'IGRAPH_DRL_2D':
        return _igraph_drl_2d(G, 50, scale)
    elif algorithm == 'IGRAPH_LGL':
        return _igraph_lgl(G, scale)
    elif algorithm == 'SPHERE':
        return _sphere_layout(num_nodes, scale)
    elif algorithm == 'SPECTRAL_3D':
        return _spectral_layout_3d(G, scale)
    elif algorithm == 'SPIRAL_3D':
        return _spiral_layout_3d(num_nodes, scale)
    elif algorithm == 'HELIX':
        return _helix_layout(num_nodes, scale)
    elif algorithm == 'CUBE':
        return _cube_layout(num_nodes, scale)
    elif algorithm == 'HIERARCHICAL_3D':
        return _hierarchical_layout_3d(G, scale)
    elif algorithm == 'BIPARTITE_3D':
        return _bipartite_layout_3d(G, scale)
    elif algorithm == 'MDS_3D':
        return _mds_layout_3d(G, scale)
    elif algorithm == 'YIFAN_HU':
        return _yifan_hu_layout(G, 50, scale)
    elif algorithm in GRAPHVIZ_ENGINES:
        return _graphviz_engine_layout(G, algorithm, 50, scale)
    elif algorithm == 'SUGIYAMA':
        return _sugiyama_layout(G, scale)
    elif algorithm == 'CIRCULAR_HIERARCHY':
        return _circular_hierarchy_layout(G, scale)
    else:
        return _random_layout(num_nodes, scale)

__all__ = [name for name in globals() if not name.startswith('__')]
