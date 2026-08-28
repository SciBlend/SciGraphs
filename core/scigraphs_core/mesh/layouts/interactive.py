"""Interactive layout iteration helpers."""

from .common import *
from .basic import *
from .networkx_layouts import *
from .forceatlas import *
from .igraph_layouts import *
from .circle_packing import *
from .hierarchical import *
from .yifan_hu import *

def _resolve_iterations(iterations, props, fallback=50):
    """The iteration budget: an explicit argument, else the panel's Iterations,
    else *fallback*. Both call sites used to hard-code 50 and drop the panel."""
    if iterations is not None:
        return max(1, int(iterations))
    if props is not None:
        try:
            return max(1, int(props.iterations))
        except (AttributeError, TypeError, ValueError):
            pass
    return fallback

def _compute_layout_for_algorithm(G, num_nodes, algorithm, scale, props=None,
                                  iterations=None):
    """Full layout for one algorithm, as an (num_nodes, 3) array. *props*
    supplies algorithm-specific parameters; without it each uses its defaults.
    *iterations* defaults to ``props.iterations``.

    The FORCEATLAS2, IGRAPH_FR, SPRING and SPRING_3D branches are here for
    direct callers only: both in-tree callers route those four to an iterative
    stepper instead, :func:`execute_layout_iteration` here and
    ``gpu_render.playback.ITERATIVE_MODELS`` there."""
    iterations = _resolve_iterations(iterations, props)
    _reset_layout_rng()

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
        # FR parameters are not passed: igraph accepts them only with a seed.
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
    elif algorithm == 'SPRING':
        return _spring_layout_2d(G, iterations, scale)
    elif algorithm == 'SPRING_3D':
        return _spring_layout_3d(G, iterations, scale)
    else:
        return _compute_static_layout(G, algorithm, num_nodes, scale,
                                      iterations=iterations)

_INTERPOLATED = ('IGRAPH_KK', 'IGRAPH_DRL', 'IGRAPH_DRL_2D', 'IGRAPH_LGL',
                 'IGRAPH_DH', 'IGRAPH_GRAPHOPT',
                 'YIFAN_HU', 'GRAPHVIZ_DOT', 'GRAPHVIZ_NEATO',
                 'GRAPHVIZ_FDP', 'GRAPHVIZ_SFDP', 'GRAPHVIZ_TWOPI',
                 'GRAPHVIZ_CIRCO', 'GRAPHVIZ_OSAGE',
                 'GRAPHVIZ_PATCHWORK', 'SPECTRAL_3D', 'MDS_3D')

def _current_positions(obj, num_nodes, scale):
    """The stored positions, or a fresh random start when there are none or the
    node count changed under us. Reshaping a stale array raises inside the modal
    operator, which has no handler for it."""
    flat = obj.get("node_positions")
    if flat is not None and len(flat) == num_nodes * 3:
        return np.array(flat).reshape(num_nodes, 3)

    if flat:
        print(f"Layout: {len(flat)} stored coordinates for {num_nodes} nodes; "
              "restarting from a random layout.")
    _reset_layout_rng()
    obj["layout_iteration"] = 0
    obj["layout_energy"] = 0.0
    return _get_layout_rng().rand(num_nodes, 3) * scale

def _target_is_stale(obj, algorithm, num_nodes):
    """True when the cached interpolation target is not the one being asked
    for. Switching algorithm mid-run otherwise eases toward the old answer."""
    return (obj.get("layout_target_algorithm") != algorithm
            or len(obj.get("layout_target_positions", ())) != num_nodes * 3)

def _interpolation_step(obj, G, num_nodes, current_pos, algorithm, scale,
                        iteration, props, iterations, total_iters):
    """Ease from where the mesh is toward a layout solved once at iteration 0.
    Returns ``(new_positions, energy, running)``."""
    # Not steppable: solve once, then ease toward the answer over 50 frames.
    if iteration == 0:
        final_pos = _compute_layout_for_algorithm(G, num_nodes, algorithm,
                                                  scale, props, iterations)
        obj["layout_target_positions"] = final_pos.flatten().tolist()
        obj["layout_target_algorithm"] = algorithm
        obj["layout_initial_positions"] = current_pos.flatten().tolist()
        obj["layout_total_iters"] = max(1, int(total_iters))

    initial_pos = np.array(obj.get("layout_initial_positions",
                                   current_pos.flatten())).reshape(num_nodes, 3)
    target_pos = np.array(obj.get("layout_target_positions",
                                  current_pos.flatten())).reshape(num_nodes, 3)

    t = min(1.0, iteration / max(1, obj.get("layout_total_iters", total_iters)))
    t_eased = 1 - (1 - t) ** 2
    new_pos = initial_pos + (target_pos - initial_pos) * t_eased

    energy = np.linalg.norm(new_pos - target_pos)
    return new_pos, energy, not (t >= 1.0 or energy < 0.001)

def _layout_step(obj, G, num_nodes, current_pos, algorithm, scale, iteration,
                 iterations, ease_iters, per_frame, props,
                 repulsion, attraction, gravity, cooling, initial_temp,
                 edge_dist):
    """One frame's worth of layout, as ``(new_positions, energy, running,
    consumed)``; *consumed* is how many iterations it advanced."""
    if algorithm in ('SPRING', 'SPRING_3D'):
        new_pos, energy = current_pos, 0.0
        for sub in range(per_frame):
            new_pos, energy = _spring_iteration_advanced(
                G, new_pos, algorithm, scale, iteration + sub,
                repulsion, attraction, gravity, cooling, initial_temp,
                edge_dist)
        return new_pos, energy, True, per_frame

    if algorithm == 'FORCEATLAS2':
        new_pos, energy = _forceatlas2_iteration(G, current_pos, scale, props)
        return new_pos, energy, True, 1

    if algorithm == 'IGRAPH_FR':
        new_pos, energy = _igraph_fr_iteration(G, current_pos, scale,
                                               iteration, props)
        return new_pos, energy, True, 1

    if algorithm in _INTERPOLATED:
        new_pos, energy, running = _interpolation_step(
            obj, G, num_nodes, current_pos, algorithm, scale, iteration,
            props, iterations, ease_iters)
        return new_pos, energy, running, per_frame

    # Static layouts: one shot on the first iteration, nothing after.
    if iteration == 0:
        new_pos = _compute_static_layout(G, algorithm, num_nodes, scale,
                                         obj=obj, iterations=iterations)
        return new_pos, 0.0, True, 1
    return current_pos, 0.0, False, 0

def execute_layout_iteration(obj, algorithm='SPRING_3D', scale=5.0, current_frame=1,
                            repulsion=1.0, attraction=1.0, gravity=0.1,
                            cooling=0.95, initial_temp=1.0, edge_dist=1.0,
                            auto_stop=0.0, props=None, edge_pairs=None,
                            iterations=None, total_iters=None, keyframe=True):
    """Run one frame of a layout, Gephi-style, returning ``(success, energy)``.
    *edge_pairs* is as in :func:`dispatcher.apply_graph_layout`; the modal
    operator recomputes it each iteration, which costs one array read.

    *iterations* is the budget for the algorithms solved in one shot and
    defaults to ``props.iterations``. *total_iters* is how many *frames* the
    non-steppable ones ease over, ``props.iterations_per_frame`` of them per
    call: this module cannot see the scene, so a caller animating a frame range
    has to pass its own frame count or the ease will end early or late.

    *keyframe* writes one key per vertex per call. That is what caches the run
    in the timeline, and what makes a large graph expensive: 5000 nodes over
    250 frames is 15000 fcurves carrying 1.25 million keys."""
    start_time = time.time()

    G, num_nodes = _build_networkx_graph(obj, edge_pairs)
    if G is None:
        return False, 0.0

    num_edges = G.number_of_edges()
    iterations = _resolve_iterations(iterations, props)
    total_iters = iterations if total_iters is None else max(1, int(total_iters))
    per_frame = 1
    if props is not None and algorithm not in ('FORCEATLAS2', 'IGRAPH_FR'):
        try:
            per_frame = max(1, int(props.iterations_per_frame))
        except (AttributeError, TypeError, ValueError):
            per_frame = 1

    current_pos = _current_positions(obj, num_nodes, scale)
    iteration = obj.get("layout_iteration", 0)
    if iteration and algorithm in _INTERPOLATED \
            and _target_is_stale(obj, algorithm, num_nodes):
        iteration = 0
        obj["layout_iteration"] = 0

    should_log = (iteration == 0) or (iteration % 50 == 0)

    try:
        new_pos, energy, running, consumed = _layout_step(
            obj, G, num_nodes, current_pos, algorithm, scale, iteration,
            iterations, total_iters * per_frame, per_frame, props,
            repulsion, attraction, gravity, cooling, initial_temp, edge_dist)
    except Exception as error:  # noqa: BLE001
        _log_layout(algorithm, num_nodes, num_edges, None, start_time, False,
                    error)
        return False, 0.0

    if not running:
        return False, energy

    prev_energy = obj.get("layout_energy", float('inf'))
    energy_change = abs(energy - prev_energy)

    if auto_stop > 0 and energy_change < auto_stop and iteration > 10:
        print(f"Auto-stop: Energy change {energy_change:.6f} < threshold {auto_stop}")
        return False, energy

    obj["node_positions"] = new_pos.flatten().tolist()
    obj["layout_iteration"] = iteration + consumed
    obj["layout_energy"] = energy

    mesh = obj.data
    if mesh and mesh.vertices:
        for i, vert in enumerate(mesh.vertices):
            if i < len(new_pos):
                vert.co = new_pos[i]
        mesh.update()

        if keyframe:
            for i, vert in enumerate(mesh.vertices):
                vert.keyframe_insert(data_path="co", frame=current_frame)

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

    obj["prev_energy"] = energy

    return True, energy

def _spring_iteration_advanced(G, current_pos, algorithm, scale, iteration,
                              repulsion_strength=1.0, attraction_strength=1.0,
                              gravity_strength=0.1, cooling_factor=0.95,
                              initial_temperature=1.0, edge_distance=1.0):
    """One step of a spring layout, as ``(new_positions, force_total)``.

    Repulsion is all-pairs, in strips so the temporaries stay bounded: 0.006 s
    per step at 400 nodes, 0.04 at 1093 and 0.67 at 5000, against 0.31, 2.5 and
    minutes for the interpreted double loop this replaces. :mod:`simulation` is
    still the one to reach for above a few thousand nodes, since its repulsion
    is O(n K) rather than O(n^2).

    The second return is the sum of the force magnitudes, not an energy: it
    grows about as n^2, plateauing at 386 on karate and 3891 on a 400-node
    grid, so ``auto_stop_threshold``, which the scene properties cap at 1.0,
    means a different thing on every graph. What ends the run is
    *cooling_factor*, not the forces: at 0.95 the temperature is 1e-9 by step
    400 and the layout freezes with the force total unchanged, and at 1.0,
    which the UI allows, it oscillates and never settles."""
    num_nodes = len(G.nodes())
    new_pos = current_pos.copy()

    k = edge_distance * scale / np.sqrt(num_nodes)

    temperature = initial_temperature * (cooling_factor ** iteration)

    forces = np.zeros_like(current_pos)
    total_energy = 0.0

    center = current_pos.mean(axis=0)

    rows = max(1, min(num_nodes, 1_400_000 // max(num_nodes, 1)))
    for lo in range(0, num_nodes, rows):
        hi = min(lo + rows, num_nodes)
        delta = current_pos[lo:hi, None, :] - current_pos[None, :, :]
        distance = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
        live = distance > 0.01
        coeff = np.zeros_like(distance)
        np.divide(repulsion_strength * k * k, distance * distance, out=coeff,
                  where=live)
        forces[lo:hi] += np.einsum("ij,ijk->ik", coeff, delta)
        total_energy += 0.5 * float((coeff * distance).sum())

    edges = np.fromiter((n for e in G.edges() for n in e), dtype=np.int64,
                        count=2 * G.number_of_edges()).reshape(-1, 2)
    if edges.size:
        src, dst = edges[:, 0], edges[:, 1]
        delta = current_pos[src] - current_pos[dst]
        distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        live = distance > 0.01
        coeff = np.zeros_like(distance)
        np.divide(attraction_strength * distance, k, out=coeff, where=live)
        pull = coeff[:, None] * delta
        for ax in range(3):
            forces[:, ax] -= np.bincount(src, weights=pull[:, ax],
                                         minlength=num_nodes)
            forces[:, ax] += np.bincount(dst, weights=pull[:, ax],
                                         minlength=num_nodes)
        total_energy += float((coeff * distance).sum())

    delta = current_pos - center
    forces -= gravity_strength * delta
    total_energy += gravity_strength * float(
        np.sqrt(np.einsum("ij,ij->i", delta, delta)).sum())

    magnitude = np.sqrt(np.einsum("ij,ij->i", forces, forces))
    step = np.zeros_like(magnitude)
    np.divide(np.minimum(magnitude, temperature), magnitude, out=step,
              where=magnitude > 0)
    new_pos += forces * step[:, None]

    return new_pos, total_energy

def _compute_static_layout(G, algorithm, num_nodes, scale, obj=None,
                           iterations=50):
    """Layouts with no iterative form, computed in one shot. Pass *obj* for
    algorithms producing more than positions: circle packing stores radii on it."""
    iterations = max(1, int(iterations))
    if algorithm == 'RANDOM':
        return _random_layout(num_nodes, scale)
    elif algorithm == 'GRID':
        return _grid_layout(num_nodes, scale)
    elif algorithm == 'CIRCLE_PACKING':
        positions, radii = _circle_packing_layout(G, iterations=500, scale=scale)
        if obj is not None:
            obj["circle_packing_radii"] = radii.tolist()
            _store_radii_as_mesh_attribute(obj, radii)
        return positions
    elif algorithm == 'IGRAPH_KK':
        return _igraph_kamada_kawai(G, scale)
    elif algorithm == 'IGRAPH_DRL':
        return _igraph_drl(G, iterations, scale)
    elif algorithm == 'IGRAPH_DRL_2D':
        return _igraph_drl_2d(G, iterations, scale)
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
        return _yifan_hu_layout(G, iterations, scale)
    elif algorithm in GRAPHVIZ_ENGINES:
        return _graphviz_engine_layout(G, algorithm, iterations, scale)
    elif algorithm == 'SUGIYAMA':
        return _sugiyama_layout(G, scale)
    elif algorithm == 'CIRCULAR_HIERARCHY':
        return _circular_hierarchy_layout(G, scale)
    else:
        return _random_layout(num_nodes, scale)

__all__ = [name for name in globals() if not name.startswith('__')]
