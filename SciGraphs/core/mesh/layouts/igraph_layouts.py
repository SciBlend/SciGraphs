"""igraph-based layout algorithms and helpers."""

from .common import *
from .basic import _random_layout
from .networkx_layouts import _spring_layout_2d

def _igraph_fruchterman_reingold(G, iterations, scale):
    """
    Fruchterman-Reingold layout using igraph.
    Much faster than NetworkX implementation.

    Note: This is the full-layout (non-iterative) version.
    Custom parameters (start_temp, coolexp, etc.) are only available
    in interactive mode via _igraph_fr_iteration().
    """
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Spring 2D")
        return _spring_layout_2d(G, iterations, scale)

    import time
    start = time.time()
    print(f"Computing igraph Fruchterman-Reingold layout for {len(G.nodes())} nodes...")

    # Convert NetworkX to igraph
    g_igraph = _nx_to_igraph(G)

    # igraph's Fruchterman-Reingold in non-iterative mode (without seed)
    # only accepts 'niter' and 'dim' parameters.
    # Other parameters (coolexp, maxdelta, area, repulserad, start_temp)
    # are only valid in iterative mode with seed.
    params = {'niter': iterations, 'dim': 3}

    # Compute layout (3D) - simple call without extra parameters
    layout = g_igraph.layout_fruchterman_reingold(**params)

    # Convert to numpy array
    positions = np.array(layout.coords) * scale

    print(f"  Fruchterman-Reingold completed in {time.time() - start:.2f}s")
    return positions

def _igraph_kamada_kawai(G, scale, maxiter=None, epsilon=None, kkconst=None):
    """
    Kamada-Kawai layout using igraph.
    Deterministic, good for reproducible layouts.
    """
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Spring 2D")
        return _spring_layout_2d(G, 50, scale)

    import time
    start = time.time()
    print(f"Computing igraph Kamada-Kawai layout for {len(G.nodes())} nodes...")

    # Convert NetworkX to igraph
    g_igraph = _nx_to_igraph(G)

    # Prepare parameters
    params = {'dim': 3}
    if maxiter is not None and maxiter > 0:
        params['maxiter'] = maxiter
    if epsilon is not None and epsilon > 0:
        params['epsilon'] = epsilon
    if kkconst is not None and kkconst > 0:
        params['kkconst'] = kkconst

    # Compute layout (3D)
    layout = g_igraph.layout_kamada_kawai(**params)

    # Convert to numpy array
    positions = np.array(layout.coords) * scale

    print(f"  Kamada-Kawai completed in {time.time() - start:.2f}s")
    return positions

def _build_drl_options(preset='default',
                       edge_cut=None,
                       init_iterations=None, init_temperature=None,
                       init_attraction=None, init_damping_mult=None,
                       liquid_iterations=None, liquid_temperature=None,
                       liquid_attraction=None, liquid_damping_mult=None,
                       expansion_iterations=None, expansion_temperature=None,
                       expansion_attraction=None, expansion_damping_mult=None,
                       cooldown_iterations=None, cooldown_temperature=None,
                       cooldown_attraction=None, cooldown_damping_mult=None,
                       crunch_iterations=None, crunch_temperature=None,
                       crunch_attraction=None, crunch_damping_mult=None,
                       simmer_iterations=None, simmer_temperature=None,
                       simmer_attraction=None, simmer_damping_mult=None):
    """
    Build a DrL options dict from individual parameters.

    If only a preset is given ('default', 'coarsen', 'coarsest', 'refine',
    'final'), returns the preset string directly.
    If any phase parameter is explicitly set (not None), returns a custom dict
    starting from the preset defaults, overridden with the supplied values.

    DrL has 6 sequential phases:
      1. init       - random initial placement
      2. liquid     - coarse-grained movement
      3. expansion  - spread out clusters
      4. cooldown   - reduce movement
      5. crunch     - fine-tune dense areas
      6. simmer     - final stabilization

    Each phase has: iterations, temperature, attraction, damping_mult.
    Global: edge_cut (0-1, higher = more edges cut in late stages).
    """
    # Collect all explicitly set overrides
    overrides = {}
    local = locals()
    for phase in ('init', 'liquid', 'expansion', 'cooldown', 'crunch', 'simmer'):
        for param in ('iterations', 'temperature', 'attraction', 'damping_mult'):
            key = f'{phase}_{param}'
            val = local.get(key)
            if val is not None:
                overrides[key] = val
    if edge_cut is not None:
        overrides['edge_cut'] = edge_cut

    # If no overrides, just return the preset string
    if not overrides:
        return preset

    # Build full options dict from preset defaults, then apply overrides
    # Default preset values (from igraph/DrL source)
    PRESETS = {
        'default': {
            'edge_cut': 32.0/40.0,
            'init_iterations': 0, 'init_temperature': 2000,
            'init_attraction': 10, 'init_damping_mult': 1.0,
            'liquid_iterations': 200, 'liquid_temperature': 2000,
            'liquid_attraction': 10, 'liquid_damping_mult': 1.0,
            'expansion_iterations': 200, 'expansion_temperature': 2000,
            'expansion_attraction': 2, 'expansion_damping_mult': 1.0,
            'cooldown_iterations': 200, 'cooldown_temperature': 2000,
            'cooldown_attraction': 1, 'cooldown_damping_mult': 0.1,
            'crunch_iterations': 50, 'crunch_temperature': 250,
            'crunch_attraction': 1, 'crunch_damping_mult': 0.25,
            'simmer_iterations': 100, 'simmer_temperature': 250,
            'simmer_attraction': 0.5, 'simmer_damping_mult': 0.0,
        },
        'coarsen': {
            'edge_cut': 32.0/40.0,
            'init_iterations': 0, 'init_temperature': 2000,
            'init_attraction': 10, 'init_damping_mult': 1.0,
            'liquid_iterations': 200, 'liquid_temperature': 2000,
            'liquid_attraction': 2, 'liquid_damping_mult': 1.0,
            'expansion_iterations': 200, 'expansion_temperature': 2000,
            'expansion_attraction': 10, 'expansion_damping_mult': 1.0,
            'cooldown_iterations': 200, 'cooldown_temperature': 2000,
            'cooldown_attraction': 1, 'cooldown_damping_mult': 0.1,
            'crunch_iterations': 50, 'crunch_temperature': 250,
            'crunch_attraction': 1, 'crunch_damping_mult': 0.25,
            'simmer_iterations': 100, 'simmer_temperature': 250,
            'simmer_attraction': 0.5, 'simmer_damping_mult': 0.0,
        },
        'coarsest': {
            'edge_cut': 32.0/40.0,
            'init_iterations': 0, 'init_temperature': 2000,
            'init_attraction': 10, 'init_damping_mult': 1.0,
            'liquid_iterations': 200, 'liquid_temperature': 2000,
            'liquid_attraction': 2, 'liquid_damping_mult': 1.0,
            'expansion_iterations': 200, 'expansion_temperature': 2000,
            'expansion_attraction': 10, 'expansion_damping_mult': 1.0,
            'cooldown_iterations': 200, 'cooldown_temperature': 2000,
            'cooldown_attraction': 1, 'cooldown_damping_mult': 0.1,
            'crunch_iterations': 200, 'crunch_temperature': 250,
            'crunch_attraction': 1, 'crunch_damping_mult': 0.25,
            'simmer_iterations': 100, 'simmer_temperature': 250,
            'simmer_attraction': 0.5, 'simmer_damping_mult': 0.0,
        },
        'refine': {
            'edge_cut': 32.0/40.0,
            'init_iterations': 0, 'init_temperature': 50,
            'init_attraction': 0.5, 'init_damping_mult': 0.0,
            'liquid_iterations': 0, 'liquid_temperature': 2000,
            'liquid_attraction': 2, 'liquid_damping_mult': 1.0,
            'expansion_iterations': 50, 'expansion_temperature': 500,
            'expansion_attraction': 0.1, 'expansion_damping_mult': 0.25,
            'cooldown_iterations': 50, 'cooldown_temperature': 200,
            'cooldown_attraction': 1, 'cooldown_damping_mult': 0.1,
            'crunch_iterations': 50, 'crunch_temperature': 250,
            'crunch_attraction': 1, 'crunch_damping_mult': 0.25,
            'simmer_iterations': 25, 'simmer_temperature': 250,
            'simmer_attraction': 0.5, 'simmer_damping_mult': 0.0,
        },
        'final': {
            'edge_cut': 0.0,
            'init_iterations': 0, 'init_temperature': 50,
            'init_attraction': 0.5, 'init_damping_mult': 0.0,
            'liquid_iterations': 0, 'liquid_temperature': 2000,
            'liquid_attraction': 2, 'liquid_damping_mult': 1.0,
            'expansion_iterations': 50, 'expansion_temperature': 2000,
            'expansion_attraction': 2, 'expansion_damping_mult': 1.0,
            'cooldown_iterations': 50, 'cooldown_temperature': 200,
            'cooldown_attraction': 1, 'cooldown_damping_mult': 0.1,
            'crunch_iterations': 50, 'crunch_temperature': 250,
            'crunch_attraction': 1, 'crunch_damping_mult': 0.25,
            'simmer_iterations': 25, 'simmer_temperature': 250,
            'simmer_attraction': 0.5, 'simmer_damping_mult': 0.0,
        },
    }

    base = PRESETS.get(preset, PRESETS['default']).copy()
    base.update(overrides)
    return base

def _igraph_drl(G, iterations, scale, options='default',
                weights=None, seed=None,
                edge_cut=None,
                init_iterations=None, init_temperature=None,
                init_attraction=None, init_damping_mult=None,
                liquid_iterations=None, liquid_temperature=None,
                liquid_attraction=None, liquid_damping_mult=None,
                expansion_iterations=None, expansion_temperature=None,
                expansion_attraction=None, expansion_damping_mult=None,
                cooldown_iterations=None, cooldown_temperature=None,
                cooldown_attraction=None, cooldown_damping_mult=None,
                crunch_iterations=None, crunch_temperature=None,
                crunch_attraction=None, crunch_damping_mult=None,
                simmer_iterations=None, simmer_temperature=None,
                simmer_attraction=None, simmer_damping_mult=None):
    """
    DrL (Distributed Recursive Layout) using igraph in 3D.
    Extremely fast multilevel force-directed algorithm for large graphs.

    Args:
        G: NetworkX graph
        iterations: ignored (DrL uses per-phase iterations), kept for API compat
        scale: output scale
        options: preset name ('default','coarsen','coarsest','refine','final')
                 or custom dict. Overridden by any explicit phase parameter.
        weights: edge weight attribute name (str) or sequence of weights
        seed: initial layout as list of [x,y,z] per node, or None for random
        edge_cut: 0-1, higher = more aggressive edge cutting in late phases

        Per-phase parameters (6 phases: init, liquid, expansion,
        cooldown, crunch, simmer):
            {phase}_iterations: number of iterations in this phase
            {phase}_temperature: start temperature for this phase
            {phase}_attraction: attraction multiplier for this phase
            {phase}_damping_mult: damping multiplier for this phase
    """
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Random")
        return _random_layout(len(G.nodes()), scale)

    import time
    start_total = time.time()
    print(f"Computing igraph DrL 3D layout for {len(G.nodes())} nodes, {len(G.edges())} edges...")

    # Build options dict from individual params or preset
    drl_options = _build_drl_options(
        preset=options if isinstance(options, str) else 'default',
        edge_cut=edge_cut,
        init_iterations=init_iterations, init_temperature=init_temperature,
        init_attraction=init_attraction, init_damping_mult=init_damping_mult,
        liquid_iterations=liquid_iterations, liquid_temperature=liquid_temperature,
        liquid_attraction=liquid_attraction, liquid_damping_mult=liquid_damping_mult,
        expansion_iterations=expansion_iterations, expansion_temperature=expansion_temperature,
        expansion_attraction=expansion_attraction, expansion_damping_mult=expansion_damping_mult,
        cooldown_iterations=cooldown_iterations, cooldown_temperature=cooldown_temperature,
        cooldown_attraction=cooldown_attraction, cooldown_damping_mult=cooldown_damping_mult,
        crunch_iterations=crunch_iterations, crunch_temperature=crunch_temperature,
        crunch_attraction=crunch_attraction, crunch_damping_mult=crunch_damping_mult,
        simmer_iterations=simmer_iterations, simmer_temperature=simmer_temperature,
        simmer_attraction=simmer_attraction, simmer_damping_mult=simmer_damping_mult,
    )
    if isinstance(options, dict):
        drl_options = options

    # Log the actual options being used
    if isinstance(drl_options, dict):
        print(f"  DrL options (from UI):")
        for phase in ('init', 'liquid', 'expansion', 'cooldown', 'crunch', 'simmer'):
            it = drl_options.get(f'{phase}_iterations', '?')
            te = drl_options.get(f'{phase}_temperature', '?')
            at = drl_options.get(f'{phase}_attraction', '?')
            da = drl_options.get(f'{phase}_damping_mult', '?')
            print(f"    {phase:10s}: iter={it}, temp={te}, attr={at}, damp={da}")
        print(f"    edge_cut: {drl_options.get('edge_cut', '?')}")
    else:
        print(f"  DrL preset: {drl_options}")

    # Convert NetworkX to igraph
    t0 = time.time()
    g_igraph = _nx_to_igraph(G)
    print(f"  [DEBUG] NetworkX -> igraph conversion: {time.time() - t0:.3f}s")

    # Prepare layout kwargs
    layout_kwargs = {'dim': 3, 'options': drl_options}
    if weights is not None:
        layout_kwargs['weights'] = weights
    if seed is not None:
        layout_kwargs['seed'] = seed

    # Compute layout
    t0 = time.time()
    layout = g_igraph.layout_drl(**layout_kwargs)
    t_layout = time.time() - t0
    print(f"  [DEBUG] DrL layout computation: {t_layout:.3f}s")

    # Convert to numpy array
    positions = np.array(layout.coords)

    # Center and scale (no std normalization - raw DrL proportions preserved)
    positions = positions - positions.mean(axis=0)
    positions = positions * scale

    t_total = time.time() - start_total
    print(f"  DrL 3D completed in {t_total:.2f}s (layout: {t_layout:.2f}s = {100*t_layout/t_total:.1f}%)")
    return positions

def _igraph_drl_2d(G, iterations, scale, options='default',
                   weights=None, seed=None,
                   edge_cut=None,
                   init_iterations=None, init_temperature=None,
                   init_attraction=None, init_damping_mult=None,
                   liquid_iterations=None, liquid_temperature=None,
                   liquid_attraction=None, liquid_damping_mult=None,
                   expansion_iterations=None, expansion_temperature=None,
                   expansion_attraction=None, expansion_damping_mult=None,
                   cooldown_iterations=None, cooldown_temperature=None,
                   cooldown_attraction=None, cooldown_damping_mult=None,
                   crunch_iterations=None, crunch_temperature=None,
                   crunch_attraction=None, crunch_damping_mult=None,
                   simmer_iterations=None, simmer_temperature=None,
                   simmer_attraction=None, simmer_damping_mult=None):
    """
    DrL (Distributed Recursive Layout) using igraph in 2D.
    Same as _igraph_drl but outputs 2D positions (z=0).
    Faster than 3D version (~5-6x speedup).
    See _igraph_drl docstring for full parameter descriptions.
    """
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Random")
        return _random_layout(len(G.nodes()), scale)

    import time
    start_total = time.time()
    print(f"Computing igraph DrL 2D layout for {len(G.nodes())} nodes, {len(G.edges())} edges...")

    # Build options dict from individual params or preset
    drl_options = _build_drl_options(
        preset=options if isinstance(options, str) else 'default',
        edge_cut=edge_cut,
        init_iterations=init_iterations, init_temperature=init_temperature,
        init_attraction=init_attraction, init_damping_mult=init_damping_mult,
        liquid_iterations=liquid_iterations, liquid_temperature=liquid_temperature,
        liquid_attraction=liquid_attraction, liquid_damping_mult=liquid_damping_mult,
        expansion_iterations=expansion_iterations, expansion_temperature=expansion_temperature,
        expansion_attraction=expansion_attraction, expansion_damping_mult=expansion_damping_mult,
        cooldown_iterations=cooldown_iterations, cooldown_temperature=cooldown_temperature,
        cooldown_attraction=cooldown_attraction, cooldown_damping_mult=cooldown_damping_mult,
        crunch_iterations=crunch_iterations, crunch_temperature=crunch_temperature,
        crunch_attraction=crunch_attraction, crunch_damping_mult=crunch_damping_mult,
        simmer_iterations=simmer_iterations, simmer_temperature=simmer_temperature,
        simmer_attraction=simmer_attraction, simmer_damping_mult=simmer_damping_mult,
    )
    if isinstance(options, dict):
        drl_options = options

    # Convert NetworkX to igraph
    g_igraph = _nx_to_igraph(G)

    # Prepare layout kwargs
    layout_kwargs = {'dim': 2, 'options': drl_options}
    if weights is not None:
        layout_kwargs['weights'] = weights
    if seed is not None:
        layout_kwargs['seed'] = seed

    # Compute layout
    t0 = time.time()
    layout = g_igraph.layout_drl(**layout_kwargs)
    t_layout = time.time() - t0
    print(f"  [DEBUG] DrL layout computation: {t_layout:.3f}s")

    # Convert to 3D numpy array (z=0)
    positions_2d = np.array(layout.coords)
    positions = np.zeros((len(positions_2d), 3))
    positions[:, :2] = positions_2d

    # Center and scale (no std normalization - raw DrL proportions preserved)
    positions = positions - positions.mean(axis=0)
    positions = positions * scale

    t_total = time.time() - start_total
    print(f"  DrL 2D completed in {t_total:.2f}s (layout: {t_layout:.2f}s = {100*t_layout/t_total:.1f}%)")
    return positions

def _igraph_lgl(G, scale, maxiter=150, maxdelta=None, area=None, coolexp=1.5,
                repulserad=None, cellsize=None):
    """
    Large Graph Layout (LGL) using igraph.
    Designed specifically for very large graphs.
    """
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Random")
        return _random_layout(len(G.nodes()), scale)

    import time
    start = time.time()
    print(f"Computing igraph LGL layout for {len(G.nodes())} nodes...")

    # Convert NetworkX to igraph
    g_igraph = _nx_to_igraph(G)

    # Prepare parameters
    params = {}
    if maxiter is not None:
        params['maxiter'] = maxiter
    if maxdelta is not None and maxdelta > 0:
        params['maxdelta'] = maxdelta
    if area is not None and area > 0:
        params['area'] = area
    if coolexp is not None:
        params['coolexp'] = coolexp
    if repulserad is not None and repulserad > 0:
        params['repulserad'] = repulserad
    if cellsize is not None and cellsize > 0:
        params['cellsize'] = cellsize

    # Compute layout (2D only, add z=0)
    layout = g_igraph.layout_lgl(**params)

    # Convert to 3D numpy array
    coords_2d = np.array(layout.coords)
    positions = np.zeros((len(coords_2d), 3))
    positions[:, :2] = coords_2d

    # Center and scale (no std normalization - raw LGL proportions preserved)
    positions = positions - positions.mean(axis=0)
    positions = positions * scale

    print(f"  LGL completed in {time.time() - start:.2f}s")
    return positions

def _igraph_davidson_harel(G, iterations, scale, maxiter=10, fineiter=0, cool_fact=0.95,
                           weight_node_dist=1.0, weight_border=0.0, weight_edge_lengths=1.0,
                           weight_edge_crossings=1.0, weight_node_edge_dist=1.0):
    """
    Davidson-Harel layout using igraph.
    Simulated annealing approach, good quality but slower.
    """
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Spring 3D")
        return _spring_layout_3d(G, iterations, scale)

    import time
    start = time.time()
    print(f"Computing igraph Davidson-Harel layout for {len(G.nodes())} nodes...")

    g_igraph = _nx_to_igraph(G)

    # Davidson-Harel supports 2D only in most igraph versions
    layout = g_igraph.layout_davidson_harel(
        maxiter=maxiter, fineiter=fineiter, cool_fact=cool_fact,
        weight_node_dist=weight_node_dist, weight_border=weight_border,
        weight_edge_lengths=weight_edge_lengths, weight_edge_crossings=weight_edge_crossings,
        weight_node_edge_dist=weight_node_edge_dist
    )

    # Convert to 3D
    coords_2d = np.array(layout.coords)
    positions = np.zeros((len(coords_2d), 3))
    positions[:, :2] = coords_2d * scale

    print(f"  Davidson-Harel completed in {time.time() - start:.2f}s")
    return positions

def _igraph_graphopt(G, iterations, scale, niter=500, node_charge=0.001, node_mass=30.0,
                     spring_length=0.0, spring_constant=1.0, max_sa_movement=5.0):
    """
    Graphopt layout using igraph.
    Energy-based layout similar to spring layouts but faster.
    """
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Spring 3D")
        return _spring_layout_3d(G, iterations, scale)

    import time
    start = time.time()
    print(f"Computing igraph Graphopt layout for {len(G.nodes())} nodes...")

    g_igraph = _nx_to_igraph(G)

    # Graphopt is 2D
    layout = g_igraph.layout_graphopt(
        niter=niter, node_charge=node_charge, node_mass=node_mass,
        spring_length=spring_length, spring_constant=spring_constant,
        max_sa_movement=max_sa_movement
    )

    # Convert to 3D
    coords_2d = np.array(layout.coords)
    positions = np.zeros((len(coords_2d), 3))
    positions[:, :2] = coords_2d * scale

    print(f"  Graphopt completed in {time.time() - start:.2f}s")
    return positions

def _igraph_fr_iteration(G, current_pos, scale, iteration, props=None):
    """
    Execute a small batch of Fruchterman-Reingold iterations using igraph.
    Returns (new_positions, energy).

    VERIFIED PARAMETERS (with seed):
    - niter: number of iterations (REQUIRED)
    - seed: initial positions 2D only (REQUIRED)
    - start_temp: initial temperature (OPTIONAL)
    - dim: must be 2 for 2D seed

    NOT ACCEPTED with seed: maxdelta, area, repulserad, coolexp
    """
    if not IGRAPH_AVAILABLE:
        return _spring_layout_2d(G, 50, scale), 0.0

    # Convert to igraph
    g_igraph = _nx_to_igraph(G)

    # Convert current 3D positions to 2D for igraph (MUST be 2D)
    seed_2d = current_pos[:, :2].tolist()

    # Execute small batch (configurable iterations per frame)
    iterations_per_frame = props.iterations_per_frame if props and hasattr(props, 'iterations_per_frame') else 10

    # Apply cooling with start_temp if available
    params = {'niter': iterations_per_frame, 'seed': seed_2d, 'dim': 2}

    if props and hasattr(props, 'igraph_fr_start_temp') and props.igraph_fr_start_temp > 0:
        # Reduce temperature over iterations for cooling
        temp = props.igraph_fr_start_temp * (0.95 ** (iteration / 10))
        if temp > 0.01:
            params['start_temp'] = temp

    layout = g_igraph.layout_fruchterman_reingold(**params)

    # Convert back to 3D
    new_pos = np.zeros_like(current_pos)
    coords_2d = np.array(layout.coords)
    new_pos[:, :2] = coords_2d
    new_pos[:, 2] = current_pos[:, 2]

    # Calculate energy
    energy = np.linalg.norm(new_pos - current_pos)

    return new_pos, energy

__all__ = [name for name in globals() if not name.startswith('__')]
