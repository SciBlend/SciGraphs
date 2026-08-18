"""igraph-based layout algorithms and helpers."""

from .common import *
from .basic import _random_layout
from .networkx_layouts import _spring_layout_2d

def _igraph_fruchterman_reingold(G, iterations, scale):
    """Fruchterman-Reingold via igraph, much faster than NetworkX. One-shot form:
    start_temp, coolexp and the rest only reach :func:`_igraph_fr_iteration`."""
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Spring 2D")
        return _spring_layout_2d(G, iterations, scale)

    import time
    start = time.time()
    print(f"Computing igraph Fruchterman-Reingold layout for {len(G.nodes())} nodes...")

    g_igraph = _nx_to_igraph(G)

    # Unseeded, igraph's FR takes only niter and dim; the rest are rejected.
    params = {'niter': iterations, 'dim': 3}

    layout = g_igraph.layout_fruchterman_reingold(**params)

    positions = np.array(layout.coords) * scale

    print(f"  Fruchterman-Reingold completed in {time.time() - start:.2f}s")
    return positions

def _igraph_kamada_kawai(G, scale, maxiter=None, epsilon=None, kkconst=None):
    """Kamada-Kawai via igraph. Deterministic, so layouts reproduce exactly."""
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Spring 2D")
        return _spring_layout_2d(G, 50, scale)

    import time
    start = time.time()
    print(f"Computing igraph Kamada-Kawai layout for {len(G.nodes())} nodes...")

    g_igraph = _nx_to_igraph(G)

    params = {'dim': 3}
    if maxiter is not None and maxiter > 0:
        params['maxiter'] = maxiter
    if epsilon is not None and epsilon > 0:
        params['epsilon'] = epsilon
    if kkconst is not None and kkconst > 0:
        params['kkconst'] = kkconst

    layout = g_igraph.layout_kamada_kawai(**params)

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
    """Build a DrL options dict, or pass a bare preset name through. With no
    phase parameter set, returns *preset* unchanged ('default', 'coarsen',
    'coarsest', 'refine', 'final'); set one and you get that preset's defaults
    with your values on top. The six phases run init, liquid, expansion,
    cooldown, crunch, simmer, each taking iterations, temperature, attraction
    and damping_mult; edge_cut is global, 0 to 1, cutting more edges as it rises."""
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

    if not overrides:
        return preset

    # Transcribed from the igraph DrL source.
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
    """DrL (Distributed Recursive Layout) in 3D via igraph, multilevel and the
    fastest thing here on large graphs. *iterations* is ignored (DrL counts per
    phase); *options* is a preset name or dict; *seed* is [x, y, z] per node."""
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Random")
        return _random_layout(len(G.nodes()), scale)

    import time
    start_total = time.time()
    print(f"Computing igraph DrL 3D layout for {len(G.nodes())} nodes, {len(G.edges())} edges...")

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

    t0 = time.time()
    g_igraph = _nx_to_igraph(G)
    print(f"  [DEBUG] NetworkX -> igraph conversion: {time.time() - t0:.3f}s")

    layout_kwargs = {'dim': 3, 'options': drl_options}
    if weights is not None:
        layout_kwargs['weights'] = weights
    if seed is not None:
        layout_kwargs['seed'] = seed

    t0 = time.time()
    layout = g_igraph.layout_drl(**layout_kwargs)
    t_layout = time.time() - t0
    print(f"  [DEBUG] DrL layout computation: {t_layout:.3f}s")

    # Not std-normalized: DrL's own proportions carry the cluster structure.
    positions = np.array(layout.coords)
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
    """:func:`_igraph_drl` in 2D with z = 0, 5 to 6 times faster; same params."""
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Random")
        return _random_layout(len(G.nodes()), scale)

    import time
    start_total = time.time()
    print(f"Computing igraph DrL 2D layout for {len(G.nodes())} nodes, {len(G.edges())} edges...")

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

    g_igraph = _nx_to_igraph(G)

    layout_kwargs = {'dim': 2, 'options': drl_options}
    if weights is not None:
        layout_kwargs['weights'] = weights
    if seed is not None:
        layout_kwargs['seed'] = seed

    t0 = time.time()
    layout = g_igraph.layout_drl(**layout_kwargs)
    t_layout = time.time() - t0
    print(f"  [DEBUG] DrL layout computation: {t_layout:.3f}s")

    positions_2d = np.array(layout.coords)
    positions = np.zeros((len(positions_2d), 3))
    positions[:, :2] = positions_2d

    positions = positions - positions.mean(axis=0)
    positions = positions * scale

    t_total = time.time() - start_total
    print(f"  DrL 2D completed in {t_total:.2f}s (layout: {t_layout:.2f}s = {100*t_layout/t_total:.1f}%)")
    return positions

def _igraph_lgl(G, scale, maxiter=150, maxdelta=None, area=None, coolexp=1.5,
                repulserad=None, cellsize=None):
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Random")
        return _random_layout(len(G.nodes()), scale)

    import time
    start = time.time()
    print(f"Computing igraph LGL layout for {len(G.nodes())} nodes...")

    g_igraph = _nx_to_igraph(G)

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

    # LGL is 2D only.
    layout = g_igraph.layout_lgl(**params)

    coords_2d = np.array(layout.coords)
    positions = np.zeros((len(coords_2d), 3))
    positions[:, :2] = coords_2d

    positions = positions - positions.mean(axis=0)
    positions = positions * scale

    print(f"  LGL completed in {time.time() - start:.2f}s")
    return positions

def _igraph_davidson_harel(G, iterations, scale, maxiter=10, fineiter=0, cool_fact=0.95,
                           weight_node_dist=1.0, weight_border=0.0, weight_edge_lengths=1.0,
                           weight_edge_crossings=1.0, weight_node_edge_dist=1.0):
    """Davidson-Harel via igraph: simulated annealing, good results, slow."""
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Spring 3D")
        return _spring_layout_3d(G, iterations, scale)

    import time
    start = time.time()
    print(f"Computing igraph Davidson-Harel layout for {len(G.nodes())} nodes...")

    g_igraph = _nx_to_igraph(G)

    layout = g_igraph.layout_davidson_harel(
        maxiter=maxiter, fineiter=fineiter, cool_fact=cool_fact,
        weight_node_dist=weight_node_dist, weight_border=weight_border,
        weight_edge_lengths=weight_edge_lengths, weight_edge_crossings=weight_edge_crossings,
        weight_node_edge_dist=weight_node_edge_dist
    )

    coords_2d = np.array(layout.coords)
    positions = np.zeros((len(coords_2d), 3))
    positions[:, :2] = coords_2d * scale

    print(f"  Davidson-Harel completed in {time.time() - start:.2f}s")
    return positions

def _igraph_graphopt(G, iterations, scale, niter=500, node_charge=0.001, node_mass=30.0,
                     spring_length=0.0, spring_constant=1.0, max_sa_movement=5.0):
    """Graphopt via igraph: energy-based like the spring layouts, but faster."""
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Spring 3D")
        return _spring_layout_3d(G, iterations, scale)

    import time
    start = time.time()
    print(f"Computing igraph Graphopt layout for {len(G.nodes())} nodes...")

    g_igraph = _nx_to_igraph(G)

    layout = g_igraph.layout_graphopt(
        niter=niter, node_charge=node_charge, node_mass=node_mass,
        spring_length=spring_length, spring_constant=spring_constant,
        max_sa_movement=max_sa_movement
    )

    coords_2d = np.array(layout.coords)
    positions = np.zeros((len(coords_2d), 3))
    positions[:, :2] = coords_2d * scale

    print(f"  Graphopt completed in {time.time() - start:.2f}s")
    return positions

def _igraph_fr_iteration(G, current_pos, scale, iteration, props=None):
    """A short batch of Fruchterman-Reingold, as ``(new_positions, energy)``.
    Seeded FR in igraph takes only niter, seed, dim and optionally start_temp,
    rejecting maxdelta, area, repulserad and coolexp; the seed must be 2D, so
    dim is 2 and Z carries over untouched."""
    if not IGRAPH_AVAILABLE:
        return _spring_layout_2d(G, 50, scale), 0.0

    g_igraph = _nx_to_igraph(G)

    seed_2d = current_pos[:, :2].tolist()

    iterations_per_frame = props.iterations_per_frame if props and hasattr(props, 'iterations_per_frame') else 10

    params = {'niter': iterations_per_frame, 'seed': seed_2d, 'dim': 2}

    if props and hasattr(props, 'igraph_fr_start_temp') and props.igraph_fr_start_temp > 0:
        # Cool across calls, since each one restarts igraph's own schedule.
        temp = props.igraph_fr_start_temp * (0.95 ** (iteration / 10))
        if temp > 0.01:
            params['start_temp'] = temp

    layout = g_igraph.layout_fruchterman_reingold(**params)

    new_pos = np.zeros_like(current_pos)
    coords_2d = np.array(layout.coords)
    new_pos[:, :2] = coords_2d
    new_pos[:, 2] = current_pos[:, 2]

    energy = np.linalg.norm(new_pos - current_pos)

    return new_pos, energy

__all__ = [name for name in globals() if not name.startswith('__')]
