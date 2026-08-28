"""igraph-based layout algorithms and helpers.

python-igraph draws its randomness from the stdlib :mod:`random` module, which
nothing here seeds: the caller must do it (``_reset_layout_rng`` in
:mod:`.common` does) or FR, DrL, LGL, DH and Graphopt all differ run to run.
None of these functions reseed, so an externally seeded run is reproducible.
"""

from .common import *
from .basic import _random_layout
from .networkx_layouts import _spring_layout_2d, _spring_layout_3d

_SLOW_ABOVE = {
    'Davidson-Harel': 1000, 'Kamada-Kawai': 3000, 'Graphopt': 3000,
    'DrL 3D': 3000, 'DrL 2D': 5000, 'Fruchterman-Reingold': 20000, 'LGL': 100000,
}

def _igraph_warn_if_slow(name, num_nodes):
    limit = _SLOW_ABOVE.get(name)
    if limit is not None and num_nodes > limit:
        print(f"  Warning: {name} gets slow past ~{limit} nodes ({num_nodes} here); "
              f"LGL and DrL 2D are the scalable options")

def _igraph_fit_positions(coords, scale):
    """Center on the origin and fit the widest axis into ``[-scale, scale]``.

    Each igraph algorithm picks its own coordinate range, so a fixed *scale*
    used to mean anything from 8 to 2600 units across; the node-radius and
    camera defaults assume the ~10 units a scale of 5.0 is supposed to give.
    *scale* stays an exact linear multiplier and the layout's own proportions
    are untouched, since the fit is uniform across the three axes."""
    coords = np.asarray(coords, dtype=float)
    positions = np.zeros((len(coords), 3))
    if len(coords) == 0:
        return positions
    positions[:, :coords.shape[1]] = coords

    positions -= positions.mean(axis=0)
    extent = np.abs(positions).max()
    if extent > 0:
        positions *= scale / extent
    return positions

def _igraph_defaults_for_none(params, function):
    """Replace every None in *params* with *function*'s own default. igraph
    rejects None with a TypeError, and a caller that maps an "auto" zero to None
    (as the dispatcher does for graphopt's spring_length) would hit it."""
    import inspect
    defaults = inspect.signature(function).parameters
    return {key: (defaults[key].default if value is None else value)
            for key, value in params.items()}

def _igraph_fruchterman_reingold(G, iterations, scale, start_temp=None):
    """Fruchterman-Reingold via igraph, much faster than NetworkX.

    igraph 0.8 rewrote this call and dropped ``coolexp``, ``maxdelta``, ``area``
    and ``repulserad``; each is a TypeError today. ``start_temp`` is the only
    survivor and subsumes maxdelta, being the largest move allowed along one
    axis in one step. *start_temp* is a fraction of igraph's own default,
    ``sqrt(n)/10``."""
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Spring 2D")
        return _spring_layout_2d(G, iterations, scale)

    import time
    start = time.time()
    num_nodes = len(G.nodes())
    print(f"Computing igraph Fruchterman-Reingold layout for {num_nodes} nodes...")
    _igraph_warn_if_slow('Fruchterman-Reingold', num_nodes)

    g_igraph = _nx_to_igraph(G)

    # Unseeded, igraph's FR takes only niter and dim; the rest are rejected.
    params = {'niter': iterations, 'dim': 3}
    if start_temp is not None and start_temp > 0:
        params['start_temp'] = start_temp * math.sqrt(num_nodes) / 10.0

    layout = g_igraph.layout_fruchterman_reingold(**params)

    positions = _igraph_fit_positions(layout.coords, scale)

    print(f"  Fruchterman-Reingold completed in {time.time() - start:.2f}s")
    return positions

def _igraph_kamada_kawai(G, scale, maxiter=None, epsilon=None, kkconst=None):
    """Kamada-Kawai via igraph. Deterministic, so layouts reproduce exactly."""
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Spring 2D")
        return _spring_layout_2d(G, 50, scale)

    import time
    start = time.time()
    num_nodes = len(G.nodes())
    print(f"Computing igraph Kamada-Kawai layout for {num_nodes} nodes...")
    _igraph_warn_if_slow('Kamada-Kawai', num_nodes)

    g_igraph = _nx_to_igraph(G)

    params = {'dim': 3}
    if maxiter is not None and maxiter > 0:
        params['maxiter'] = maxiter
    if epsilon is not None and epsilon > 0:
        params['epsilon'] = epsilon
    if kkconst is not None and kkconst > 0:
        params['kkconst'] = kkconst

    layout = g_igraph.layout_kamada_kawai(**params)

    positions = _igraph_fit_positions(layout.coords, scale)

    print(f"  Kamada-Kawai completed in {time.time() - start:.2f}s")
    return positions

_DRL_PHASES = ('init', 'liquid', 'expansion', 'cooldown', 'crunch', 'simmer')
_DRL_KEYS = ('edge_cut',) + tuple(
    f'{phase}_{param}'
    for phase in _DRL_PHASES
    for param in ('iterations', 'temperature', 'attraction', 'damping_mult'))

_DRL_MAX_DAMPING = 1.5

class _DrLOptions:
    """Carrier for igraph's 25 DrL parameters, readable only as attributes.

    A dict does not work. python-igraph's CONVERT_DRL_OPTION macro
    (src/_igraph/convert.c) reads every name through the mapping protocol and
    then unconditionally through ``getattr``; on a dict that second lookup
    raises AttributeError, and ``igraphmodule_PyObject_to_real_t`` bails on the
    pending exception without storing. edge_cut is converted first, so a dict
    delivers edge_cut and silently drops the other 24. An attribute-only object
    never raises, so all 25 arrive; verified by reproducing igraph's own
    'coarsen', 'coarsest' and 'default' presets to the bit."""

    __slots__ = _DRL_KEYS

    def __init__(self, values):
        for key in _DRL_KEYS:
            value = values[key]
            if key.endswith('_iterations'):
                value = int(value)
            else:
                value = float(value)
            if key.endswith('_damping_mult'):
                value = min(max(value, 0.0), _DRL_MAX_DAMPING)
            setattr(self, key, value)

def _log_drl_options(options):
    if not isinstance(options, _DrLOptions):
        print(f"  DrL preset: {options}")
        return
    print("  DrL options (from UI):")
    for phase in _DRL_PHASES:
        print(f"    {phase:10s}: "
              f"iter={getattr(options, f'{phase}_iterations')}, "
              f"temp={getattr(options, f'{phase}_temperature')}, "
              f"attr={getattr(options, f'{phase}_attraction')}, "
              f"damp={getattr(options, f'{phase}_damping_mult')}")
    print(f"    edge_cut: {options.edge_cut}")

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
    """Build DrL options, or pass a bare preset name through. With no phase
    parameter set, returns *preset* unchanged ('default', 'coarsen', 'coarsest',
    'refine', 'final'), which is the path the dispatcher takes with no scene
    properties; set one and you get that preset's values with yours on top, as a
    :class:`_DrLOptions`. *preset* may also be a dict, read as overrides. The six
    phases run init, liquid, expansion, cooldown, crunch, simmer, each taking
    iterations, temperature, attraction and damping_mult; edge_cut is global,
    0 to 1, cutting more edges as it rises."""
    overrides = {}
    local = locals()
    for key in _DRL_KEYS:
        val = local.get(key)
        if val is not None:
            overrides[key] = val

    if isinstance(preset, dict):
        for key, val in preset.items():
            if key in _DRL_KEYS and val is not None:
                overrides.setdefault(key, val)
        preset = 'default'

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
            'simmer_iterations': 0, 'simmer_temperature': 250,
            'simmer_attraction': 0.5, 'simmer_damping_mult': 0.0,
        },
        'final': {
            'edge_cut': 32.0/40.0,
            'init_iterations': 0, 'init_temperature': 50,
            'init_attraction': 0.5, 'init_damping_mult': 0.0,
            'liquid_iterations': 0, 'liquid_temperature': 2000,
            'liquid_attraction': 2, 'liquid_damping_mult': 1.0,
            'expansion_iterations': 50, 'expansion_temperature': 50,
            'expansion_attraction': 0.1, 'expansion_damping_mult': 0.25,
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
    return _DrLOptions(base)

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
    """DrL (Distributed Recursive Layout, the OpenOrd algorithm of Martin et al.
    2011) in 3D via igraph. Built for big graphs but not the fastest here: on
    5000 nodes it takes 152 s against LGL's 2.5 s and FR's 11.5 s, and it holds
    around 660 MB whatever the size. *iterations* is ignored (DrL counts per
    phase); *options* is a preset name or a dict of overrides; *seed* is
    [x, y, z] per node.

    igraph's 3D DrL answers only to the six per-phase iteration counts. Measured
    on 0.11.9 and 1.0.0: edge_cut, every temperature, attraction and
    damping_mult leave the coordinates bit-identical here, and 'coarsen', which
    differs from 'default' in attractions alone, reproduces 'default' exactly.
    :func:`_igraph_drl_2d` honours all 25."""
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Random")
        return _random_layout(len(G.nodes()), scale)

    import time
    start_total = time.time()
    num_nodes = len(G.nodes())
    print(f"Computing igraph DrL 3D layout for {num_nodes} nodes, {len(G.edges())} edges...")
    _igraph_warn_if_slow('DrL 3D', num_nodes)

    drl_options = _build_drl_options(
        preset=options,
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
    _log_drl_options(drl_options)
    if isinstance(drl_options, _DrLOptions):
        print("    (3D DrL reads the iteration counts only; use DrL 2D for the rest)")

    t0 = time.time()
    g_igraph = _nx_to_igraph(G)
    print(f"  [DEBUG] NetworkX -> igraph conversion: {time.time() - t0:.3f}s")

    layout_kwargs = {'dim': 3, 'options': drl_options}
    if weights is not None:
        layout_kwargs['weights'] = weights
    if seed is not None:
        layout_kwargs['seed'] = [list(row[:3]) for row in seed]

    t0 = time.time()
    layout = g_igraph.layout_drl(**layout_kwargs)
    t_layout = time.time() - t0
    print(f"  [DEBUG] DrL layout computation: {t_layout:.3f}s")

    positions = _igraph_fit_positions(layout.coords, scale)

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
    """:func:`_igraph_drl` in 2D with z = 0; same params. Measured 1.8x to 4.9x
    faster than the 3D form, not the 5 to 6 times once claimed."""
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Random")
        return _random_layout(len(G.nodes()), scale)

    import time
    start_total = time.time()
    num_nodes = len(G.nodes())
    print(f"Computing igraph DrL 2D layout for {num_nodes} nodes, {len(G.edges())} edges...")
    _igraph_warn_if_slow('DrL 2D', num_nodes)

    drl_options = _build_drl_options(
        preset=options,
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
    _log_drl_options(drl_options)

    g_igraph = _nx_to_igraph(G)

    layout_kwargs = {'dim': 2, 'options': drl_options}
    if weights is not None:
        layout_kwargs['weights'] = weights
    if seed is not None:
        layout_kwargs['seed'] = [list(row[:2]) for row in seed]

    t0 = time.time()
    layout = g_igraph.layout_drl(**layout_kwargs)
    t_layout = time.time() - t0
    print(f"  [DEBUG] DrL layout computation: {t_layout:.3f}s")

    positions = _igraph_fit_positions(layout.coords, scale)

    t_total = time.time() - start_total
    print(f"  DrL 2D completed in {t_total:.2f}s (layout: {t_layout:.2f}s = {100*t_layout/t_total:.1f}%)")
    return positions

def _igraph_lgl(G, scale, maxiter=150, maxdelta=None, area=None, coolexp=1.5,
                repulserad=None, cellsize=None):
    """Large Graph Layout (Adai et al. 2004) via igraph, and the fastest here:
    2.5 s on 5000 nodes. Planar only, so z is always 0; the UI calls it 3D."""
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Random")
        return _random_layout(len(G.nodes()), scale)

    import time
    start = time.time()
    num_nodes = len(G.nodes())
    print(f"Computing igraph LGL layout for {num_nodes} nodes...")
    _igraph_warn_if_slow('LGL', num_nodes)

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

    positions = _igraph_fit_positions(layout.coords, scale)

    print(f"  LGL completed in {time.time() - start:.2f}s")
    return positions

def _igraph_davidson_harel(G, iterations, scale, maxiter=10, fineiter=0, cool_fact=0.95,
                           weight_node_dist=1.0, weight_border=0.0, weight_edge_lengths=1.0,
                           weight_edge_crossings=1.0, weight_node_edge_dist=1.0):
    """Davidson-Harel via igraph (Davidson and Harel 1996): simulated annealing,
    good results, and the slowest here, over 420 s on 5000 nodes. Planar, so z
    is always 0."""
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Spring 3D")
        return _spring_layout_3d(G, iterations, scale)

    import time
    start = time.time()
    num_nodes = len(G.nodes())
    print(f"Computing igraph Davidson-Harel layout for {num_nodes} nodes...")
    _igraph_warn_if_slow('Davidson-Harel', num_nodes)

    g_igraph = _nx_to_igraph(G)

    params = {'maxiter': maxiter, 'fineiter': fineiter, 'cool_fact': cool_fact,
              'weight_node_dist': weight_node_dist, 'weight_border': weight_border,
              'weight_edge_lengths': weight_edge_lengths,
              'weight_edge_crossings': weight_edge_crossings,
              'weight_node_edge_dist': weight_node_edge_dist}
    params = _igraph_defaults_for_none(params, _igraph_davidson_harel)

    layout = g_igraph.layout_davidson_harel(**params)

    positions = _igraph_fit_positions(layout.coords, scale)

    print(f"  Davidson-Harel completed in {time.time() - start:.2f}s")
    return positions

def _igraph_graphopt(G, iterations, scale, niter=500, node_charge=0.001, node_mass=30.0,
                     spring_length=0.0, spring_constant=1.0, max_sa_movement=5.0):
    """Graphopt via igraph: energy-based like the spring layouts, but faster.
    Planar, so z is always 0."""
    if not IGRAPH_AVAILABLE:
        print("igraph not available, falling back to Spring 3D")
        return _spring_layout_3d(G, iterations, scale)

    import time
    start = time.time()
    num_nodes = len(G.nodes())
    print(f"Computing igraph Graphopt layout for {num_nodes} nodes...")
    _igraph_warn_if_slow('Graphopt', num_nodes)

    g_igraph = _nx_to_igraph(G)

    params = {'niter': niter, 'node_charge': node_charge, 'node_mass': node_mass,
              'spring_length': spring_length, 'spring_constant': spring_constant,
              'max_sa_movement': max_sa_movement}
    params = _igraph_defaults_for_none(params, _igraph_graphopt)

    layout = g_igraph.layout_graphopt(**params)

    positions = _igraph_fit_positions(layout.coords, scale)

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
