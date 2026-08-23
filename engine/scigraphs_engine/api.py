# A Graph is an immutable recipe over the arrays and never copies them.
# Channels cache per (channel, attribute); coreness is 768 ms at 2M nodes.
# Filtering masks without reindexing, so a chained filter measures everything.

from __future__ import annotations

import numpy as np

from . import edge_styles as _edge_styles
from . import filters as _filters
from . import mesh as _mesh
from . import palette as _palette
from . import simplify as _simplify
from .settings import Clause, Settings
from .source import ArraySource

__all__ = [
    "BackendUnavailable", "Camera", "ChannelResult", "ChannelSpec",
    "ChannelUnavailable", "Condition", "EngineError", "Geometry", "Graph",
    "Image", "Report", "UnknownChannel", "channel", "channel_names",
    "channel_specs", "colormap_names", "gpu_available",
]


class EngineError(Exception):
    """Base class for the errors this API raises."""


class UnknownChannel(EngineError):
    """A name that is neither a channel nor an attribute of the graph."""


class ChannelUnavailable(EngineError):
    """A channel that cannot be computed here; ``reason`` names the fix."""

    def __init__(self, name, reason):
        super().__init__(f"channel {name!r} is unavailable: {reason}")
        self.channel = name
        self.reason = reason


class BackendUnavailable(EngineError):
    """No GPU backend could draw this. ``reason`` says which stage failed."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


# Exceptions only: unlisted engine ids keep their id lower-cased as the name.
_RENAMED = {
    'BETWEEN': "betweenness",
    'EDGE_MINDEG': "rich_club",
}

# What a channel needs beyond the graph; filters.NEEDS_* keyed the other way.
_NEEDS = {}
for _name in _filters.NEEDS_WEIGHT:
    _NEEDS[_name] = "weight"
for _name in _filters.NEEDS_GROUPS:
    _NEEDS[_name] = "groups"
for _name in _filters.NEEDS_SEEDS:
    _NEEDS[_name] = "seeds"
_NEEDS['ATTR'] = "attr"
_NEEDS['EDGE_ATTR'] = "attr"


# Optional package per channel. CORE and COMPONENT have numpy fallbacks.
_IGRAPH_ONLY = frozenset(('BETWEEN', 'ARTICULATION', 'BRIDGE'))
_SCIPY_ONLY = frozenset(('CROWDING', 'SUPPORT', 'OVERLAP', 'HOPS'))
_EITHER = frozenset(('CLUSTERING', 'PAGERANK'))


class ChannelSpec:
    """One channel as a menu or a ``--help`` listing needs it."""

    __slots__ = ("name", "id", "domain", "label", "group", "description",
                 "needs", "is_flag")

    def __init__(self, name, cid, domain, label, group, description, needs,
                 is_flag):
        self.name = name
        self.id = cid
        self.domain = domain
        self.label = label
        self.group = group
        self.description = description
        self.needs = needs
        self.is_flag = is_flag

    def __repr__(self):
        return (f"ChannelSpec({self.name!r}, domain={self.domain!r}"
                + (f", needs={self.needs!r}" if self.needs else "") + ")")


def _build_registry():
    out = {}
    for group, entries in _filters.CHANNEL_GROUPS:
        for cid, label, domain, description in entries:
            name = _RENAMED.get(cid, cid.lower())
            out[name] = ChannelSpec(
                name, cid, 'edge' if domain == 'EDGE' else 'node', label,
                group, description, _NEEDS.get(cid),
                cid in _filters.FLAG_CHANNELS)
    return out


CHANNELS = _build_registry()

# Every accepted spelling -> canonical name; engine ids included for recipes.
_ALIASES = {}
for _n, _spec in CHANNELS.items():
    _ALIASES[_n] = _n
    _ALIASES[_spec.id.lower()] = _n


def channel_names(domain=None):
    """Return every channel name, optionally restricted to 'node' or 'edge'."""
    return tuple(sorted(n for n, s in CHANNELS.items()
                        if domain is None or s.domain == domain))


def channel_specs(domain=None):
    return tuple(s for s in CHANNELS.values()
                 if domain is None or s.domain == domain)


def colormap_names():
    return _palette.colormap_names()


_OPS = ("gte", "lte", "between", "outside", "eq")


class Condition:
    __slots__ = ("clause", "spec", "text")

    def __init__(self, clause, spec, text):
        self.clause = clause
        self.spec = spec
        self.text = text

    def __repr__(self):
        return f"<{self.text}>"


class ChannelRef:
    """A channel, not yet compared to anything."""

    __slots__ = ("name", "attr")

    def __init__(self, name, attr=""):
        self.name = name
        self.attr = attr or ""

    def _cond(self, lo, hi, invert, text):
        spec = CHANNELS.get(self.name)
        clause = Clause(channel=spec.id if spec else self.name,
                        attr_name=self.attr,
                        range_min=float(lo), range_max=float(hi),
                        invert=bool(invert))
        return Condition(clause, spec, text)

    def __ge__(self, value):
        return self._cond(value, 1.0, False, f"{self._label()} >= {value:g}")

    def __le__(self, value):
        return self._cond(0.0, value, False, f"{self._label()} <= {value:g}")

    def between(self, lo, hi):
        return self._cond(lo, hi, False, f"{lo:g} <= {self._label()} <= {hi:g}")

    def outside(self, lo, hi):
        return self._cond(lo, hi, True, f"{self._label()} outside [{lo:g}, {hi:g}]")

    def is_true(self):
        return self._cond(0.5, 1.0, False, f"{self._label()} is true")

    def is_false(self):
        return self._cond(0.0, 0.5, False, f"{self._label()} is false")

    def _label(self):
        return f"{self.name}[{self.attr}]" if self.attr else self.name

    def __repr__(self):
        return f"channel({self._label()!r})"

    def __gt__(self, value):
        raise TypeError(
            "use >= : the filter range is inclusive at both ends, so a strict "
            "> cannot be expressed and accepting it would move the boundary "
            "by one value without saying so")

    def __lt__(self, value):
        raise TypeError("use <= ; see the message for >")


def channel(name, attr=""):
    """Return a ChannelRef, as in ``channel("attr", "degree") >= 0.5``: ``attr``
    names an attribute the kwargs shorthand would read as a channel."""
    resolved = _ALIASES.get(str(name).lower())
    if resolved is None and str(name).lower() not in ("attr", "edge_attr"):
        raise UnknownChannel(
            f"no channel {name!r}; have {list(channel_names())}")
    return ChannelRef(resolved or str(name).lower(), attr)


class ChannelResult:
    """``values`` is [0, 1] or None; ``native`` is (lo, hi) in data units."""

    __slots__ = ("name", "domain", "values", "native", "available", "reason",
                 "is_flag")

    def __init__(self, name, domain, values, native, reason, is_flag):
        self.name = name
        self.domain = domain
        self.values = values
        self.native = native
        self.available = values is not None
        self.reason = reason
        self.is_flag = is_flag

    def to_native(self, t):
        lo, hi = self.native
        return lo + float(t) * (hi - lo)

    def from_native(self, value):
        """Convert a value in the units of the data to a `filter` threshold."""
        lo, hi = self.native
        if hi <= lo:
            return 0.0
        return float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0))

    def __repr__(self):
        if not self.available:
            return f"<ChannelResult {self.name!r} unavailable: {self.reason}>"
        return (f"<ChannelResult {self.name!r} {self.domain} "
                f"n={self.values.size} native {self.native[0]:g}"
                f"..{self.native[1]:g}>")


class Report:
    """What the pipeline kept and what it threw away."""

    __slots__ = ("nodes_in", "nodes_out", "edges_in", "edges_out", "clauses",
                 "backbone", "warnings", "detail")

    def __init__(self, nodes_in, nodes_out, edges_in, edges_out, clauses,
                 backbone, warnings, detail=None):
        self.nodes_in = int(nodes_in)
        self.nodes_out = int(nodes_out)
        self.edges_in = int(edges_in)
        self.edges_out = int(edges_out)
        self.clauses = tuple(clauses)     # (text, kept_count) per clause
        self.backbone = backbone          # None or the simplify stats dict
        self.warnings = tuple(warnings)
        self.detail = dict(detail or {})

    def to_dict(self):
        return {
            "nodes": {"in": self.nodes_in, "out": self.nodes_out},
            "edges": {"in": self.edges_in, "out": self.edges_out},
            "clauses": [{"clause": t, "kept": k} for t, k in self.clauses],
            "backbone": self.backbone,
            "warnings": list(self.warnings),
            "detail": dict(self.detail),
        }

    def __str__(self):
        pct_n = 100.0 * self.nodes_out / max(self.nodes_in, 1)
        pct_e = 100.0 * self.edges_out / max(self.edges_in, 1)
        lines = [
            f"nodes  {self.nodes_out:>9,} / {self.nodes_in:<9,} ({pct_n:5.1f}%)",
            f"edges  {self.edges_out:>9,} / {self.edges_in:<9,} ({pct_e:5.1f}%)",
        ]
        for text, kept in self.clauses:
            lines.append(f"  clause  {text:<34} "
                         + ("SKIPPED" if kept is None else f"kept {kept:,}"))
        if self.backbone:
            bb = self.backbone
            lines.append(
                f"  backbone kept {bb['kept']:,} of {bb['total']:,} edges "
                f"carrying {100.0 * bb['weight_frac']:.1f}% of the weight")
        for key, value in self.detail.items():
            lines.append(f"  {key}: {value}")
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)

    def __repr__(self):
        return f"<Report {self.nodes_out}/{self.nodes_in} nodes>"


EDGE_STYLES = {
    "straight": 'STRAIGHT',
    "curved": 'CURVED',
    "quadratic": 'QUADRATIC',
    "arc": 'ARC',
    "orthogonal": 'ORTHOGONAL',
    "tapered": 'TAPERED',
    "bundled": 'BUNDLED',
    "hierarchical": 'HIERARCHICAL',
}

# Everything `style()` accepts that is not an edge_styles parameter.
_APPEARANCE_DEFAULTS = {
    "nodes": "sphere",
    "edges": "straight",
    "edge_tier": "ribbon",        # 'ribbon' | 'line' | 'none'
    "node_radius": None,
    "edge_radius_scale": 0.45,    # of node_radius; the add-on's default
    "node_color": (0.3, 0.7, 1.0, 1.0),
    "edge_color": (0.5, 0.5, 0.55, 0.35),
    "color_by": None,             # channel or attribute name
    "colormap": "viridis",
    "size_by": None,
    "size_max_mult": 4.0,
    "edge_width_by": None,        # EDGE attribute name
    "edge_width_max_mult": 4.0,
    "arrows": False,              # True | False | 'flat'
    "arrow_size": 1.0,
    "max_nodes": None,
    "max_edges": None,
}

_STYLE_KEYS = set(_APPEARANCE_DEFAULTS) | (
    set(_edge_styles.STYLE_PARAM_DEFAULTS) - {"style_type"})


class _Base:
    """One graph's arrays and one channel cache, shared by every derived Graph."""

    __slots__ = ("source", "num_nodes", "weight", "groups", "seeds", "_cache",
                 "_settings", "_hierarchy")

    def __init__(self, source, weight, groups, seeds):
        self.source = source
        self.num_nodes = int(source.num_vertices)
        self.weight = weight or ""
        self.groups = groups or ""
        self.seeds = seeds or _filters.SEED_ATTR
        self._cache = {}
        self._hierarchy = None
        # Always carries coarsen_attr so its readers decline, not raise.
        self._settings = Settings({"coarsen_attr": self.groups})

    @property
    def edges(self):
        return self.source.edges()

    @property
    def num_edges(self):
        e = self.source.edges()
        return 0 if e is None else int(e.shape[0])

    def derived(self, weight=None, groups=None, seeds=None):
        """Fresh base and cache: weighted channels differ under another weight."""
        return _Base(self.source,
                     self.weight if weight is None else weight,
                     self.groups if groups is None else groups,
                     self.seeds if seeds is None else seeds)

    def measure(self, name, attr=""):
        spec = CHANNELS[name]
        key = (spec.id, attr, self.weight, self.groups, self.seeds)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        result = self._compute(spec, attr)
        self._cache[key] = result
        return result

    def _compute(self, spec, attr):
        # channel_values reads the HOPS seed attribute out of attr_name.
        attr_name = attr
        if spec.id == 'HOPS' and not attr_name:
            attr_name = self.seeds
        values, lo, hi, domain = _filters.channel_values(
            self.source, self._settings, spec.id, attr_name,
            group_attr=self.groups, weight_attr=self.weight)
        reason = None if values is not None else self._reason(spec, attr)
        return ChannelResult(spec.name, spec.domain, values, (lo, hi), reason,
                             spec.is_flag)

    def _reason(self, spec, attr):
        if spec.needs == "attr":
            domain = 'EDGE' if spec.domain == 'edge' else 'POINT'
            have = tuple(self.source.scalar_names(domain))
            if not attr:
                return (f"needs an attribute name; this graph has "
                        f"{list(have)} on {domain}")
            return (f"no usable {domain} attribute {attr!r}; this graph has "
                    f"{list(have)}")
        if spec.needs == "weight" and not self.weight:
            return ("needs an EDGE weight attribute: "
                    "Graph.from_arrays(..., weight='weight')")
        if spec.needs == "groups" and not self.groups:
            return ("needs a grouping attribute: "
                    "Graph.from_arrays(..., groups='community')")
        if spec.needs == "seeds":
            return (f"needs a seed set: a boolean POINT attribute named "
                    f"{self.seeds!r} with at least one True"
                    + ("" if _filters.SCIPY_AVAILABLE
                       else ", and the 'scipy' extra"))
        if self.num_edges == 0:
            return "the graph has no edges"
        if spec.id in _IGRAPH_ONLY and not _filters.IGRAPH_AVAILABLE:
            return "needs the 'igraph' extra: pip install 'scigraphs-engine[igraph]'"
        if spec.id in _SCIPY_ONLY and not _filters.SCIPY_AVAILABLE:
            return "needs the 'scipy' extra: pip install 'scigraphs-engine[scipy]'"
        if spec.id in _EITHER and not (_filters.IGRAPH_AVAILABLE
                                       or _filters.SCIPY_AVAILABLE):
            return ("needs the 'igraph' or 'scipy' extra: "
                    "pip install 'scigraphs-engine[igraph]'")
        if spec.needs == "weight":
            return (f"the EDGE attribute {self.weight!r} is missing or does "
                    f"not have one value per edge")
        if spec.needs == "groups":
            return (f"the POINT attribute {self.groups!r} is missing or does "
                    f"not have one value per node")
        return "the graph does not support this channel"


_GROUP_CANDIDATES = ("community", "cluster_id", "group", "conference")
_WEIGHT_CANDIDATES = ("weight", "weights", "value", "strength", "count",
                      "passengers", "flights")


def _guess(source, domain, candidates):
    names = set(source.scalar_names(domain))
    for name in candidates:
        if name in names:
            return name
    return ""


class Graph:
    """An immutable graph plus everything asked of it: build, ``filter``,
    ``simplify``, ``style``, then ``geometry()`` or ``render()``."""

    __slots__ = ("_base", "_conditions", "_backbone", "_style", "_masks",
                 "_report")

    def __init__(self, base, conditions=(), backbone=None, style=None):
        self._base = base
        self._conditions = tuple(conditions)
        self._backbone = backbone
        self._style = dict(_APPEARANCE_DEFAULTS) if style is None else dict(style)
        self._masks = None
        self._report = None

    @classmethod
    def from_arrays(cls, coords, edges=None, *, node_attrs=None,
                    edge_attrs=None, colors=None, weight=None, groups=None,
                    seeds=None, key=None):
        """Return a Graph over plain arrays: ``coords`` (N, 3), ``edges`` (E, 2)
        of node indices, ``node_attrs``/``edge_attrs`` ``{name: array}``. The
        EDGE ``weight`` and POINT ``groups`` are guessed by name if omitted."""
        source = ArraySource(coords, edges, point_attrs=node_attrs,
                             edge_attrs=edge_attrs, colors=colors, key=key)
        return cls.from_source(source, weight=weight, groups=groups,
                               seeds=seeds)

    @classmethod
    def from_source(cls, source, *, weight=None, groups=None, seeds=None):
        if weight is None:
            weight = _guess(source, 'EDGE', _WEIGHT_CANDIDATES)
        if groups is None:
            groups = _guess(source, 'POINT', _GROUP_CANDIDATES)
        return cls(_Base(source, weight, groups, seeds))

    @property
    def source(self):
        return self._base.source

    @property
    def num_nodes(self):
        """Nodes in the graph as given, before any filter."""
        return self._base.num_nodes

    @property
    def num_edges(self):
        return self._base.num_edges

    @property
    def attributes(self):
        """Attribute names the weighted, grouped and seeded channels read."""
        return {"weight": self._base.weight, "groups": self._base.groups,
                "seeds": self._base.seeds}

    def node_radius(self):
        """Return the radius each node is drawn at: ``node_radius``, or 1.2%
        of the layout's diagonal, since a fixed 0.1 vanishes at 10^4."""
        return self._node_radius()

    def __repr__(self):
        # Never evaluates masks: a coreness clause would stall the REPL.
        if self._masks is None:
            pending = (f", {len(self._conditions)} clause(s) pending"
                       if self._conditions else "")
            return (f"<Graph {self.num_nodes:,} nodes, {self.num_edges:,} "
                    f"edges{pending}, edges={self._style['edges']!r}>")
        node_mask, edge_mask = self._masks
        return (f"<Graph {int(node_mask.sum()):,}/{self.num_nodes:,} nodes, "
                f"{int(edge_mask.sum()):,}/{self.num_edges:,} edges, "
                f"{len(self._conditions)} clause(s), "
                f"edges={self._style['edges']!r}>")

    def _evolve(self, **changes):
        out = Graph(changes.get("base", self._base),
                    changes.get("conditions", self._conditions),
                    changes.get("backbone", self._backbone),
                    changes.get("style", self._style))
        return out

    def using(self, *, weight=None, groups=None, seeds=None):
        """Return a Graph reading weight/groups/seeds from other attributes."""
        # seeds is unchecked: scalar_names excludes the booleans a seed set is.
        for name, domain in ((weight, 'EDGE'), (groups, 'POINT')):
            if name and name not in self.source.scalar_names(domain):
                raise UnknownChannel(
                    f"no {domain} attribute {name!r}; this graph has "
                    f"{list(self.source.scalar_names(domain))}")
        return self._evolve(base=self._base.derived(weight, groups, seeds))

    def style(self, **kwargs):
        """Return a Graph that draws differently; never changes what survives."""
        unknown = sorted(set(kwargs) - _STYLE_KEYS)
        if unknown:
            raise KeyError(f"unknown style key(s): {unknown}; known: "
                           f"{sorted(_STYLE_KEYS)}")
        if "edges" in kwargs and kwargs["edges"] not in EDGE_STYLES:
            raise ValueError(f"unknown edge style {kwargs['edges']!r}; have "
                             f"{sorted(EDGE_STYLES)}")
        if "nodes" in kwargs and kwargs["nodes"] not in ("sphere", "point"):
            raise ValueError(f"nodes must be 'sphere' or 'point', got "
                             f"{kwargs['nodes']!r}")
        merged = dict(self._style)
        merged.update(kwargs)
        return self._evolve(style=merged)

    def filter(self, *conditions, on_missing="raise", **kwargs):
        """Return a Graph with more clauses, ANDed: ``g.filter(core__gte=0.5)``
        or ``g.filter(bridge=True)``. Thresholds are normalized to [0, 1]; see
        ``measure(name).from_native``. ``on_missing`` is "raise", "skip" or
        "empty", and raises by default: a skipped filter still looks right."""
        if on_missing not in ("raise", "skip", "empty"):
            raise ValueError("on_missing must be 'raise', 'skip' or 'empty'")
        built = list(conditions)
        for key, value in kwargs.items():
            built.append(self._condition_from_kwarg(key, value))
        for cond in built:
            if not isinstance(cond, Condition):
                raise TypeError(
                    f"filter() takes conditions and channel__op=value keywords, "
                    f"not {type(cond).__name__}; build one with "
                    f"channel('core') >= 0.5")
        merged = self._conditions + tuple(
            (c, on_missing) for c in built)
        return self._evolve(conditions=merged)

    def _condition_from_kwarg(self, key, value):
        name, _, op = key.partition("__")
        if op and op not in _OPS:
            raise ValueError(
                f"unknown comparison {op!r} in {key!r}; use one of "
                f"{list(_OPS)}. There is no __gt or __lt: the range is "
                f"inclusive at both ends (see channel.__gt__).")
        ref = self._resolve(name)
        if not op:
            if isinstance(value, bool):
                return ref.is_true() if value else ref.is_false()
            raise ValueError(
                f"{key}={value!r} needs a comparison: {name}__gte=..., "
                f"{name}__lte=..., {name}__between=(lo, hi). A bare "
                f"{name}=True/False is only for the flag channels "
                f"{list(_filters.FLAG_CHANNELS)}.")
        if op == "gte":
            return ref >= value
        if op == "lte":
            return ref <= value
        if op == "eq":
            return ref.is_true() if value else ref.is_false()
        lo, hi = value
        return ref.between(lo, hi) if op == "between" else ref.outside(lo, hi)

    def _resolve(self, name):
        """A channel name wins over an attribute of the same spelling."""
        lowered = str(name).lower()
        canonical = _ALIASES.get(lowered)
        if canonical is not None:
            return ChannelRef(canonical)
        if name in self.source.scalar_names('POINT'):
            return ChannelRef("attr", name)
        if name in self.source.scalar_names('EDGE'):
            return ChannelRef("edge_attr", name)
        raise UnknownChannel(
            f"{name!r} is neither a channel nor an attribute of this graph. "
            f"Channels: {list(channel_names())}. "
            f"POINT attributes: {list(self.source.scalar_names('POINT'))}. "
            f"EDGE attributes: {list(self.source.scalar_names('EDGE'))}.")

    def simplify(self, backbone=None, *, k=3, alpha=0.05, sample=0.25):
        """Drop edges structurally: ``backbone`` is "disparity", "topk", "mst",
        "sample" or None. Disparity is per node, not a weight threshold."""
        if backbone is None:
            return self._evolve(backbone=None)
        mode = str(backbone).upper()
        if mode not in ('TOPK', 'DISPARITY', 'MST', 'SAMPLE', 'ALL'):
            raise ValueError(
                f"unknown backbone {backbone!r}; have 'disparity', 'topk', "
                f"'mst', 'sample'")
        return self._evolve(backbone={"mode": mode, "k": int(k),
                                      "alpha": float(alpha),
                                      "sample": float(sample)})

    def measure(self, name, attr=""):
        """Return one ChannelResult, cached; check ``.available``, do not catch."""
        resolved = self._resolve(name) if not attr else channel(name, attr)
        return self._base.measure(resolved.name, resolved.attr)

    def available_channels(self):
        """Return ``{name: None or the reason it is unavailable}``, computing all."""
        out = {}
        for name, spec in CHANNELS.items():
            attr = ""
            if spec.needs == "attr":
                names = self.source.scalar_names(
                    'EDGE' if spec.domain == 'edge' else 'POINT')
                attr = names[0] if names else ""
            result = self._base.measure(name, attr)
            out[name] = None if result.available else result.reason
        return out

    def masks(self):
        """Return ``(node_mask, edge_mask)`` over the original index space.
        A removed node takes its edges with it."""
        if self._masks is None:
            self._masks = self._compute_masks()
        return self._masks

    @property
    def node_mask(self):
        return self.masks()[0]

    @property
    def edge_mask(self):
        return self.masks()[1]

    @property
    def nodes(self):
        """Surviving nodes, indexed into the arrays as supplied."""
        return np.flatnonzero(self.masks()[0]).astype(np.int32)

    @property
    def edge_ids(self):
        return np.flatnonzero(self.masks()[1]).astype(np.int32)

    @property
    def edges(self):
        """The surviving edges, as (E, 2) node indices in the original space."""
        edges = self._base.edges
        if edges is None:
            return np.empty((0, 2), dtype=np.int32)
        return edges[self.masks()[1]]

    def _compute_masks(self):
        base = self._base
        n = base.num_nodes
        edges = base.edges
        e = 0 if edges is None else int(edges.shape[0])
        node_mask = np.ones(n, dtype=bool)
        edge_mask = np.ones(e, dtype=bool)
        clause_rows = []
        warnings = []

        for cond, on_missing in self._conditions:
            spec = cond.spec
            result = base.measure(spec.name, cond.clause.attr_name)
            if not result.available:
                if on_missing == "raise":
                    raise ChannelUnavailable(spec.name, result.reason)
                if on_missing == "skip":
                    warnings.append(f"skipped {cond.text}: {result.reason}")
                    clause_rows.append((cond.text, None))
                    continue
                warnings.append(f"{cond.text} kept nothing: {result.reason}")
                if spec.domain == 'edge':
                    edge_mask[:] = False
                else:
                    node_mask[:] = False
                clause_rows.append((cond.text, 0))
                continue
            keep = _filters.slot_mask(result.values, cond.clause)
            if spec.domain == 'edge':
                if keep.size != e:
                    warnings.append(
                        f"skipped {cond.text}: the channel has {keep.size} "
                        f"values for {e} edges")
                    continue
                edge_mask &= keep
            else:
                if keep.size != n:
                    warnings.append(
                        f"skipped {cond.text}: the channel has {keep.size} "
                        f"values for {n} nodes")
                    continue
                node_mask &= keep
            clause_rows.append((cond.text, int(keep.sum())))

        backbone_stats = None
        if self._backbone is not None and edges is not None and e:
            weights = None
            if base.weight:
                weights = self.source.edge_scalar(base.weight)
            if weights is None and self._backbone["mode"] in ('DISPARITY',
                                                              'TOPK', 'MST'):
                warnings.append(
                    f"backbone {self._backbone['mode'].lower()!r} ran with "
                    f"uniform weights: no EDGE weight attribute "
                    f"(pass weight=... to from_arrays)")
            bb_mask, backbone_stats = _simplify.backbone_mask(
                edges, weights, self._backbone["mode"], self._backbone["k"],
                self._backbone["alpha"], self._backbone["sample"])
            if backbone_stats and backbone_stats.get("degraded"):
                warnings.append(backbone_stats["degraded"])
            if bb_mask is not None:
                edge_mask &= bb_mask

        if e and not node_mask.all():
            edge_mask &= node_mask[edges[:, 0]] & node_mask[edges[:, 1]]

        self._report = Report(n, int(node_mask.sum()), e, int(edge_mask.sum()),
                              clause_rows, backbone_stats, warnings)
        return node_mask, edge_mask

    def report(self):
        self.masks()
        return self._report

    def compact(self):
        """Return a new Graph over only the survivors, indices renumbered. It
        changes what the channels mean: ``degree`` is now among the survivors."""
        node_mask, edge_mask = self.masks()
        keep = np.flatnonzero(node_mask)
        remap = np.full(self.num_nodes, -1, dtype=np.int64)
        remap[keep] = np.arange(keep.size)
        edges = self._base.edges
        new_edges = None
        if edges is not None and edge_mask.any():
            new_edges = remap[edges[edge_mask]].astype(np.int32)

        source = self.source
        point_attrs = {name: np.asarray(source.point_scalar(name))[node_mask]
                       for name in source.scalar_names('POINT')
                       if source.point_scalar(name) is not None}
        edge_attrs = {name: np.asarray(source.edge_scalar(name))[edge_mask]
                      for name in source.scalar_names('EDGE')
                      if source.edge_scalar(name) is not None}
        colors = source.point_colors()
        node_mask_src = source.node_mask()
        compacted = ArraySource(
            source.coords()[node_mask], new_edges,
            node_mask=None if node_mask_src is None else node_mask_src[node_mask],
            point_attrs=point_attrs, edge_attrs=edge_attrs,
            colors=None if colors is None else colors[node_mask])
        base = _Base(compacted, self._base.weight, self._base.groups,
                     self._base.seeds)
        return Graph(base, (), None, self._style)

    def hierarchy(self, detect=None, *, levels=6, base_radius=None,
                  algorithm=None):
        """Return the community tree hierarchical bundling routes along, or [].
        The detector is ``detect=``, then ``set_community_detector()``, then the
        bundled one, since a GPLv3 Infomap cannot be a dependency."""
        from . import communities as _communities
        if detect is None:
            detect = _simplify.community_detector()
        if detect is None:
            detect = _communities.detect
            # The bundled detector has no 'infomap'; name what it implements.
            if algorithm is None:
                algorithm = "louvain"
        if algorithm is None:
            algorithm = "infomap"

        coords = self.source.coords().astype(np.float64)
        edges = self._base.edges
        if edges is None or edges.shape[0] == 0:
            return []
        labels = None
        if self._base.groups:
            labels = self.source.point_scalar(self._base.groups)
        if labels is None:
            labels = detect(edges, self.num_nodes, None)
        if labels is None:
            return []
        colors = self._node_colors().astype(np.float64)
        weights = None
        if self._base.weight:
            weights = self.source.edge_scalar(self._base.weight)
        radius = base_radius if base_radius is not None else self._node_radius()
        return _simplify.build_hierarchy(
            coords, edges, np.asarray(labels), colors, weights, radius,
            algorithm=algorithm, max_levels=int(levels), detect=detect)

    def _node_radius(self):
        given = self._style["node_radius"]
        if given is not None:
            return float(given)
        coords = self.source.coords()
        if coords.shape[0] == 0:
            return 1.0
        span = coords.max(axis=0) - coords.min(axis=0)
        diagonal = float(np.linalg.norm(span))
        return (diagonal * 0.012) if diagonal > 1e-9 else 1.0

    def _node_values(self, name):
        """Return a [0, 1] per-node channel for color or size, or None."""
        if not name:
            return None
        result = self.measure(name)
        if not result.available:
            return None
        if result.domain != 'node' or result.values.size != self.num_nodes:
            return None
        return result.values

    def _node_colors(self):
        """Return (N, 4) colors for every node: bundling indexes them by id."""
        n = self.num_nodes
        by = self._style["color_by"]
        values = self._node_values(by)
        if values is not None:
            return _palette.apply(values, self._style["colormap"])
        supplied = self.source.point_colors()
        if supplied is not None and supplied.shape[0] == n:
            return np.ascontiguousarray(supplied, dtype=np.float32)
        return _palette.solid(self._style["node_color"], n)

    def geometry(self):
        """Return drawable geometry; ``specs`` is in draw order."""
        return _build_geometry(self)

    def render(self, camera=None, size=(1280, 720), *,
               background=(0.05, 0.05, 0.07, 1.0), lighting=None):
        """Return an Image, or raise BackendUnavailable if nothing can draw;
        ``camera`` defaults to ``Camera.fit(self)`` at this size."""
        return _render(self, camera, size, background, lighting)

    def recipe(self):
        """Return everything asked of this graph as JSON-able data for ``replay``."""
        return {
            "version": 1,
            "attributes": {"weight": self._base.weight,
                           "groups": self._base.groups,
                           "seeds": self._base.seeds},
            "clauses": [dict(c.clause.as_dict(), text=c.text)
                        for c, _ in self._conditions],
            "backbone": None if self._backbone is None else dict(self._backbone),
            "style": {k: v for k, v in self._style.items()
                      if v != _APPEARANCE_DEFAULTS.get(k, object())},
        }

    def replay(self, recipe):
        if int(recipe.get("version", 1)) != 1:
            raise ValueError(f"recipe version {recipe.get('version')} is not 1")
        attrs = recipe.get("attributes") or {}
        out = self.using(weight=attrs.get("weight") or None,
                         groups=attrs.get("groups") or None,
                         seeds=attrs.get("seeds") or None)
        conditions = []
        for entry in recipe.get("clauses", ()):
            clause = Clause.from_dict(entry)
            spec = CHANNELS.get(_ALIASES.get(clause.channel.lower(), ""))
            if spec is None:
                raise UnknownChannel(f"recipe names channel "
                                     f"{clause.channel!r}")
            conditions.append((Condition(clause, spec,
                                         entry.get("text", clause.channel)),
                               "raise"))
        out = out._evolve(conditions=tuple(conditions))
        backbone = recipe.get("backbone")
        if backbone:
            out = out._evolve(backbone=dict(backbone))
        style = recipe.get("style")
        return out.style(**style) if style else out


class Geometry:
    __slots__ = ("specs", "nodes", "node_coords", "node_colors", "node_radii",
                 "edges", "segments", "report", "style")

    def __init__(self, specs, nodes, node_coords, node_colors, node_radii,
                 edges, segments, report, style):
        self.specs = tuple(s for s in specs if s is not None)
        self.nodes = nodes
        self.node_coords = node_coords
        self.node_colors = node_colors
        self.node_radii = node_radii
        self.edges = edges
        self.segments = segments
        self.report = report
        self.style = style

    @property
    def vertex_count(self):
        return sum(spec.vertex_count for spec in self.specs)

    def validate(self):
        """Return ``[(index, reason), ...]`` for every invalid spec."""
        return [(i, reason) for i, reason
                in ((i, s.validate()) for i, s in enumerate(self.specs))
                if reason]

    def __repr__(self):
        return (f"<Geometry {len(self.specs)} spec(s), "
                f"{self.nodes.size:,} nodes, {self.edges.shape[0]:,} edges, "
                f"{self.segments:,} segments, {self.vertex_count:,} vertices>")


def _stride_to(idx, ceiling):
    """Subsample by striding, not a random draw, so renders repeat."""
    if not ceiling or idx.size <= ceiling:
        return idx, 1
    stride = int(np.ceil(idx.size / float(ceiling)))
    return idx[::stride], stride


def _build_geometry(graph):
    style = graph._style
    base = graph._base
    source = graph.source
    coords = source.coords()
    edges = base.edges
    node_mask_src = source.node_mask()
    node_keep, edge_keep = graph.masks()
    report = graph.report()
    warnings = list(report.warnings)
    detail = {}

    # Only real nodes become nodes: a baked curve point drawn as a sphere shows.
    node_visible = node_keep if node_mask_src is None else (node_keep & node_mask_src)
    point_idx = np.flatnonzero(node_visible)
    point_idx, stride = _stride_to(point_idx, style["max_nodes"])
    if stride > 1:
        detail["node subsample"] = f"every {stride} of {int(node_visible.sum()):,}"

    colors_all = graph._node_colors()
    base_radius = graph._node_radius()
    size_norm = graph._node_values(style["size_by"])
    if style["size_by"] and size_norm is None:
        warnings.append(f"size_by={style['size_by']!r} is unavailable; "
                        f"nodes are drawn at one size")
    if size_norm is not None:
        radii = (base_radius * (1.0 + size_norm[point_idx]
                                * (float(style["size_max_mult"]) - 1.0))
                 ).astype(np.float32)
    else:
        radii = np.full(point_idx.size, base_radius, dtype=np.float32)

    if node_mask_src is None:
        edge_vis = node_visible
    else:
        edge_vis = (~node_mask_src) | node_visible

    logical = np.empty((0, 2), dtype=np.int32)
    mesh_ids = np.empty(0, dtype=np.int32)
    if edges is not None and edges.shape[0]:
        recovered = _edge_styles.recover_logical_edges(edges, node_mask_src)
        if recovered is None:
            warnings.append("the mesh is not clean polylines; edges are drawn "
                            "as stored rather than as recovered node pairs")
            logical, mesh_ids = edges, np.arange(edges.shape[0], dtype=np.int32)
        else:
            logical, mesh_ids = recovered
        keep = (edge_vis[logical[:, 0]] & edge_vis[logical[:, 1]]
                & edge_keep[mesh_ids])
        logical, mesh_ids = logical[keep], mesh_ids[keep]
        kept_idx, estride = _stride_to(np.arange(logical.shape[0]),
                                       style["max_edges"])
        if estride > 1:
            logical, mesh_ids = logical[kept_idx], mesh_ids[kept_idx]
            detail["edge subsample"] = f"every {estride}"

    edge_widths = None
    if style["edge_width_by"]:
        raw = source.edge_scalar(style["edge_width_by"])
        if raw is None:
            warnings.append(f"edge_width_by={style['edge_width_by']!r} is not "
                            f"an EDGE attribute of this graph")
        else:
            from .channels import scale_to_unit
            unit = scale_to_unit(raw, 'LOG')
            edge_widths = (1.0 + unit[mesh_ids]
                           * (float(style["edge_width_max_mult"]) - 1.0)
                           ).astype(np.float32)

    hierarchy = None
    if style["edges"] == "hierarchical" and logical.shape[0]:
        if base._hierarchy is None:
            base._hierarchy = graph.hierarchy()
        hierarchy = base._hierarchy
        if not hierarchy:
            warnings.append("hierarchical bundling fell back to straight "
                            "edges: no community tree could be built")

    params = _edge_styles.default_style_params(
        **{k: v for k, v in style.items()
           if k in _edge_styles.STYLE_PARAM_DEFAULTS})
    params["style_type"] = EDGE_STYLES[style["edges"]]

    styled = None
    if logical.shape[0]:
        styled = _edge_styles.tessellate(
            coords, logical.astype(np.int64), params,
            edge_widths=edge_widths, hierarchy=hierarchy,
            node_colors=colors_all,
            edge_color=np.asarray(style["edge_color"], dtype=np.float32))

    specs = []
    edge_radius = base_radius * float(style["edge_radius_scale"])
    segments = 0
    if styled is not None:
        segments = int(styled["seg_a"].shape[0])
        col_a, col_b = styled.get("col_a"), styled.get("col_b")
        tier = style["edge_tier"]
        if tier == "ribbon":
            specs.append(_mesh.ribbon_spec(
                styled["seg_a"], styled["seg_b"], style["edge_color"],
                (styled["scale_a"] * edge_radius).astype(np.float32),
                (styled["scale_b"] * edge_radius).astype(np.float32),
                col_a, col_b))
        elif tier == "line":
            specs.append(_mesh.segment_line_spec(
                styled["seg_a"], styled["seg_b"], col_a, col_b))

    if point_idx.size:
        if style["nodes"] == "sphere":
            specs.append(_mesh.sphere_spec(coords[point_idx],
                                           colors_all[point_idx], radii))
        else:
            for spec, _weight in _mesh.point_specs(
                    coords[point_idx], colors_all[point_idx]):
                specs.append(spec)

    if style["arrows"] and styled is not None \
            and styled.get("arrow_pos") is not None:
        node_r = np.full(graph.num_nodes, base_radius, dtype=np.float32)
        if size_norm is not None:
            node_r *= 1.0 + size_norm * (float(style["size_max_mult"]) - 1.0)
        target = styled["arrow_target"]
        head_r = (styled["arrow_width"] * edge_radius).astype(np.float32)
        if style["arrows"] == "flat":
            specs.append(_mesh.arrow_flat_spec(
                styled["arrow_pos"] - styled["arrow_dir"], styled["arrow_pos"],
                head_r, node_r[target], style["edge_color"],
                float(style["arrow_size"])))
        else:
            specs.append(_mesh.arrow_spec(
                styled["arrow_pos"], styled["arrow_dir"], head_r,
                node_r[target], style["edge_color"],
                float(style["arrow_size"])))

    detail["radius"] = f"{base_radius:.4g} world units per node"
    if hierarchy:
        from . import communities as _communities
        detail["hierarchy"] = (f"{len(hierarchy)} level(s) by "
                               f"{_communities.name()}")
    geom_report = Report(report.nodes_in, int(point_idx.size),
                         report.edges_in, int(logical.shape[0]),
                         report.clauses, report.backbone, warnings,
                         dict(report.detail, **detail))
    return Geometry(specs, point_idx, coords[point_idx], colors_all[point_idx],
                    radii, logical, segments, geom_report, dict(style))


class Camera:
    """A view and a projection, or a fit deferred until the aspect is known."""

    __slots__ = ("view", "proj", "near", "far", "_fit")

    def __init__(self, view=None, proj=None, near=0.1, far=1000.0, _fit=None):
        self.view = None if view is None else np.asarray(view, np.float32).reshape(4, 4)
        self.proj = None if proj is None else np.asarray(proj, np.float32).reshape(4, 4)
        self.near = float(near)
        self.far = float(far)
        self._fit = _fit

    @classmethod
    def fit(cls, graph, *, fov=45.0, margin=1.15, direction=(0.0, -1.0, 0.35),
            up=(0.0, 0.0, 1.0)):
        """Frame every drawn node once the size is known. Takes a Graph, a
        Geometry or an (N, 3) array; a Graph fits the survivors, not the input."""
        coords = _coords_of(graph)
        return cls(_fit={"coords": coords, "fov": float(fov),
                         "margin": float(margin), "direction": direction,
                         "up": up})

    @classmethod
    def look_at(cls, eye, target, *, up=(0.0, 0.0, 1.0), fov=45.0, near=None,
                far=None):
        """Return an explicit camera, deferred in the aspect ratio only."""
        return cls(_fit={"eye": eye, "target": target, "up": up,
                         "fov": float(fov), "near": near, "far": far})

    @property
    def resolved(self):
        return self.view is not None and self.proj is not None

    def resolve(self, size):
        """Return a concrete Camera for a (width, height) frame."""
        if self.resolved:
            return self
        from . import _camera_math as cm
        spec = self._fit or {}
        if "coords" in spec:
            return cm.fit_view(spec["coords"], size, spec["fov"],
                               spec["margin"], spec["direction"], spec["up"])
        return cm.look_at_camera(spec["eye"], spec["target"], spec["up"],
                                 spec["fov"], size, spec.get("near"),
                                 spec.get("far"))

    @property
    def view_proj(self):
        if not self.resolved:
            raise EngineError("this Camera is deferred; call resolve(size)")
        return (self.proj @ self.view).astype(np.float32)

    def project(self, points, size):
        """Project (N, 3) world points to pixels: x right, y down, z in [0, 1]."""
        cam = self.resolve(size)
        width, height = size
        p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        h = np.concatenate([p, np.ones((p.shape[0], 1), np.float32)], axis=1)
        clip = h @ cam.view_proj.T
        w = np.where(np.abs(clip[:, 3:4]) < 1e-12, 1e-12, clip[:, 3:4])
        ndc = clip[:, :3] / w
        out = np.empty_like(ndc)
        out[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * width
        out[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
        out[:, 2] = ndc[:, 2]
        return out

    def __repr__(self):
        return ("<Camera resolved>" if self.resolved
                else "<Camera deferred until resolve(size)>")


def _coords_of(thing):
    if isinstance(thing, Graph):
        coords = thing.source.coords()
        mask = thing.masks()[0]
        return coords[mask] if mask.any() else coords
    if isinstance(thing, Geometry):
        return thing.node_coords
    return np.asarray(thing, dtype=np.float32).reshape(-1, 3)


class Image:
    __slots__ = ("array", "stats")

    def __init__(self, array, stats):
        self.array = array
        self.stats = dict(stats)

    @property
    def size(self):
        return (int(self.array.shape[1]), int(self.array.shape[0]))

    @property
    def width(self):
        return int(self.array.shape[1])

    @property
    def height(self):
        return int(self.array.shape[0])

    def save(self, path):
        """Write a PNG to ``path``. Needs Pillow, included in the 'wgpu' extra."""
        try:
            from PIL import Image as _PIL
        except ImportError as exc:
            raise BackendUnavailable(
                "saving a PNG needs Pillow: pip install pillow") from exc
        data = (np.clip(self.array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        _PIL.fromarray(data, mode="RGBA").save(str(path))
        return path

    def __repr__(self):
        return (f"<Image {self.width}x{self.height}, "
                f"{self.stats.get('draws', 0)} draw(s)>")


# The `lights` uniform the sphere/ribbon/arrow shaders declare, in order:
# key_dir, key_col (w = rim), fill_dir, fill_col, ambient.
DEFAULT_LIGHTING = {
    "key_dir": (0.0, 0.0, 1.0),
    "key_col": (0.9, 0.9, 0.9),
    "fill_dir": (0.0, 0.0, 1.0),
    "fill_col": (0.0, 0.0, 0.0),
    "ambient": (0.25, 0.25, 0.25),
    "rim": 0.15,
}


def _lighting_block(lighting):
    rig = dict(DEFAULT_LIGHTING)
    rig.update(lighting or {})
    kd, kc = rig["key_dir"], rig["key_col"]
    fd, fc = rig["fill_dir"], rig["fill_col"]
    amb = rig["ambient"]
    return np.array([kd[0], kd[1], kd[2], 0.0,
                     kc[0], kc[1], kc[2], float(rig["rim"]),
                     fd[0], fd[1], fd[2], 0.0,
                     fc[0], fc[1], fc[2], 0.0,
                     amb[0], amb[1], amb[2], 0.0], dtype=np.float32)


def gpu_available():
    """Return ``(True, name)`` if something can draw, else ``(False, reason)``."""
    try:
        from .backends import wgpu as _backend
    except ImportError as exc:
        return False, (f"no GPU backend: install with "
                       f"pip install \"scigraphs-engine[wgpu]\" ({exc})")
    try:
        ctx = _backend.acquire()
    except Exception as exc:            # noqa: BLE001 - any adapter failure
        return False, f"wgpu is installed but no device could be acquired: {exc}"
    return True, getattr(ctx, "describe", lambda: "wgpu")()


def _render(graph, camera, size, background, lighting):
    try:
        from .backends import wgpu as _backend
    except ImportError as exc:
        raise BackendUnavailable(
            f"no GPU backend: install with "
            f"pip install \"scigraphs-engine[wgpu]\". "
            f"Graph.geometry() returns the MeshSpecs if you want to draw "
            f"them yourself. ({exc})") from exc

    geometry = graph.geometry()
    width, height = int(size[0]), int(size[1])
    cam = (Camera.fit(graph) if camera is None else camera).resolve((width, height))

    try:
        ctx = _backend.acquire()
    except Exception as exc:            # noqa: BLE001 - adapter/device failure
        raise BackendUnavailable(
            f"wgpu is installed but no device could be acquired: {exc}") from exc

    target = _backend.Target(ctx, width, height)
    renderer = _backend.Renderer(ctx, target)
    backend_camera = _backend.camera.Camera(cam.view, cam.proj, cam.near,
                                            cam.far)
    lights = renderer.block_buffer("lights", _lighting_block(lighting))

    uploaded, skipped = [], []
    for spec in geometry.specs:
        mesh = renderer.upload(spec)
        if mesh is None:
            skipped.append(spec.shader.base)
        else:
            uploaded.append(mesh)
    try:
        stats = renderer.render(uploaded, backend_camera,
                                clear_color=tuple(background),
                                blocks={"lights": lights})
        array = target.read()
    finally:
        for mesh in uploaded:
            mesh.destroy()
        target.destroy()

    stats = dict(stats)
    stats["skipped_shaders"] = tuple(skipped)
    stats["report"] = geometry.report
    if skipped:
        # In stats, not a log: an incomplete picture must not read as empty.
        stats["warning"] = (f"this backend has no shader for {sorted(set(skipped))}; "
                            f"those specs were not drawn")
    return Image(array, stats)
