from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import blf
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from scigraphs_core.coloring import colormaps

from ...core.render import lod
from ...core.visualization import text_overlay as t_ov
from .labels import _font_id


RGBA = Tuple[float, float, float, float]

ANCHORS = ('TOP_LEFT', 'TOP_RIGHT', 'BOTTOM_LEFT', 'BOTTOM_RIGHT')

NORM_LABELS = {
    'LINEAR': "linear",
    'LOG': "log10",
    'RANK': "rank equalized",
    'QUANTILE': "quantile",
}


@dataclass
class ColorRamp:
    """A continuous ramp. ``vmin``/``vmax`` are raw data units whatever the
    normalization; ``samples`` are the raw per-node values, which RANK and
    QUANTILE need because their tick positions come from the distribution and
    cannot be derived from the bounds alone."""

    vmin: float = 0.0
    vmax: float = 1.0
    colormap: str = "viridis"
    reverse: bool = False
    norm: str = 'LINEAR'
    gamma: float = 1.0
    samples: Optional[np.ndarray] = None
    log_eps: Optional[float] = None
    title: str = ""
    units: str = ""
    caption: Optional[str] = None
    ticks: Optional[Sequence[float]] = None
    max_ticks: int = 0
    decimals: Optional[int] = None
    bar_w: float = 22.0
    bar_h: float = 150.0
    horizontal: bool = False

    @classmethod
    def from_plan(cls, plan, samples=None, **kwargs):
        """Build from a :class:`colormaps.NormalizationPlan`, which already
        holds the exact bounds the shader used."""
        return cls(
            vmin=float(plan.data_lo), vmax=float(plan.data_hi),
            norm=str(plan.mode), gamma=float(plan.gamma),
            log_eps=float(plan.eps), samples=samples, **kwargs)


@dataclass
class Categories:
    """Discrete classes: communities, components, node types."""

    entries: Sequence[Tuple[str, RGBA]] = ()
    title: str = ""
    max_rows: int = 12


@dataclass
class SizeKey:
    """Marks at real radii. ``marks`` is ``(value, radius_ref_px)``, and the
    caller supplies both because only the caller knows which size ramp is in
    play (see :func:`marks_linear_radius` and :func:`marks_sqrt_area`).

    Printing the mapping as a number would be worse than useless: readers judge
    a disc by its radius and the data lives in the area, so the two disagree by
    a square. Drawing the actual marks is the only honest key."""

    marks: Sequence[Tuple[float, float]] = ()
    title: str = ""
    color: RGBA = (0.82, 0.82, 0.82, 1.0)
    decimals: Optional[int] = None


@dataclass
class MissingSlot:
    """The NaN swatch. Without it an unmeasured node and a node at the floor of
    the ramp are the same pixel colour, and no reader can tell them apart."""

    color: RGBA = (0.30, 0.30, 0.30, 1.0)
    label: str = "no data"
    count: Optional[int] = None


@dataclass
class LegendSpec:
    """Everything :func:`draw` needs. Blocks stack top to bottom in the order
    ramp, categories, size, missing; any of them may be None."""

    ramp: Optional[ColorRamp] = None
    categories: Optional[Categories] = None
    size: Optional[SizeKey] = None
    missing: Optional[MissingSlot] = None

    anchor: str = 'BOTTOM_RIGHT'
    margin: float = 24.0
    offset: Tuple[float, float] = (0.0, 0.0)

    text_color: RGBA = (0.92, 0.92, 0.92, 1.0)
    dim_color: RGBA = (0.72, 0.72, 0.72, 1.0)
    background_color: RGBA = (0.06, 0.06, 0.06, 0.72)
    frame_color: RGBA = (0.55, 0.55, 0.55, 0.55)

    title_pt: float = 13.0
    label_pt: float = 11.0
    caption_pt: float = 10.0

    font_path: Optional[str] = None
    scale: float = 1.0


def spec_is_empty(spec) -> bool:
    return spec is None or not (spec.ramp or spec.categories
                                or spec.size or spec.missing)


def marks_linear_radius(values, base_px, size_max_mult, vmin, vmax):
    """Radii for the per-node size path, which is affine in the *radius*:
    ``r = base * (1 + norm * (mult - 1))`` (batches._radii_for). Area therefore
    grows as the square, and the marks show that rather than assert it."""
    lo, hi = float(vmin), float(vmax)
    span = (hi - lo) if hi > lo else 1.0
    out = []
    for v in values:
        norm = min(max((float(v) - lo) / span, 0.0), 1.0)
        out.append((float(v),
                    float(base_px) * (1.0 + norm * (float(size_max_mult) - 1.0))))
    return out


def marks_sqrt_area(values, max_px, vmax):
    """Radii for the coarsened-supernode path, which ramps by sqrt against the
    largest group (simplify._group_radii), so area is proportional to value."""
    top = float(vmax) if vmax > 0.0 else 1.0
    return [(float(v), float(max_px) * math.sqrt(max(float(v), 0.0) / top))
            for v in values]


def quantile_marks(values, radius_fn, quantiles=(0.05, 0.5, 0.95)):
    """Three marks at data quantiles. The extremes sit just inside the range
    because the true min and max are usually single outliers."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return []
    picks = [float(np.quantile(arr, q)) for q in quantiles]
    uniq = []
    for v in picks:
        if not uniq or abs(v - uniq[-1]) > 1e-12:
            uniq.append(v)
    return radius_fn(uniq)


class _Mapper:
    """Position along the ramp for a value, and back. Mirrors
    colormaps.normalize_values exactly, including the gamma applied last."""

    def __init__(self, ramp: ColorRamp):
        self.mode = (ramp.norm or 'LINEAR').upper()
        if self.mode not in colormaps.NORM_MODE_IDS:
            self.mode = 'LINEAR'
        self.lo = float(ramp.vmin)
        self.hi = float(ramp.vmax)
        if not math.isfinite(self.lo) or not math.isfinite(self.hi):
            self.lo, self.hi = 0.0, 1.0
        if self.hi <= self.lo:
            self.hi = self.lo + 1e-12
        gamma = float(ramp.gamma or 1.0)
        self.gamma = gamma if (math.isfinite(gamma) and gamma > 0.0) else 1.0

        self.value_axis = True
        self.floored = False
        self._table = None

        if self.mode == 'LOG':
            self.eps = self._log_eps(ramp)
            self._llo = math.log10(max(self.lo, self.eps))
            self._lhi = math.log10(max(self.hi, self.eps))
            if self._lhi <= self._llo:
                self._lhi = self._llo + 1e-9
            self.floored = self.lo <= 0.0
        elif self.mode in colormaps.BAKED_NORM_MODES:
            self._table = self._build_table(ramp)
            self.value_axis = self._table is not None

    def _log_eps(self, ramp):
        if ramp.log_eps is not None and ramp.log_eps > 0.0:
            return float(ramp.log_eps)
        eps_fn = getattr(colormaps, "_log_epsilon", None)
        if eps_fn is not None and ramp.samples is not None:
            arr = np.asarray(ramp.samples, dtype=np.float64).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size:
                return float(eps_fn(np.clip(arr, self.lo, self.hi), self.hi))
        return max(self.lo, 1e-12) if self.lo > 0.0 else 1e-12

    def _build_table(self, ramp):
        """``(values, positions)`` for the baked modes, monotone in both."""
        if ramp.samples is None:
            return None
        arr = np.asarray(ramp.samples, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return None
        work = np.clip(arr, self.lo, self.hi)
        if self.mode == 'QUANTILE':
            uniq = np.unique(work)
            if uniq.size < 2:
                return None
            return uniq, np.linspace(0.0, 1.0, uniq.size)
        order = np.sort(work)
        uniq, inverse, counts = np.unique(order, return_inverse=True,
                                          return_counts=True)
        if uniq.size < 2:
            return None
        cumulative = np.cumsum(counts)
        mid = (cumulative - counts + cumulative - 1) * 0.5
        return uniq, mid / float(order.size - 1)

    def t_of(self, value: float) -> float:
        v = float(value)
        if not math.isfinite(v):
            return float('nan')
        if self.mode == 'LOG':
            t = (math.log10(max(v, self.eps)) - self._llo) / (self._lhi - self._llo)
        elif self._table is not None:
            vals, pos = self._table
            t = float(np.interp(v, vals, pos))
        else:
            t = (v - self.lo) / (self.hi - self.lo)
        t = min(max(t, 0.0), 1.0)
        return t ** (1.0 / self.gamma) if self.gamma != 1.0 else t

    def v_of(self, t: float) -> float:
        t = min(max(float(t), 0.0), 1.0)
        if self.gamma != 1.0:
            t = t ** self.gamma
        if self.mode == 'LOG':
            return 10.0 ** (self._llo + t * (self._lhi - self._llo))
        if self._table is not None:
            vals, pos = self._table
            return float(np.interp(t, pos, vals))
        return self.lo + t * (self.hi - self.lo)


_MANTISSA_SETS = ((1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 5), (1, 3), (1,))
_STEP_LADDER = (1.0, 2.0, 5.0)


def _nice_step(raw: float) -> float:
    """Smallest 1/2/5 x 10^k at or above ``raw``."""
    if not math.isfinite(raw) or raw <= 0.0:
        return 1.0
    exp = math.floor(math.log10(raw))
    frac = raw / (10.0 ** exp)
    for m in _STEP_LADDER:
        if frac <= m * (1.0 + 1e-9):
            return m * (10.0 ** exp)
    return 10.0 ** (exp + 1)


def _next_step(step: float) -> float:
    exp = math.floor(math.log10(step) + 1e-9)
    frac = step / (10.0 ** exp)
    for m in _STEP_LADDER:
        if frac < m * (1.0 - 1e-9):
            return m * (10.0 ** exp)
    return 10.0 ** (exp + 1)


def _snap(value: float, step: float) -> float:
    """Kill the 0.30000000000000004 that a float multiply leaves behind, or
    the tick prints four decimals of noise."""
    if step <= 0.0:
        return value
    digits = max(0, -int(math.floor(math.log10(step))) + 2)
    return round(round(value / step) * step, min(digits, 12))


def linear_ticks(lo: float, hi: float, max_count: int) -> List[float]:
    """1-2-5 ticks inside ``[lo, hi]``, densest set that fits."""
    if hi <= lo or max_count < 2:
        return [lo, hi]
    step = _nice_step((hi - lo) / float(max_count))
    for _ in range(24):
        first = math.ceil(lo / step - 1e-9)
        last = math.floor(hi / step + 1e-9)
        count = int(last - first) + 1
        if count <= max_count:
            if count >= 2:
                return [_snap((first + i) * step, step) for i in range(count)]
            break
        step = _next_step(step)
    return [lo, hi]


def log_ticks(lo: float, hi: float, max_count: int, eps: float) -> List[float]:
    """Decade ticks, subdivided by mantissa when a span under a couple of
    decades would otherwise show two labels, thinned by whole decades when it
    would show forty."""
    lo = max(lo, eps)
    hi = max(hi, lo * (1.0 + 1e-9))
    e0 = math.floor(math.log10(lo))
    e1 = math.ceil(math.log10(hi))
    exps = range(int(e0), int(e1) + 1)

    for mantissas in _MANTISSA_SETS:
        vals = [m * (10.0 ** e) for e in exps for m in mantissas]
        vals = [v for v in vals if lo * (1.0 - 1e-9) <= v <= hi * (1.0 + 1e-9)]
        vals = sorted(set(vals))
        if 2 <= len(vals) <= max_count:
            return vals

    decades = [10.0 ** e for e in exps
               if lo * (1.0 - 1e-9) <= 10.0 ** e <= hi * (1.0 + 1e-9)]
    for stride in (2, 3, 4, 5, 6, 10, 20, 50):
        thinned = decades[::-1][::stride][::-1]
        if 2 <= len(thinned) <= max_count:
            return thinned
    return [lo, hi]


def _tick_settings(values: Sequence[float], decimals: Optional[int],
                   prefix: str = "", suffix: str = ""):
    """Pick a format from the tick spacing. Two decimals on a ramp of 1e-5
    prints five identical zeros, and eight decimals on a ramp of thousands
    prints noise."""
    finite = [abs(v) for v in values if math.isfinite(v) and v != 0.0]
    magnitude = max(finite) if finite else 1.0

    ordered = sorted(v for v in values if math.isfinite(v))
    gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b - a > 0.0]
    step = min(gaps) if gaps else magnitude

    fmt = 'FLOAT'
    if decimals is None:
        if magnitude >= 1e5 or magnitude < 1e-3:
            fmt, decimals = 'SCIENTIFIC', 1
        else:
            decimals = int(min(max(math.ceil(-math.log10(step)) + 0, 0), 6))
            if step >= 1.0 and all(abs(v - round(v)) < 1e-9 for v in ordered):
                fmt, decimals = 'INTEGER', 0

    return t_ov.TextOverlaySettings(
        size_mode='FIXED', fixed_size=12, size_scale=1.0, max_distance=0.0,
        text_color=(1.0, 1.0, 1.0), background_enabled=False,
        background_color=(0.0, 0.0, 0.0), background_alpha=0.0,
        depth_occlusion=False, filter_enabled=False, filter_attribute="",
        filter_operator='GREATER', filter_value=0.0,
        format_type=fmt, float_decimals=int(decimals),
        format_prefix=prefix, format_suffix=suffix,
        thousands_separator=(fmt == 'INTEGER' and magnitude >= 1e4),
    )


def _log_texts(values, units):
    """One format per decade, not one for the whole ramp. A shared decimal
    count turns 1000 into "1000.000" the moment 0.001 is also on the bar."""
    out = []
    for v in values:
        exp = math.floor(math.log10(abs(v))) if v else 0
        if v == 0.0 or -4 <= exp <= 5:
            settings = _tick_settings([v], max(0, -exp), suffix=units)
        else:
            settings = t_ov.TextOverlaySettings(
                size_mode='FIXED', fixed_size=12, size_scale=1.0,
                max_distance=0.0, text_color=(1.0, 1.0, 1.0),
                background_enabled=False, background_color=(0.0, 0.0, 0.0),
                background_alpha=0.0, depth_occlusion=False,
                filter_enabled=False, filter_attribute="",
                filter_operator='GREATER', filter_value=0.0,
                format_type='SCIENTIFIC', float_decimals=0, format_suffix=units)
        out.append(t_ov.format_value(v, settings))
    return out


def bound_ticks(ramp: ColorRamp,
                mapper: Optional[_Mapper] = None) -> List[Tuple[float, str]]:
    """The two ends of the bar, labelled with the real data bounds.

    Round ticks stop short of the ends by construction: a 1..248 range gets
    50 to 200, so the top of the bar is a colour the reader cannot put a number
    on. These pin both ends. They are drawn dimmer, because the round numbers
    are what you read the scale by and these are what you read the limits by.
    """
    mapper = mapper or _Mapper(ramp)
    if ramp.ticks or not mapper.value_axis:
        return []
    lo, hi = float(mapper.lo), float(mapper.hi)
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return []
    settings = _tick_settings([lo, hi], ramp.decimals, suffix=ramp.units)
    out = []
    for v in (lo, hi):
        t = mapper.t_of(v)
        if not math.isnan(t):
            out.append((t, t_ov.format_value(v, settings)))
    return out


def plan_ticks(ramp: ColorRamp, max_count: int,
               mapper: Optional[_Mapper] = None) -> List[Tuple[float, str]]:
    """``(position_in_0_1, text)`` for the ramp, ordered low to high. Exposed
    so a test can check placement without touching the GPU."""
    mapper = mapper or _Mapper(ramp)
    max_count = max(2, int(max_count))

    percent = False
    decades = False
    if ramp.ticks:
        values = [float(v) for v in ramp.ticks]
    elif not mapper.value_axis:
        values = [t * 100.0 for t in linear_ticks(0.0, 1.0, max_count)]
        percent = True
    elif mapper.mode == 'LOG':
        values = log_ticks(mapper.lo, mapper.hi, max_count, mapper.eps)
        decades = ramp.decimals is None
    elif mapper.mode in colormaps.BAKED_NORM_MODES:
        values = [mapper.v_of(t) for t in linear_ticks(0.0, 1.0, max_count)]
    else:
        values = linear_ticks(mapper.lo, mapper.hi, max_count)

    if percent:
        settings = _tick_settings(values, None, suffix="%")
        return [(v / 100.0, t_ov.format_value(v, settings)) for v in values]

    values = [v for v in values if math.isfinite(v)]
    if decades:
        texts = _log_texts(values, ramp.units)
    else:
        settings = _tick_settings(values, ramp.decimals, suffix=ramp.units)
        texts = [t_ov.format_value(v, settings) for v in values]

    out = []
    for v, text in zip(values, texts):
        t = mapper.t_of(v)
        if not math.isnan(t):
            out.append((t, text))
    return out


def _thin(ticks, positions_px, min_gap):
    """Greedy from the low end. Fewer readable ticks beat a black smear."""
    if not ticks:
        return []
    kept = []
    last = None
    for i, (tick, y) in enumerate(zip(ticks, positions_px)):
        if last is None or (y - last) >= min_gap:
            kept.append((i, tick, y))
            last = y
    top = len(ticks) - 1
    if kept[-1][0] != top:
        top_y = positions_px[top]
        if len(kept) >= 2 and (top_y - kept[-2][2]) >= min_gap:
            kept[-1] = (top, ticks[top], top_y)
        elif (top_y - kept[-1][2]) >= min_gap:
            kept.append((top, ticks[top], top_y))
    return [(tick, y) for _i, tick, y in kept]


def _rects_batch(shader, rects):
    coords, indices = [], []
    for k, (x0, y0, x1, y1) in enumerate(rects):
        base = k * 4
        coords.extend(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
        indices.extend(((base, base + 1, base + 2), (base, base + 2, base + 3)))
    return batch_for_shader(shader, 'TRIS', {"pos": coords}, indices=indices)


def _fill(rects, color):
    if not rects:
        return
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = _rects_batch(shader, rects)
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _outline(rect, color, w):
    x0, y0, x1, y1 = rect
    _fill([(x0, y0, x1, y0 + w), (x0, y1 - w, x1, y1),
           (x0, y0 + w, x0 + w, y1 - w), (x1 - w, y0 + w, x1, y1 - w)], color)


def _disc(cx, cy, r, color, segments=48):
    if r <= 0.0:
        return
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    coords = [(cx, cy)]
    indices = []
    for i in range(segments + 1):
        a = 2.0 * math.pi * i / segments
        coords.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        if i:
            indices.append((0, i, i + 1))
    batch = batch_for_shader(shader, 'TRIS', {"pos": coords}, indices=indices)
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _ramp_lut(ramp: ColorRamp, steps=256):
    """Straight from sample_colormap; see the module note on colour encoding."""
    try:
        lut = colormaps.sample_colormap(ramp.colormap, steps, ramp.reverse)
    except Exception:  # noqa: BLE001 - unknown colormap name: grey ramp
        grey = np.linspace(0.0, 1.0, steps)
        lut = np.stack([grey, grey, grey, np.ones(steps)], axis=1)
    lut = np.asarray(lut, dtype=np.float32)
    if lut.ndim != 2 or lut.shape[1] < 4:
        lut = np.pad(lut.reshape(-1, lut.shape[-1]),
                     ((0, 0), (0, 4 - lut.shape[-1])), constant_values=1.0)
    return lut


def _gradient_bar(rect, lut, gamma=1.0, horizontal=False):
    """One SMOOTH_COLOR batch rather than 256 UNIFORM_COLOR draws: the strip is
    per-frame overlay work and 256 draw calls a frame is not free."""
    x0, y0, x1, y1 = rect
    n = lut.shape[0]
    shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    coords, colors, indices = [], [], []
    for i in range(n):
        u = i / float(n - 1)
        t = u ** (1.0 / gamma) if gamma != 1.0 else u
        rgba = tuple(float(c) for c in lut[i][:4])
        if horizontal:
            x = x0 + (x1 - x0) * t
            coords.extend(((x, y0), (x, y1)))
        else:
            y = y0 + (y1 - y0) * t
            coords.extend(((x0, y), (x1, y)))
        colors.extend((rgba, rgba))
        if i:
            base = (i - 1) * 2
            indices.extend(((base, base + 1, base + 3),
                            (base, base + 3, base + 2)))
    batch = batch_for_shader(shader, 'TRIS', {"pos": coords, "color": colors},
                             indices=indices)
    shader.bind()
    batch.draw(shader)


def _text(fid, x, y, pt, text, color):
    blf.size(fid, pt)
    blf.color(fid, color[0], color[1], color[2], color[3])
    blf.position(fid, x, y, 0.0)
    blf.draw(fid, text)


def _measure(fid, pt, text):
    blf.size(fid, pt)
    return blf.dimensions(fid, text)


_PAD = 13.0
_BLOCK_GAP = 13.0
_TICK_LEN = 5.0
_TICK_GAP = 4.0
_LINE_GAP = 6.0
_SWATCH = 12.0
_ROW_GAP = 5.0
_MARK_GAP = 8.0
_FRAME_W = 1.0


def _line_h(fid, pt):
    return _measure(fid, pt, "Mg")[1]


def _ramp_caption(ramp, mapper):
    """What the ramp is, in the cases where it is not obvious. LINEAR is the
    default and naming it costs a line that never varies."""
    if ramp.caption is not None:
        return ramp.caption
    caption = ramp.colormap
    if mapper.mode != 'LINEAR':
        caption += f", {NORM_LABELS.get(mapper.mode, mapper.mode)}"
    if mapper.floored:
        caption += ", <=0 floored"
    if not mapper.value_axis:
        caption += ", percentile axis"
    return caption


def _pin_bounds(ramp, mapper, kept, span_px, gap_px):
    """Put the real bounds back on the bar, evicting whatever they land on.

    Bounds win the collision: a round tick has a neighbour half a step away
    saying nearly the same thing, an end label has nothing behind it.
    ``gap_px`` is whatever collides, which is line height down a vertical bar
    and label width along a horizontal one.
    """
    bounds = bound_ticks(ramp, mapper)
    if not bounds:
        return kept, set()
    bp = [span_px * t for t, _ in bounds]
    kept = [(tick, p) for tick, p in kept
            if all(abs(p - b) >= gap_px for b in bp)]
    return (sorted(list(zip(bounds, bp)) + kept, key=lambda kv: kv[1]),
            {id(t) for t in bounds})


def _ramp_block(spec, ramp, fid, s):
    title_pt, label_pt, cap_pt = (spec.title_pt * s, spec.label_pt * s,
                                  spec.caption_pt * s)
    bar_w, bar_h = ramp.bar_w * s, ramp.bar_h * s
    mapper = _Mapper(ramp)

    label_h = _line_h(fid, label_pt)
    max_count = ramp.max_ticks or int(bar_h / max(label_h * 1.9, 1.0)) + 1
    ticks = plan_ticks(ramp, max(2, min(max_count, 12)), mapper)

    ys = [bar_h * t for t, _ in ticks]
    kept = _thin(ticks, ys, label_h * 1.6)
    kept, ends = _pin_bounds(ramp, mapper, kept, bar_h, label_h * 1.6)

    widths = [_measure(fid, label_pt, text)[0] for (_t, text), _y in kept]
    tick_w = max(widths) if widths else 0.0
    row_w = bar_w + (_TICK_LEN + _TICK_GAP) * s + tick_w

    caption = _ramp_caption(ramp, mapper)

    title_h = _line_h(fid, title_pt) + _LINE_GAP * 1.8 * s if ramp.title else 0.0
    cap_h = _line_h(fid, cap_pt) + _LINE_GAP * s if caption else 0.0
    over = label_h * 0.5

    width = max(row_w,
                _measure(fid, title_pt, ramp.title)[0] if ramp.title else 0.0,
                _measure(fid, cap_pt, caption)[0] if caption else 0.0)
    payload = {"ticks": kept, "ends": ends, "bar_w": bar_w, "bar_h": bar_h,
               "caption": caption, "lut": _ramp_lut(ramp),
               "gamma": mapper.gamma, "title": ramp.title,
               "title_h": title_h, "cap_h": cap_h, "over": over}
    return width, title_h + over + bar_h + over + cap_h, payload


def _ramp_block_h(spec, ramp, fid, s):
    """The same ramp lying down. A tall key eats the height of a wide figure,
    and a graph is usually wider than it is tall."""
    title_pt, label_pt, cap_pt = (spec.title_pt * s, spec.label_pt * s,
                                  spec.caption_pt * s)
    bar_len, bar_th = ramp.bar_h * s, ramp.bar_w * s
    mapper = _Mapper(ramp)
    label_h = _line_h(fid, label_pt)

    ticks = plan_ticks(ramp, 12, mapper)
    widest = max((_measure(fid, label_pt, text)[0] for _t, text in ticks),
                 default=0.0)
    gap = widest + _TICK_GAP * 2.0 * s
    xs = [bar_len * t for t, _ in ticks]
    kept = _thin(ticks, xs, gap)
    kept, ends = _pin_bounds(ramp, mapper, kept, bar_len, gap)

    caption = _ramp_caption(ramp, mapper)
    title_h = _line_h(fid, title_pt) + _LINE_GAP * s if ramp.title else 0.0
    cap_h = _line_h(fid, cap_pt) + _LINE_GAP * s if caption else 0.0
    axis_h = (_TICK_LEN + _TICK_GAP) * s + label_h

    over = widest * 0.5
    width = max(bar_len + over * 2.0,
                _measure(fid, title_pt, ramp.title)[0] if ramp.title else 0.0,
                _measure(fid, cap_pt, caption)[0] if caption else 0.0)
    payload = {"ticks": kept, "ends": ends, "bar_len": bar_len,
               "bar_th": bar_th, "caption": caption, "lut": _ramp_lut(ramp),
               "gamma": mapper.gamma, "title": ramp.title,
               "title_h": title_h, "cap_h": cap_h, "over": over,
               "axis_h": axis_h}
    return width, title_h + bar_th + axis_h + cap_h, payload


def _draw_ramp_h(spec, payload, x, y_top, fid, s):
    title_pt, label_pt, cap_pt = (spec.title_pt * s, spec.label_pt * s,
                                  spec.caption_pt * s)
    y = y_top
    if payload["title"]:
        y -= payload["title_h"]
        _text(fid, x, y + _LINE_GAP * s * 0.5, title_pt, payload["title"],
              spec.text_color)

    x0 = x + payload["over"]
    bar_len, bar_th = payload["bar_len"], payload["bar_th"]
    y -= bar_th
    rect = (x0, y, x0 + bar_len, y + bar_th)
    _gradient_bar(rect, payload["lut"], payload["gamma"], horizontal=True)
    _outline(rect, spec.frame_color, _FRAME_W * s)

    marks = []
    for (t, _label), _p in payload["ticks"]:
        tx = x0 + bar_len * t
        marks.append((tx - _FRAME_W * s * 0.5, y - _TICK_LEN * s,
                      tx + _FRAME_W * s * 0.5, y))
    _fill(marks, spec.text_color)

    ends = payload.get("ends") or set()
    label_y = y - (_TICK_LEN + _TICK_GAP) * s - _line_h(fid, label_pt)
    for tick, _p in payload["ticks"]:
        t, text = tick
        tw = _measure(fid, label_pt, text)[0]
        color = spec.dim_color if id(tick) in ends else spec.text_color
        _text(fid, x0 + bar_len * t - tw * 0.5, label_y, label_pt, text, color)

    if payload["caption"]:
        _text(fid, x, label_y - payload["cap_h"] + _LINE_GAP * s * 0.5,
              cap_pt, payload["caption"], spec.dim_color)


def _draw_ramp(spec, payload, x, y_top, fid, s):
    title_pt, label_pt, cap_pt = (spec.title_pt * s, spec.label_pt * s,
                                  spec.caption_pt * s)
    y = y_top
    if payload["title"]:
        y -= payload["title_h"]
        _text(fid, x, y + _LINE_GAP * s * 0.5, title_pt, payload["title"],
              spec.text_color)

    y -= payload["over"]
    bar_w, bar_h = payload["bar_w"], payload["bar_h"]
    bar_bottom = y - bar_h
    rect = (x, bar_bottom, x + bar_w, y)
    _gradient_bar(rect, payload["lut"], payload["gamma"])
    _outline(rect, spec.frame_color, _FRAME_W * s)

    tick_x = x + bar_w
    label_x = tick_x + (_TICK_LEN + _TICK_GAP) * s
    marks = []
    for (t, text), _y in payload["ticks"]:
        ty = bar_bottom + bar_h * t
        marks.append((tick_x, ty - _FRAME_W * s * 0.5,
                      tick_x + _TICK_LEN * s, ty + _FRAME_W * s * 0.5))
    _fill(marks, spec.text_color)
    ends = payload.get("ends") or set()
    for tick, _y in payload["ticks"]:
        t, text = tick
        ty = bar_bottom + bar_h * t
        th = _measure(fid, label_pt, text)[1]
        color = spec.dim_color if id(tick) in ends else spec.text_color
        _text(fid, label_x, ty - th * 0.5, label_pt, text, color)

    if payload["caption"]:
        _text(fid, x,
              bar_bottom - payload["over"] - payload["cap_h"]
              + _LINE_GAP * s * 0.5,
              cap_pt, payload["caption"], spec.dim_color)


def _rows_block(spec, rows, title, fid, s):
    """Shared by the category stack and the missing slot: swatch plus label."""
    title_pt, label_pt = spec.title_pt * s, spec.label_pt * s
    row_h = max(_SWATCH * s, _line_h(fid, label_pt))
    title_h = _line_h(fid, title_pt) + _LINE_GAP * s if title else 0.0

    label_w = max((_measure(fid, label_pt, text)[0] for text, _c in rows),
                  default=0.0)
    width = max(_SWATCH * s + _ROW_GAP * s + label_w,
                _measure(fid, title_pt, title)[0] if title else 0.0)
    height = title_h + len(rows) * row_h + max(0, len(rows) - 1) * _ROW_GAP * s
    return width, height, {"rows": rows, "title": title, "row_h": row_h,
                           "title_h": title_h}


def _draw_rows(spec, payload, x, y_top, fid, s):
    title_pt, label_pt = spec.title_pt * s, spec.label_pt * s
    y = y_top
    if payload["title"]:
        y -= payload["title_h"]
        _text(fid, x, y + _LINE_GAP * s * 0.5, title_pt, payload["title"],
              spec.text_color)

    row_h = payload["row_h"]
    sw = _SWATCH * s
    for text, color in payload["rows"]:
        y -= row_h
        cy = y + row_h * 0.5
        rect = (x, cy - sw * 0.5, x + sw, cy + sw * 0.5)
        _fill([rect], tuple(color))
        _outline(rect, spec.frame_color, _FRAME_W * s)
        th = _measure(fid, label_pt, text)[1]
        _text(fid, x + sw + _ROW_GAP * s, cy - th * 0.5, label_pt, text,
              spec.text_color)
        y -= _ROW_GAP * s


def _size_block(spec, key: SizeKey, fid, s):
    title_pt, label_pt = spec.title_pt * s, spec.label_pt * s
    marks = sorted(((float(v), max(float(r), 0.5) * s) for v, r in key.marks),
                   key=lambda m: m[1])
    if not marks:
        return 0.0, 0.0, None

    settings = _tick_settings([v for v, _r in marks], key.decimals)
    texts = [t_ov.format_value(v, settings) for v, _r in marks]
    label_h = _line_h(fid, label_pt)
    title_h = _line_h(fid, title_pt) + _LINE_GAP * s if key.title else 0.0

    cells = [max(2.0 * r, _measure(fid, label_pt, txt)[0])
             for (_v, r), txt in zip(marks, texts)]
    width = sum(cells) + _MARK_GAP * s * max(0, len(cells) - 1)
    width = max(width, _measure(fid, title_pt, key.title)[0] if key.title else 0.0)
    disc_h = 2.0 * max(r for _v, r in marks)
    height = title_h + disc_h + _LINE_GAP * s + label_h
    return width, height, {"marks": marks, "texts": texts, "cells": cells,
                           "title": key.title, "title_h": title_h,
                           "label_h": label_h, "disc_h": disc_h,
                           "color": key.color}


def _draw_size(spec, payload, x, y_top, fid, s):
    title_pt, label_pt = spec.title_pt * s, spec.label_pt * s
    y = y_top
    if payload["title"]:
        y -= payload["title_h"]
        _text(fid, x, y + _LINE_GAP * s * 0.5, title_pt, payload["title"],
              spec.text_color)

    base = y - payload["disc_h"] - _LINE_GAP * s - payload["label_h"]
    cx = x
    for (_v, r), text, cell in zip(payload["marks"], payload["texts"],
                                   payload["cells"]):
        centre = cx + cell * 0.5
        disc_y = base + payload["label_h"] + _LINE_GAP * s + r
        _disc(centre, disc_y, r, payload["color"])
        tw = _measure(fid, label_pt, text)[0]
        _text(fid, centre - tw * 0.5, base, label_pt, text, spec.text_color)
        cx += cell + _MARK_GAP * s


def _missing_rows(slot: MissingSlot):
    label = slot.label
    if slot.count is not None:
        label = f"{label} ({int(slot.count)})"
    return [(label, tuple(slot.color))]


def _category_rows(cats: Categories):
    rows = [(str(name), tuple(color)) for name, color in cats.entries]
    limit = max(1, int(cats.max_rows))
    if len(rows) > limit:
        hidden = len(rows) - (limit - 1)
        rows = rows[:limit - 1] + [(f"+{hidden} more", (0.0, 0.0, 0.0, 0.0))]
    return rows


def _layout(spec, fid, s, avail_h):
    """Blocks top to bottom, then the box around them. Returns
    ``(box_w, box_h, [(kind, payload, width, height)])``."""
    blocks = []
    if spec.ramp is not None:
        ramp = spec.ramp
        if avail_h > 0.0 and not ramp.horizontal:
            budget = avail_h / s - 2.0 * _PAD - 80.0
            if ramp.bar_h > budget:
                ramp = ColorRamp(**{**ramp.__dict__, "bar_h": max(48.0, budget)})
        build = _ramp_block_h if ramp.horizontal else _ramp_block
        blocks.append(('ramp',) + build(spec, ramp, fid, s))
    if spec.categories is not None and spec.categories.entries:
        w, h, p = _rows_block(spec, _category_rows(spec.categories),
                              spec.categories.title, fid, s)
        blocks.append(('rows', w, h, p))
    if spec.size is not None and spec.size.marks:
        w, h, p = _size_block(spec, spec.size, fid, s)
        if p is not None:
            blocks.append(('size', w, h, p))
    if spec.missing is not None:
        w, h, p = _rows_block(spec, _missing_rows(spec.missing), "", fid, s)
        blocks.append(('rows', w, h, p))

    blocks = [(k, w, h, p) for k, w, h, p in blocks if h > 0.0]
    if not blocks:
        return 0.0, 0.0, []

    content_w = max(w for _k, w, _h, _p in blocks)
    content_h = (sum(h for _k, _w, h, _p in blocks)
                 + _BLOCK_GAP * s * (len(blocks) - 1))
    return (content_w + 2.0 * _PAD * s, content_h + 2.0 * _PAD * s, blocks)


def _origin(spec, width, height, box_w, box_h, s):
    margin = spec.margin * s
    left = margin if spec.anchor.endswith('LEFT') else width - box_w - margin
    bottom = (height - box_h - margin if spec.anchor.startswith('TOP')
              else margin)
    return (left + spec.offset[0] * s, bottom + spec.offset[1] * s)


def measure(scene, width, height, spec) -> Tuple[float, float]:
    """Box size in buffer pixels, without drawing. Useful for keeping other
    overlays clear of the legend."""
    if spec_is_empty(spec):
        return (0.0, 0.0)
    s = lod.px_scale(height) * float(spec.scale or 1.0)
    fid = _font_id(_font_path(scene, spec))
    box_w, box_h, _blocks = _layout(spec, fid, s, float(height))
    return (box_w, box_h)


def _font_path(scene, spec):
    if spec.font_path is not None:
        return spec.font_path
    props = getattr(scene, "scigraphs", None) if scene is not None else None
    if props is None:
        return ""
    try:
        return t_ov.get_font_path(props)
    except AttributeError:
        return ""


def draw(scene, width, height, spec) -> None:
    """Draw ``spec`` into the framebuffer that is already bound.

    ``width``/``height`` are that buffer's pixel size, origin bottom left, the
    convention blf uses. ``scene`` is only consulted for the font path and may
    be None. The one entry point serves both call sites: the viewport
    POST_PIXEL handler calls it directly, and the F12 path calls it through
    :func:`composite`.

    Sizes come out of the spec in reference pixels and are scaled by
    ``lod.px_scale(height)``, so the legend covers the same fraction of the
    frame in a 1080p viewport and a 4K render.
    """
    if spec_is_empty(spec) or width <= 0 or height <= 0:
        return

    from mathutils import Matrix

    s = lod.px_scale(height) * float(spec.scale or 1.0)
    fid = _font_id(_font_path(scene, spec))
    box_w, box_h, blocks = _layout(spec, fid, s, float(height))
    if not blocks:
        return
    x0, y0 = _origin(spec, width, height, box_w, box_h, s)

    prev_blend = gpu.state.blend_get()
    gpu.state.blend_set('ALPHA')

    ortho = Matrix((
        (2.0 / width, 0.0, 0.0, -1.0),
        (0.0, 2.0 / height, 0.0, -1.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))
    try:
        with gpu.matrix.push_pop():
            with gpu.matrix.push_pop_projection():
                gpu.matrix.load_matrix(Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(ortho)

                box = (x0, y0, x0 + box_w, y0 + box_h)
                if spec.background_color[3] > 0.0:
                    _fill([box], tuple(spec.background_color))
                if spec.frame_color[3] > 0.0:
                    _outline(box, tuple(spec.frame_color), _FRAME_W * s)

                x = x0 + _PAD * s
                y = y0 + box_h - _PAD * s
                for kind, _w, h, payload in blocks:
                    if kind == 'ramp':
                        drawer = (_draw_ramp_h if spec.ramp is not None
                                  and spec.ramp.horizontal else _draw_ramp)
                        drawer(spec, payload, x, y, fid, s)
                    elif kind == 'rows':
                        _draw_rows(spec, payload, x, y, fid, s)
                    elif kind == 'size':
                        _draw_size(spec, payload, x, y, fid, s)
                    y -= h + _BLOCK_GAP * s
    finally:
        gpu.state.blend_set(prev_blend)


def composite(scene, width, height, spec, color):
    """Straight-alpha composite of the legend over an ``(h, w, 4)`` float
    array, for the F12 path. Same arithmetic as engine._composite_labels, which
    runs before Blender's premultiply, so this must too."""
    if spec_is_empty(spec):
        return color

    offscreen = gpu.types.GPUOffScreen(int(width), int(height))
    try:
        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0, 0.0, 0.0, 0.0))
            draw(scene, int(width), int(height), spec)
            over = np.array(fb.read_color(0, 0, int(width), int(height), 4, 0,
                                          'FLOAT'),
                            dtype=np.float32).reshape(int(height), int(width), 4)
    finally:
        offscreen.free()

    color = np.asarray(color, dtype=np.float32)
    oa = over[..., 3:4]
    ca = color[..., 3:4]
    out_a = oa + ca * (1.0 - oa)
    out_rgb = over[..., :3] * oa + color[..., :3] * ca * (1.0 - oa)
    rgb = np.divide(out_rgb, out_a, out=np.zeros_like(out_rgb),
                    where=out_a > 1e-6)
    return np.concatenate([rgb, out_a], axis=-1).astype(np.float32)


def _ramp_source(scene, st):
    """Who decided the colors: the render settings, or the Coloring bake.

    Two paths produce a colormapped graph and only one of them is the render's
    own. Under ATTRIBUTE the engine maps the attribute per frame, so ``st``
    describes it. Under VERTEX the colors were baked into the color attribute
    by Coloring, and ``st`` knows nothing about them, which is how the key came
    up empty on almost every real scene.

    The bake settings are a record of the last bake, not of what is in the
    layer. Editing them without re-baking leaves the key describing colors the
    mesh no longer has.
    """
    mode = getattr(st, "color_mode", "")
    if mode == 'ATTRIBUTE':
        return dict(attr=str(getattr(st, "attr_name", "")),
                    colormap=str(getattr(st, "colormap", "viridis")),
                    reverse=bool(getattr(st, "reverse_colormap", False)),
                    norm=str(getattr(st, "norm_mode", 'LINEAR')),
                    gamma=float(getattr(st, "norm_gamma", 1.0)),
                    vmin=None, vmax=None)
    if mode != 'VERTEX':
        return None
    cp = getattr(scene, "scigraphs_coloring", None)
    attr = str(getattr(cp, "attribute_name", "")) if cp else ""
    if not attr:
        return None
    locked = not bool(getattr(cp, "auto_range", True))
    return dict(attr=attr,
                colormap=str(getattr(cp, "colormap", "viridis")),
                reverse=bool(getattr(cp, "reverse", False)),
                norm=str(getattr(cp, "color_norm", 'LINEAR')),
                gamma=float(getattr(cp, "color_gamma", 1.0)),
                vmin=float(getattr(cp, "vmin", 0.0)) if locked else None,
                vmax=float(getattr(cp, "vmax", 1.0)) if locked else None)


def spec_from_scene(scene, obj, st=None):
    """A :class:`LegendSpec` for what the engine is currently drawing, or None.

    Reads the settings that produced the colors, so the key cannot describe a
    mapping the nodes are not using. Returns None rather than an empty box when
    there is nothing to key.
    """
    if not bool(getattr(scene, "scigraphs_preview_legend", False)):
        return None
    from . import attributes
    if st is None:
        from .host import settings_from_scene
        st = settings_from_scene(scene)
    src = _ramp_source(scene, st)
    if src is None:
        return None
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return None
    values, vmin, vmax = attributes.read_value_channel(
        mesh, src["attr"], len(mesh.vertices))
    if values is None:
        return None
    if src["vmin"] is not None:
        vmin, vmax = src["vmin"], src["vmax"]

    missing = None
    unmeasured = int(np.count_nonzero(~np.isfinite(values)))
    if unmeasured:
        missing = MissingSlot(count=unmeasured)

    ramp = ColorRamp(
        vmin=vmin, vmax=vmax,
        colormap=src["colormap"],
        reverse=src["reverse"],
        norm=src["norm"],
        gamma=src["gamma"],
        samples=values,
        title=src["attr"],
        horizontal=(getattr(scene, "scigraphs_preview_legend_orient",
                            'VERTICAL') == 'HORIZONTAL'),
    )

    size = None
    if bool(getattr(st, "size_by_attr", False)):
        base = float(getattr(st, "impostor_radius", 0.1))
        mult = float(getattr(st, "size_max_mult", 1.0))
        finite = values[np.isfinite(values)]
        if finite.size and mult > 1.0:
            qs = np.quantile(finite, (0.0, 0.5, 1.0))
            size = SizeKey(marks=marks_linear_radius(qs, base, mult, vmin, vmax),
                           title="Size")

    transparent = (0.0, 0.0, 0.0, 0.0)
    boxed = bool(getattr(scene, "scigraphs_preview_legend_box", True))
    spec = LegendSpec(
        ramp=ramp, size=size, missing=missing,
        anchor=str(getattr(scene, "scigraphs_preview_legend_anchor",
                           'BOTTOM_RIGHT')),
        offset=(float(getattr(scene, "scigraphs_preview_legend_x", 0.0)),
                float(getattr(scene, "scigraphs_preview_legend_y", 0.0))),
        scale=float(getattr(scene, "scigraphs_preview_legend_scale", 1.0)),
        background_color=(LegendSpec.background_color if boxed else transparent),
        frame_color=(LegendSpec.frame_color if boxed else transparent),
    )
    return None if spec_is_empty(spec) else spec


_HANDLE = None


_WARNED = False


def _warn_once(exc):
    """Swallowing a draw failure keeps the viewport alive and hides real bugs:
    a shadowed name read as "the key silently does nothing". Say it once, since
    this runs per frame."""
    global _WARNED
    if not _WARNED:
        _WARNED = True
        print(f"SciGraphs: color key failed to draw: {exc!r}")


def _overlay_callback():
    """The key in every shading mode, not only under a live render.

    Every shading mode, this handler alone. It used to step aside under
    Rendered and let view_draw place the key, but view_draw runs before
    Blender's overlays and they cover whatever it drew: with the graph object
    selected the key ended up under its own vertices. POST_PIXEL runs last.
    """
    import bpy
    ctx = bpy.context
    area = getattr(ctx, "area", None)
    region = getattr(ctx, "region", None)
    space = getattr(ctx, "space_data", None)
    if area is None or area.type != 'VIEW_3D' or region is None:
        return
    shading = getattr(space, "shading", None)
    scene = getattr(ctx, "scene", None)
    if shading is None or scene is None:
        return

    try:
        from .state import is_graph_object
        obj = next((o for o in scene.objects
                    if is_graph_object(o) and not o.hide_get()), None)
        if obj is None:
            return
        spec = spec_from_scene(scene, obj)
        if spec is not None:
            draw(scene, region.width, region.height, spec)
    except Exception as exc:  # noqa: BLE001 - must not break the viewport
        _warn_once(exc)


def enable_overlay():
    global _HANDLE
    import bpy
    if _HANDLE is None:
        _HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _overlay_callback, (), 'WINDOW', 'POST_PIXEL')


def disable_overlay():
    global _HANDLE
    import bpy
    if _HANDLE is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_HANDLE, 'WINDOW')
        _HANDLE = None
