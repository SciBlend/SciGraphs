"""Colormap catalog and value-to-RGBA helpers.

Two execution paths:

1. If matplotlib is available, we delegate to ``matplotlib.cm`` so users get
   the full quality of every supported colormap.
2. Otherwise we fall back to a numpy-only piecewise linear interpolation
   between a small set of control points hard-coded for each colormap. This
   way the addon stays usable inside vanilla Blender installations that do
   not bundle matplotlib.

Quick lookups (``QUICK_COLORMAPS``) are designed for the floating toolbar:
they cover the most common use cases (perceptual sequential and diverging
maps) and always have a fallback definition so the chips never silently
fail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Colormap metadata
# ---------------------------------------------------------------------------

# Each entry: (identifier, label, family).
COLORMAP_CATALOG: Tuple[Tuple[str, str, str], ...] = (
    # Perceptual sequential
    ("viridis", "Viridis", "Perceptual"),
    ("plasma", "Plasma", "Perceptual"),
    ("inferno", "Inferno", "Perceptual"),
    ("magma", "Magma", "Perceptual"),
    ("cividis", "Cividis", "Perceptual"),
    ("turbo", "Turbo", "Perceptual"),
    # Sequential
    ("Greys", "Greys", "Sequential"),
    ("Blues", "Blues", "Sequential"),
    ("Greens", "Greens", "Sequential"),
    ("Reds", "Reds", "Sequential"),
    ("Oranges", "Oranges", "Sequential"),
    ("Purples", "Purples", "Sequential"),
    ("YlOrRd", "Yellow-Orange-Red", "Sequential"),
    ("YlGnBu", "Yellow-Green-Blue", "Sequential"),
    ("BuPu", "Blue-Purple", "Sequential"),
    ("GnBu", "Green-Blue", "Sequential"),
    ("hot", "Hot", "Sequential"),
    ("cool", "Cool", "Sequential"),
    ("copper", "Copper", "Sequential"),
    ("bone", "Bone", "Sequential"),
    # Diverging
    ("coolwarm", "Cool-Warm", "Diverging"),
    ("bwr", "Blue-White-Red", "Diverging"),
    ("seismic", "Seismic", "Diverging"),
    ("RdBu", "Red-Blue", "Diverging"),
    ("RdYlBu", "Red-Yellow-Blue", "Diverging"),
    ("RdYlGn", "Red-Yellow-Green", "Diverging"),
    ("Spectral", "Spectral", "Diverging"),
    ("PiYG", "Pink-Yellow-Green", "Diverging"),
    ("PRGn", "Purple-Green", "Diverging"),
    ("BrBG", "Brown-Blue-Green", "Diverging"),
    # Cyclic
    ("hsv", "HSV", "Cyclic"),
    ("twilight", "Twilight", "Cyclic"),
    ("twilight_shifted", "Twilight Shifted", "Cyclic"),
)


# Subset surfaced as instant chips in the floating toolbar.
QUICK_COLORMAPS: Tuple[str, ...] = (
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "turbo",
    "cividis",
    "coolwarm",
    "RdYlBu",
)


# Per-colormap icon so the gizmo chip can hint at the family even before the
# user reads the tooltip. Keep them inside the standard Blender icon set.
COLORMAP_ICONS = {
    "viridis": "COLORSET_03_VEC",
    "plasma": "COLORSET_05_VEC",
    "inferno": "COLORSET_01_VEC",
    "magma": "COLORSET_06_VEC",
    "turbo": "COLORSET_04_VEC",
    "cividis": "COLORSET_08_VEC",
    "coolwarm": "COLORSET_02_VEC",
    "RdYlBu": "COLORSET_07_VEC",
}


# ---------------------------------------------------------------------------
# Numpy-only fallback control points
# ---------------------------------------------------------------------------

# Control stops for each colormap, sampled (uniformly) so np.interp can
# rebuild a continuous gradient. The exact colors come from publicly known
# matplotlib LUTs (Viridis & friends) and follow the same orientation as
# matplotlib (low value -> first stop).
_FALLBACK_STOPS: dict = {
    "viridis": [
        (0.267, 0.005, 0.329),
        (0.229, 0.322, 0.546),
        (0.127, 0.566, 0.551),
        (0.369, 0.789, 0.383),
        (0.993, 0.906, 0.144),
    ],
    "plasma": [
        (0.050, 0.030, 0.528),
        (0.435, 0.000, 0.659),
        (0.798, 0.281, 0.469),
        (0.974, 0.581, 0.246),
        (0.940, 0.975, 0.131),
    ],
    "inferno": [
        (0.001, 0.000, 0.014),
        (0.258, 0.039, 0.406),
        (0.578, 0.148, 0.404),
        (0.865, 0.317, 0.228),
        (0.988, 1.000, 0.645),
    ],
    "magma": [
        (0.001, 0.000, 0.014),
        (0.232, 0.060, 0.439),
        (0.551, 0.162, 0.507),
        (0.867, 0.335, 0.413),
        (0.987, 0.991, 0.749),
    ],
    "cividis": [
        (0.000, 0.135, 0.305),
        (0.236, 0.298, 0.426),
        (0.490, 0.504, 0.502),
        (0.751, 0.706, 0.430),
        (1.000, 0.928, 0.372),
    ],
    "turbo": [
        (0.189, 0.072, 0.232),
        (0.192, 0.513, 0.984),
        (0.216, 0.941, 0.482),
        (0.940, 0.716, 0.200),
        (0.479, 0.016, 0.011),
    ],
    "Greys": [
        (1.000, 1.000, 1.000),
        (0.800, 0.800, 0.800),
        (0.550, 0.550, 0.550),
        (0.300, 0.300, 0.300),
        (0.000, 0.000, 0.000),
    ],
    "Blues": [
        (0.969, 0.984, 1.000),
        (0.776, 0.859, 0.937),
        (0.420, 0.682, 0.839),
        (0.129, 0.443, 0.710),
        (0.031, 0.188, 0.420),
    ],
    "Greens": [
        (0.969, 0.988, 0.961),
        (0.776, 0.914, 0.753),
        (0.455, 0.769, 0.463),
        (0.137, 0.545, 0.270),
        (0.000, 0.267, 0.106),
    ],
    "Reds": [
        (1.000, 0.961, 0.941),
        (0.988, 0.733, 0.631),
        (0.984, 0.416, 0.290),
        (0.796, 0.094, 0.114),
        (0.404, 0.000, 0.051),
    ],
    "Oranges": [
        (1.000, 0.961, 0.922),
        (0.992, 0.815, 0.635),
        (0.992, 0.553, 0.235),
        (0.851, 0.282, 0.004),
        (0.498, 0.153, 0.016),
    ],
    "Purples": [
        (0.988, 0.984, 0.992),
        (0.855, 0.855, 0.922),
        (0.620, 0.604, 0.784),
        (0.416, 0.318, 0.640),
        (0.247, 0.000, 0.490),
    ],
    "YlOrRd": [
        (1.000, 1.000, 0.800),
        (0.996, 0.851, 0.463),
        (0.992, 0.553, 0.235),
        (0.937, 0.231, 0.173),
        (0.502, 0.000, 0.149),
    ],
    "YlGnBu": [
        (1.000, 1.000, 0.851),
        (0.780, 0.914, 0.706),
        (0.255, 0.714, 0.769),
        (0.141, 0.408, 0.674),
        (0.031, 0.114, 0.345),
    ],
    "BuPu": [
        (0.969, 0.988, 0.992),
        (0.749, 0.827, 0.902),
        (0.549, 0.588, 0.776),
        (0.549, 0.318, 0.639),
        (0.302, 0.000, 0.294),
    ],
    "GnBu": [
        (0.969, 0.988, 0.941),
        (0.733, 0.894, 0.808),
        (0.400, 0.761, 0.643),
        (0.243, 0.502, 0.776),
        (0.031, 0.251, 0.506),
    ],
    "hot": [
        (0.042, 0.000, 0.000),
        (0.500, 0.000, 0.000),
        (1.000, 0.500, 0.000),
        (1.000, 1.000, 0.333),
        (1.000, 1.000, 1.000),
    ],
    "cool": [
        (0.000, 1.000, 1.000),
        (0.250, 0.750, 1.000),
        (0.500, 0.500, 1.000),
        (0.750, 0.250, 1.000),
        (1.000, 0.000, 1.000),
    ],
    "copper": [
        (0.000, 0.000, 0.000),
        (0.300, 0.187, 0.119),
        (0.600, 0.375, 0.239),
        (0.900, 0.563, 0.358),
        (1.000, 0.625, 0.397),
    ],
    "bone": [
        (0.000, 0.000, 0.000),
        (0.220, 0.220, 0.305),
        (0.440, 0.500, 0.550),
        (0.700, 0.766, 0.766),
        (1.000, 1.000, 1.000),
    ],
    "coolwarm": [
        (0.230, 0.299, 0.754),
        (0.545, 0.622, 0.919),
        (0.866, 0.866, 0.866),
        (0.957, 0.620, 0.479),
        (0.706, 0.016, 0.150),
    ],
    "bwr": [
        (0.000, 0.000, 1.000),
        (0.500, 0.500, 1.000),
        (1.000, 1.000, 1.000),
        (1.000, 0.500, 0.500),
        (1.000, 0.000, 0.000),
    ],
    "seismic": [
        (0.000, 0.000, 0.300),
        (0.000, 0.000, 1.000),
        (1.000, 1.000, 1.000),
        (1.000, 0.000, 0.000),
        (0.500, 0.000, 0.000),
    ],
    "RdBu": [
        (0.404, 0.000, 0.122),
        (0.937, 0.541, 0.384),
        (0.969, 0.969, 0.969),
        (0.420, 0.682, 0.839),
        (0.020, 0.188, 0.380),
    ],
    "RdYlBu": [
        (0.647, 0.000, 0.149),
        (0.992, 0.682, 0.380),
        (1.000, 1.000, 0.749),
        (0.510, 0.789, 0.753),
        (0.192, 0.211, 0.584),
    ],
    "RdYlGn": [
        (0.647, 0.000, 0.149),
        (0.992, 0.682, 0.380),
        (1.000, 1.000, 0.749),
        (0.569, 0.812, 0.376),
        (0.000, 0.408, 0.216),
    ],
    "Spectral": [
        (0.620, 0.004, 0.259),
        (0.965, 0.427, 0.263),
        (1.000, 1.000, 0.749),
        (0.400, 0.761, 0.647),
        (0.369, 0.310, 0.635),
    ],
    "PiYG": [
        (0.557, 0.004, 0.322),
        (0.945, 0.541, 0.769),
        (0.969, 0.969, 0.969),
        (0.722, 0.882, 0.525),
        (0.153, 0.392, 0.098),
    ],
    "PRGn": [
        (0.250, 0.000, 0.294),
        (0.600, 0.439, 0.671),
        (0.969, 0.969, 0.969),
        (0.498, 0.737, 0.553),
        (0.000, 0.267, 0.106),
    ],
    "BrBG": [
        (0.329, 0.188, 0.020),
        (0.749, 0.506, 0.176),
        (0.961, 0.961, 0.961),
        (0.353, 0.706, 0.675),
        (0.000, 0.235, 0.188),
    ],
    "hsv": [
        (1.000, 0.000, 0.000),
        (1.000, 1.000, 0.000),
        (0.000, 1.000, 0.000),
        (0.000, 1.000, 1.000),
        (0.000, 0.000, 1.000),
        (1.000, 0.000, 1.000),
        (1.000, 0.000, 0.000),
    ],
    "twilight": [
        (0.886, 0.851, 0.886),
        (0.486, 0.475, 0.694),
        (0.110, 0.114, 0.353),
        (0.482, 0.290, 0.349),
        (0.886, 0.851, 0.886),
    ],
    "twilight_shifted": [
        (0.110, 0.114, 0.353),
        (0.482, 0.290, 0.349),
        (0.886, 0.851, 0.886),
        (0.486, 0.475, 0.694),
        (0.110, 0.114, 0.353),
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def colormap_items_for_enum() -> List[Tuple[str, str, str]]:
    """Return ``(identifier, label, description)`` tuples for an EnumProperty."""
    items = []
    for ident, label, family in COLORMAP_CATALOG:
        description = f"{family} colormap"
        items.append((ident, label, description))
    return items


def colormap_exists(name: str) -> bool:
    """True when ``name`` has at least a fallback definition."""
    return name in _FALLBACK_STOPS


def sample_colormap(name: str, samples: int = 8, reverse: bool = False) -> np.ndarray:
    """Sample ``samples`` colors from a colormap as an ``(N, 4)`` RGBA array."""
    samples = max(2, int(samples))
    norm = np.linspace(0.0, 1.0, samples, dtype=float)
    if reverse:
        norm = norm[::-1]
    return _resolve_rgba(name, norm)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# EnumProperty-ready description of every normalization mode.
NORM_MODES: Tuple[Tuple[str, str, str], ...] = (
    (
        "LINEAR",
        "Linear",
        "Map the raw value range onto the colormap. Fast and faithful, but "
        "heavy-tailed data (betweenness, degree, PageRank) collapses into the "
        "bottom of the ramp",
    ),
    (
        "LOG",
        "Log10",
        "Normalize log10(value). Spreads out multiplicative data. Zero and "
        "negative samples are floored to a small epsilon and share the lowest "
        "color",
    ),
    (
        "RANK",
        "Rank",
        "Replace each value by its rank / (n-1) (histogram equalization). The "
        "robust choice for skewed data: colors are spread uniformly across the "
        "ramp no matter how extreme the tail",
    ),
    (
        "QUANTILE",
        "Quantile",
        "Rank computed over the distinct values instead of the samples, so a "
        "large block of identical values does not dominate the ramp",
    ),
)

NORM_MODE_IDS: Tuple[str, ...] = tuple(mode[0] for mode in NORM_MODES)

DEFAULT_NORM_MODE = "LINEAR"

# Transforms that depend on the whole distribution instead of a single sample.
# No shader math can express them, so the normalized result is baked into a
# helper float attribute the material reads instead of the source attribute.
BAKED_NORM_MODES: Tuple[str, ...] = ("RANK", "QUANTILE")

# Suffix of the baked helper attribute written for BAKED_NORM_MODES.
NORM_ATTRIBUTE_SUFFIX = "_scignorm"

# log10 is undefined at 0 and below, so LOG floors every sample at ``eps`` and
# zeros/negatives share the lowest color. How far below the data ``eps`` sits
# matters: a fixed "three decades below the smallest positive sample" floor
# handed 54% of the ramp to a dozen zeros. Solving gap / (gap + span) = f for
# span = log10(hi) - log10(min_positive) instead gives
# gap = f * span / (1 - f) decades, bounding the non-positive bucket at
# ``_LOG_ZERO_RAMP_FRACTION`` of the ramp whatever the data span.
_LOG_EPS_DECADES = 1e-3
_LOG_EPS_FLOOR = 1e-30
_LOG_ZERO_RAMP_FRACTION = 0.1
_LOG_MIN_GAP_DECADES = 0.05


@dataclass
class NormalizationPlan:
    """Bounds and parameters the shader needs to reproduce :func:`normalize_values`.

    ``lo``/``hi`` are the post-transform ``Map Range`` bounds: raw units for
    ``LINEAR``, log10 units for ``LOG``, ``(0.0, 1.0)`` for the baked modes.
    ``data_lo``/``data_hi`` keep the bounds in raw data units for the UI.
    """

    mode: str = DEFAULT_NORM_MODE
    lo: float = 0.0
    hi: float = 1.0
    eps: float = _LOG_EPS_FLOOR
    gamma: float = 1.0
    baked: bool = False
    data_lo: float = 0.0
    data_hi: float = 1.0


def _as_float_array(values) -> np.ndarray:
    """Coerce anything iterable (including generators) into a 1D float array."""
    if isinstance(values, np.ndarray):
        return values.astype(float, copy=False)
    return np.asarray(list(values), dtype=float)


def _effective_bounds(
    arr: np.ndarray,
    finite: np.ndarray,
    vmin: Optional[float],
    vmax: Optional[float],
    clip_low_pct: float,
    clip_high_pct: float,
) -> Tuple[float, float]:
    """Data bounds after optional percentile clipping.

    The percentile pass runs only when a slider actually moved, so the default
    path stays bit-for-bit identical to plain ``nanmin``/``nanmax`` rather than
    relying on ``nanpercentile(x, 0) == nanmin(x)``.
    """
    low = float(clip_low_pct)
    high = float(clip_high_pct)

    if low > 0.0 or high < 100.0:
        low = min(max(low, 0.0), 100.0)
        high = min(max(high, 0.0), 100.0)
        if high < low:
            low, high = high, low
        sample = arr[finite]
        auto_lo = float(np.nanpercentile(sample, low))
        auto_hi = float(np.nanpercentile(sample, high))
    else:
        auto_lo = float(np.nanmin(arr[finite]))
        auto_hi = float(np.nanmax(arr[finite]))

    lo = auto_lo if vmin is None else float(vmin)
    hi = auto_hi if vmax is None else float(vmax)
    if hi == lo:
        hi = lo + 1e-9
    return lo, hi


def _log_epsilon(values: np.ndarray, upper: float) -> float:
    """Floor applied before log10 (see the LOG notes above)."""
    positive = values[values > 0.0]
    if positive.size == 0:
        return _LOG_EPS_FLOOR

    pos_min = float(positive.min())
    if not bool((values <= 0.0).any()):
        # Nothing to floor: eps only has to stay under the data, never binding.
        return max(pos_min * _LOG_EPS_DECADES, _LOG_EPS_FLOOR)

    hi = max(float(upper), pos_min)
    span = math.log10(hi) - math.log10(pos_min)
    gap = span * _LOG_ZERO_RAMP_FRACTION / (1.0 - _LOG_ZERO_RAMP_FRACTION)
    gap = max(gap, _LOG_MIN_GAP_DECADES)
    return max(pos_min * (10.0 ** -gap), _LOG_EPS_FLOOR)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Mid-rank of every sample, ties sharing their average rank.

    Ordinal ranking (``argsort(argsort(x))``) gives two identical values two
    different colors, which the shader cannot reproduce.
    """
    _uniq, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    starts = cumulative - counts
    mid = (starts + cumulative - 1) * 0.5
    return mid[inverse.ravel()]


def normalize_values(
    values: Iterable[float],
    mode: str = DEFAULT_NORM_MODE,
    vmin: float | None = None,
    vmax: float | None = None,
    gamma: float = 1.0,
    clip_low_pct: float = 0.0,
    clip_high_pct: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, NormalizationPlan]:
    """Normalize ``values`` into ``[0, 1]``.

    Returns ``(norm, finite_mask, plan)``, non-finite samples coming back as
    ``NaN`` and ``False``. The shader mirrors this order: percentile clip,
    ``vmin``/``vmax`` override, transform, gamma; colormap reversal happens at
    sampling time, not here.
    """
    arr = _as_float_array(values)
    mode = (mode or DEFAULT_NORM_MODE).upper()
    if mode not in NORM_MODE_IDS:
        mode = DEFAULT_NORM_MODE
    try:
        gamma = float(gamma)
    except (TypeError, ValueError):
        gamma = 1.0
    if not math.isfinite(gamma) or gamma <= 0.0:
        gamma = 1.0

    baked = mode in BAKED_NORM_MODES

    if arr.size == 0:
        return (
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=bool),
            NormalizationPlan(mode=mode, gamma=gamma, baked=baked),
        )

    finite = np.isfinite(arr)
    if not finite.any():
        return (
            np.full(arr.shape, np.nan, dtype=float),
            finite,
            NormalizationPlan(mode=mode, gamma=gamma, baked=baked),
        )

    data_lo, data_hi = _effective_bounds(
        arr, finite, vmin, vmax, clip_low_pct, clip_high_pct
    )

    norm = np.full(arr.shape, np.nan, dtype=float)
    work = np.clip(arr[finite], data_lo, data_hi)
    eps = _LOG_EPS_FLOOR

    if mode == "LOG":
        eps = _log_epsilon(work, data_hi)
        plan_lo = math.log10(max(data_lo, eps))
        plan_hi = math.log10(max(data_hi, eps))
        if plan_hi == plan_lo:
            plan_hi = plan_lo + 1e-9
        norm[finite] = (np.log10(np.maximum(work, eps)) - plan_lo) / (plan_hi - plan_lo)
    elif mode == "RANK":
        plan_lo, plan_hi = 0.0, 1.0
        denom = float(work.size - 1) if work.size > 1 else 1.0
        norm[finite] = _average_ranks(work) / denom
    elif mode == "QUANTILE":
        plan_lo, plan_hi = 0.0, 1.0
        uniq, inverse = np.unique(work, return_inverse=True)
        denom = float(uniq.size - 1) if uniq.size > 1 else 1.0
        norm[finite] = inverse.ravel().astype(float) / denom
    else:  # LINEAR
        plan_lo, plan_hi = data_lo, data_hi
        norm[finite] = (work - plan_lo) / (plan_hi - plan_lo)

    norm[finite] = np.clip(norm[finite], 0.0, 1.0)

    if gamma != 1.0:
        norm[finite] = np.power(norm[finite], 1.0 / gamma)

    plan = NormalizationPlan(
        mode=mode,
        lo=float(plan_lo),
        hi=float(plan_hi),
        eps=float(eps),
        gamma=float(gamma),
        baked=baked,
        data_lo=float(data_lo),
        data_hi=float(data_hi),
    )
    return norm, finite, plan


def norm_to_rgba(
    norm: Iterable[float],
    cmap_name: str = "viridis",
    reverse: bool = False,
    finite: Optional[np.ndarray] = None,
    nan_color: Sequence[float] = (0.30, 0.30, 0.30, 1.0),
) -> np.ndarray:
    """Map already-normalized ``[0, 1]`` values onto an ``(N, 4)`` RGBA array."""
    arr = _as_float_array(norm)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=float)

    if finite is None:
        finite = np.isfinite(arr)
    if not finite.any():
        result = np.zeros((arr.size, 4), dtype=float)
        result[:] = nan_color
        return result

    work = np.where(finite, arr, 0.0)
    if reverse:
        work = 1.0 - work

    rgba = _resolve_rgba(cmap_name, work)
    rgba[~finite] = nan_color
    return rgba


def values_to_rgba(
    values: Iterable[float],
    cmap_name: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    reverse: bool = False,
    nan_color: Sequence[float] = (0.30, 0.30, 0.30, 1.0),
    norm_mode: str = DEFAULT_NORM_MODE,
    gamma: float = 1.0,
    clip_low_pct: float = 0.0,
    clip_high_pct: float = 100.0,
) -> np.ndarray:
    """Map a sequence of floats to an ``(N, 4)`` RGBA array.

    Non-finite values are mapped to ``nan_color`` instead of raising, and an
    empty ``values`` gives an empty ``(0, 4)`` array. The defaults (``LINEAR``,
    ``gamma=1``, no clip) reproduce the historical linear normalization.
    """
    arr = _as_float_array(values)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=float)

    finite = np.isfinite(arr)
    if not finite.any():
        result = np.zeros((arr.size, 4), dtype=float)
        result[:] = nan_color
        return result

    norm, finite, _plan = normalize_values(
        arr,
        mode=norm_mode,
        vmin=vmin,
        vmax=vmax,
        gamma=gamma,
        clip_low_pct=clip_low_pct,
        clip_high_pct=clip_high_pct,
    )
    return norm_to_rgba(norm, cmap_name, reverse, finite, nan_color)


def normalize_range(
    values: Iterable[float],
    vmin: float | None = None,
    vmax: float | None = None,
    clip_low_pct: float = 0.0,
    clip_high_pct: float = 100.0,
) -> Tuple[float, float]:
    """Compute the ``(vmin, vmax)`` pair used by ``values_to_rgba``.

    Always in raw data units, whatever the normalization mode. Returns
    ``(0.0, 1.0)`` when there is no finite sample.
    """
    arr = _as_float_array(values)
    if arr.size == 0:
        return (0.0, 1.0)

    finite = np.isfinite(arr)
    if not finite.any():
        return (0.0, 1.0)

    return _effective_bounds(arr, finite, vmin, vmax, clip_low_pct, clip_high_pct)


def norm_mode_items_for_enum() -> List[Tuple[str, str, str]]:
    """Return ``(identifier, label, description)`` tuples for an EnumProperty."""
    return [(ident, label, description) for ident, label, description in NORM_MODES]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _resolve_rgba(cmap_name: str, norm: np.ndarray) -> np.ndarray:
    """Try matplotlib first, otherwise interpolate the fallback stops."""
    try:
        from matplotlib import colormaps as mpl_colormaps

        cmap_name_resolved = cmap_name
        if cmap_name_resolved not in mpl_colormaps:
            cmap_name_resolved = "viridis"
        cmap = mpl_colormaps.get_cmap(cmap_name_resolved)
        rgba = np.asarray(cmap(np.clip(norm, 0.0, 1.0)), dtype=float)
        if rgba.ndim == 1:
            rgba = rgba.reshape(1, -1)
        return rgba
    except Exception:  # pylint: disable=broad-except
        # Anything could go wrong reaching matplotlib (missing install,
        # incompatible version, deprecated API). Fall back to numpy stops.
        return _fallback_rgba(cmap_name, norm)


def _fallback_rgba(cmap_name: str, norm: np.ndarray) -> np.ndarray:
    pts = _FALLBACK_STOPS.get(cmap_name)
    if pts is None:
        pts = _FALLBACK_STOPS["viridis"]

    stops = np.linspace(0.0, 1.0, len(pts))
    arr = np.clip(np.asarray(norm, dtype=float), 0.0, 1.0)

    r = np.interp(arr, stops, [p[0] for p in pts])
    g = np.interp(arr, stops, [p[1] for p in pts])
    b = np.interp(arr, stops, [p[2] for p in pts])
    a = np.ones_like(arr)
    return np.stack([r, g, b, a], axis=-1)
