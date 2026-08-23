# Color, so a default render is not one flat blue. Four colormaps as
# interpolated 11-point tables rather than 30 MB of matplotlib and a font cache
# to turn a float into a hue; within 1.5/255 per channel of its viridis.

import numpy as np

# viridis and plasma: Nathaniel Smith and Stefan van der Walt, CC0.
VIRIDIS = (
    (0.267, 0.005, 0.329), (0.283, 0.141, 0.458), (0.254, 0.265, 0.530),
    (0.208, 0.372, 0.553), (0.164, 0.471, 0.558), (0.128, 0.567, 0.551),
    (0.135, 0.659, 0.518), (0.267, 0.749, 0.441), (0.478, 0.821, 0.318),
    (0.741, 0.873, 0.150), (0.993, 0.906, 0.144),
)
PLASMA = (
    (0.050, 0.030, 0.528), (0.254, 0.014, 0.615), (0.417, 0.000, 0.658),
    (0.562, 0.052, 0.641), (0.692, 0.166, 0.564), (0.798, 0.280, 0.470),
    (0.881, 0.392, 0.383), (0.949, 0.518, 0.296), (0.988, 0.653, 0.212),
    (0.988, 0.809, 0.145), (0.940, 0.975, 0.131),
)
COOLWARM = (
    (0.230, 0.299, 0.754), (0.406, 0.537, 0.934), (0.603, 0.731, 0.999),
    (0.788, 0.846, 0.939), (0.865, 0.865, 0.865), (0.929, 0.797, 0.727),
    (0.945, 0.657, 0.538), (0.897, 0.480, 0.376), (0.796, 0.276, 0.238),
    (0.706, 0.016, 0.150), (0.706, 0.016, 0.150),
)
GRAY = tuple((v, v, v) for v in np.linspace(0.10, 0.95, 11))

COLORMAPS = {
    "viridis": VIRIDIS,
    "plasma": PLASMA,
    "coolwarm": COOLWARM,
    "gray": GRAY,
    "grey": GRAY,
}


def colormap_names():
    """The names ``apply`` accepts, without the spelling duplicate."""
    return ("viridis", "plasma", "coolwarm", "gray")


def apply(values, name="viridis", alpha=1.0):
    """(N,) in [0, 1] -> (N, 4) float32 RGBA. Non-finite entries land at the
    bottom: a NaN color uploads without complaint and rasterizes as a hole."""
    table = COLORMAPS.get(str(name).lower())
    if table is None:
        raise KeyError(f"unknown colormap {name!r}; have "
                       f"{list(colormap_names())}")
    stops = np.asarray(table, dtype=np.float32)
    t = np.asarray(values, dtype=np.float32).ravel()
    t = np.where(np.isfinite(t), t, 0.0)
    t = np.clip(t, 0.0, 1.0) * (stops.shape[0] - 1)
    lo = np.clip(t.astype(np.int32), 0, stops.shape[0] - 2)
    frac = (t - lo)[:, None]
    rgb = stops[lo] * (1.0 - frac) + stops[lo + 1] * frac
    out = np.empty((t.size, 4), dtype=np.float32)
    out[:, :3] = rgb
    out[:, 3] = float(alpha)
    return out


def solid(color, count):
    rgba = np.asarray(color, dtype=np.float32).ravel()
    if rgba.size == 3:
        rgba = np.concatenate([rgba, [1.0]]).astype(np.float32)
    if rgba.size != 4:
        raise ValueError(f"color must be 3 or 4 components, got {rgba.size}")
    return np.tile(rgba, (int(count), 1))
