# The three mappings onto [0, 1] differ on purpose and must not be unified.
# size_normalized sends a constant to zeros over the caller's range, so the
# colormap keys on the numbers the legend prints. channel_normalized sends it
# to ones over its own range; at zero, a connected graph's one component size
# emptied the viewport. scale_to_unit adds LOG and RANK for heavy tails.

import numpy as np


def value_channel(values, num_vertices=None):
    """``(values32, vmin, vmax)``, or ``(None, 0.0, 1.0)`` on no attribute, bad
    length, or nothing finite. The bounds skip NaN, which poisons both ends."""
    if values is None:
        return None, 0.0, 1.0

    arr = np.asarray(values)
    if num_vertices is not None and arr.size != num_vertices:
        return None, 0.0, 1.0

    finite = np.isfinite(arr)
    if not finite.any():
        return None, 0.0, 1.0

    return (arr.astype(np.float32),
            float(arr[finite].min()), float(arr[finite].max()))


def size_normalized(values, vmin, vmax, num_vertices):
    """Affine map onto [0, 1] over the caller's ``vmin``/``vmax``. A constant
    becomes zeros, and nothing is clipped. Non-finite samples come back as 0.0.

    ``_ch_attr`` is the one producer that hands back its own native range, so
    its values skip ``channel_normalized`` in ``channel_values`` and reach
    ``slot_mask`` raw. NaN is false against both ends of a clause, so an
    unmeasured node used to vanish from an ordinary slot and *appear* in an
    inverted one -- alone among the channels, the rest of which sort those
    nodes to the bottom. The host's ``normalized_values`` maps the same array
    the same way and multiplies it into a node radius."""
    if values is None:
        return None
    if vmax > vmin:
        norm = (values - vmin) / (vmax - vmin)
    else:
        norm = np.zeros(num_vertices, dtype=np.float32)
    return np.where(np.isfinite(values), norm, 0.0).astype(np.float32)


def channel_normalized(values):
    """Onto [0, 1] over the channel's own range; a constant becomes ones.
    Non-finite entries land at the bottom, since NaN compares false anywhere."""
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.ones(values.size, dtype=np.float32)
    lo = float(values[finite].min())
    hi = float(values[finite].max())
    if hi <= lo:
        return np.ones(values.size, dtype=np.float32)
    out = (np.where(finite, values, lo) - lo) / (hi - lo)
    return out.astype(np.float32)


def scale_to_unit(values, mode='LINEAR'):
    """Magnitudes onto [0, 1]. ``LINEAR`` is min-max, which squashes a heavy
    tail to minimum width; ``LOG`` is min-max of the log, shifted positive;
    ``RANK`` is sorted position, ties sharing a rank. Non-finite map to 0."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    out = np.zeros(values.size, dtype=np.float32)
    if not finite.any():
        return out

    if mode == 'RANK':
        # One rank per distinct value, so equal weights get equal widths.
        vals = values[finite]
        uniq, inv = np.unique(vals, return_inverse=True)
        if uniq.size > 1:
            out[finite] = (inv / (uniq.size - 1)).astype(np.float32)
        return out

    work = values.copy()
    if mode == 'LOG':
        lo = float(work[finite].min())
        hi = float(work[finite].max())
        if lo > 0.0:
            # Plain log: log1p barely separates weights under 1.
            work = np.log(work)
        else:
            # Shift positive; the thousandth-of-span offset fixes three decades.
            span = max(hi - lo, 1e-12)
            work = np.log(np.maximum(work - lo, 0.0) + span * 1e-3)
        finite = np.isfinite(work) & finite

    if not finite.any():
        return out
    vmin = float(work[finite].min())
    vmax = float(work[finite].max())
    if vmax <= vmin:
        return out
    out[finite] = ((work[finite] - vmin) / (vmax - vmin)).astype(np.float32)
    return out
