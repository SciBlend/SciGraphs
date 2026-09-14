"""Per-frame animation for graph objects: positions, and the things Blender
cannot keyframe.
"""

from __future__ import annotations

import bpy
import numpy as np
from bpy.app.handlers import persistent

_KEY_PREFIX = "scigraphs_anim_"



def positions(obj, keys, interpolation='BEZIER'):
    """Animate node positions from `[(frame, (n, 3) array), ...]`."""
    if not keys:
        return []
    mesh = obj.data
    count = len(mesh.vertices)

    stages = []
    for frame, pos in keys:
        arr = np.asarray(pos, dtype=np.float64)
        if arr.shape != (count, 3):
            raise ValueError(
                f"{obj.name}: key at frame {frame} has shape {arr.shape}, "
                f"expected ({count}, 3)")
        stages.append((int(frame), arr))
    stages.sort(key=lambda kv: kv[0])

    mesh.vertices.foreach_set("co", stages[-1][1].ravel())
    mesh.update()

    if mesh.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)

    made = []
    for index, (frame, arr) in enumerate(stages):
        key = obj.shape_key_add(name=f"sg_stage_{index:03d}", from_mix=False)
        for i, co in enumerate(arr):
            key.data[i].co = co
        made.append((frame, key))

    for index, (frame, key) in enumerate(made):
        before = made[index - 1][0] if index else None
        after = made[index + 1][0] if index + 1 < len(made) else None

        key.value = 1.0
        key.keyframe_insert("value", frame=frame)
        if before is not None:
            key.value = 0.0
            key.keyframe_insert("value", frame=before)
        else:
            key.value = 1.0
            key.keyframe_insert("value", frame=max(1, min(frame, 1)))
        if after is not None:
            key.value = 0.0
            key.keyframe_insert("value", frame=after)
        key.value = 1.0 if after is None else 0.0

    action = getattr(getattr(mesh.shape_keys, "animation_data", None),
                     "action", None)
    if action is not None:
        for fcurve in _action_fcurves(action):
            for point in fcurve.keyframe_points:
                point.interpolation = interpolation
    return made


def _action_fcurves(action):
    """F-curves of an action, across Blender 4.4+ slotted actions too."""
    curves = list(getattr(action, "fcurves", ()))
    if curves:
        return curves
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for bag in getattr(strip, "channelbags", ()):
                curves.extend(bag.fcurves)
    return curves



def attribute(obj, name, keys, domain='POINT', data_type='FLOAT'):
    """Animate a mesh attribute from `[(frame, values), ...]`."""
    mesh = obj.data
    attr = mesh.attributes.get(name)
    if attr is None:
        attr = mesh.attributes.new(name, data_type, domain)
    size = len(attr.data)

    frames, flat = [], []
    for frame, values in sorted(keys, key=lambda kv: kv[0]):
        arr = np.asarray(values, dtype=np.float32).ravel()
        if len(arr) != size:
            raise ValueError(
                f"{obj.name}.{name}: key at frame {frame} has {len(arr)} "
                f"values for {size} {domain.lower()} elements")
        frames.append(int(frame))
        flat.extend(float(v) for v in arr)

    obj[f"{_KEY_PREFIX}{name}"] = {
        "frames": frames,
        "values": flat,
        "domain": domain,
        "size": size,
    }
    apply_frame(obj, bpy.context.scene.frame_current)
    return frames


def _animated_attributes(obj):
    for key in obj.keys():
        if key.startswith(_KEY_PREFIX):
            yield key[len(_KEY_PREFIX):], obj[key]


def apply_frame(obj, frame):
    """Write every animated attribute of `obj` at `frame`. Linear between keys."""
    mesh = getattr(obj, "data", None)
    if mesh is None or not hasattr(mesh, "attributes"):
        return 0

    written = 0
    for name, record in _animated_attributes(obj):
        frames = list(record["frames"])
        size = int(record["size"])
        flat = np.asarray(list(record["values"]), dtype=np.float32)
        if not frames or size <= 0 or len(flat) != len(frames) * size:
            continue
        table = flat.reshape(len(frames), size)

        if frame <= frames[0]:
            values = table[0]
        elif frame >= frames[-1]:
            values = table[-1]
        else:
            upper = int(np.searchsorted(frames, frame, side='left'))
            lower = upper - 1
            span = frames[upper] - frames[lower]
            t = 0.0 if span == 0 else (frame - frames[lower]) / span
            values = table[lower] * (1.0 - t) + table[upper] * t

        attr = mesh.attributes.get(name)
        if attr is None or len(attr.data) != size:
            continue
        attr.data.foreach_set("value", np.ascontiguousarray(values,
                                                            dtype=np.float32))
        written += 1

    if written:
        mesh.update()
    return written


@persistent
def _sg_anim_frame_change(scene, depsgraph=None):
    """Write animated attributes for every object that has them."""
    for obj in scene.objects:
        if obj.type != 'MESH':
            continue
        try:
            apply_frame(obj, scene.frame_current)
        except Exception:  # noqa: BLE001 - never let a handler break playback
            continue


def ensure_handler():
    """Install the frame-change handler if it is not already there."""
    handlers = bpy.app.handlers.frame_change_post
    if _sg_anim_frame_change not in handlers:
        handlers.append(_sg_anim_frame_change)
    render_handlers = bpy.app.handlers.render_pre
    if _sg_anim_frame_change not in render_handlers:
        render_handlers.append(_sg_anim_frame_change)
    return True


def remove_handler():
    for handlers in (bpy.app.handlers.frame_change_post,
                     bpy.app.handlers.render_pre):
        if _sg_anim_frame_change in handlers:
            handlers.remove(_sg_anim_frame_change)
    return True


def clear(obj):
    """Drop every stored animation from `obj`: attributes and shape keys."""
    for key in [k for k in obj.keys() if k.startswith(_KEY_PREFIX)]:
        del obj[key]
    if obj.data.shape_keys is not None:
        for key in [k for k in obj.data.shape_keys.key_blocks
                    if k.name.startswith("sg_stage_") or k.name == "Basis"]:
            obj.shape_key_remove(key)
    return True
