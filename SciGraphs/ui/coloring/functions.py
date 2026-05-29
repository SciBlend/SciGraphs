"""Helper functions used by coloring operators.

Operators stay thin wrappers around these functions so the actual mesh and
material wiring lives in one easily testable place. Functions never read
from ``bpy.context`` directly: they receive the context (or the resolved
object) explicitly.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import bpy
import numpy as np

from ...core.coloring.attributes import (
    SCALAR_DATA_TYPES,
    attribute_value_range,
    color_domain_for,
    find_attribute,
    first_scalar_attribute,
    list_scalar_attributes,
    read_attribute_values,
    values_for_color_domain,
)
from ...core.coloring.colormaps import (
    QUICK_COLORMAPS,
    colormap_exists,
    sample_colormap,
    values_to_rgba,
)


# Material name reused so we don't pile up duplicates per object.
_AUTO_MATERIAL_NAME_TEMPLATE = "{obj}_SciGraphsColor"

# Maximum number of stops a ColorRamp accepts in Blender.
_COLOR_RAMP_MAX_STOPS = 32


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def coloring_props(context):
    """Return the scene-level ``SCIGRAPHS_PG_coloring`` (or ``None``)."""
    scene = getattr(context, "scene", None)
    if scene is None:
        return None
    return getattr(scene, "scigraphs_coloring", None)


def active_mesh_object(context):
    """Return the active object if it is a usable mesh, else ``None``."""
    obj = getattr(context, "active_object", None)
    if obj is None or obj.type != 'MESH':
        return None
    return obj


def has_scalar_attributes(obj) -> bool:
    """True when ``obj`` is a mesh with at least one scalar attribute."""
    if obj is None or obj.type != 'MESH':
        return False
    return bool(list_scalar_attributes(obj.data))


def resolve_attribute_name(props, obj) -> str:
    """Pick the best attribute to use given the user's choice + the mesh."""
    if obj is None or obj.type != 'MESH':
        return ""

    available = [name for name, *_ in list_scalar_attributes(obj.data)]
    if not available:
        return ""

    # Priority order: explicit string -> enum value -> first available.
    candidate = (props.attribute_name or "").strip()
    if candidate and candidate in available:
        return candidate

    enum_value = getattr(props, "attribute_enum", "") or ""
    if enum_value and enum_value != "__NONE__" and enum_value in available:
        return enum_value

    first = first_scalar_attribute(obj.data)
    if first is None:
        return ""
    return first[0]  # pylint: disable=unsubscriptable-object


def cycle_attribute_name(current: str, names: List[str], step: int = 1) -> str:
    """Return the next attribute name in ``names`` after ``current``."""
    if not names:
        return ""
    if current not in names:
        return names[0]
    index = (names.index(current) + step) % len(names)
    return names[index]


def quick_colormap_index(name: str) -> int:
    """Return the 0-based index of ``name`` inside ``QUICK_COLORMAPS`` (-1 if missing)."""
    try:
        return QUICK_COLORMAPS.index(name)
    except ValueError:
        return -1


# ---------------------------------------------------------------------------
# Range computation
# ---------------------------------------------------------------------------

def update_property_range(props, obj, attribute_name: str) -> Tuple[Optional[float], Optional[float]]:
    """Refresh ``vmin``/``vmax`` on the property group from a mesh attribute."""
    vmin, vmax = attribute_value_range(obj.data, attribute_name) if obj else (None, None)
    if vmin is None or vmax is None:
        return (None, None)

    props.vmin = float(vmin)
    props.vmax = float(vmax)
    props.last_vmin = float(vmin)
    props.last_vmax = float(vmax)
    return (vmin, vmax)


# ---------------------------------------------------------------------------
# Color writing
# ---------------------------------------------------------------------------

def _select_color_domain(props, source_domain: str, mesh) -> str:
    explicit = (props.color_domain or "AUTO").upper()
    if explicit == 'AUTO':
        return color_domain_for(source_domain, mesh)
    if explicit == 'CORNER' and len(getattr(mesh, "loops", [])) == 0:
        return 'POINT'
    return explicit


def _color_attribute_name(props, attribute_name: str) -> str:
    explicit = (props.color_attribute_name or "").strip()
    return explicit or f"{attribute_name}_color"


def write_color_attribute(
    obj,
    attribute_name: str,
    color_name: str,
    color_domain: str,
    rgba: np.ndarray,
) -> Tuple[bool, str]:
    """Create or replace the color attribute and fill it with ``rgba``."""
    mesh = obj.data
    if rgba.size == 0:
        return False, "Attribute returned no values to color"

    if color_name in mesh.color_attributes:
        try:
            mesh.color_attributes.remove(mesh.color_attributes[color_name])
        except RuntimeError:
            return False, f"Could not replace color attribute '{color_name}'"

    try:
        color = mesh.color_attributes.new(
            name=color_name,
            type='FLOAT_COLOR',
            domain=color_domain,
        )
    except RuntimeError:
        try:
            color = mesh.color_attributes.new(
                name=color_name,
                type='BYTE_COLOR',
                domain=color_domain,
            )
        except RuntimeError as exc:
            return False, f"Failed to create color attribute: {exc}"

    count = min(len(color.data), rgba.shape[0])
    for i in range(count):
        c = rgba[i]
        color.data[i].color = (
            float(c[0]), float(c[1]), float(c[2]), float(c[3]),
        )

    try:
        mesh.color_attributes.active_color = color
    except (AttributeError, RuntimeError):
        pass

    obj["scigraphs_last_color_attribute"] = color_name
    obj["scigraphs_last_source_attribute"] = attribute_name
    return True, color_name


# ---------------------------------------------------------------------------
# Material auto-setup
# ---------------------------------------------------------------------------

def _add_colorramp_stops(color_ramp, samples) -> None:
    """Populate a ColorRamp element list from an ``(N, 4)`` RGBA numpy array."""
    elements = color_ramp.elements
    while len(elements) > 1:
        elements.remove(elements[-1])

    count = min(int(samples.shape[0]), _COLOR_RAMP_MAX_STOPS)
    if count < 2:
        count = 2

    elements[0].position = 0.0
    elements[0].color = tuple(float(v) for v in samples[0])

    for i in range(1, count):
        position = i / (count - 1)
        if i < len(elements):
            element = elements[i]
            element.position = position
        else:
            element = elements.new(position)
        element.color = tuple(float(v) for v in samples[i])


def _mesh_has_attribute(obj, name: str) -> bool:
    if obj is None or not name or obj.type != 'MESH':
        return False
    return name in obj.data.attributes


def _gate_attribute_available(obj, name: str) -> bool:
    """True when ``name`` will be readable by the shader on this object.

    ``scigraphs_is_node`` lives only on the geometry produced by the
    SciGraphs_Viz Geometry Nodes tree (it is injected by
    ``_stamp_node_marker_after_realize``) so we look at the modifier instead
    of the source mesh.
    """
    if name == "scigraphs_is_node":
        return obj is not None and hasattr(obj, "modifiers") and (
            obj.modifiers.get("SciGraphs_Viz") is not None
        )
    return _mesh_has_attribute(obj, name)


def build_color_material(
    obj,
    source_attribute_name: str,
    colormap: str,
    vmin: float,
    vmax: float,
    reverse: bool,
    fallback_color_layer: Optional[str] = None,
    nodes_only: bool = False,
    edge_color: Optional[Sequence[float]] = None,
    intersection_attribute: str = "is_intersection",
) -> Optional[bpy.types.Material]:
    """Build (or refresh) a shader graph that maps a float attribute to color.

    The shader reads ``source_attribute_name`` via ``ShaderNodeAttribute``,
    normalises it via ``Map Range`` against the given ``vmin``/``vmax`` and
    feeds the result into a ``ColorRamp`` whose stops match the requested
    colormap. The output drives the Base Color of a Principled BSDF.

    Optional behaviour:

    * ``fallback_color_layer`` – name of an explicit FLOAT_COLOR attribute
      sampled in parallel and mixed in only when the source attribute is
      missing on the rendered geometry (e.g. on Geometry Nodes-realised
      tubes whose custom data was stripped).
    * ``nodes_only`` – when True and the mesh exposes the
      ``intersection_attribute`` (e.g. ``is_intersection`` on OSMnx
      graphs), the colormap is only applied where that attribute equals 1.
      Everything else (intermediate curve points, tube vertices, …) is
      painted with ``edge_color`` so the colormap is not diluted by
      non-node geometry.
    """
    if obj is None:
        return None

    if vmax == vmin:
        vmax = vmin + 1e-9

    mat_name = _AUTO_MATERIAL_NAME_TEMPLATE.format(obj=obj.name)
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    for node in list(nodes):
        nodes.remove(node)

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (820, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (560, 0)

    attr_node = nodes.new("ShaderNodeAttribute")
    attr_node.attribute_name = source_attribute_name
    attr_node.attribute_type = 'GEOMETRY'
    attr_node.location = (-720, 60)

    map_range = nodes.new("ShaderNodeMapRange")
    map_range.data_type = 'FLOAT'
    map_range.interpolation_type = 'LINEAR'
    map_range.clamp = True
    map_range.location = (-440, 60)
    map_range.inputs['From Min'].default_value = float(vmin)
    map_range.inputs['From Max'].default_value = float(vmax)
    map_range.inputs['To Min'].default_value = 0.0
    map_range.inputs['To Max'].default_value = 1.0

    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.location = (-180, 60)
    color_ramp.color_ramp.interpolation = 'LINEAR'
    samples = sample_colormap(colormap, samples=_COLOR_RAMP_MAX_STOPS, reverse=reverse)
    _add_colorramp_stops(color_ramp.color_ramp, samples)

    links.new(attr_node.outputs['Fac'], map_range.inputs['Value'])
    links.new(map_range.outputs['Result'], color_ramp.inputs['Fac'])

    # Build the path that ultimately feeds Base Color, optionally mixing in a
    # fallback color layer when the source attribute is missing.
    color_socket = color_ramp.outputs['Color']

    if fallback_color_layer:
        vc_node = nodes.new("ShaderNodeAttribute")
        vc_node.attribute_name = fallback_color_layer
        vc_node.attribute_type = 'GEOMETRY'
        vc_node.location = (-180, -200)

        is_missing = nodes.new("ShaderNodeMath")
        is_missing.operation = 'LESS_THAN'
        is_missing.location = (-440, -200)
        is_missing.inputs[1].default_value = 1e-6
        links.new(attr_node.outputs['Fac'], is_missing.inputs[0])

        fallback_mix = nodes.new("ShaderNodeMixRGB")
        fallback_mix.blend_type = 'MIX'
        fallback_mix.location = (120, -100)
        links.new(is_missing.outputs['Value'], fallback_mix.inputs['Fac'])
        links.new(color_ramp.outputs['Color'], fallback_mix.inputs['Color1'])
        links.new(vc_node.outputs['Color'], fallback_mix.inputs['Color2'])

        color_socket = fallback_mix.outputs['Color']

    # Optional gating: only color real intersection nodes when requested.
    if nodes_only and _gate_attribute_available(obj, intersection_attribute):
        ec = tuple(edge_color) if edge_color is not None else (0.18, 0.18, 0.20, 1.0)
        if len(ec) == 3:
            ec = (ec[0], ec[1], ec[2], 1.0)

        intersection_node = nodes.new("ShaderNodeAttribute")
        intersection_node.attribute_name = intersection_attribute
        intersection_node.attribute_type = 'GEOMETRY'
        intersection_node.location = (60, 220)

        # Compare attribute > 0.5 -> treat as a real node.
        is_node = nodes.new("ShaderNodeMath")
        is_node.operation = 'GREATER_THAN'
        is_node.location = (260, 220)
        is_node.inputs[1].default_value = 0.5
        links.new(intersection_node.outputs['Fac'], is_node.inputs[0])

        gate_mix = nodes.new("ShaderNodeMixRGB")
        gate_mix.blend_type = 'MIX'
        gate_mix.location = (380, 80)
        gate_mix.inputs['Color1'].default_value = ec  # used when not a node
        links.new(is_node.outputs['Value'], gate_mix.inputs['Fac'])
        links.new(color_socket, gate_mix.inputs['Color2'])

        color_socket = gate_mix.outputs['Color']

    links.new(color_socket, bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    return mat


SCIGRAPHS_NODE_MARKER_ATTR = "scigraphs_is_node"


def _stamp_node_marker_after_realize(node_group, marker_name: str = SCIGRAPHS_NODE_MARKER_ATTR) -> bool:
    """Insert ``Store Named Attribute`` after every ``Realize Instances``.

    Realised sphere geometry gets ``marker_name = 1`` on POINT, so the
    coloring shader can tell those vertices apart from edge-tube vertices,
    which inherit ``is_intersection`` from the source curve points and would
    otherwise be coloured as if they were nodes.
    """
    if node_group is None:
        return False

    nodes = node_group.nodes
    links = node_group.links
    inserted = False

    for node in list(nodes):
        bl_idname = getattr(node, "bl_idname", "")
        if bl_idname != 'GeometryNodeRealizeInstances':
            continue

        try:
            geo_out = node.outputs['Geometry']
        except KeyError:
            continue

        downstream_links = list(geo_out.links)
        if not downstream_links:
            continue

        already_marked = False
        for link in downstream_links:
            tgt = link.to_node
            if (
                getattr(tgt, "bl_idname", "") == 'GeometryNodeStoreNamedAttribute'
                and tgt.inputs.get('Name')
                and getattr(tgt.inputs['Name'], 'default_value', '') == marker_name
            ):
                already_marked = True
                break
        if already_marked:
            continue

        store = nodes.new('GeometryNodeStoreNamedAttribute')
        store.data_type = 'FLOAT'
        store.domain = 'POINT'
        try:
            store.inputs['Name'].default_value = marker_name
        except KeyError:
            nodes.remove(store)
            continue

        # Find the value input across Blender versions.
        value_input = store.inputs.get('Value')
        if value_input is None:
            for candidate in store.inputs:
                if candidate.name.startswith('Value'):
                    value_input = candidate
                    break
        if value_input is not None:
            try:
                value_input.default_value = 1.0
            except (TypeError, RuntimeError):
                pass

        store.location = (node.location[0] + 200, node.location[1])

        for link in downstream_links:
            target = link.to_socket
            links.remove(link)
            links.new(store.outputs['Geometry'], target)
        links.new(geo_out, store.inputs['Geometry'])

        inserted = True

    return inserted


def wire_color_into_visual_setup(obj, source_attribute_name: str, material: bpy.types.Material) -> bool:
    """Hook the coloring material into the SciGraphs_Viz Geometry Nodes tree.

    * Records the protected attribute on the object so future GN rebuilds do
      not strip it.
    * Sets the Material input on every ``GeometryNodeSetMaterial`` node in
      the active visualisation tree.
    * Bypasses any pre-existing ``GeometryNodeRemoveAttribute`` node whose
      Name matches the source attribute so it survives the pipeline.
    * Stamps ``scigraphs_is_node = 1`` on realised sphere geometry so the
      shader can gate by node-vs-tube without trusting ``is_intersection``,
      which otherwise leaks into the first tube vertex of every street.

    Returns ``True`` when the modifier exists and was successfully patched.
    """
    if obj is None or material is None:
        return False

    obj["scigraphs_color_attr"] = source_attribute_name

    mod = obj.modifiers.get("SciGraphs_Viz") if hasattr(obj, "modifiers") else None
    if mod is None or not getattr(mod, "node_group", None):
        return False

    node_group = mod.node_group
    nodes = node_group.nodes
    links = node_group.links

    patched = False

    for node in list(nodes):
        node_type = getattr(node, "type", "") or getattr(node, "bl_idname", "")
        if node_type in {'SET_MATERIAL', 'GeometryNodeSetMaterial'}:
            try:
                node.inputs['Material'].default_value = material
                patched = True
            except (KeyError, RuntimeError):
                continue
        elif node_type in {'REMOVE_ATTRIBUTE', 'GeometryNodeRemoveAttribute'}:
            try:
                name_input = node.inputs.get('Name')
                current_name = getattr(name_input, "default_value", "") if name_input else ""
            except AttributeError:
                continue
            if current_name in {source_attribute_name, f"{source_attribute_name}_color"}:
                try:
                    geo_in = node.inputs['Geometry']
                    geo_out = node.outputs['Geometry']
                except KeyError:
                    continue
                if not geo_in.is_linked:
                    continue
                upstream_socket = geo_in.links[0].from_socket
                downstream_links = list(geo_out.links)
                for link in downstream_links:
                    target = link.to_socket
                    links.remove(link)
                    links.new(upstream_socket, target)
                nodes.remove(node)
                patched = True

    if _stamp_node_marker_after_realize(node_group):
        patched = True

    return patched


def has_visual_setup(obj) -> bool:
    """Return True when ``obj`` has the ``SciGraphs_Viz`` modifier installed."""
    if obj is None or not hasattr(obj, "modifiers"):
        return False
    mod = obj.modifiers.get("SciGraphs_Viz")
    return mod is not None and getattr(mod, "node_group", None) is not None


# ---------------------------------------------------------------------------
# Reapplying coloring after the visual setup is rebuilt
# ---------------------------------------------------------------------------

def reapply_coloring_after_viz_rebuild(obj) -> bool:
    """Restore the coloring wiring after a ``SciGraphs_Viz`` rebuild.

    Geometry rebuilds (centrality, edge style, ...) recreate the GN tree
    from scratch which wipes both the material on the ``Set Material``
    node and the ``scigraphs_is_node`` marker we stamped after every
    ``Realize Instances``. We rely on:

    * ``obj["scigraphs_color_attr"]`` – the attribute the user was coloring
      by, set by :func:`wire_color_into_visual_setup`.
    * ``bpy.data.materials[<obj>_SciGraphsColor]`` – the material
      previously built by :func:`build_color_material`.

    When both are available, we re-run :func:`wire_color_into_visual_setup`
    on the freshly rebuilt tree so the user keeps seeing exactly the same
    colors with no manual re-apply.
    """
    if obj is None or obj.type != 'MESH':
        return False

    color_attr = obj.get("scigraphs_color_attr", "") or ""
    if not color_attr:
        return False

    mat_name = _AUTO_MATERIAL_NAME_TEMPLATE.format(obj=obj.name)
    material = bpy.data.materials.get(mat_name)
    if material is None:
        return False

    return wire_color_into_visual_setup(obj, color_attr, material)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def apply_coloring(context) -> Tuple[bool, str]:
    """Apply the current coloring settings to the active mesh object."""
    obj = active_mesh_object(context)
    if obj is None:
        return False, "Active object is not a mesh"

    props = coloring_props(context)
    if props is None:
        return False, "Coloring settings not registered"

    attribute_name = resolve_attribute_name(props, obj)
    if not attribute_name:
        return False, "No scalar attribute available on this mesh"

    src_attr = find_attribute(obj.data, attribute_name)
    if src_attr is None:
        return False, f"Attribute '{attribute_name}' not found on mesh"

    if src_attr.data_type not in SCALAR_DATA_TYPES:
        return False, f"Attribute '{attribute_name}' is not scalar ({src_attr.data_type})"

    source_values = read_attribute_values(obj.data, attribute_name)
    if source_values.size == 0:
        return False, f"Attribute '{attribute_name}' has no data"

    if not colormap_exists(props.colormap):
        # Catalog should always include the value, but be defensive.
        return False, f"Unknown colormap '{props.colormap}'"

    vmin = None if props.auto_range else float(props.vmin)
    vmax = None if props.auto_range else float(props.vmax)

    color_domain = _select_color_domain(props, src_attr.domain, obj.data)
    target_values = values_for_color_domain(
        obj.data, src_attr.domain, source_values, color_domain
    )
    if target_values.size == 0:
        return False, "Could not map attribute values to the chosen color domain"

    rgba = values_to_rgba(
        target_values,
        cmap_name=props.colormap,
        vmin=vmin,
        vmax=vmax,
        reverse=props.reverse,
    )
    if props.opacity < 1.0:
        rgba[:, 3] *= float(props.opacity)

    color_name = _color_attribute_name(props, attribute_name)
    ok, message = write_color_attribute(
        obj, attribute_name, color_name, color_domain, rgba
    )
    if not ok:
        return False, message

    eff_vmin, eff_vmax = (vmin, vmax)
    if eff_vmin is None or eff_vmax is None:
        from ...core.coloring.colormaps import normalize_range
        eff_vmin, eff_vmax = normalize_range(target_values)
    props.last_vmin = float(eff_vmin)
    props.last_vmax = float(eff_vmax)

    if props.auto_setup_material:
        # When the SciGraphs_Viz GN modifier is installed we mark realised
        # sphere geometry with `scigraphs_is_node = 1`, so the shader gates
        # against that instead of `is_intersection`. The latter would also be
        # 1 on the tube vertex that closes against an intersection, which is
        # exactly the "edge segment that took the node color" symptom.
        gate_attr = (
            SCIGRAPHS_NODE_MARKER_ATTR if has_visual_setup(obj) else "is_intersection"
        )
        material = build_color_material(
            obj,
            source_attribute_name=attribute_name,
            colormap=props.colormap,
            vmin=float(eff_vmin),
            vmax=float(eff_vmax),
            reverse=props.reverse,
            fallback_color_layer=color_name,
            nodes_only=bool(getattr(props, "nodes_only", False)),
            edge_color=tuple(getattr(props, "edge_color", (0.18, 0.18, 0.20, 1.0))),
            intersection_attribute=gate_attr,
        )
        if material is not None:
            wire_color_into_visual_setup(obj, attribute_name, material)

    return True, (
        f"{attribute_name} -> {props.colormap}"
        f"  [{eff_vmin:.4g} … {eff_vmax:.4g}] on {color_domain.title()}"
    )


def remove_coloring(context) -> Tuple[bool, str]:
    """Remove the most recent SciGraphs color attribute from the active mesh."""
    obj = active_mesh_object(context)
    if obj is None:
        return False, "Active object is not a mesh"

    color_name = obj.get("scigraphs_last_color_attribute", "") or ""
    if color_name and color_name in obj.data.color_attributes:
        try:
            obj.data.color_attributes.remove(obj.data.color_attributes[color_name])
        except RuntimeError as exc:
            return False, f"Could not remove '{color_name}': {exc}"
        return True, f"Removed color attribute '{color_name}'"
    return False, "No SciGraphs color attribute to remove"


def attribute_names(obj) -> List[str]:
    """Convenience wrapper returning just the names of scalar attributes."""
    if obj is None or obj.type != 'MESH':
        return []
    return [name for name, *_ in list_scalar_attributes(obj.data)]


def quick_colormap_at(index: int) -> Optional[str]:
    """Return the colormap identifier for a 0-based chip index."""
    if 0 <= index < len(QUICK_COLORMAPS):
        return QUICK_COLORMAPS[index]
    return None


def quick_colormap_iter() -> Iterable[Tuple[int, str]]:
    """Yield ``(index, colormap_name)`` pairs for every chip."""
    for index, name in enumerate(QUICK_COLORMAPS):
        yield index, name
