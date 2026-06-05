"""Reference generator for reproducible SciGraphs pipelines.

Introspects the live Blender property groups, the declarative pipeline
``SCHEMA`` and the operator registry to produce an exhaustive Markdown
reference of every option a pipeline JSON/YAML can set. This keeps the
authoring documentation in sync with the code instead of being hand-maintained.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .schema import SCHEMA
from .registry import SCENE_PROPERTY_GROUPS, get_registry


def _bpy():
    try:
        import bpy
        return bpy
    except ImportError:
        return None


def _enum_items(prop) -> List[str]:
    """Return the identifiers of an enum property, if any."""
    items = getattr(prop, "enum_items", None)
    if not items:
        return []
    out: List[str] = []
    for item in items:
        identifier = getattr(item, "identifier", "")
        if identifier:
            out.append(identifier)
    return out


def _default_value(prop) -> str:
    """Return a printable default for a property definition."""
    for attr in ("default", "default_array"):
        if hasattr(prop, attr):
            try:
                value = getattr(prop, attr)
            except (AttributeError, TypeError):
                continue
            if value is None:
                continue
            if attr == "default_array":
                try:
                    return str(tuple(value))
                except TypeError:
                    continue
            return str(value)
    return ""


def _iter_group_properties(group) -> List[Tuple[str, Any]]:
    """Yield (name, property_definition) for a PointerProperty group instance."""
    rna = getattr(type(group), "bl_rna", None)
    if rna is None:
        return []
    rows: List[Tuple[str, Any]] = []
    for prop in rna.properties:
        if prop.identifier == "rna_type":
            continue
        if getattr(prop, "is_readonly", False):
            continue
        rows.append((prop.identifier, prop))
    rows.sort(key=lambda item: item[0])
    return rows


def _group_table(group) -> List[str]:
    """Build a Markdown table for one scene property group."""
    lines = ["| Property | Type | Default | Enum values |", "| --- | --- | --- | --- |"]
    for name, prop in _iter_group_properties(group):
        ptype = getattr(prop, "type", "")
        default = _default_value(prop)
        enum_vals = ", ".join(_enum_items(prop)) if ptype == "ENUM" else ""
        lines.append(f"| `{name}` | {ptype} | {default} | {enum_vals} |")
    return lines


def _schema_section_table(section: Dict[str, Any]) -> List[str]:
    """Build a Markdown table for one declarative schema section."""
    props = section.get("properties", {})
    lines = ["| Field | Type | Default | Enum |", "| --- | --- | --- | --- |"]
    for key, definition in props.items():
        ptype = definition.get("type", "")
        default = definition.get("default", "")
        enum = ", ".join(str(v) for v in definition.get("enum", []))
        lines.append(f"| `{key}` | {ptype} | {default} | {enum} |")
    return lines


def generate_reference_markdown() -> str:
    """Generate the complete pipeline-options reference as a Markdown string."""
    lines: List[str] = []
    lines.append("# SciGraphs pipeline options reference")
    lines.append("")
    lines.append(
        "Auto-generated reference of every option exposed to reproducible "
        "pipelines. Regenerate it from the Reproducibility panel "
        "(Export Options Reference) or via "
        "`SciGraphs.core.repro.reference.generate_reference_markdown()`."
    )
    lines.append("")

    # 1. Declarative schema sections.
    lines.append("## Declarative stages")
    lines.append("")
    lines.append(
        "These typed sections cover the common workflow. Every field below is "
        "applied by the executor."
    )
    lines.append("")
    for section_name in (
        "meta", "dataset", "analysis", "layout", "visual", "render", "exports",
    ):
        section = SCHEMA.get(section_name)
        if not section:
            continue
        lines.append(f"### `{section_name}`")
        lines.append("")
        lines.extend(_schema_section_table(section))
        lines.append("")
        nested = section.get("properties", {})
        for key, definition in nested.items():
            if definition.get("type") == "object" and "properties" in definition:
                lines.append(f"#### `{section_name}.{key}`")
                lines.append("")
                lines.extend(_schema_section_table(definition))
                lines.append("")

    # 2. Generic ops + registry shortcuts.
    lines.append("## Generic ops")
    lines.append("")
    lines.append(
        "The `ops` array calls any operator by `id` (a bl_idname or a "
        "registry shortcut), with `props` (operator keyword arguments) and "
        "`scene_props` (scene state). `scene_props` may be flat (applied to "
        "`scene.scigraphs`) or keyed by property group."
    )
    lines.append("")
    lines.append("### Property-group keys for `scene_props`")
    lines.append("")
    lines.append("| Group key | Scene attribute |")
    lines.append("| --- | --- |")
    for group_key, scene_attr in SCENE_PROPERTY_GROUPS.items():
        lines.append(f"| `{group_key}` | `scene.{scene_attr}` |")
    lines.append("")

    registry = get_registry()
    shortcuts = getattr(registry, "_shortcuts", {})
    if shortcuts:
        lines.append("### Registry shortcuts")
        lines.append("")
        lines.append("| Shortcut | Operator |")
        lines.append("| --- | --- |")
        for short, full in sorted(shortcuts.items()):
            lines.append(f"| `{short}` | `{full}` |")
        lines.append("")

    # 3. Scene property groups (requires bpy).
    bpy = _bpy()
    lines.append("## Scene property groups")
    lines.append("")
    if bpy is None:
        lines.append(
            "_Property-group tables require a running Blender session. Run the "
            "generator from inside Blender to include them._"
        )
        lines.append("")
        return "\n".join(lines)

    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        lines.append("_No active scene available to introspect._")
        lines.append("")
        return "\n".join(lines)

    for group_key, scene_attr in SCENE_PROPERTY_GROUPS.items():
        group = getattr(scene, scene_attr, None)
        if group is None:
            continue
        lines.append(f"### `{group_key}` (`scene.{scene_attr}`)")
        lines.append("")
        lines.extend(_group_table(group))
        lines.append("")

    return "\n".join(lines)


def write_reference_markdown(filepath: str) -> str:
    """Write the generated reference to ``filepath`` and return the content."""
    import os

    content = generate_reference_markdown()
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as handle:
        handle.write(content)
    return content
