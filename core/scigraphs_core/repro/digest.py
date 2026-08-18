"""Compact exports of the pipeline schema, for tools that generate specs.

Both artifacts derive from ``SCHEMA``, so neither can describe a field that no
longer exists. ``build_digest`` is a prose listing: each field's type, default
and allowed values, plus the one-line description saying when the default is
the wrong choice. ``build_json_schema`` is the same thing as JSON Schema, fit
for editor autocompletion or for constrained generation. Neither includes the
``ops`` section: it executes operators, and nothing that writes a spec from
outside should know it exists.
"""

from .schema import SCHEMA

# Sections a generated specification may use. `ops` is absent by design; see
# safety.py for why.
ALLOWED_SECTIONS = (
    "meta", "dataset", "analysis", "layout",
    "visual", "labels", "world", "lighting", "render", "exports",
)

_TYPES = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
}


def build_digest():
    """Compact, readable description of every usable field."""
    lines = []
    for name in ALLOWED_SECTIONS:
        section = SCHEMA.get(name)
        if not section:
            continue
        required = section.get("required") or []
        header = "## %s" % name
        if required:
            header += "  (required: %s)" % ", ".join(required)
        lines.append(header)

        for field, definition in (section.get("properties") or {}).items():
            if definition.get("type") == "object" and "properties" in definition:
                for sub, subdef in definition["properties"].items():
                    lines.append("  " + _field_line("%s.%s" % (field, sub), subdef))
                continue
            lines.append("  " + _field_line(field, definition))
        lines.append("")
    return "\n".join(lines).rstrip()


def _field_line(name, definition):
    parts = ["%s: %s" % (name, _TYPES.get(definition.get("type", ""), "any"))]
    if "default" in definition:
        parts.append("default %r" % (definition["default"],))
    enum = definition.get("enum")
    if enum:
        shown = enum if len(enum) <= 12 else list(enum[:12]) + ["..."]
        parts.append("one of: %s" % ", ".join(str(v) for v in shown))
    description = definition.get("description")
    if description:
        parts.append("-- %s" % description)
    return "; ".join(parts)


def build_json_schema():
    """JSON Schema for endpoints that support constrained decoding."""
    properties = {}
    for name in ALLOWED_SECTIONS:
        section = SCHEMA.get(name)
        if not section:
            continue
        properties[name] = _section_schema(section)

    return {
        "type": "object",
        "properties": properties,
        "required": ["meta"],
        # A generated spec must not invent sections; `ops` in particular would be
        # stripped afterwards, so there is no point letting it be produced.
        "additionalProperties": False,
    }


def _section_schema(section):
    props = {}
    for field, definition in (section.get("properties") or {}).items():
        props[field] = _field_schema(definition)
    out = {"type": "object", "properties": props, "additionalProperties": False}
    if section.get("required"):
        out["required"] = list(section["required"])
    return out


def _field_schema(definition):
    kind = definition.get("type", "string")
    if kind == "object" and "properties" in definition:
        return _section_schema(definition)
    out = {"type": kind}
    if definition.get("enum"):
        out["enum"] = list(definition["enum"])
    if kind == "array" and "items" in definition:
        item_type = (definition["items"] or {}).get("type")
        if item_type:
            out["items"] = {"type": item_type}
    return out
