"""Emit a real JSON Schema for pipeline specifications.

    python3 scripts/docs/build_json_schema.py

Writes docs/reference/pipeline.schema.json, translated from the add-on's own
`SCHEMA`. Editors that understand `$schema` autocomplete field names, and
`additionalProperties: false` stops constrained decoding inventing one."""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# scigraphs_core is a wheel built from core/, so its package root is core/
# and not the repository root.
CORE = os.path.join(ROOT, "core")
if CORE not in sys.path:
    sys.path.insert(0, CORE)

OUT = os.path.join(ROOT, "docs", "reference", "pipeline.schema.json")

TYPES = {"string": "string", "integer": "integer", "number": "number",
         "boolean": "boolean", "array": "array", "object": "object"}


def convert_field(spec):
    out = {}
    ftype = TYPES.get(spec.get("type"))
    if ftype:
        out["type"] = ftype
    if "enum" in spec:
        out["enum"] = list(spec["enum"])
    if "description" in spec:
        out["description"] = spec["description"]
    if "default" in spec:
        out["default"] = spec["default"]
    if spec.get("type") == "array" and "items" in spec:
        out["items"] = convert_field(spec["items"])
    if spec.get("type") == "object" and "properties" in spec:
        out["properties"] = {k: convert_field(v)
                             for k, v in spec["properties"].items()}
        out["additionalProperties"] = False
    return out


def convert_section(name, section):
    if section.get("type") == "array":
        item = section.get("items") or {}
        node = {"type": "array", "items": {
            "type": "object",
            "properties": {k: convert_field(v)
                           for k, v in (item.get("properties") or {}).items()},
            "additionalProperties": False,
        }}
        if item.get("required"):
            node["items"]["required"] = list(item["required"])
        return node

    node = {
        "type": "object",
        "properties": {k: convert_field(v)
                       for k, v in (section.get("properties") or {}).items()},
        # The whole reason this file exists: reject names that are not fields.
        "additionalProperties": False,
    }
    if section.get("required"):
        node["required"] = list(section["required"])
    return node


def main():
    from scigraphs_core.repro.schema import SCHEMA

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://sciblend.github.io/SciGraphs/pipeline.schema.json",
        "title": "SciGraphs pipeline specification",
        "description": (
            "A declarative description of a graph visualization workflow: "
            "ingest, analyze, lay out, style, light, render and export."
        ),
        "type": "object",
        "properties": {name: convert_section(name, body)
                       for name, body in SCHEMA.items()},
        "required": ["meta"],
        "additionalProperties": False,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2)
        handle.write("\n")

    fields = sum(len(v.get("properties", {})) for v in schema["properties"].values())
    print("wrote %s -- %d sections, %d fields"
          % (os.path.relpath(OUT, ROOT), len(schema["properties"]), fields))
    return 0


if __name__ == "__main__":
    sys.exit(main())
