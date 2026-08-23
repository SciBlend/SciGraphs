"""Generate the model-facing skill from the real schema.

    python3 scripts/docs/build_skill.py

Writes docs/reference/scigraphs-pipeline-skill.md, one self-contained file a user
attaches to whatever assistant they already use. The field list comes from
`SCHEMA`; what a schema cannot express (which fields exclude each other, why
RANK, why a layout ruins a street network) is written by hand below."""

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

OUT = os.path.join(ROOT, "docs", "reference", "_scigraphs-pipeline-skill.md")
INCLUDE = os.path.join(ROOT, "docs", "reference", "_skill-download.qmd")

# Sections in the order the executor runs them.
ORDER = ("meta", "dataset", "analysis", "layout", "visual",
         "labels", "world", "lighting", "render", "exports", "ops")


def field_line(name, spec):
    """One compact line per field: name, type, enum, default, purpose."""
    bits = [f"`{name}`", spec.get("type", "")]
    enum = spec.get("enum")
    if enum:
        # Quote strings: a bare ["8","16","32"] reads as numeric and the model
        # emits 16 instead of "16", which fails validation.
        bits.append("one of: " + " | ".join(
            ('"%s"' % v) if isinstance(v, str) else str(v) for v in enum))
    if "default" in spec:
        bits.append(f"default {spec['default']!r}")
    line = " ".join(b for b in bits if b)
    desc = spec.get("description")
    if desc:
        line += f". {desc}"
    return "- " + line


def schema_block():
    from scigraphs_core.repro.schema import SCHEMA

    out = []
    for name in ORDER:
        section = SCHEMA.get(name)
        if not section:
            continue
        out.append(f"### `{name}`")
        required = section.get("required")
        if required:
            out.append(f"Required: {', '.join('`%s`' % r for r in required)}")
        props = section.get("properties")
        if not props:
            # `ops` is an array, so its fields live under items.
            item_props = (section.get("items") or {}).get("properties")
            if item_props:
                out.append("An array of objects, each with:")
                for key, spec in item_props.items():
                    out.append(field_line(key, spec))
            out.append("")
            continue
        for key, spec in props.items():
            if spec.get("type") == "object" and "properties" in spec:
                out.append(f"- `{key}` - object:")
                for sub, subspec in spec["properties"].items():
                    out.append("  " + field_line(sub, subspec))
                continue
            out.append(field_line(key, spec))
        out.append("")
    return "\n".join(out)


RULES = """
## What the user says, and the field that means

Look a request up here before writing anything. Every failure observed while
testing this file came from improvising a field name instead of finding the real
one.

| The request says | The field is |
| --- | --- |
| depth of field, bokeh, blurred background | `render.dof_fstop` |
| lens, focal length, "50mm", wide angle | `render.camera_lens` |
| orthographic, isometric, flat, no perspective | `render.camera_ortho` |
| communities, clusters, modularity, Louvain, Infomap | `analysis.clustering.algorithm` |
| labels, annotate, name the nodes, show names | the `labels` section |
| "the 10 most central", top N, only the biggest | `labels.max_count`, never `max_distance`, which is a camera distance |
| dark background, background color, backdrop | `world.color` and `world.strength` |
| print quality, dpi, publication | `render.dpi`, `render.color_depth` |
| logarithmic, log scale | `visual.color_norm` set to `LOG` |
| bigger nodes, node size, thicker edges | `visual.node_radius_rel`, `visual.edge_radius_rel` |
| size by degree, scale nodes by a value | `visual.node_size` naming the attribute |
| ambient occlusion, contact shadows | `render.ambient_occlusion` with `render.raytracing`, EEVEE only |
| soft shadows, lighting, brighter | the `lighting` section |
| transparent background, alpha | `render.transparent` |
| curved edges, bundled edges | `visual.edge_style` |
| sphere, cube, icosphere nodes | `visual.node_glyph` |

If a request names something that is not in this table and not in the field list,
do not invent a field for it. Say which part you could not express.

## Rules that the field list does not tell you

These are the mistakes to avoid. They are not type errors, so nothing will catch
them for you.

**The `layout` block is required for some sources and forbidden for others.**

- `gexf`, `graphml`, `csv`, `sql`: these carry no coordinates, so the nodes would
  all sit at the origin. You **must** include a `layout` block. `SPRING_3D` with
  `scale` 5 and 150 iterations is a good default.
- `osmnx`, `city2graph`: these arrive with real coordinates, and those
  coordinates are the data. **Never** add a `layout`, it overwrites them and
  throws the geography away.
- `suitesparse`: include a layout unless the matrix ships a coordinate file.
  `SPECTRAL_3D` suits most matrices.

**Always set `color_norm` when you color by a centrality.** Centrality measures
are heavy-tailed. Measured on the graphs shipped with SciGraphs, over 90% of
nodes fall in the lowest tenth of the range, so without a transform almost every
node renders the same dark color and the encoding carries no information. Write `"color_norm": "RANK"` for betweenness
and degree, `"QUANTILE"` when many nodes tie, `"LOG"` for a quantity that varies
smoothly over orders of magnitude.

`RANK` equalizes the histogram, so a tenth of the nodes always land in the
brightest tenth of the ramp. That is right when the ordering is what matters, and
wrong when the point is that a small minority dominates -- a road network, say,
where the interesting fact is which few streets carry the through traffic. There,
keep the ramp linear and set `color_clip_percentile` to `[0, 99]` instead: the
concentration survives and one extreme node no longer compresses everything.

**Prefer `node_radius_rel` and `edge_radius_rel` to the absolute forms.** They are
fractions of the graph's own extent, so they work at any `layout.scale`. Around
`0.02` and `0.003` suit a graph of tens of nodes; around `0.004` and `0.0014` suit
a city street network of thousands.

**Set `render.view_transform` to `"Standard"`** for anything where color encodes
a value. Blender's default tone-maps the image, so the rendered color of a node
stops matching the colormap entry for its value.

**Choose one source per specification.** The `dataset` fields are
source-specific: `filepath` for file sources, `query` and `method` for `osmnx`,
`matrix_name` for `suitesparse`, `nodes_query`/`edges_query` for `sql`. Never mix
them.

**Do not set both a field and its alternative.** `node_radius` versus
`node_radius_rel`, `color_vmin`/`color_vmax` versus a `color_norm` that derives
its own domain, `render.camera` versus `frame_camera`.

**Engine options live in `render`, never in `lighting`.** `lighting` holds one
sun and nothing else, its only fields are `sun_energy`, `sun_angle`,
`sun_rotation` and `replace`. Everything about how the image is computed belongs
to `render`:

- EEVEE only: `render.raytracing`, `render.ambient_occlusion`,
  `render.ao_distance`, `render.clamp_indirect`.
- Cycles only: `render.adaptive_threshold`, `render.max_bounces`,
  `render.denoiser`, `render.denoise`.

`render.engine` accepts exactly three values: `CYCLES`, `BLENDER_EEVEE`,
`BLENDER_WORKBENCH`. Copy one of those three characters for character.

**You do not have the `scene_props` names.** `ops[].scene_props` reaches about
520 Blender properties that are deliberately not listed here. Never guess one:
say you need the pipeline options reference instead.

**Avoid `ops` unless asked.** It invokes arbitrary Blender operators and is the
one part of a specification that is not just data. If you use it, say so plainly
in your reply so the user knows to read it before running.

**Metric attribute names.** `analysis.metrics` produces attributes named
`centrality_<metric>`, but in `visual` you reference them by the short name:
`"node_color": "betweenness"`, not `"centrality_betweenness"`.

## Output contract

Reply with **one JSON object and nothing else**, no explanation before or after,
no markdown fence, no comments. JSON has no comment syntax; a `//` line makes the
file unparseable.

Never emit `""`, `null` or a placeholder for a field you have no value for -
omit the field entirely and its default applies. An empty string fails
validation just as an invented name does.

Include only the fields the request needs. A short, correct specification is
better than an exhaustive one. `meta.title` is the only required field, but a
useful specification almost always has `dataset`, `visual` and `render` too.

Paths beginning `//` resolve against the folder holding the specification.

## Worked examples

A file-based graph. Note the `layout` block, without it every node sits at
the origin:

```json
{
  "meta": {"title": "karate", "seed": 42, "output_dir": "//repro/karate"},
  "dataset": {"source": "gexf", "filepath": "//data/karate.gexf"},
  "analysis": {"metrics": ["degree", "betweenness"]},
  "layout": {"algorithm": "SPRING_3D", "scale": 5.0, "iterations": 150},
  "visual": {"node_color": "betweenness", "colormap": "inferno", "color_norm": "RANK",
             "node_radius_rel": 0.022, "edge_radius_rel": 0.0035},
  "labels": {"rank_by": "betweenness", "max_count": 12},
  "lighting": {"sun_energy": 3.0, "sun_angle": 180.0},
  "render": {"engine": "CYCLES", "samples": 96, "output": "figure.png",
             "view_transform": "Standard"}
}
```

A street network. Note there is **no** `layout` block, the coordinates come
with the data:

```json
{
  "meta": {"title": "granada_walk", "seed": 42, "output_dir": "//repro/granada_walk"},
  "dataset": {"source": "osmnx", "method": "PLACE", "query": "Granada, Spain", "network_type": "walk"},
  "analysis": {"metrics": ["degree", "betweenness"]},
  "visual": {"node_color": "betweenness", "colormap": "plasma", "color_norm": "RANK",
             "node_radius_rel": 0.004, "edge_radius_rel": 0.0014},
  "lighting": {"sun_energy": 3.0, "sun_angle": 180.0},
  "render": {"engine": "CYCLES", "samples": 64, "output": "figure.png",
             "view_transform": "Standard"}
}
```

A matrix rendered with the real-time engine:

```json
{
  "meta": {"title": "bcsstk09", "seed": 42, "output_dir": "//repro/bcsstk09"},
  "dataset": {"source": "suitesparse", "matrix_name": "HB/bcsstk09"},
  "analysis": {"metrics": ["degree", "eigenvector"]},
  "layout": {"algorithm": "SPECTRAL_3D", "scale": 6.0, "iterations": 1},
  "visual": {"node_color": "eigenvector", "colormap": "cividis", "color_norm": "LOG",
             "node_radius_rel": 0.01, "edge_radius_rel": 0.0022},
  "lighting": {"sun_energy": 3.0, "sun_angle": 180.0},
  "render": {"engine": "BLENDER_EEVEE", "samples": 128, "output": "figure.png",
             "view_transform": "Standard", "raytracing": true,
             "ambient_occlusion": true, "ao_distance": 2.0}
}
```

## Before you answer

1. One source, and its fields only.
2. Is the source `gexf`, `graphml`, `csv`, `sql` or `suitesparse`? Then a
   `layout` block is present. Is it `osmnx` or `city2graph`? Then there is none.
3. Every field is in the section the list above puts it in, engine options go in
   `render`, not `lighting`.
4. `color_norm` set if coloring by a centrality.
5. Every field name and enum value copied from the list above, never invented.
6. Enum values shown in quotes are strings: write `"16"`, not `16`.
7. Engine-specific fields match the engine.
8. One JSON object, no prose, no fence, no comments.
"""


HEADER = """# SciGraphs pipeline specifications, assistant instructions

You write **SciGraphs pipeline specifications**: declarative JSON files that
drive a whole graph-visualization workflow inside Blender, loading data,
computing metrics, laying out, styling, lighting, rendering and exporting, and
that reproduce the same figure from the same seed.

Attach this file to your assistant, then describe the figure you want. It replies
with the specification, which you save as `.json` and run: drag it onto Blender's
3D Viewport, or

```bash
blender -b --python-expr "import bpy; bpy.ops.scigraphs.run_pipeline(filepath='/path/to/spec.json')"
```

Generated from the add-on's own schema, so the field list below cannot drift from
what the software accepts.

---

## The fields

Only `meta.title` is required. Sections run in the order listed.

"""


def main():
    import base64

    text = HEADER + schema_block() + RULES
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        handle.write(text)

    # Embedded, not served, so the page works offline and from a checkout.
    uri = ("data:text/markdown;base64,"
           + base64.b64encode(text.encode("utf-8")).decode("ascii"))
    with open(INCLUDE, "w", encoding="utf-8") as handle:
        handle.write(
            "::: {.callout-tip}\n"
            "## Let an assistant write it for you\n\n"
            "Attach this file to whatever assistant you already use, Claude, "
            "ChatGPT, Copilot, a local model, then describe the figure you "
            "want and it replies with the specification.\n\n"
            "::: {.gallery-actions}\n"
            '<a class="download-spec" download="scigraphs-pipeline-skill.md" '
            'href="%s">Download the assistant instructions</a>\n'
            ":::\n\n"
            "It carries the field list generated from this same schema plus the "
            "rules a field list cannot express. Measured on ten requests "
            "against a 7B local model: none of the "
            "specifications it produced without this file were usable. With it "
            "and the schema below constraining the output, nine of ten both "
            "validated and matched the request.\n\n"
            "One caution: a specification is not inert data. The `ops` block "
            "invokes arbitrary Blender operators, so read a generated file "
            "before running it.\n"
            ":::\n" % uri)
    words = len(text.split())
    print("wrote %s -- %d lines, ~%d words (~%d tokens)"
          % (os.path.relpath(OUT, ROOT), text.count("\n") + 1, words, words * 4 // 3))
    print("wrote %s" % os.path.relpath(INCLUDE, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
