"""Ask a local model for a pipeline specification, from the terminal.

    python3 scripts/repro/ask.py "the walk network of Lliria colored by betweenness"
    python3 scripts/repro/ask.py "..." -o my_spec.json --run

Sends the assistant instructions plus your request to Ollama, constrains the
reply to the published JSON Schema so the model cannot invent a field, checks
what comes back and writes it out; --run executes it in Blender. Needs Ollama
running and a model pulled (`ollama pull qwen2.5-coder:7b`)."""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# scigraphs_core is a wheel built from core/, so its package root is core/.
# Without this the model answers and the import below throws the answer away.
CORE = os.path.join(ROOT, "core")
if CORE not in sys.path:
    sys.path.insert(0, CORE)

SKILL = os.path.join(ROOT, "docs", "reference", "_scigraphs-pipeline-skill.md")
JSON_SCHEMA = os.path.join(ROOT, "docs", "reference", "pipeline.schema.json")
OLLAMA = "http://localhost:11434/api/generate"


def generate(model, request, skill, schema, seed):
    body = {
        "model": model,
        "prompt": skill + "\n\n---\n\nRequest: " + request,
        "stream": False,
        "options": {"temperature": 0, "seed": seed},
    }
    if schema is not None:
        body["format"] = schema
    req = urllib.request.Request(
        OLLAMA, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            return json.loads(response.read())["response"]
    except urllib.error.URLError as exc:
        sys.exit("Could not reach Ollama at %s (%s). Is it running?"
                 % (OLLAMA, exc))


def parse(reply):
    """The reply is JSON, but a model may still wrap it in prose or a fence."""
    text = reply.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.split("\n", 1)[1] if text.startswith("json") else text
    start = text.find("{")
    if start < 0:
        sys.exit("The model did not reply with JSON:\n%s" % reply[:400])
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            try:
                return json.loads(text[start:i + 1])
            except json.JSONDecodeError as exc:
                sys.exit("The reply is not valid JSON: %s" % exc)
    sys.exit("The reply contains an unterminated JSON object.")


def review(spec):
    """Report what is wrong, and what is merely worth a second look."""
    from scigraphs_core.repro.schema import SCHEMA, validate_pipeline

    errors, notes = [], []
    try:
        validate_pipeline(spec)
    except Exception as exc:
        errors.append(str(exc))

    for section, body in spec.items():
        if section not in SCHEMA:
            errors.append("section '%s' does not exist" % section)
            continue
        known = SCHEMA[section].get("properties") or {}
        if isinstance(body, dict):
            for field in body:
                if known and field not in known:
                    errors.append("'%s.%s' does not exist" % (section, field))

    source = (spec.get("dataset") or {}).get("source")
    if source in ("osmnx", "city2graph") and "layout" in spec:
        errors.append("a layout on '%s' overwrites the real coordinates" % source)
    if source in ("gexf", "graphml", "csv", "sql") and "layout" not in spec:
        errors.append("'%s' has no coordinates and no layout: every node would "
                      "sit at the origin" % source)

    visual = spec.get("visual") or {}
    if (visual.get("node_color") in ("betweenness", "degree", "closeness",
                                     "eigenvector", "pagerank")
            and visual.get("color_norm", "LINEAR") == "LINEAR"):
        notes.append("coloring by a centrality without color_norm; RANK usually "
                     "reads far better")
    labels = spec.get("labels") or {}
    if labels.get("max_distance") and "max_count" not in labels:
        notes.append("labels.max_distance is a distance from the camera, not a "
                     "number of labels. If you asked for the top N, you want "
                     "max_count")

    if spec.get("ops"):
        notes.append("this specification uses `ops`, which invokes Blender "
                     "operators directly -- read it before running")
    return errors, notes


def main():
    parser = argparse.ArgumentParser(
        description="Ask a local model for a SciGraphs pipeline specification.")
    parser.add_argument("request", help="what you want, in plain language")
    parser.add_argument("-o", "--out", default="spec.json")
    parser.add_argument("-m", "--model", default="qwen2.5-coder:7b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--free", action="store_true",
                        help="do not constrain the reply to the schema")
    parser.add_argument("--run", action="store_true",
                        help="execute the result in Blender afterwards")
    args = parser.parse_args()

    for path in (SKILL, JSON_SCHEMA):
        if not os.path.isfile(path):
            sys.exit("missing %s -- run scripts/docs/build_skill.py and "
                     "scripts/docs/build_json_schema.py" % path)

    skill = open(SKILL, encoding="utf-8").read()
    schema = None if args.free else json.load(open(JSON_SCHEMA, encoding="utf-8"))

    print("asking %s..." % args.model, file=sys.stderr)
    spec = parse(generate(args.model, args.request, skill, schema, args.seed))

    errors, notes = review(spec)
    for note in notes:
        print("  note:  %s" % note, file=sys.stderr)
    for error in errors:
        print("  ERROR: %s" % error, file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2)
        handle.write("\n")
    print("wrote %s (%d sections)" % (args.out, len(spec)), file=sys.stderr)

    if errors:
        print("Not run: fix the errors above first.", file=sys.stderr)
        return 1
    if not args.run:
        return 0

    print("running in Blender...", file=sys.stderr)
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    return subprocess.call([
        "blender", "-b", "--python-expr",
        "import bpy, sys; sys.exit(0 if 'FINISHED' in "
        "bpy.ops.scigraphs.run_pipeline(filepath=sys.argv[-1]) else 1)",
        "--", os.path.abspath(args.out),
    ], env=env)


if __name__ == "__main__":
    sys.exit(main())
