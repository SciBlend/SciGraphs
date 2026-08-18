"""Bounds on a pipeline spec from an untrusted source (model, drop, email). A spec
is not inert, the executor runs it, so unbounded parts go before a human approves:
``ops`` is arbitrary operators, ``connection_string`` and its SQL siblings carry
credentials. Paths and place names stay."""

import copy
import os

DENIED_SECTIONS = ("ops",)
DENIED_FIELDS = {
    "dataset": ("connection_string", "nodes_query", "edges_query"),
}

# Shown to the user before run (disk / network).
EFFECT_FIELDS = (
    ("dataset", "filepath", "reads"),
    ("dataset", "query", "downloads from OpenStreetMap"),
    ("dataset", "matrix_name", "downloads from SuiteSparse"),
    ("world", "hdri", "reads"),
    ("meta", "output_dir", "writes into"),
)


class SafetyReport:
    def __init__(self):
        self.removed = []
        self.effects = []

    @property
    def ok(self):
        return not self.removed

    def __repr__(self):  # pragma: no cover
        return "SafetyReport(removed=%r, effects=%r)" % (self.removed, self.effects)


def sanitize(spec):
    """Copy with unbounded parts stripped, plus a report; strip rather than reject."""
    report = SafetyReport()
    if not isinstance(spec, dict):
        return {}, report

    clean = copy.deepcopy(spec)

    for section in DENIED_SECTIONS:
        if section in clean:
            del clean[section]
            report.removed.append(section)

    for section, fields in DENIED_FIELDS.items():
        block = clean.get(section)
        if not isinstance(block, dict):
            continue
        for field in fields:
            if field in block:
                del block[field]
                report.removed.append("%s.%s" % (section, field))

    report.effects = describe_effects(clean)
    return clean, report


def describe_effects(spec, base_dir=None):
    """What running would read, download, and write (paths absolute)."""
    effects = []
    if not isinstance(spec, dict):
        return effects

    for section, field, verb in EFFECT_FIELDS:
        block = spec.get(section)
        if not isinstance(block, dict):
            continue
        value = block.get(field)
        if not value or not isinstance(value, str):
            continue
        shown = value
        if verb in ("reads", "writes into"):
            shown = _resolve(value, base_dir)
        effects.append((verb, shown))

    return effects


def _resolve(path, base_dir):
    """Resolve a path the way the executor will."""
    if path.startswith("//"):
        path = path[2:]
        if base_dir:
            return os.path.abspath(os.path.join(base_dir, path))
        return "<specification folder>/" + path
    if os.path.isabs(path):
        return path
    if base_dir:
        return os.path.abspath(os.path.join(base_dir, path))
    return path


# Fields that apply per dataset source (models fill every schema field).
SOURCE_FIELDS = {
    "osmnx": ("source", "method", "query", "network_type", "simplify",
              "cache", "retain_all"),
    "gexf": ("source", "filepath", "auto_layout"),
    "graphml": ("source", "filepath", "auto_layout"),
    "csv": ("source", "filepath", "auto_layout"),
    "suitesparse": ("source", "matrix_name"),
    "city2graph": ("source", "bbox", "layers"),
    "sql": ("source",),
}

# Real coordinates already; a layout would overwrite geography.
POSITIONED_SOURCES = ("osmnx", "city2graph")


def normalize(spec):
    """Drop inapplicable fields and common model mistakes. Returns (clean, notes)."""
    notes = []
    if not isinstance(spec, dict):
        return {}, notes

    clean = copy.deepcopy(spec)

    dataset = clean.get("dataset")
    source = dataset.get("source") if isinstance(dataset, dict) else None

    if isinstance(dataset, dict) and source in SOURCE_FIELDS:
        keep = SOURCE_FIELDS[source]
        for field in [f for f in dataset if f not in keep]:
            del dataset[field]
            notes.append("dataset.%s does not apply to source '%s'" % (field, source))

    if source in POSITIONED_SOURCES and "layout" in clean:
        del clean["layout"]
        notes.append(
            "removed layout: a %s dataset carries its own coordinates, and a "
            "layout would overwrite the geography" % source
        )

    for section in list(clean):
        block = clean[section]
        if not isinstance(block, dict):
            continue
        for field in [f for f, v in block.items() if _is_empty(v)]:
            del block[field]
        if not block:
            del clean[section]
            notes.append("removed empty %s section" % section)

    return clean, notes


def _is_empty(value):
    return value is None or value == "" or value == []


# Models sometimes put Overpass-style text in a place-name query.
QUERY_LANGUAGE_MARKERS = ("[", "]", "=", "highway", "way(", "node(", "->")


def check_semantics(spec):
    """Problems the schema misses but that will fail at run time."""
    problems = []
    if not isinstance(spec, dict):
        return problems

    dataset = spec.get("dataset") or {}
    source = dataset.get("source")

    if source == "osmnx":
        method = dataset.get("method", "PLACE")
        query = dataset.get("query") or ""
        if not query:
            problems.append(
                "dataset.query is empty; OSMnx needs a place name "
                "(method is %s)" % method
            )
        elif method == "PLACE" and any(m in query for m in QUERY_LANGUAGE_MARKERS):
            problems.append(
                "dataset.query looks like a query language expression, not a "
                "place name: %r" % query[:60]
            )

    if source in ("gexf", "graphml", "csv") and not dataset.get("filepath"):
        problems.append("dataset.filepath is empty; nothing to read")

    if source == "suitesparse" and not dataset.get("matrix_name"):
        problems.append("dataset.matrix_name is empty; nothing to download")

    labels = spec.get("labels") or {}
    if labels.get("max_distance") and not labels.get("max_count"):
        problems.append(
            "labels.max_distance is a distance from the camera, not a count. "
            "To label the most important nodes set labels.max_count"
        )

    return problems
