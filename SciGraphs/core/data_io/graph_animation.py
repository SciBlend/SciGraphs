"""The `.sgraphs` file: a graph, how it moves, and who made it.

Every row is tagged with what it is, and positions refer to nodes by name:

    record,source,target,stage,frame,x,y,z
    meta,format,sgraphs/1,,,,,
    meta,nodes,5,,,,,
    edge,a,b,,,,,
    pos,a,,0,1,-2.0,0.0,0.0

``meta`` is a header field, key in ``source`` and value in ``target``. The
first must be ``format``. ``nodes``, ``edges`` and ``stages`` are checked
against the body. Unknown keys are kept and ignored.

``edge`` connects ``source`` to ``target`` and declares both as nodes.
``node`` declares one node on its own, for isolated nodes or attributes.
``pos`` places ``source`` at ``(x, y, z)`` during ``stage``. ``frame`` pins
the stage to a frame and ``size`` multiplies the node radius; give either for
every row or none.

Any further column is an attribute: on ``edge`` rows an edge attribute, on
``node`` rows a node attribute.

Stages, not frames. Blender interpolates between the poses.

Nothing here imports bpy, so the reader also runs outside Blender.
"""

from __future__ import annotations

import csv
from collections import OrderedDict

RESERVED = ("record", "source", "target", "stage", "frame", "x", "y", "z",
            "size")

RECORDS = ("meta", "edge", "node", "pos")

EXTENSION = ".sgraphs"
FORMAT_ID = "sgraphs"
FORMAT_VERSION = 1

META_KEYS = ("format", "generator", "blender", "created", "title", "nodes",
             "edges", "stages", "directed", "frame_start", "frame_end", "fps",
             "space", "up_axis")


def scigraphs_version():
    """The extension's version string, read from its manifest."""
    from pathlib import Path

    here = Path(__file__).resolve()
    for parent in here.parents:
        manifest = parent / "blender_manifest.toml"
        if not manifest.is_file():
            continue
        for line in manifest.read_text().splitlines():
            if line.strip().startswith("version"):
                _, _, value = line.partition("=")
                return value.strip().strip('"\'')
    return "unknown"


class GraphAnimationError(ValueError):
    """The file is not a usable graph animation, with the reason attached."""


class GraphAnimation:
    """Topology, attributes and motion, as read from one file.

    Attributes:
        nodes: node names in declaration order, which is the vertex order.
        edges: ``(source, target, attrs)`` triples.
        node_attrs: ``{node: {name: value}}``.
        stages: ``[{node: (x, y, z)}]``, in stage order.
        frames: ``[int]`` aligned with ``stages``, or None when the file does
            not pin them.
        sizes: ``[{node: float}]`` aligned with ``stages``, or None.
        meta: the header fields. ``format`` is always present; a file with no
            header reads as ``sgraphs/0``.
    
    """

    def __init__(self, nodes, edges, node_attrs, stages, frames, sizes=None,
                 meta=None):
        self.nodes = nodes
        self.edges = edges
        self.node_attrs = node_attrs
        self.stages = stages
        self.frames = frames
        self.sizes = sizes
        self.meta = meta or {}

    def __repr__(self):
        return (f"GraphAnimation({len(self.nodes)} nodes, "
                f"{len(self.edges)} edges, {len(self.stages)} stages)")

    def positions(self, stage):
        """Stage ``stage`` as a list of ``(x, y, z)`` in node order."""
        poses = self.stages[stage]
        return [poses[name] for name in self.nodes]

    def frames_over(self, first, last):
        """The frame of each stage, honouring the file or spreading evenly."""
        if self.frames is not None:
            return list(self.frames)
        if len(self.stages) == 1:
            return [first]
        span = max(last - first, 1)
        step = span / (len(self.stages) - 1)
        return [first + int(round(i * step)) for i in range(len(self.stages))]


def _number(row, key, record, line):
    text = row.get(key)
    if text in (None, ""):
        raise GraphAnimationError(
            f"line {line}: a '{record}' row needs a value in '{key}'")
    try:
        return float(text)
    except ValueError:
        raise GraphAnimationError(
            f"line {line}: '{key}' is {text!r}, which is not a number") from None


def _extras(row, fieldnames):
    """The attribute columns of a row, skipping blanks."""
    out = {}
    for key in fieldnames:
        if key in RESERVED or key is None:
            continue
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except ValueError:
            out[key] = value
        else:
            out[key] = int(number) if number.is_integer() else number
    return out


def read(path):
    """Parse a graph-animation file, or raise `GraphAnimationError`."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        if "record" not in fieldnames:
            raise GraphAnimationError(
                "no 'record' column: this is not a graph-animation file. A "
                "plain trajectory (stage,node,x,y,z) goes through "
                "Layout > Animation instead")

        meta = OrderedDict()
        seen_data = False
        nodes = OrderedDict()
        edges = []
        node_attrs = {}
        by_stage = OrderedDict()
        by_size = OrderedDict()
        stage_frames = {}

        for line, row in enumerate(reader, start=2):
            kind = (row.get("record") or "").strip().lower()
            if not kind or kind.startswith("#"):
                continue
            if kind not in RECORDS:
                raise GraphAnimationError(
                    f"line {line}: record {kind!r} is not one of {RECORDS}")

            source = (row.get("source") or "").strip()
            if not source:
                raise GraphAnimationError(
                    f"line {line}: a '{kind}' row needs a name in 'source'")

            if kind == "meta":
                if seen_data:
                    raise GraphAnimationError(
                        f"line {line}: header field {source!r} comes after the "
                        f"data; every 'meta' row belongs at the top of the file")
                if not meta and source != "format":
                    raise GraphAnimationError(
                        f"line {line}: the first header field must be "
                        f"'format', not {source!r}")
                meta[source] = (row.get("target") or "").strip()
                if source == "format":
                    _check_format(meta["format"], line)
                continue

            seen_data = True

            if kind == "edge":
                target = (row.get("target") or "").strip()
                if not target:
                    raise GraphAnimationError(
                        f"line {line}: an 'edge' row needs a 'target'")
                nodes.setdefault(source)
                nodes.setdefault(target)
                edges.append((source, target, _extras(row, fieldnames)))

            elif kind == "node":
                nodes.setdefault(source)
                attrs = _extras(row, fieldnames)
                if attrs:
                    node_attrs.setdefault(source, {}).update(attrs)

            else:
                stage = int(_number(row, "stage", kind, line))
                point = (_number(row, "x", kind, line),
                         _number(row, "y", kind, line),
                         _number(row, "z", kind, line))
                by_stage.setdefault(stage, {})[source] = point
                if row.get("size") not in (None, ""):
                    by_size.setdefault(stage, {})[source] = _number(
                        row, "size", kind, line)
                if row.get("frame") not in (None, ""):
                    frame = int(_number(row, "frame", kind, line))
                    if stage_frames.setdefault(stage, frame) != frame:
                        raise GraphAnimationError(
                            f"line {line}: stage {stage} is pinned to frame "
                            f"{stage_frames[stage]} elsewhere and to {frame} "
                            f"here")

    if "format" not in meta:
        meta["format"] = f"{FORMAT_ID}/0"

    if not nodes:
        raise GraphAnimationError("the file declares no nodes")

    _check_counts(meta, nodes, [e for e in edges], by_stage)

    _validate(nodes, by_stage)

    order = sorted(by_stage)
    stages = [by_stage[s] for s in order]
    frames = None
    if stage_frames:
        if len(stage_frames) != len(order):
            missing = sorted(set(order) - set(stage_frames))
            raise GraphAnimationError(
                f"stages {missing} have no 'frame' while others do; pin every "
                f"stage or none, because a half-pinned file re-times itself")
        frames = [stage_frames[s] for s in order]
        if any(b <= a for a, b in zip(frames, frames[1:])):
            raise GraphAnimationError(
                f"the pinned frames do not increase: {frames}")

    sizes = None
    if by_size:
        missing = [s for s in order
                   if set(by_size.get(s, {})) != set(nodes)]
        if missing:
            raise GraphAnimationError(
                f"stages {missing[:5]} have an incomplete 'size' column while "
                f"others have one; a partial column would draw the rows that "
                f"lack it at zero radius")
        sizes = [by_size[s] for s in order]

    return GraphAnimation(list(nodes), edges, node_attrs, stages, frames,
                          sizes, dict(meta))


def _check_format(value, line):
    """Refuse a file this reader does not understand, and say which it is."""
    name, _, version = value.partition("/")
    if name != FORMAT_ID:
        raise GraphAnimationError(
            f"line {line}: format is {value!r}; this reads "
            f"{FORMAT_ID!r} files")
    try:
        major = int(version.split(".")[0])
    except ValueError:
        raise GraphAnimationError(
            f"line {line}: format {value!r} has no version number") from None
    if major > FORMAT_VERSION:
        raise GraphAnimationError(
            f"line {line}: this file is {FORMAT_ID}/{major} and this build "
            f"reads up to {FORMAT_ID}/{FORMAT_VERSION}. Update SciGraphs "
            f"rather than let an older reader guess at newer records")


def _check_counts(meta, nodes, edges, stages):
    """The declared sizes against the real ones."""
    for key, got in (("nodes", len(nodes)), ("edges", len(edges)),
                     ("stages", len(stages))):
        if key not in meta or meta[key] == "":
            continue
        try:
            want = int(meta[key])
        except ValueError:
            raise GraphAnimationError(
                f"header field {key!r} is {meta[key]!r}, not a number") from None
        if want != got:
            raise GraphAnimationError(
                f"the header says {want} {key} and the file contains {got}. "
                f"Either it is truncated, or it was edited by hand and the "
                f"'meta,{key},...' row was not updated")


def _validate(nodes, by_stage):
    known = set(nodes)
    for stage in sorted(by_stage):
        placed = set(by_stage[stage])

        stray = sorted(placed - known)
        if stray:
            raise GraphAnimationError(
                f"stage {stage} places node(s) {stray[:5]} that no 'edge' or "
                f"'node' row declares")

        absent = sorted(known - placed)
        if absent:
            raise GraphAnimationError(
                f"stage {stage} places {len(placed)} of {len(known)} nodes "
                f"(missing {absent[:5]}{'...' if len(absent) > 5 else ''}); a "
                f"pose needs every node, not most of them")

    if len(by_stage) == 1:
        raise GraphAnimationError(
            "only one stage: that is a layout, not an animation. Import it as "
            "a graph, or add the poses it moves to")


def write(path, nodes, edges, stages, frames=None, node_attrs=None,
          edge_attrs=None, sizes=None, precision=5, title=None,
          directed=False, fps=None, blender=None, meta=None):
    """Write a `.sgraphs` file. The inverse of `read`, and used to test it."""
    import datetime
    node_attrs = node_attrs or {}
    edge_attrs = edge_attrs or {}

    extra = []
    for mapping in list(node_attrs.values()) + list(edge_attrs.values()):
        for key in mapping:
            if key not in extra and key not in RESERVED:
                extra.append(key)

    if frames is not None and len(frames) != len(stages):
        raise ValueError(f"{len(frames)} frames for {len(stages)} stages")
    if sizes is not None and len(sizes) != len(stages):
        raise ValueError(f"{len(sizes)} size maps for {len(stages)} stages")

    header = OrderedDict()
    header["format"] = f"{FORMAT_ID}/{FORMAT_VERSION}"
    header["generator"] = f"SciGraphs {scigraphs_version()}"
    if blender:
        header["blender"] = str(blender)
    header["created"] = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if title:
        header["title"] = str(title)
    header["nodes"] = len(nodes)
    header["edges"] = len(edges)
    header["stages"] = len(stages)
    header["directed"] = "true" if directed else "false"
    if frames is not None:
        header["frame_start"] = frames[0]
        header["frame_end"] = frames[-1]
    if fps:
        header["fps"] = fps
    header["space"] = "local"
    header["up_axis"] = "Z"
    for key, value in (meta or {}).items():
        if key not in header:
            header[key] = value

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESERVED) + extra,
                                restval="")
        writer.writeheader()

        for key, value in header.items():
            writer.writerow({"record": "meta", "source": key,
                             "target": value})

        for source, target in edges:
            row = {"record": "edge", "source": source, "target": target}
            row.update(edge_attrs.get((source, target), {}))
            writer.writerow(row)

        for node in nodes:
            attrs = node_attrs.get(node)
            touched = any(node in (a, b) for a, b in edges)
            if attrs or not touched:
                row = {"record": "node", "source": node}
                row.update(attrs or {})
                writer.writerow(row)

        for index, poses in enumerate(stages):
            for node in nodes:
                x, y, z = poses[node]
                row = {"record": "pos", "source": node, "stage": index,
                       "x": f"{x:.{precision}f}", "y": f"{y:.{precision}f}",
                       "z": f"{z:.{precision}f}"}
                if frames is not None:
                    row["frame"] = frames[index]
                if sizes is not None:
                    row["size"] = f"{sizes[index][node]:.{precision}f}"
                writer.writerow(row)

    return path
