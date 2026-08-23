"""Compare two pipeline runs: provenance manifests, or exported coordinates.

Manifest mode reports the spec hash, the seed, a per-dependency version diff and
a digest table keyed by each artifact's path *relative to its own run's
output_dir*, so byte-identical runs in different directories still match.
Coordinate mode reports per-node deviation and topology change.

    python3 scripts/repro/compare_runs.py manifests A/run_manifest.json B/run_manifest.json
    python3 scripts/repro/compare_runs.py positions A/positions.csv B/positions.csv

Exits 0 when the runs agree, 1 when they do not, 2 on a usage or read error.
Needs no bpy."""

import argparse
import csv
import json
import math
import os
import statistics
import sys


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _output_root(manifest, manifest_path):
    """Directory the recorded artifact paths are relative to: the manifest's
    own record, else its directory, else the paths' longest common prefix."""
    recorded = manifest.get("output_dir") or ""
    if recorded:
        return recorded

    here = os.path.dirname(os.path.abspath(manifest_path))
    paths = [o.get("path", "") for o in manifest.get("outputs", [])]
    if paths and all(p.startswith(here + os.sep) for p in paths):
        return here
    if len(paths) > 1:
        common = os.path.commonpath(paths)
        return common if os.path.basename(common) else here
    if len(paths) == 1:
        return os.path.dirname(paths[0])
    return here


def _relativize(path, root):
    """Path relative to root, or the basename when it lies outside root."""
    if not path:
        return path
    try:
        rel = os.path.relpath(path, root)
    except ValueError:  # different drives on Windows
        return os.path.basename(path)
    if rel.startswith(".."):
        return os.path.basename(path)
    return rel.replace(os.sep, "/")


def _artifact_table(manifest, root):
    table = {}
    for entry in manifest.get("outputs", []):
        table[_relativize(entry.get("path", ""), root)] = {
            "hash": entry.get("hash"),
            "size": entry.get("size"),
            "type": entry.get("type"),
        }
    return table


def _input_table(manifest, root):
    table = {}
    for entry in manifest.get("inputs", []):
        path = entry.get("path", "")
        # Network inputs are identifiers, not filesystem paths; keep them whole.
        key = path if "://" in path else _relativize(path, root)
        table[key] = {"hash": entry.get("hash"), "pinned": entry.get("pinned")}
    return table


def compare_manifest_files(path_a, path_b, verbose=True):
    man_a, man_b = _load(path_a), _load(path_b)
    root_a, root_b = _output_root(man_a, path_a), _output_root(man_b, path_b)

    report = {
        "manifest_a": os.path.abspath(path_a),
        "manifest_b": os.path.abspath(path_b),
        "output_dir_a": root_a,
        "output_dir_b": root_b,
        "spec_hash_a": man_a.get("pipeline_hash"),
        "spec_hash_b": man_b.get("pipeline_hash"),
        "spec_hash_agree": man_a.get("pipeline_hash") == man_b.get("pipeline_hash"),
        "seed_a": man_a.get("seed"),
        "seed_b": man_b.get("seed"),
        "seed_agree": man_a.get("seed") == man_b.get("seed"),
        "success_a": man_a.get("success"),
        "success_b": man_b.get("success"),
    }

    env_a = man_a.get("environment", {})
    env_b = man_b.get("environment", {})
    env_diff = []
    for key in sorted(set(env_a) | set(env_b)):
        if key == "timestamp":
            continue  # always differs; not a reproducibility signal
        if env_a.get(key) != env_b.get(key):
            env_diff.append({"field": key, "a": env_a.get(key), "b": env_b.get(key)})
    report["environment_differences"] = env_diff

    deps_a = {d["name"]: d["version"] for d in man_a.get("dependencies", [])}
    deps_b = {d["name"]: d["version"] for d in man_b.get("dependencies", [])}
    dep_diff = []
    for name in sorted(set(deps_a) | set(deps_b)):
        va, vb = deps_a.get(name), deps_b.get(name)
        if va != vb:
            dep_diff.append({"package": name, "a": va, "b": vb})
    report["dependency_count_a"] = len(deps_a)
    report["dependency_count_b"] = len(deps_b)
    report["dependency_differences"] = dep_diff

    in_a = _input_table(man_a, root_a)
    in_b = _input_table(man_b, root_b)
    input_rows = []
    for key in sorted(set(in_a) | set(in_b)):
        a, b = in_a.get(key), in_b.get(key)
        if a is None:
            verdict = "only_in_b"
        elif b is None:
            verdict = "only_in_a"
        elif a["hash"] is None or b["hash"] is None:
            verdict = "unhashed"  # network input: no digest was ever recorded
        elif a["hash"] == b["hash"]:
            verdict = "identical"
        else:
            verdict = "differs"
        input_rows.append({
            "path": key,
            "hash_a": (a or {}).get("hash"),
            "hash_b": (b or {}).get("hash"),
            "verdict": verdict,
        })
    report["inputs"] = input_rows

    # Keyed relative to each run's own output_dir.
    out_a = _artifact_table(man_a, root_a)
    out_b = _artifact_table(man_b, root_b)
    rows = []
    for key in sorted(set(out_a) | set(out_b)):
        a, b = out_a.get(key), out_b.get(key)
        if a is None:
            verdict = "only_in_b"
        elif b is None:
            verdict = "only_in_a"
        elif a["hash"] == b["hash"]:
            verdict = "identical"
        else:
            verdict = "differs"
        rows.append({
            "path": key,
            "hash_a": (a or {}).get("hash"),
            "hash_b": (b or {}).get("hash"),
            "size_a": (a or {}).get("size"),
            "size_b": (b or {}).get("size"),
            "verdict": verdict,
        })
    report["artifacts"] = rows
    report["artifact_count_a"] = len(out_a)
    report["artifact_count_b"] = len(out_b)
    report["artifacts_identical"] = bool(rows) and all(
        r["verdict"] == "identical" for r in rows
    )

    steps_a = man_a.get("steps", [])
    steps_b = man_b.get("steps", [])
    report["steps_a"] = [
        {"name": s.get("name"), "status": s.get("status"),
         "duration_ms": s.get("duration_ms")} for s in steps_a
    ]
    report["steps_b"] = [
        {"name": s.get("name"), "status": s.get("status"),
         "duration_ms": s.get("duration_ms")} for s in steps_b
    ]

    report["reproducible"] = bool(
        report["spec_hash_agree"]
        and report["seed_agree"]
        and report["artifacts_identical"]
        and not any(r["verdict"] == "differs" for r in input_rows)
    )

    if verbose:
        _print_manifest_report(report)
    return report


def _print_manifest_report(r):
    def mark(ok):
        return "OK  " if ok else "FAIL"

    print("=" * 78)
    print("A: %s" % r["manifest_a"])
    print("B: %s" % r["manifest_b"])
    print("   output_dir A: %s" % r["output_dir_a"])
    print("   output_dir B: %s" % r["output_dir_b"])
    print("-" * 78)
    print("%s spec hash    %s" % (mark(r["spec_hash_agree"]), r["spec_hash_a"]))
    if not r["spec_hash_agree"]:
        print("              vs %s" % r["spec_hash_b"])
    print("%s seed         %s vs %s" % (mark(r["seed_agree"]), r["seed_a"], r["seed_b"]))
    print("     success      %s vs %s" % (r["success_a"], r["success_b"]))

    print("-" * 78)
    print("environment (%d differing field(s))" % len(r["environment_differences"]))
    for d in r["environment_differences"]:
        print("     %-22s %s | %s" % (d["field"], d["a"], d["b"]))

    print("-" * 78)
    print("dependencies: %d recorded in A, %d in B, %d differing"
          % (r["dependency_count_a"], r["dependency_count_b"],
             len(r["dependency_differences"])))
    for d in r["dependency_differences"]:
        print("     %-24s %-14s | %-14s" % (d["package"], d["a"], d["b"]))

    print("-" * 78)
    print("inputs")
    for row in r["inputs"]:
        print("     %-10s %s" % (row["verdict"], row["path"]))

    print("-" * 78)
    print("artifacts (path relative to each run's output_dir)")
    print("     %-9s %-34s %-16s %-16s" % ("verdict", "path", "digest A", "digest B"))
    for row in r["artifacts"]:
        ha = (row["hash_a"] or "-")[:16]
        hb = (row["hash_b"] or "-")[:16]
        print("     %-9s %-34s %-16s %-16s" % (row["verdict"], row["path"], ha, hb))
    if not r["artifacts"]:
        print("     (no outputs recorded in either manifest)")

    print("-" * 78)
    print("VERDICT: %s" % ("reproducible" if r["reproducible"] else "NOT reproducible"))
    print("=" * 78)


def _read_positions(path):
    """Read a positions.csv export into {node_id: (x, y, z)}, taking columns
    by name from the header row or the first four positionally."""
    rows = {}
    order = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        first = None
        for record in reader:
            if not record or all(not c.strip() for c in record):
                continue
            if first is None:
                first = record
                lowered = [c.strip().lower() for c in record]
                if any(c in ("x", "pos_x", "px") for c in lowered):
                    idx_id = 0
                    for cand in ("id", "node", "node_id", "name", "label"):
                        if cand in lowered:
                            idx_id = lowered.index(cand)
                            break
                    idx = [idx_id]
                    for axis in ("x", "y", "z"):
                        for cand in (axis, "pos_" + axis, "p" + axis):
                            if cand in lowered:
                                idx.append(lowered.index(cand))
                                break
                    if len(idx) == 4:
                        columns = idx
                        continue
                columns = [0, 1, 2, 3]
                # first row was data, not a header: fall through and use it
            try:
                node = record[columns[0]].strip()
                xyz = tuple(float(record[columns[i]]) for i in (1, 2, 3))
            except (ValueError, IndexError):
                continue
            rows[node] = xyz
            order.append(node)
    return rows, order


def _edges_from_gexf(path):
    """Undirected edge set of a GEXF export, using only the standard library."""
    import xml.etree.ElementTree as ET

    edges = set()
    root = ET.parse(path).getroot()
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "edge":
            continue
        src = element.get("source")
        tgt = element.get("target")
        if src is not None and tgt is not None:
            edges.add(tuple(sorted((str(src), str(tgt)))))
    return edges


def _read_edges(path):
    """Read an edge export beside positions.csv: an explicit edge CSV first,
    then any GEXF or GraphML in the same directory."""
    base = os.path.dirname(path)
    for name in ("edges.csv", "edge_list.csv"):
        candidate = os.path.join(base, name)
        if os.path.isfile(candidate):
            edges = set()
            with open(candidate, "r", encoding="utf-8", newline="") as handle:
                for record in csv.reader(handle):
                    if len(record) >= 2:
                        a, b = record[0].strip(), record[1].strip()
                        if a and b:
                            edges.add(tuple(sorted((a, b))))
            return edges, candidate

    try:
        siblings = sorted(os.listdir(base))
    except OSError:
        return None, None
    for name in siblings:
        if name.lower().endswith((".gexf", ".graphml")):
            candidate = os.path.join(base, name)
            try:
                return _edges_from_gexf(candidate), candidate
            except Exception:
                continue
    return None, None


def compare_positions(path_a, path_b, verbose=True):
    pos_a, order_a = _read_positions(path_a)
    pos_b, order_b = _read_positions(path_b)

    shared = sorted(set(pos_a) & set(pos_b))
    only_a = sorted(set(pos_a) - set(pos_b))
    only_b = sorted(set(pos_b) - set(pos_a))

    deviations = []
    per_axis_max = [0.0, 0.0, 0.0]
    for node in shared:
        a, b = pos_a[node], pos_b[node]
        for i in range(3):
            per_axis_max[i] = max(per_axis_max[i], abs(a[i] - b[i]))
        deviations.append(math.dist(a, b))

    with open(path_a, "rb") as fh:
        bytes_a = fh.read()
    with open(path_b, "rb") as fh:
        bytes_b = fh.read()

    edges_a, edge_path_a = _read_edges(path_a)
    edges_b, edge_path_b = _read_edges(path_b)

    report = {
        "positions_a": os.path.abspath(path_a),
        "positions_b": os.path.abspath(path_b),
        "byte_identical": bytes_a == bytes_b,
        "n_nodes_a": len(pos_a),
        "n_nodes_b": len(pos_b),
        "n_shared": len(shared),
        "only_in_a": only_a[:20],
        "only_in_b": only_b[:20],
        "node_set_identical": not only_a and not only_b,
        "row_order_identical": order_a == order_b,
        "max_deviation": max(deviations) if deviations else None,
        "median_deviation": statistics.median(deviations) if deviations else None,
        "mean_deviation": statistics.fmean(deviations) if deviations else None,
        "max_axis_deviation": per_axis_max,
        "n_nodes_moved": sum(1 for d in deviations if d > 0.0),
    }

    if edges_a is not None and edges_b is not None:
        degrees_a = {}
        degrees_b = {}
        for u, v in edges_a:
            degrees_a[u] = degrees_a.get(u, 0) + 1
            degrees_a[v] = degrees_a.get(v, 0) + 1
        for u, v in edges_b:
            degrees_b[u] = degrees_b.get(u, 0) + 1
            degrees_b[v] = degrees_b.get(v, 0) + 1
        report["edges_source_a"] = edge_path_a
        report["edges_source_b"] = edge_path_b
        report["n_edges_a"] = len(edges_a)
        report["n_edges_b"] = len(edges_b)
        report["edge_set_identical"] = edges_a == edges_b
        report["degree_sequence_identical"] = (
            sorted(degrees_a.values()) == sorted(degrees_b.values())
        )
        report["topology_unchanged"] = (
            report["node_set_identical"]
            and report["edge_set_identical"]
            and report["degree_sequence_identical"]
        )
    else:
        # No edge export beside the positions: node identity is all we can check.
        report["edge_set_identical"] = None
        report["degree_sequence_identical"] = None
        report["topology_unchanged"] = None

    if verbose:
        _print_position_report(report)
    return report


def _print_position_report(r):
    print("=" * 78)
    print("A: %s" % r["positions_a"])
    print("B: %s" % r["positions_b"])
    print("-" * 78)
    print("byte-identical file        : %s" % r["byte_identical"])
    print("nodes                      : %d in A, %d in B, %d shared"
          % (r["n_nodes_a"], r["n_nodes_b"], r["n_shared"]))
    print("node set identical         : %s" % r["node_set_identical"])
    print("row order identical        : %s" % r["row_order_identical"])
    if r["max_deviation"] is None:
        print("no shared nodes: no deviation computed")
    else:
        print("nodes with any deviation   : %d / %d" % (r["n_nodes_moved"], r["n_shared"]))
        print("max  |deviation| (euclid)  : %.17g" % r["max_deviation"])
        print("median |deviation|         : %.17g" % r["median_deviation"])
        print("mean |deviation|           : %.17g" % r["mean_deviation"])
        print("max per-axis |dx|,|dy|,|dz|: %.17g, %.17g, %.17g" % tuple(r["max_axis_deviation"]))
    if r["topology_unchanged"] is None:
        print("topology                   : no edge export found beside positions.csv")
    else:
        print("edges                      : %d in A, %d in B" % (r["n_edges_a"], r["n_edges_b"]))
        print("edge set identical         : %s" % r["edge_set_identical"])
        print("degree sequence identical  : %s" % r["degree_sequence_identical"])
        print("topology unchanged         : %s" % r["topology_unchanged"])
    print("=" * 78)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="compare_runs.py",
        description="Compare two SciGraphs pipeline runs.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_man = sub.add_parser("manifests", help="compare two run_manifest.json files")
    p_man.add_argument("a")
    p_man.add_argument("b")
    p_man.add_argument("--json", metavar="PATH", help="also write the report as JSON")

    p_pos = sub.add_parser("positions", help="compare two positions.csv exports")
    p_pos.add_argument("a")
    p_pos.add_argument("b")
    p_pos.add_argument("--json", metavar="PATH", help="also write the report as JSON")
    p_pos.add_argument("--tolerance", type=float, default=0.0,
                       help="max euclidean deviation still counted as agreement")

    args = parser.parse_args(argv)

    try:
        if args.mode == "manifests":
            report = compare_manifest_files(args.a, args.b)
            agreed = report["reproducible"]
        else:
            report = compare_positions(args.a, args.b)
            worst = report["max_deviation"]
            agreed = (
                report["node_set_identical"]
                and worst is not None
                and worst <= args.tolerance
            )
    except (OSError, ValueError, KeyError) as exc:
        print("compare_runs: %s" % exc, file=sys.stderr)
        return 2

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    return 0 if agreed else 1


if __name__ == "__main__":
    sys.exit(main())
