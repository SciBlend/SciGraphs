#!/usr/bin/env python3
"""Generate SciGraphs reproducible pipeline files.

Examples:
    python3 scripts/generate_pipeline.py --preset osmnx --output examples/pipelines/my_pipeline.json
    python3 scripts/generate_pipeline.py --preset gexf --title figure_3 --seed 123 --output figure_3.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _base_meta(title: str, seed: int, output_dir: str) -> dict[str, Any]:
    return {
        "title": title,
        "seed": seed,
        "output_dir": output_dir,
        "description": "Generated SciGraphs reproducible pipeline",
        "version": "1.0",
    }


def build_pipeline(
    preset: str,
    title: str,
    seed: int,
    output_dir: str,
    dataset_path: str,
    place: str,
) -> dict[str, Any]:
    """Return a pipeline dictionary for a named preset."""
    pipeline: dict[str, Any] = {"meta": _base_meta(title, seed, output_dir)}

    if preset == "minimal":
        return pipeline

    if preset == "osmnx":
        pipeline.update(
            {
                "dataset": {
                    "source": "osmnx",
                    "method": "PLACE",
                    "query": place,
                    "network_type": "walk",
                    "simplify": True,
                    "cache": True,
                },
                "analysis": {
                    "metrics": ["degree", "betweenness", "closeness"],
                    "clustering": {"algorithm": "louvain", "resolution": 1.0},
                    "normalize": True,
                },
                "layout": {"algorithm": "YIFAN_HU", "scale": 5.0, "iterations": 50},
                "visual": {
                    "setup_geometry_nodes": True,
                    "node_color": "betweenness",
                    "node_size": "degree",
                    "edge_style": "GEPHI_DEFAULT",
                    "colormap": "viridis",
                },
            }
        )
    elif preset == "gexf":
        pipeline.update(
            {
                "dataset": {
                    "source": "gexf",
                    "filepath": dataset_path,
                    "auto_layout": True,
                },
                "layout": {"algorithm": "SPRING_3D", "scale": 5.0, "iterations": 100},
                "visual": {
                    "setup_geometry_nodes": True,
                    "node_color": "degree",
                    "edge_style": "CYTOSCAPE_BEZIER",
                },
            }
        )
    elif preset == "suitesparse":
        pipeline.update(
            {
                "dataset": {"source": "suitesparse", "matrix_name": "HB/bcsstk01"},
                "analysis": {
                    "metrics": ["degree", "betweenness", "eigenvector"],
                    "clustering": {"algorithm": "leiden", "resolution": 1.0},
                    "normalize": True,
                },
                "layout": {"algorithm": "IGRAPH_DRL", "scale": 10.0, "iterations": 100},
                "visual": {
                    "setup_geometry_nodes": True,
                    "node_color": "eigenvector",
                    "node_size": "degree",
                    "edge_style": "GRAPHVIZ_SPLINE",
                },
            }
        )
    else:
        raise ValueError(f"Unknown preset: {preset}")

    pipeline["render"] = {
        "engine": "CYCLES",
        "resolution": [3840, 2160],
        "samples": 256,
        "output": f"{title}.png",
        "transparent": False,
        "denoise": True,
    }
    pipeline["exports"] = {
        "graph": "graph.gexf",
        "positions": "positions.csv",
        "statistics": "statistics.txt",
    }
    return pipeline


def save_pipeline(pipeline: dict[str, Any], output: Path) -> None:
    """Save a pipeline as JSON or YAML based on file extension."""
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise SystemExit("PyYAML is required to write YAML. Use a .json output instead.") from exc

        output.write_text(
            yaml.safe_dump(pipeline, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        return

    output.write_text(json.dumps(pipeline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a SciGraphs reproducible pipeline file.")
    parser.add_argument(
        "--preset",
        choices=("minimal", "osmnx", "gexf", "suitesparse"),
        default="osmnx",
        help="Pipeline preset to generate.",
    )
    parser.add_argument("--title", default="generated_pipeline", help="Pipeline title.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument(
        "--output-dir",
        default="//repro/generated_pipeline",
        help="Output directory stored in the pipeline spec.",
    )
    parser.add_argument(
        "--dataset-path",
        default="//data/network.gexf",
        help="Dataset path for file-based presets.",
    )
    parser.add_argument(
        "--place",
        default="Burjassot, Valencia, Spain",
        help="Place query for the OSMnx preset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/pipelines/generated_pipeline.json"),
        help="Pipeline file to write (.json, .yaml, or .yml).",
    )
    args = parser.parse_args()

    pipeline = build_pipeline(
        preset=args.preset,
        title=args.title,
        seed=args.seed,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        place=args.place,
    )
    save_pipeline(pipeline, args.output)
    print(f"Pipeline written to {args.output}")


if __name__ == "__main__":
    main()
