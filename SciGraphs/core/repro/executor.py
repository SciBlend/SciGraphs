# Pipeline executor for reproducible SciGraphs workflows
#
# Executes declarative pipelines with logging, timing, error handling
# and artifact generation.

import datetime
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schema import PipelineSchema, DatasetSpec, AnalysisSpec, LayoutSpec, VisualSpec, RenderSpec, ExportsSpec, OpSpec
from .parser import save_canonical, canonicalize_pipeline
from .determinism import set_pipeline_seed, get_seed_context
from .provenance import (
    ProvenanceManifest, create_manifest, add_input, add_output,
    add_step, finalize_manifest, save_manifest
)
from .registry import get_registry, call_operator, prepare_operator_props


@dataclass
class ExecutionResult:
    """Result of a pipeline execution."""
    success: bool
    pipeline_hash: str
    output_dir: str
    manifest_path: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_ms: int = 0


class PipelineExecutor:
    """
    Executes reproducible pipelines from declarative specifications.

    Handles:
    - Deterministic seeding
    - Step-by-step execution with timing
    - Error handling and recovery
    - Artifact and provenance generation
    """

    def __init__(self, stop_on_error: bool = True, verbose: bool = True):
        """
        Initialize executor.

        Args:
            stop_on_error: Whether to stop execution on first error
            verbose: Whether to log execution progress
        """
        self.stop_on_error = stop_on_error
        self.verbose = verbose
        self.logger = logging.getLogger("scigraphs.repro")
        self._bpy = None
        # Friendly metric name -> actual mesh attribute produced by analysis.
        self._metric_attributes: Dict[str, str] = {}

    def _get_bpy(self):
        """Lazy import of bpy."""
        if self._bpy is None:
            try:
                import bpy
                self._bpy = bpy
            except ImportError:
                raise RuntimeError("Pipeline execution requires Blender (bpy)")
        return self._bpy

    def _log(self, message: str, level: str = "info") -> None:
        """Log a message if verbose."""
        if self.verbose:
            getattr(self.logger, level)(message)
            print(f"[SciGraphs Repro] {message}")

    def _prepare_output_dir(self, output_dir: str) -> str:
        """Prepare output directory, resolving // paths."""
        bpy = self._get_bpy()

        if output_dir.startswith("//"):
            # Blender-style relative path
            blend_path = bpy.data.filepath
            if blend_path:
                base_dir = os.path.dirname(blend_path)
                output_dir = os.path.join(base_dir, output_dir[2:])
            else:
                # No blend file - use temp directory
                import tempfile
                output_dir = os.path.join(tempfile.gettempdir(), "scigraphs_repro", output_dir[2:])

        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def execute(
        self,
        schema: PipelineSchema,
        raw_dict: Dict[str, Any],
        pipeline_hash: str,
    ) -> ExecutionResult:
        """
        Execute a validated pipeline.

        Args:
            schema: Parsed pipeline schema
            raw_dict: Original pipeline dictionary
            pipeline_hash: Pre-computed pipeline hash

        Returns:
            ExecutionResult with status and artifacts
        """
        bpy = self._get_bpy()

        self._metric_attributes = {}

        result = ExecutionResult(
            success=False,
            pipeline_hash=pipeline_hash,
            output_dir="",
        )

        # Prepare output directory
        output_dir = self._prepare_output_dir(schema.meta.output_dir)
        result.output_dir = output_dir

        self._log(f"Executing pipeline: {schema.meta.title}")
        self._log(f"Output directory: {output_dir}")
        self._log(f"Pipeline hash: {pipeline_hash[:16]}...")

        # Set deterministic seed
        seed = schema.meta.seed
        set_pipeline_seed(seed)
        self._log(f"Set global seed: {seed}")

        # Create provenance manifest
        manifest = create_manifest(pipeline_hash, schema.meta.title, seed)

        # Save canonical pipeline
        canonical_path = os.path.join(output_dir, "pipeline.normalized.json")
        save_canonical(raw_dict, canonical_path)
        result.artifacts.append(canonical_path)

        # Execute stages
        try:
            # Dataset stage
            if schema.dataset:
                self._execute_dataset(schema.dataset, manifest, result)
                if result.errors and self.stop_on_error:
                    raise RuntimeError(result.errors[-1])

            # Analysis stage
            if schema.analysis:
                self._execute_analysis(schema.analysis, manifest, result)
                if result.errors and self.stop_on_error:
                    raise RuntimeError(result.errors[-1])

            # Layout stage
            if schema.layout:
                self._execute_layout(schema.layout, seed, manifest, result)
                if result.errors and self.stop_on_error:
                    raise RuntimeError(result.errors[-1])

            # Visual stage
            if schema.visual:
                self._execute_visual(schema.visual, manifest, result)
                if result.errors and self.stop_on_error:
                    raise RuntimeError(result.errors[-1])

            # Generic ops
            if schema.ops:
                self._execute_ops(schema.ops, manifest, result)
                if result.errors and self.stop_on_error:
                    raise RuntimeError(result.errors[-1])

            # Render stage
            if schema.render:
                self._execute_render(schema.render, output_dir, manifest, result)
                if result.errors and self.stop_on_error:
                    raise RuntimeError(result.errors[-1])

            # Exports stage
            if schema.exports:
                self._execute_exports(schema.exports, output_dir, manifest, result)
                if result.errors and self.stop_on_error:
                    raise RuntimeError(result.errors[-1])

            result.success = len(result.errors) == 0

        except Exception as e:
            result.errors.append(str(e))
            self._log(f"Pipeline failed: {e}", "error")

        # Finalize manifest
        finalize_manifest(manifest, result.success)
        manifest.warnings = result.warnings

        # Save manifest
        manifest_path = os.path.join(output_dir, "run_manifest.json")
        save_manifest(manifest, manifest_path)
        result.manifest_path = manifest_path
        result.artifacts.append(manifest_path)

        # Save execution log
        log_path = os.path.join(output_dir, "run.log")
        self._save_log(log_path, schema, result)
        result.artifacts.append(log_path)

        self._log(f"Pipeline {'completed successfully' if result.success else 'failed'}")
        return result

    def _execute_dataset(
        self,
        spec: DatasetSpec,
        manifest: ProvenanceManifest,
        result: ExecutionResult,
    ) -> None:
        """Execute dataset loading stage."""
        self._log(f"Loading dataset: {spec.source}")
        start_time = datetime.datetime.now(datetime.timezone.utc)

        registry = get_registry()
        status = "success"
        error = None

        try:
            if spec.source == "osmnx":
                # Import OSMnx graph
                scene_props = {
                    "osmnx_download_method": spec.method or "PLACE",
                    "osmnx_place_name": spec.query or "",
                    "osmnx_network_type": spec.network_type,
                    "osmnx_simplify": spec.simplify,
                    "osmnx_retain_all": spec.retain_all,
                }
                res = call_operator("scigraphs.import_osm_graph", scene_props=scene_props)
                if res["status"] == "error":
                    raise RuntimeError(res.get("error", "Unknown error"))

                # Mark as network source
                add_input(manifest, f"osmnx://{spec.query}", source="network", pinned=spec.cache)

            elif spec.source in ("gexf", "graphml"):
                # Import from file
                if not spec.filepath:
                    raise ValueError(f"filepath required for {spec.source} source")
                scene_props = {
                    "filepath": spec.filepath,
                    "source_column": "0",
                    "target_column": "1",
                    "use_geospatial": False,
                    "auto_layout_on_import": False,
                }
                res = call_operator("scigraphs.create_graph", scene_props=scene_props)
                if res["status"] == "error":
                    raise RuntimeError(res.get("error", "Unknown error"))
                add_input(manifest, spec.filepath, source="file", pinned=True)

            elif spec.source == "suitesparse":
                # Import from SuiteSparse Matrix Collection
                if not spec.matrix_name:
                    raise ValueError("matrix_name required for suitesparse source")
                props = {"matrix_name": spec.matrix_name}
                res = call_operator("scigraphs.import_suitesparse", props)
                if res["status"] == "error":
                    raise RuntimeError(res.get("error", "Unknown error"))
                add_input(manifest, f"suitesparse://{spec.matrix_name}", source="network", pinned=True)

            elif spec.source == "csv":
                # Import from CSV
                if not spec.filepath:
                    raise ValueError("filepath required for csv source")
                props = {"filepath": spec.filepath}
                res = call_operator("scigraphs.import_csv", props)
                if res["status"] == "error":
                    raise RuntimeError(res.get("error", "Unknown error"))
                add_input(manifest, spec.filepath, source="file", pinned=True)

            elif spec.source == "sql":
                # Import from SQL database
                props = {
                    "connection_string": spec.connection_string or "",
                    "nodes_query": spec.nodes_query or "",
                    "edges_query": spec.edges_query or "",
                }
                res = call_operator("scigraphs.import_sql", props)
                if res["status"] == "error":
                    raise RuntimeError(res.get("error", "Unknown error"))
                add_input(manifest, f"sql://{spec.connection_string}", source="network", pinned=False)

            else:
                raise ValueError(f"Unknown dataset source: {spec.source}")

        except Exception as e:
            status = "error"
            error = str(e)
            result.errors.append(f"Dataset loading failed: {e}")
            self._log(f"Dataset error: {e}", "error")

        end_time = datetime.datetime.now(datetime.timezone.utc)
        add_step(manifest, "dataset", f"load_{spec.source}", start_time, end_time, status, error)

    def _execute_analysis(
        self,
        spec: AnalysisSpec,
        manifest: ProvenanceManifest,
        result: ExecutionResult,
    ) -> None:
        """Execute analysis stage."""
        self._log(f"Running analysis: metrics={spec.metrics}")
        start_time = datetime.datetime.now(datetime.timezone.utc)
        status = "success"
        error = None

        try:
            # Compute metrics. The real operator is ``scigraphs.calculate_centrality``
            # which takes a ``method`` enum and stores the result on a mesh
            # attribute named ``centrality_<method>``. We remember that mapping so
            # the visual stage can resolve friendly names like "betweenness" to the
            # attribute that was actually produced.
            if spec.metrics:
                method_map = {
                    "degree": "degree",
                    "in_degree": "degree",
                    "out_degree": "degree",
                    "betweenness": "betweenness",
                    "betweenness_centrality": "betweenness",
                    "closeness": "closeness",
                    "closeness_centrality": "closeness",
                    "eigenvector": "eigenvector",
                    "eigenvector_centrality": "eigenvector",
                }
                for metric in spec.metrics:
                    metric_lower = metric.lower()
                    method = method_map.get(metric_lower)
                    if method is None:
                        self._log(
                            f"Unsupported metric: {metric} (supported: "
                            f"{', '.join(sorted(set(method_map.values())))})",
                            "warning",
                        )
                        result.warnings.append(f"Unsupported metric: {metric}")
                        continue

                    res = call_operator("scigraphs.calculate_centrality", {"method": method})
                    if res.get("status") == "error":
                        result.warnings.append(f"Metric {metric} failed: {res.get('error')}")
                    else:
                        attr_name = f"centrality_{method}"
                        self._metric_attributes[metric_lower] = attr_name
                        self._metric_attributes[method] = attr_name

            # Clustering (community detection)
            if spec.clustering:
                props = {"algorithm": spec.clustering.algorithm.lower()}
                res = call_operator("scigraphs.apply_clustering", props)
                if res.get("status") == "error":
                    result.warnings.append(f"Clustering failed: {res.get('error')}")
                else:
                    self._metric_attributes["community"] = "community"

        except Exception as e:
            status = "error"
            error = str(e)
            result.errors.append(f"Analysis failed: {e}")
            self._log(f"Analysis error: {e}", "error")

        end_time = datetime.datetime.now(datetime.timezone.utc)
        add_step(manifest, "analysis", "compute_metrics", start_time, end_time, status, error)

    def _execute_layout(
        self,
        spec: LayoutSpec,
        global_seed: int,
        manifest: ProvenanceManifest,
        result: ExecutionResult,
    ) -> None:
        """Execute layout stage."""
        self._log(f"Applying layout: {spec.algorithm}")
        start_time = datetime.datetime.now(datetime.timezone.utc)
        status = "success"
        error = None

        try:
            # Determine seed
            layout_seed = spec.seed if spec.seed is not None else global_seed

            props = {
                "algorithm": spec.algorithm,
                "scale": spec.scale,
                "iterations": spec.iterations,
            }
            if spec.k is not None:
                props["k"] = spec.k
            if spec.dimension == 2:
                props["dim"] = 2

            res = call_operator("scigraphs.apply_layout", props)
            if res.get("status") == "error":
                raise RuntimeError(res.get("error", "Unknown error"))

        except Exception as e:
            status = "error"
            error = str(e)
            result.errors.append(f"Layout failed: {e}")
            self._log(f"Layout error: {e}", "error")

        end_time = datetime.datetime.now(datetime.timezone.utc)
        add_step(manifest, "layout", f"apply_{spec.algorithm}", start_time, end_time, status, error)

    def _resolve_visual_attribute(self, name: str) -> Optional[str]:
        """Resolve a friendly attribute name to a real mesh attribute.

        Handles the centrality naming convention (``betweenness`` ->
        ``centrality_betweenness``) and verifies the attribute exists on the
        active graph object.
        """
        bpy = self._get_bpy()
        obj = bpy.context.active_object
        if obj is None or obj.type != 'MESH':
            return None

        attrs = obj.data.attributes
        name_lower = name.lower()

        candidates = [name]
        mapped = self._metric_attributes.get(name_lower)
        if mapped:
            candidates.append(mapped)
        candidates.append(f"centrality_{name_lower}")

        for cand in candidates:
            if cand in attrs:
                return cand
        return None

    def _apply_node_size_attribute(self, attr: str, result: ExecutionResult) -> None:
        """Drive node size from a mesh attribute via the interactive GN tree."""
        bpy = self._get_bpy()
        obj = bpy.context.active_object
        viz = getattr(bpy.context.scene, "scigraphs_viz", None)
        if viz is None or obj is None:
            result.warnings.append("Node sizing unavailable (no viz settings)")
            return

        try:
            from ..mesh import geometry
        except Exception:  # pragma: no cover - import guard
            result.warnings.append("Node sizing unavailable (geometry module)")
            return

        mod = obj.modifiers.get("SciGraphs_Viz")
        node_group = mod.node_group if mod else None
        is_interactive = bool(node_group and node_group.name.startswith("SciGraphs_Interactive"))
        if not is_interactive and hasattr(geometry, "setup_interactive_geometry_nodes"):
            try:
                geometry.setup_interactive_geometry_nodes(obj)
            except Exception as e:  # pragma: no cover - Blender runtime
                result.warnings.append(f"Could not build interactive tree: {e}")

        try:
            viz.node_scale_attribute = attr
        except (TypeError, ValueError):
            result.warnings.append(
                f"Node size attribute '{attr}' not selectable; skipping sizing"
            )
            return

        if hasattr(geometry, "update_geometry_nodes_parameters"):
            try:
                geometry.update_geometry_nodes_parameters(obj)
            except Exception as e:  # pragma: no cover - Blender runtime
                result.warnings.append(f"Node sizing update failed: {e}")

    def _execute_visual(
        self,
        spec: VisualSpec,
        manifest: ProvenanceManifest,
        result: ExecutionResult,
    ) -> None:
        """Execute visualization stage."""
        self._log("Applying visual settings")
        start_time = datetime.datetime.now(datetime.timezone.utc)
        status = "success"
        error = None

        try:
            # Setup geometry nodes
            if spec.setup_geometry_nodes:
                res = call_operator("scigraphs.setup_visualization")
                if res.get("status") == "error":
                    result.warnings.append(f"Geometry nodes setup failed: {res.get('error')}")

            # Node coloring (driven through the coloring toolbar settings +
            # the real ``scigraphs.color_apply`` operator).
            if spec.node_color:
                attr = self._resolve_visual_attribute(spec.node_color)
                if attr is None:
                    result.warnings.append(
                        f"Node color attribute '{spec.node_color}' not found on graph"
                    )
                else:
                    coloring = getattr(bpy.context.scene, "scigraphs_coloring", None)
                    if coloring is not None:
                        coloring.attribute_name = attr
                        try:
                            coloring.attribute_enum = attr
                        except (TypeError, ValueError):
                            pass
                        try:
                            coloring.colormap = spec.colormap
                        except (TypeError, ValueError):
                            result.warnings.append(f"Unknown colormap: {spec.colormap}")
                        coloring.auto_range = True
                    res = call_operator("scigraphs.color_apply")
                    if res.get("status") == "error":
                        result.warnings.append(f"Node coloring failed: {res.get('error')}")

            # Node sizing by attribute (interactive geometry-nodes tree).
            if spec.node_size:
                attr = self._resolve_visual_attribute(spec.node_size)
                if attr is None:
                    result.warnings.append(
                        f"Node size attribute '{spec.node_size}' not found on graph"
                    )
                else:
                    self._apply_node_size_attribute(attr, result)

            # Edge style
            if spec.edge_style:
                props = {"preset": spec.edge_style}
                res = call_operator("scigraphs.apply_edge_style_preset", props)
                if res.get("status") == "error":
                    result.warnings.append(f"Edge style failed: {res.get('error')}")

        except Exception as e:
            status = "error"
            error = str(e)
            result.errors.append(f"Visual setup failed: {e}")
            self._log(f"Visual error: {e}", "error")

        end_time = datetime.datetime.now(datetime.timezone.utc)
        add_step(manifest, "visual", "setup_visualization", start_time, end_time, status, error)

    def _execute_ops(
        self,
        ops: List[OpSpec],
        manifest: ProvenanceManifest,
        result: ExecutionResult,
    ) -> None:
        """Execute generic operator calls."""
        for i, op in enumerate(ops):
            self._log(f"Executing op {i+1}/{len(ops)}: {op.id}")
            start_time = datetime.datetime.now(datetime.timezone.utc)
            status = "success"
            error = None

            try:
                res = call_operator(op.id, op.props, op.scene_props)
                if res.get("status") == "error":
                    raise RuntimeError(res.get("error", "Unknown error"))
            except Exception as e:
                status = "error"
                error = str(e)
                result.errors.append(f"Op {op.id} failed: {e}")
                self._log(f"Op error: {e}", "error")
                if self.stop_on_error:
                    break

            end_time = datetime.datetime.now(datetime.timezone.utc)
            add_step(manifest, f"op_{i}", op.id, start_time, end_time, status, error)

    def _execute_render(
        self,
        spec: RenderSpec,
        output_dir: str,
        manifest: ProvenanceManifest,
        result: ExecutionResult,
    ) -> None:
        """Execute render stage."""
        self._log(f"Rendering with {spec.engine}")
        start_time = datetime.datetime.now(datetime.timezone.utc)
        status = "success"
        error = None

        bpy = self._get_bpy()

        try:
            # Set render engine
            bpy.context.scene.render.engine = spec.engine

            # Set resolution
            bpy.context.scene.render.resolution_x = spec.resolution[0]
            bpy.context.scene.render.resolution_y = spec.resolution[1]

            # Set samples
            if spec.engine == "CYCLES":
                bpy.context.scene.cycles.samples = spec.samples
            elif spec.engine == "BLENDER_EEVEE_NEXT":
                bpy.context.scene.eevee.taa_render_samples = spec.samples

            # Set output path
            output_path = os.path.join(output_dir, spec.output)
            bpy.context.scene.render.filepath = output_path

            # Set transparency
            bpy.context.scene.render.film_transparent = spec.transparent

            # Set denoising
            if spec.engine == "CYCLES":
                bpy.context.scene.cycles.use_denoising = spec.denoise

            # Set camera if specified
            if spec.camera:
                cam_obj = bpy.data.objects.get(spec.camera)
                if cam_obj and cam_obj.type == 'CAMERA':
                    bpy.context.scene.camera = cam_obj
                else:
                    result.warnings.append(f"Camera '{spec.camera}' not found")

            # Render
            bpy.ops.render.render(write_still=True)

            # Record output
            add_output(manifest, output_path, "render")
            result.artifacts.append(output_path)

        except Exception as e:
            status = "error"
            error = str(e)
            result.errors.append(f"Render failed: {e}")
            self._log(f"Render error: {e}", "error")

        end_time = datetime.datetime.now(datetime.timezone.utc)
        add_step(manifest, "render", f"render_{spec.engine}", start_time, end_time, status, error)

    def _execute_exports(
        self,
        spec: ExportsSpec,
        output_dir: str,
        manifest: ProvenanceManifest,
        result: ExecutionResult,
    ) -> None:
        """Execute exports stage."""
        self._log("Exporting artifacts")
        start_time = datetime.datetime.now(datetime.timezone.utc)
        status = "success"
        error = None

        bpy = self._get_bpy()

        try:
            # Export graph
            if spec.graph:
                output_path = os.path.join(output_dir, spec.graph)
                export_format = "GEXF"
                if output_path.lower().endswith(".graphml"):
                    export_format = "GRAPHML"
                elif output_path.lower().endswith(".json"):
                    export_format = "JSON"
                elif output_path.lower().endswith(".csv"):
                    export_format = "CSV"
                scene_props = {
                    "export_filepath": output_path,
                    "export_format": export_format,
                }
                res = call_operator("scigraphs.export_graph", scene_props=scene_props)
                if res.get("status") == "success":
                    add_output(manifest, output_path, "graph")
                    result.artifacts.append(output_path)
                else:
                    result.warnings.append(f"Graph export failed: {res.get('error')}")

            # Export positions
            if spec.positions:
                output_path = os.path.join(output_dir, spec.positions)
                res = call_operator(
                    "scigraphs.export_positions",
                    scene_props={"export_filepath": output_path},
                )
                if res.get("status") == "success":
                    add_output(manifest, output_path, "positions")
                    result.artifacts.append(output_path)
                else:
                    result.warnings.append(f"Positions export failed: {res.get('error')}")

            # Export statistics
            if spec.statistics:
                output_path = os.path.join(output_dir, spec.statistics)
                res = call_operator(
                    "scigraphs.generate_statistics_report",
                    scene_props={"export_filepath": output_path},
                )
                if res.get("status") == "success":
                    add_output(manifest, output_path, "statistics")
                    result.artifacts.append(output_path)
                else:
                    result.warnings.append(f"Statistics export failed: {res.get('error')}")

            # Save blend file copy
            if spec.blend:
                output_path = os.path.join(output_dir, spec.blend)
                bpy.ops.wm.save_as_mainfile(filepath=output_path, copy=True)
                add_output(manifest, output_path, "blend")
                result.artifacts.append(output_path)

        except Exception as e:
            status = "error"
            error = str(e)
            result.errors.append(f"Export failed: {e}")
            self._log(f"Export error: {e}", "error")

        end_time = datetime.datetime.now(datetime.timezone.utc)
        add_step(manifest, "exports", "export_artifacts", start_time, end_time, status, error)

    def _save_log(
        self,
        log_path: str,
        schema: PipelineSchema,
        result: ExecutionResult,
    ) -> None:
        """Save execution log to file."""
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"SciGraphs Pipeline Execution Log\n")
            f.write(f"================================\n\n")
            f.write(f"Pipeline: {schema.meta.title}\n")
            f.write(f"Hash: {result.pipeline_hash}\n")
            f.write(f"Seed: {schema.meta.seed}\n")
            f.write(f"Output: {result.output_dir}\n")
            f.write(f"Success: {result.success}\n")
            f.write(f"Duration: {result.duration_ms}ms\n\n")

            if result.warnings:
                f.write("Warnings:\n")
                for w in result.warnings:
                    f.write(f"  - {w}\n")
                f.write("\n")

            if result.errors:
                f.write("Errors:\n")
                for e in result.errors:
                    f.write(f"  - {e}\n")
                f.write("\n")

            f.write("Artifacts:\n")
            for a in result.artifacts:
                f.write(f"  - {a}\n")


def run_pipeline(
    source,
    base_dir: Optional[str] = None,
    stop_on_error: bool = True,
    verbose: bool = True,
) -> ExecutionResult:
    """
    Convenience function to parse and execute a pipeline.

    Args:
        source: Pipeline file path, content string, or dictionary
        base_dir: Base directory for resolving paths
        stop_on_error: Whether to stop on first error
        verbose: Whether to log progress

    Returns:
        ExecutionResult
    """
    from .parser import parse_pipeline

    schema, raw_dict, pipeline_hash = parse_pipeline(source, base_dir)
    executor = PipelineExecutor(stop_on_error=stop_on_error, verbose=verbose)
    return executor.execute(schema, raw_dict, pipeline_hash)
