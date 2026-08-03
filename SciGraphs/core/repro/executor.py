# Pipeline executor for reproducible SciGraphs workflows
#
# Executes declarative pipelines with logging, timing, error handling
# and artifact generation.

import datetime
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schema import (
    PipelineSchema, DatasetSpec, AnalysisSpec, LayoutSpec, VisualSpec,
    LabelsSpec, WorldSpec, LightingSpec, RenderSpec, ExportsSpec, OpSpec,
)
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
        # Configured as its own section, executed inside the render stage,
        # where the camera is final.
        self._pending_labels: Optional[LabelsSpec] = None

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

    def _collect_warnings(self, res: Dict[str, Any], result: ExecutionResult) -> None:
        """Append any scene-property warnings from an operator result."""
        for warning in res.get("warnings", []) or []:
            result.warnings.append(warning)

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
        self._pending_labels = (
            schema.labels if schema.labels and schema.labels.enabled else None
        )

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
            # Start from an empty scene: leftovers from the startup file get
            # rendered, or picked up as the active object.
            if getattr(schema.meta, "clear_scene", True):
                self._clear_scene(result)

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

            # World stage
            if schema.world:
                self._execute_world(schema.world, manifest, result)
                if result.errors and self.stop_on_error:
                    raise RuntimeError(result.errors[-1])

            # Lighting stage
            if schema.lighting:
                self._execute_lighting(schema.lighting, manifest, result)
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

    def _clear_scene(self, result: ExecutionResult) -> None:
        """Remove every object, so the run starts from a known empty scene."""
        bpy = self._get_bpy()
        try:
            removed = 0
            for obj in list(bpy.data.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
                removed += 1
            if removed:
                self._log(f"Cleared scene: {removed} object(s) removed")
        except Exception as e:  # pragma: no cover - defensive
            result.warnings.append(f"Scene clear failed: {e}")

    def _apply_color_mapping(self, spec: VisualSpec, coloring, result: ExecutionResult) -> None:
        """Normalization, explicit domain and ramp options.

        `auto_range` must be off for `vmin`/`vmax` to reach the shader.
        """
        explicit = spec.color_vmin is not None or spec.color_vmax is not None
        coloring.auto_range = not explicit
        if spec.color_vmin is not None and hasattr(coloring, "vmin"):
            coloring.vmin = float(spec.color_vmin)
        if spec.color_vmax is not None and hasattr(coloring, "vmax"):
            coloring.vmax = float(spec.color_vmax)

        pairs = [
            ("color_norm", spec.color_norm),
            ("color_gamma", spec.color_gamma),
            ("reverse", spec.colormap_reverse),
            ("opacity", spec.color_opacity),
        ]
        clip = list(spec.color_clip_percentile or [0.0, 100.0])
        if len(clip) == 2:
            pairs += [("clip_low_pct", float(clip[0])),
                      ("clip_high_pct", float(clip[1]))]
        if spec.edge_base_color:
            rgba = list(spec.edge_base_color)
            while len(rgba) < 4:
                rgba.append(1.0)
            pairs.append(("edge_color", tuple(rgba[:4])))

        for name, value in pairs:
            if value is None:
                continue
            if not hasattr(coloring, name):
                result.warnings.append(f"coloring.{name} unavailable on this build")
                continue
            try:
                setattr(coloring, name, value)
            except (TypeError, ValueError) as e:
                result.warnings.append(f"coloring.{name}={value!r} rejected: {e}")

    def _apply_glyph_settings(self, spec: VisualSpec, result: ExecutionResult) -> None:
        """Stage glyph shape, resolution and radii on the graph object.

        The node tree reads these as object custom properties at build time, so
        they must be written before `setup_visualization` runs. The `_rel` radii
        are fractions of the graph extent, resolved here against the bounds.
        """
        bpy = self._get_bpy()
        obj = bpy.context.active_object
        if obj is None or obj.type != 'MESH':
            return

        shape_index = {
            "SPHERE": 0, "ICOSPHERE": 1, "CUBE": 2, "CONE": 3, "CYLINDER": 4,
        }
        if spec.node_glyph:
            obj["scigraphs_node_shape_index"] = shape_index.get(spec.node_glyph, 0)
        if spec.node_resolution is not None:
            obj["scigraphs_node_resolution"] = int(spec.node_resolution)
        if spec.edge_resolution is not None:
            obj["scigraphs_edge_resolution"] = int(spec.edge_resolution)
        obj["scigraphs_node_shade_smooth"] = bool(spec.node_shade_smooth)
        obj["scigraphs_edge_profile"] = spec.edge_profile or "ROUND"

        radius = None
        if spec.node_radius_rel is not None or spec.edge_radius_rel is not None:
            bounds = self._graph_bounds()
            if bounds is None:
                result.warnings.append(
                    "Relative sizes need graph bounds; none available"
                )
            else:
                radius = bounds[1]

        node_r = spec.node_radius
        if spec.node_radius_rel is not None and radius:
            node_r = radius * float(spec.node_radius_rel)
        if node_r is not None:
            obj["scigraphs_node_size"] = float(node_r)

        edge_r = spec.edge_radius
        if spec.edge_radius_rel is not None and radius:
            edge_r = radius * float(spec.edge_radius_rel)
        if edge_r is not None:
            obj["scigraphs_edge_thickness"] = float(edge_r)

        if node_r is not None or edge_r is not None:
            self._log(
                "Glyph radii: node %s, edge %s"
                % (f"{node_r:.4f}" if node_r else "-",
                   f"{edge_r:.4f}" if edge_r else "-")
            )

    def _apply_material_overrides(self, spec: VisualSpec, obj, result: ExecutionResult) -> None:
        """Roughness and metallic on the graph's Principled BSDF."""
        materials = [s.material for s in obj.material_slots if s.material]
        if not materials and obj.data.materials:
            materials = [m for m in obj.data.materials if m]
        if not materials:
            result.warnings.append("No material to apply roughness/metallic to")
            return
        for material in materials:
            if not material.use_nodes:
                continue
            bsdf = material.node_tree.nodes.get("Principled BSDF")
            if bsdf is None:
                continue
            if spec.material_roughness is not None and "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = float(spec.material_roughness)
            if spec.material_metallic is not None and "Metallic" in bsdf.inputs:
                bsdf.inputs["Metallic"].default_value = float(spec.material_metallic)

    def _apply_render_quality(self, spec: RenderSpec, result: ExecutionResult) -> None:
        """Color management, image format and engine-specific quality.

        Every setting is hasattr-guarded: the EEVEE Next rewrite dropped many
        property names, and an unsupported one warns rather than aborts.
        """
        bpy = self._get_bpy()
        scene = bpy.context.scene

        def _set(owner, name, value, label):
            if value is None:
                return
            if not hasattr(owner, name):
                result.warnings.append(f"{label} unsupported on this build")
                return
            try:
                setattr(owner, name, value)
            except Exception as e:
                result.warnings.append(f"{label} rejected: {e}")

        # -- color management. view_transform first: valid `look` values are
        # scoped to it, so the reverse order rejects every look.
        _set(scene.view_settings, "view_transform", spec.view_transform, "view_transform")
        _set(scene.view_settings, "look", spec.look, "look")
        _set(scene.view_settings, "exposure", spec.exposure, "exposure")
        _set(scene.view_settings, "gamma", spec.gamma, "gamma")

        # -- image
        _set(scene.render, "resolution_percentage", spec.resolution_percentage, "resolution_percentage")
        _set(scene.render.image_settings, "file_format", spec.file_format, "file_format")
        _set(scene.render.image_settings, "color_depth", spec.color_depth, "color_depth")
        if spec.dpi is not None:
            _set(scene.render, "ppm_factor", float(spec.dpi), "dpi")

        # -- reconstruction filter. The two properties are not aliased: Cycles
        # ignores render.filter_size and reads cycles.filter_width (on 5.2).
        if spec.filter_width is not None:
            if spec.engine == "CYCLES":
                _set(scene.cycles, "filter_width", float(spec.filter_width), "filter_width")
            else:
                _set(scene.render, "filter_size", float(spec.filter_width), "filter_width")

        if spec.engine == "CYCLES":
            _set(scene.cycles, "adaptive_threshold", spec.adaptive_threshold, "adaptive_threshold")
            _set(scene.cycles, "max_bounces", spec.max_bounces, "max_bounces")
            if spec.denoiser and spec.denoiser != "AUTO":
                _set(scene.cycles, "denoiser", spec.denoiser, "denoiser")
        elif spec.engine == "BLENDER_EEVEE":
            # use_fast_gi does nothing with use_raytracing off, so enable the
            # parent rather than silently no-op.
            wants_gi = bool(spec.ambient_occlusion) or bool(spec.raytracing)
            if wants_gi:
                _set(scene.eevee, "use_raytracing", True, "raytracing")
            elif spec.raytracing is not None:
                _set(scene.eevee, "use_raytracing", bool(spec.raytracing), "raytracing")
            if spec.ambient_occlusion:
                _set(scene.eevee, "use_fast_gi", True, "ambient_occlusion")
                _set(scene.eevee, "fast_gi_method", "AMBIENT_OCCLUSION_ONLY", "fast_gi_method")
                _set(scene.eevee, "fast_gi_distance", spec.ao_distance, "ao_distance")
            _set(scene.eevee, "clamp_surface_indirect", spec.clamp_indirect, "clamp_indirect")

    def _execute_labels(self, spec, output_dir, manifest, result) -> None:
        """Composite node labels over the render.

        Calls `text_overlay` directly: `scigraphs.generate_text_overlay` polls
        for an active object that background mode does not provide. Must run
        after camera framing and before the render.
        """
        bpy = self._get_bpy()
        self._log("Generating labels")
        start_time = datetime.datetime.now(datetime.timezone.utc)
        status, error = "success", None

        try:
            from ..visualization import text_overlay as overlay

            if not getattr(overlay, "PIL_AVAILABLE", False):
                result.warnings.append("Labels need Pillow, which is unavailable")
                raise RuntimeError("Pillow unavailable")

            obj = bpy.context.active_object
            if obj is None or obj.type != 'MESH':
                raise RuntimeError("No graph object to label")
            scene = bpy.context.scene
            camera = scene.camera
            if camera is None:
                raise RuntimeError("Labels need a camera")

            resolution = (scene.render.resolution_x, scene.render.resolution_y)

            settings = overlay.TextOverlaySettings(
                size_mode=spec.size_mode,
                fixed_size=int(spec.font_size),
                size_scale=1.0,
                max_distance=float(spec.max_distance or 0.0),
                text_color=tuple(spec.color)[:3],
                background_enabled=bool(spec.halo),
                background_color=tuple(spec.halo_color)[:3],
                background_alpha=float(spec.halo_alpha),
                depth_occlusion=bool(spec.occlusion),
                # Ranked in _rank_labels instead: the built-in filter is a
                # single threshold, which does not carry across graphs.
                filter_enabled=False,
                filter_attribute="",
                filter_operator="GREATER",
                filter_value=0.0,
                float_decimals=int(spec.float_decimals),
            )

            projected = overlay.project_nodes_to_screen(obj, camera, scene, resolution)
            n_projected = len(projected)
            projected = [p for p in projected if p.visible]
            n_visible = len(projected)

            if spec.occlusion:
                # Takes a context, but only for evaluated_depsgraph_get(),
                # which works in background mode.
                projected = overlay.test_depth_occlusion(
                    bpy.context, obj, projected, camera
                ) or projected
                projected = [p for p in projected if not p.occluded]

            n_unoccluded = len(projected)
            projected = self._rank_labels(projected, obj, spec, result)

            # Report where labels were lost; four failures look alike.
            self._log(
                "Labels: %d projected, %d in frame, %d unoccluded, %d kept"
                % (n_projected, n_visible, n_unoccluded, len(projected))
            )

            if not projected:
                result.warnings.append(
                    "No labels survived: %d projected, %d in frame, %d unoccluded"
                    % (n_projected, n_visible, n_unoccluded)
                )
            else:
                image_path = os.path.join(output_dir, "labels.png")
                written = overlay.generate_text_image(
                    projected, resolution, settings, camera, output_path=image_path
                )
                if not written:
                    raise RuntimeError("Label image was not written")
                if not overlay.setup_compositor_overlay(scene, written):
                    raise RuntimeError("Could not wire the compositor overlay")
                scene.render.use_compositing = True
                add_output(manifest, written, "labels")
                result.artifacts.append(written)
                self._log(f"Labels: {len(projected)} placed")
        except Exception as e:
            status, error = "error", str(e)
            result.errors.append(f"Labels failed: {e}")
            self._log(f"Labels error: {e}", "error")

        end_time = datetime.datetime.now(datetime.timezone.utc)
        add_step(manifest, "labels", "generate_labels", start_time, end_time, status, error)

    def _rank_labels(self, projected, obj, spec, result):
        """Keep the highest-ranked labels, and only those above `min_value`."""
        from ..visualization import text_overlay as overlay

        attr = self._resolve_visual_attribute(spec.rank_by) or spec.rank_by
        values = overlay.get_node_attribute_values(obj, attr) or {}
        if not values:
            result.warnings.append(
                f"Label ranking attribute '{spec.rank_by}' not found; "
                "labelling by camera distance instead"
            )
            ranked = sorted(projected, key=lambda p: p.distance)
        else:
            for node in projected:
                node.attribute_value = values.get(node.name)
            if spec.min_value is not None:
                projected = [
                    p for p in projected
                    if p.attribute_value is not None
                    and p.attribute_value >= float(spec.min_value)
                ]
            ranked = sorted(
                projected,
                key=lambda p: (p.attribute_value is None, -(p.attribute_value or 0.0)),
            )

        if spec.declutter:
            ranked = self._declutter_labels(ranked, spec)

        if spec.max_count and spec.max_count > 0:
            ranked = ranked[: int(spec.max_count)]
        return ranked

    def _declutter_labels(self, ranked, spec):
        """Drop labels whose box would overlap one already accepted.

        Greedy in rank order; the overlay code itself has no collision pass and
        just overdraws. Boxes are estimated wide rather than measured with
        Pillow, which would mean loading the font a second time.
        """
        accepted = []
        boxes = []
        half_h = max(spec.font_size, 1) * 0.62
        for node in ranked:
            half_w = 0.30 * spec.font_size * max(len(node.name), 1)
            box = (node.x - half_w, node.y - half_h,
                   node.x + half_w, node.y + half_h)
            if any(box[0] < b[2] and b[0] < box[2] and
                   box[1] < b[3] and b[1] < box[3] for b in boxes):
                continue
            boxes.append(box)
            accepted.append(node)
        return accepted

    def _execute_world(self, spec, manifest, result) -> None:
        """Background color, ambient strength and optional HDRI."""
        bpy = self._get_bpy()
        self._log("Applying world")
        start_time = datetime.datetime.now(datetime.timezone.utc)
        status, error = "success", None

        try:
            world = bpy.context.scene.world
            if world is None:
                world = bpy.data.worlds.new("SciGraphsWorld")
                bpy.context.scene.world = world
            world.use_nodes = True
            nodes, links = world.node_tree.nodes, world.node_tree.links

            background = next(
                (n for n in nodes if n.type == 'BACKGROUND'), None
            )
            if background is None:
                background = nodes.new("ShaderNodeBackground")
                output = next((n for n in nodes if n.type == 'OUTPUT_WORLD'), None)
                if output is None:
                    output = nodes.new("ShaderNodeOutputWorld")
                links.new(background.outputs["Background"], output.inputs["Surface"])

            if spec.color is not None:
                rgb = list(spec.color)[:3]
                while len(rgb) < 3:
                    rgb.append(0.0)
                background.inputs["Color"].default_value = (*rgb, 1.0)
            if spec.strength is not None:
                background.inputs["Strength"].default_value = float(spec.strength)

            if spec.hdri:
                path = bpy.path.abspath(spec.hdri)
                if not os.path.isfile(path):
                    result.warnings.append(f"HDRI not found: {path}")
                else:
                    env = nodes.new("ShaderNodeTexEnvironment")
                    env.image = bpy.data.images.load(path, check_existing=True)
                    mapping = nodes.new("ShaderNodeMapping")
                    coord = nodes.new("ShaderNodeTexCoord")
                    mapping.inputs["Rotation"].default_value[2] = math.radians(
                        float(spec.hdri_rotation or 0.0)
                    )
                    links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
                    links.new(mapping.outputs["Vector"], env.inputs["Vector"])
                    links.new(env.outputs["Color"], background.inputs["Color"])
                    manifest_path = path
                    add_input(manifest, manifest_path, "hdri")
        except Exception as e:
            status, error = "error", str(e)
            result.errors.append(f"World failed: {e}")
            self._log(f"World error: {e}", "error")

        end_time = datetime.datetime.now(datetime.timezone.utc)
        add_step(manifest, "world", "apply_world", start_time, end_time, status, error)

    def _execute_lighting(self, spec, manifest, result) -> None:
        """A single sun.

        One light, not a rig: the add-on's three-point rigs put their fills at
        fixed coordinates near the origin, which does nothing at graph scale.
        """
        bpy = self._get_bpy()
        self._log("Applying lighting")
        start_time = datetime.datetime.now(datetime.timezone.utc)
        status, error = "success", None

        try:
            if spec.replace:
                for obj in [o for o in bpy.data.objects if o.type == 'LIGHT']:
                    bpy.data.objects.remove(obj, do_unlink=True)

            existing = [o for o in bpy.data.objects if o.type == 'LIGHT']
            if existing:
                sun = existing[0]
                if sun.data.type != 'SUN':
                    sun.data.type = 'SUN'
            else:
                light_data = bpy.data.lights.new("SciGraphsKey", type='SUN')
                sun = bpy.data.objects.new("SciGraphsKey", light_data)
                bpy.context.scene.collection.objects.link(sun)

            sun.data.energy = float(spec.sun_energy)
            # `angle` is the angular diameter in radians, capped at pi (5.2);
            # anything larger is silently reduced.
            sun.data.angle = min(math.radians(float(spec.sun_angle)), math.pi)
            sun.rotation_euler = tuple(spec.sun_rotation)[:3]
            self._log(
                f"Sun: energy {sun.data.energy:.2f}, "
                f"angle {math.degrees(sun.data.angle):.1f} deg"
            )
        except Exception as e:
            status, error = "error", str(e)
            result.errors.append(f"Lighting failed: {e}")
            self._log(f"Lighting error: {e}", "error")

        end_time = datetime.datetime.now(datetime.timezone.utc)
        add_step(manifest, "lighting", "apply_lighting", start_time, end_time, status, error)

    def _graph_bounds(self):
        """World-space (center, radius) over every mesh object, or None.

        Evaluated geometry, so the glyphs and edges instanced by the Geometry
        Nodes modifier count; the bare point cloud under-reports the extent.
        """
        import math

        from mathutils import Vector

        bpy = self._get_bpy()
        depsgraph = bpy.context.evaluated_depsgraph_get()

        lo = [float("inf")] * 3
        hi = [float("-inf")] * 3
        found = False
        for obj in bpy.data.objects:
            if obj.type != 'MESH':
                continue
            evaluated = obj.evaluated_get(depsgraph)
            for corner in evaluated.bound_box:
                world = evaluated.matrix_world @ Vector(corner)
                for axis in range(3):
                    lo[axis] = min(lo[axis], world[axis])
                    hi[axis] = max(hi[axis], world[axis])
                found = True
        if not found:
            return None

        center = [(lo[a] + hi[a]) * 0.5 for a in range(3)]
        radius = 0.5 * math.sqrt(sum((hi[a] - lo[a]) ** 2 for a in range(3)))
        return center, max(radius, 1e-6)

    def _frame_camera(self, spec: RenderSpec, result: ExecutionResult) -> None:
        """Place and aim a camera so the whole graph fits the frame."""
        import math

        from mathutils import Vector

        bpy = self._get_bpy()

        bounds = self._graph_bounds()
        if bounds is None:
            result.warnings.append("Nothing to frame; camera left as-is")
            return
        center, radius = bounds

        cam_obj = bpy.context.scene.camera
        if cam_obj is None or cam_obj.type != 'CAMERA':
            cam_data = bpy.data.cameras.new("SciGraphsCamera")
            cam_obj = bpy.data.objects.new("SciGraphsCamera", cam_data)
            bpy.context.scene.collection.objects.link(cam_obj)
            bpy.context.scene.camera = cam_obj

        cam = cam_obj.data
        if spec.camera_lens is not None:
            cam.lens = float(spec.camera_lens)

        # Fit the bounding sphere in the narrower of the two field angles, so
        # the graph fits regardless of the aspect ratio.
        sensor = cam.sensor_width or 36.0
        half_angle = math.atan((sensor * 0.5) / max(cam.lens, 1e-6))
        res_x = max(bpy.context.scene.render.resolution_x, 1)
        res_y = max(bpy.context.scene.render.resolution_y, 1)
        if res_y > res_x:
            half_angle = math.atan(math.tan(half_angle) * res_x / res_y)
        distance = (radius * float(spec.camera_margin)) / max(math.sin(half_angle), 1e-6)

        direction = Vector(spec.camera_direction or [0.48, -0.72, 0.50])
        if direction.length < 1e-9:
            direction = Vector((0.48, -0.72, 0.50))
        direction.normalize()

        center_vec = Vector(center)
        cam_obj.location = center_vec + direction * distance
        # Blender cameras look down local -Z; track_to gives the aiming basis.
        cam_obj.rotation_euler = (
            (cam_obj.location - center_vec).to_track_quat('Z', 'Y').to_euler()
        )

        # Orthographic drops the near/far size falloff, so a node radius that
        # encodes a value stays comparable across the frame.
        if spec.camera_ortho:
            cam.type = 'ORTHO'
            cam.ortho_scale = 2.0 * radius * float(spec.camera_margin)
        else:
            cam.type = 'PERSP'

        # Depth of field focused on the framing distance, so no focus object.
        if spec.dof_fstop is not None:
            cam.dof.use_dof = True
            cam.dof.focus_distance = distance
            cam.dof.aperture_fstop = float(spec.dof_fstop)
        else:
            cam.dof.use_dof = False

        # Flush the transform: anything reading `camera.matrix_world` before
        # the render (label projection) otherwise sees the pre-move matrix.
        bpy.context.view_layer.update()

        if not any(obj.type == 'LIGHT' for obj in bpy.data.objects):
            light_data = bpy.data.lights.new("SciGraphsKey", type='SUN')
            light_data.energy = 3.0
            light = bpy.data.objects.new("SciGraphsKey", light_data)
            bpy.context.scene.collection.objects.link(light)
            light.rotation_euler = (0.9, 0.0, 0.7)
            result.warnings.append("No light in scene; added a default sun")

        self._log(f"Framed camera at distance {distance:.2f} (radius {radius:.2f})")

    def _execute_dataset(
        self,
        spec: DatasetSpec,
        manifest: ProvenanceManifest,
        result: ExecutionResult,
    ) -> None:
        """Execute dataset loading stage."""
        bpy = self._get_bpy()
        self._log(f"Loading dataset: {spec.source}")
        start_time = datetime.datetime.now(datetime.timezone.utc)

        registry = get_registry()
        status = "success"
        error = None

        try:
            if spec.source == "osmnx":
                # Import OSMnx graph. The operator reads everything from
                # ``scene.scigraphs`` (osmnx_* properties live there).
                scene_props = {
                    "osmnx_download_method": spec.method or "PLACE",
                    "osmnx_place_name": spec.query or "",
                    "osmnx_network_type": spec.network_type,
                    "osmnx_simplify": spec.simplify,
                    "osmnx_retain_all": spec.retain_all,
                    "osmnx_use_cache": spec.cache,
                }
                res = call_operator("scigraphs.import_osm_graph", scene_props=scene_props)
                self._collect_warnings(res, result)
                if res["status"] == "error":
                    raise RuntimeError(res.get("error", "Unknown error"))

                # Mark as network source
                add_input(manifest, f"osmnx://{spec.query}", source="network", pinned=spec.cache)

            elif spec.source in ("gexf", "graphml", "csv"):
                # File-based import goes through scigraphs.create_graph, which
                # reads the file path and column mapping from scene.scigraphs.
                if not spec.filepath:
                    raise ValueError(f"filepath required for {spec.source} source")

                resolved = bpy.path.abspath(spec.filepath)
                if not os.path.isfile(resolved):
                    raise FileNotFoundError(
                        f"Dataset file not found: {resolved} "
                        f"(from filepath '{spec.filepath}'). Provide an existing "
                        f"file; example pipelines use placeholder paths."
                    )
                if spec.source == "graphml":
                    raise ValueError(
                        "GraphML files are not loadable via the file dataset "
                        "source yet. Convert to GEXF, or load OSMnx GraphML "
                        "through the OSMnx tab."
                    )

                scene_props = {
                    "filepath": resolved,
                    "use_geospatial": False,
                    "auto_layout_on_import": bool(spec.auto_layout),
                }
                # CSV needs an explicit source/target column mapping; default to
                # the first two columns when the spec does not override them.
                if spec.source == "csv":
                    scene_props.setdefault("source_column", "0")
                    scene_props.setdefault("target_column", "1")
                res = call_operator("scigraphs.create_graph", scene_props=scene_props)
                self._collect_warnings(res, result)
                if res["status"] == "error":
                    raise RuntimeError(res.get("error", "Unknown error"))
                add_input(manifest, resolved, source="file", pinned=True)

            elif spec.source == "suitesparse":
                # Import from SuiteSparse Matrix Collection. The operator reads
                # the matrix identifier from scene.scigraphs.suitesparse_id.
                if not spec.matrix_name:
                    raise ValueError("matrix_name required for suitesparse source")
                scene_props = {"suitesparse_id": spec.matrix_name}
                res = call_operator("scigraphs.download_suitesparse", scene_props=scene_props)
                self._collect_warnings(res, result)
                if res["status"] == "error":
                    raise RuntimeError(res.get("error", "Unknown error"))
                add_input(manifest, f"suitesparse://{spec.matrix_name}", source="network", pinned=True)

            elif spec.source == "sql":
                # Import from a configured SQL connection. The operator reads
                # the query and connection from scene.scigraphs.
                scene_props = {}
                if spec.nodes_query:
                    scene_props["sql_query"] = spec.nodes_query
                elif spec.edges_query:
                    scene_props["sql_query"] = spec.edges_query
                res = call_operator(
                    "scigraphs.create_graph_from_sql", scene_props=scene_props
                )
                self._collect_warnings(res, result)
                if res["status"] == "error":
                    raise RuntimeError(res.get("error", "Unknown error"))
                add_input(
                    manifest,
                    f"sql://{spec.connection_string or 'configured'}",
                    source="network",
                    pinned=False,
                )

            elif spec.source == "city2graph":
                # City2Graph / Overture import. Use a place name when given,
                # otherwise the bbox-based REST download.
                if spec.query:
                    scene_props = {"city2graph": {"geocode_place_name": spec.query}}
                    op_id = "scigraphs.c2g_load_overture_place"
                else:
                    scene_props = {}
                    if spec.bbox and len(spec.bbox) == 4:
                        scene_props = {
                            "city2graph": {
                                "c2g_use_osmnx_bbox": False,
                                "c2g_bbox_west": float(spec.bbox[0]),
                                "c2g_bbox_south": float(spec.bbox[1]),
                                "c2g_bbox_east": float(spec.bbox[2]),
                                "c2g_bbox_north": float(spec.bbox[3]),
                            }
                        }
                    op_id = "scigraphs.c2g_load_overture"
                res = call_operator(op_id, scene_props=scene_props)
                self._collect_warnings(res, result)
                if res["status"] == "error":
                    raise RuntimeError(res.get("error", "Unknown error"))
                add_input(
                    manifest,
                    f"city2graph://{spec.query or spec.bbox}",
                    source="network",
                    pinned=False,
                )

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
                # scigraphs.calculate_centrality supports these methods and
                # stores the result on a mesh attribute named centrality_<method>.
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
                        # Plain clustering coefficient has its own operator.
                        if metric_lower in ("clustering", "clustering_coefficient"):
                            res = call_operator("scigraphs.calculate_clustering")
                            self._collect_warnings(res, result)
                            if res.get("status") == "error":
                                result.warnings.append(
                                    f"Clustering coefficient failed: {res.get('error')}"
                                )
                            else:
                                self._metric_attributes["clustering"] = "clustering"
                            continue
                        self._log(
                            f"Unsupported metric: {metric} (supported: "
                            f"{', '.join(sorted(set(method_map.values())))}, clustering)",
                            "warning",
                        )
                        result.warnings.append(f"Unsupported metric: {metric}")
                        continue

                    res = call_operator("scigraphs.calculate_centrality", {"method": method})
                    self._collect_warnings(res, result)
                    if res.get("status") == "error":
                        result.warnings.append(f"Metric {metric} failed: {res.get('error')}")
                    else:
                        attr_name = f"centrality_{method}"
                        self._metric_attributes[metric_lower] = attr_name
                        self._metric_attributes[method] = attr_name

            # Clustering (community detection). The operator reads algorithm,
            # resolution and seed from scene.scigraphs; pass resolution there.
            if spec.clustering:
                # Map common modularity-based names onto the operator's set.
                algo_aliases = {"louvain": "rb", "leiden": "rb", "label_prop": "rb"}
                algo = spec.clustering.algorithm.lower()
                algo = algo_aliases.get(algo, algo)
                scene_props = {
                    "clustering_algorithm": algo,
                    "clustering_resolution": spec.clustering.resolution,
                }
                props = {
                    "algorithm": algo,
                    "resolution": spec.clustering.resolution,
                }
                res = call_operator(
                    "scigraphs.apply_clustering", props, scene_props
                )
                self._collect_warnings(res, result)
                if res.get("status") == "error":
                    result.warnings.append(f"Clustering failed: {res.get('error')}")
                else:
                    # apply_clustering stores community labels on cluster_id.
                    self._metric_attributes["community"] = "cluster_id"
                    self._metric_attributes["cluster_id"] = "cluster_id"

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
            # Determine seed (re-seed so the layout itself is deterministic).
            layout_seed = spec.seed if spec.seed is not None else global_seed
            set_pipeline_seed(layout_seed)

            # apply_layout reads algorithm/scale/iterations and all algorithm
            # specific parameters from scene.scigraphs. Map the friendly typed
            # fields onto the actual scene property names.
            scene_props: Dict[str, Any] = {
                "layout_algorithm": spec.algorithm,
                "layout_scale": spec.scale,
                "iterations": spec.iterations,
            }

            algo = (spec.algorithm or "").upper()
            if spec.gravity is not None:
                if "FORCEATLAS2" in algo or "FORCE_ATLAS2" in algo:
                    scene_props["fa2_gravity"] = spec.gravity
                else:
                    scene_props["gravity_strength"] = spec.gravity
            if spec.scaling_ratio is not None and (
                "FORCEATLAS2" in algo or "FORCE_ATLAS2" in algo
            ):
                scene_props["fa2_scaling_ratio"] = spec.scaling_ratio
            if spec.k is not None:
                # Ideal edge length under several names depending on algorithm.
                if "YIFAN" in algo:
                    scene_props["yifan_hu_spring_constant"] = spec.k
                else:
                    scene_props["sfdp_k"] = spec.k

            props = {
                "algorithm": spec.algorithm,
                "scale": spec.scale,
                "iterations": spec.iterations,
            }

            res = call_operator("scigraphs.apply_layout", props, scene_props)
            self._collect_warnings(res, result)
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
        bpy = self._get_bpy()
        self._log("Applying visual settings")
        start_time = datetime.datetime.now(datetime.timezone.utc)
        status = "success"
        error = None

        try:
            # Glyph geometry and sizes are baked into the node tree when it is
            # built, so they have to be staged on the object first.
            self._apply_glyph_settings(spec, result)

            # Setup geometry nodes
            if spec.setup_geometry_nodes:
                res = call_operator("scigraphs.setup_visualization")
                self._collect_warnings(res, result)
                if res.get("status") == "error":
                    result.warnings.append(f"Geometry nodes setup failed: {res.get('error')}")

            # Apply node/edge size ranges onto the interactive viz settings.
            # The viz tree exposes node_scale and edge_thickness as the upper
            # bounds for attribute-driven sizing.
            viz = getattr(bpy.context.scene, "scigraphs_viz", None)
            if viz is not None:
                size_map = {
                    "node_scale": spec.node_max_size,
                    "edge_thickness": spec.edge_max_width,
                }
                for attr_name, value in size_map.items():
                    if value is not None and hasattr(viz, attr_name):
                        try:
                            setattr(viz, attr_name, value)
                        except (TypeError, ValueError):
                            result.warnings.append(
                                f"Could not set viz.{attr_name}={value}"
                            )

            # Rendering preset (scientific/presentation/print).
            if spec.rendering_preset:
                res = call_operator(
                    "scigraphs.apply_rendering_preset",
                    {"preset": spec.rendering_preset},
                )
                self._collect_warnings(res, result)
                if res.get("status") == "error":
                    result.warnings.append(
                        f"Rendering preset failed: {res.get('error')}"
                    )

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
                        self._apply_color_mapping(spec, coloring, result)
                    res = call_operator("scigraphs.color_apply")
                    self._collect_warnings(res, result)
                    if res.get("status") == "error":
                        result.warnings.append(f"Node coloring failed: {res.get('error')}")

            # After coloring: the material these override does not exist until
            # color_apply has run.
            if spec.material_roughness is not None or spec.material_metallic is not None:
                obj = bpy.context.active_object
                if obj is not None and obj.type == 'MESH':
                    self._apply_material_overrides(spec, obj, result)

            # Node sizing by attribute (interactive geometry-nodes tree).
            if spec.node_size:
                attr = self._resolve_visual_attribute(spec.node_size)
                if attr is None:
                    result.warnings.append(
                        f"Node size attribute '{spec.node_size}' not found on graph"
                    )
                else:
                    self._apply_node_size_attribute(attr, result)

            # Edge width by attribute (interactive geometry-nodes tree).
            if spec.edge_width:
                attr = self._resolve_visual_attribute(spec.edge_width)
                if attr is None:
                    result.warnings.append(
                        f"Edge width attribute '{spec.edge_width}' not found on graph"
                    )
                else:
                    viz = getattr(bpy.context.scene, "scigraphs_viz", None)
                    if viz is not None and hasattr(viz, "edge_scale_attribute"):
                        try:
                            viz.edge_scale_attribute = attr
                        except (TypeError, ValueError):
                            result.warnings.append(
                                f"Edge width attribute '{attr}' not selectable"
                            )

            # Edge style
            if spec.edge_style:
                props = {"preset": spec.edge_style}
                res = call_operator("scigraphs.apply_edge_style_preset", props)
                self._collect_warnings(res, result)
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
                for warning in res.get("warnings", []) or []:
                    result.warnings.append(f"Op {op.id}: {warning}")
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

            self._apply_render_quality(spec, result)

            # A named camera is an explicit choice and is left alone.
            if getattr(spec, "frame_camera", True) and not spec.camera:
                self._frame_camera(spec, result)

            # Between framing and rendering: the projection needs the final
            # camera matrix, and the composite must be wired before the render.
            if self._pending_labels is not None:
                self._execute_labels(
                    self._pending_labels, output_dir, manifest, result
                )
                self._pending_labels = None

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
