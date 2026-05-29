# Provenance manifest generation for reproducible SciGraphs workflows
#
# Records environment, dependencies, inputs, outputs and hashes for
# full reproducibility of pipeline executions.

import datetime
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EnvironmentInfo:
    """Captured environment information."""
    platform: str
    platform_version: str
    python_version: str
    blender_version: str
    scigraphs_version: str
    timestamp: str
    hostname: str


@dataclass
class DependencyInfo:
    """Package dependency information."""
    name: str
    version: str


@dataclass
class ArtifactInfo:
    """Information about a generated artifact."""
    path: str
    hash: str
    size: int
    type: str


@dataclass
class InputInfo:
    """Information about an input file/resource."""
    path: str
    hash: Optional[str]
    source: str  # 'file', 'network', 'cache'
    pinned: bool  # Whether the data is frozen/cached


@dataclass
class StepInfo:
    """Information about an execution step."""
    name: str
    operator: str
    start_time: str
    end_time: str
    duration_ms: int
    status: str  # 'success', 'error', 'skipped'
    error: Optional[str] = None


@dataclass
class ProvenanceManifest:
    """Complete provenance manifest for a pipeline run."""
    pipeline_hash: str
    pipeline_title: str
    seed: int
    environment: EnvironmentInfo
    dependencies: List[DependencyInfo]
    inputs: List[InputInfo]
    outputs: List[ArtifactInfo]
    steps: List[StepInfo]
    started_at: str
    completed_at: str
    duration_ms: int
    success: bool
    warnings: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def _get_blender_version() -> str:
    """Get Blender version string."""
    try:
        import bpy
        return ".".join(str(v) for v in bpy.app.version)
    except ImportError:
        return "unknown"


def _get_scigraphs_version() -> str:
    """Get SciGraphs addon version."""
    try:
        # Try reading from manifest
        manifest_path = Path(__file__).parent.parent.parent.parent / "blender_manifest.toml"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                for line in f:
                    if line.startswith("version"):
                        return line.split("=")[1].strip().strip('"')
    except Exception:
        pass
    return "unknown"


def _get_dependency_versions() -> List[DependencyInfo]:
    """Get versions of key dependencies."""
    deps = []

    packages = [
        ("networkx", "networkx"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("igraph", "igraph"),
        ("osmnx", "osmnx"),
        ("shapely", "shapely"),
        ("geopandas", "geopandas"),
    ]

    for name, module in packages:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            deps.append(DependencyInfo(name=name, version=version))
        except ImportError:
            pass

    return deps


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_environment_info() -> EnvironmentInfo:
    """Capture current environment information."""
    import socket
    return EnvironmentInfo(
        platform=platform.system(),
        platform_version=platform.release(),
        python_version=sys.version.split()[0],
        blender_version=_get_blender_version(),
        scigraphs_version=_get_scigraphs_version(),
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        hostname=socket.gethostname(),
    )


def create_manifest(
    pipeline_hash: str,
    pipeline_title: str,
    seed: int,
) -> ProvenanceManifest:
    """
    Create a new provenance manifest for a pipeline run.

    Args:
        pipeline_hash: Hash of the canonical pipeline
        pipeline_title: Pipeline title from meta
        seed: Global seed used

    Returns:
        New ProvenanceManifest instance
    """
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return ProvenanceManifest(
        pipeline_hash=pipeline_hash,
        pipeline_title=pipeline_title,
        seed=seed,
        environment=get_environment_info(),
        dependencies=_get_dependency_versions(),
        inputs=[],
        outputs=[],
        steps=[],
        started_at=now,
        completed_at=now,
        duration_ms=0,
        success=False,
    )


def add_input(
    manifest: ProvenanceManifest,
    path: str,
    source: str = "file",
    pinned: bool = True,
) -> None:
    """
    Add an input file/resource to the manifest.

    Args:
        manifest: Manifest to update
        path: Path to input file
        source: Source type ('file', 'network', 'cache')
        pinned: Whether the data is frozen/reproducible
    """
    file_hash = None
    if os.path.isfile(path):
        try:
            file_hash = compute_file_hash(path)
        except Exception:
            pass

    manifest.inputs.append(InputInfo(
        path=path,
        hash=file_hash,
        source=source,
        pinned=pinned,
    ))


def add_output(
    manifest: ProvenanceManifest,
    path: str,
    artifact_type: str = "file",
) -> None:
    """
    Add an output artifact to the manifest.

    Args:
        manifest: Manifest to update
        path: Path to output file
        artifact_type: Type of artifact ('render', 'export', 'blend', etc.)
    """
    if not os.path.isfile(path):
        return

    file_hash = compute_file_hash(path)
    file_size = os.path.getsize(path)

    manifest.outputs.append(ArtifactInfo(
        path=path,
        hash=file_hash,
        size=file_size,
        type=artifact_type,
    ))


def add_step(
    manifest: ProvenanceManifest,
    name: str,
    operator: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    status: str = "success",
    error: Optional[str] = None,
) -> None:
    """
    Add an execution step to the manifest.

    Args:
        manifest: Manifest to update
        name: Step name
        operator: Operator bl_idname
        start_time: Step start time
        end_time: Step end time
        status: Step status ('success', 'error', 'skipped')
        error: Error message if status is 'error'
    """
    duration_ms = int((end_time - start_time).total_seconds() * 1000)
    manifest.steps.append(StepInfo(
        name=name,
        operator=operator,
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        duration_ms=duration_ms,
        status=status,
        error=error,
    ))


def finalize_manifest(
    manifest: ProvenanceManifest,
    success: bool,
) -> None:
    """
    Finalize a manifest after pipeline execution.

    Args:
        manifest: Manifest to finalize
        success: Whether execution succeeded
    """
    started = datetime.datetime.fromisoformat(manifest.started_at)
    completed = datetime.datetime.now(datetime.timezone.utc)

    manifest.completed_at = completed.isoformat()
    manifest.duration_ms = int((completed - started).total_seconds() * 1000)
    manifest.success = success


def save_manifest(manifest: ProvenanceManifest, filepath: str) -> None:
    """
    Save manifest to JSON file.

    Args:
        manifest: Manifest to save
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(manifest.to_dict(), f, indent=2)


def load_manifest(filepath: str) -> ProvenanceManifest:
    """
    Load manifest from JSON file.

    Args:
        filepath: Path to manifest file

    Returns:
        ProvenanceManifest instance
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Reconstruct nested dataclasses
    env = EnvironmentInfo(**data["environment"])
    deps = [DependencyInfo(**d) for d in data["dependencies"]]
    inputs = [InputInfo(**i) for i in data["inputs"]]
    outputs = [ArtifactInfo(**o) for o in data["outputs"]]
    steps = [StepInfo(**s) for s in data["steps"]]

    return ProvenanceManifest(
        pipeline_hash=data["pipeline_hash"],
        pipeline_title=data["pipeline_title"],
        seed=data["seed"],
        environment=env,
        dependencies=deps,
        inputs=inputs,
        outputs=outputs,
        steps=steps,
        started_at=data["started_at"],
        completed_at=data["completed_at"],
        duration_ms=data["duration_ms"],
        success=data["success"],
        warnings=data.get("warnings", []),
        notes=data.get("notes", ""),
    )


def compare_manifests(
    manifest1: ProvenanceManifest,
    manifest2: ProvenanceManifest,
) -> Dict[str, Any]:
    """
    Compare two manifests for reproducibility verification.

    Args:
        manifest1: First manifest
        manifest2: Second manifest

    Returns:
        Comparison report dictionary
    """
    report = {
        "same_pipeline": manifest1.pipeline_hash == manifest2.pipeline_hash,
        "same_seed": manifest1.seed == manifest2.seed,
        "same_environment": (
            manifest1.environment.blender_version == manifest2.environment.blender_version and
            manifest1.environment.scigraphs_version == manifest2.environment.scigraphs_version
        ),
        "output_differences": [],
        "input_differences": [],
    }

    # Compare inputs
    inputs1 = {i.path: i.hash for i in manifest1.inputs}
    inputs2 = {i.path: i.hash for i in manifest2.inputs}

    for path, hash1 in inputs1.items():
        hash2 = inputs2.get(path)
        if hash2 is None:
            report["input_differences"].append({"path": path, "issue": "missing_in_second"})
        elif hash1 != hash2:
            report["input_differences"].append({"path": path, "issue": "hash_mismatch"})

    for path in inputs2:
        if path not in inputs1:
            report["input_differences"].append({"path": path, "issue": "missing_in_first"})

    # Compare outputs
    outputs1 = {o.path: o.hash for o in manifest1.outputs}
    outputs2 = {o.path: o.hash for o in manifest2.outputs}

    for path, hash1 in outputs1.items():
        hash2 = outputs2.get(os.path.basename(path))  # Compare by filename
        if hash2 is None:
            # Try exact path
            hash2 = outputs2.get(path)
        if hash2 is None:
            report["output_differences"].append({"path": path, "issue": "missing_in_second"})
        elif hash1 != hash2:
            report["output_differences"].append({"path": path, "issue": "hash_mismatch"})

    report["reproducible"] = (
        report["same_pipeline"] and
        report["same_seed"] and
        len(report["output_differences"]) == 0 and
        len(report["input_differences"]) == 0
    )

    return report
