# Provenance manifests: environment, dependencies, inputs, outputs and hashes
# for a pipeline run.

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
    # Defaulted so older manifests still load.
    machine: str = "unknown"
    libc_version: str = "unknown"


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
    # Root the recorded artifact paths hang from, so two runs that wrote to
    # different directories can still be compared. Defaulted for older manifests.
    output_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def _detect_blender_version() -> str:
    """Return the host Blender's version, or "unknown" outside Blender.

    Reads sys.modules instead of importing bpy, so this module stays importable
    from a notebook or a test process. Inside Blender the answer is the same,
    since bpy is already imported before any add-on code runs.

    A plain python3 process with the bpy PyPI wheel installed but not imported
    gets "unknown". That wheel is not the Blender that ran the pipeline, and
    recording its version would make the manifest claim something untrue.
    """
    bpy = sys.modules.get("bpy")
    app = getattr(bpy, "app", None)
    version = getattr(app, "version", None)
    if version is None:
        return "unknown"
    try:
        return ".".join(str(v) for v in version)
    except TypeError:
        return "unknown"


def _get_scigraphs_version() -> str:
    """Get the SciGraphs add-on version, or "unknown"."""
    try:
        manifest_path = Path(__file__).parent.parent.parent.parent / "blender_manifest.toml"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                for line in f:
                    if line.startswith("version"):
                        return line.split("=")[1].strip().strip('"')
    except Exception:
        pass
    return "unknown"


# Their build determines the output, so they belong in the manifest whenever
# installed, even if the run never reached the stage that imports them.
_CRITICAL_DISTRIBUTIONS = (
    "scigraphs-utils",   # native Graphviz layout binding
    "pysurprise",        # native community detection
    "rustworkx",         # native betweenness
    "city2graph",
    "networkx",
    "numpy",
    "scipy",
    "pandas",
    "igraph",
    "python-igraph",
    "osmnx",
    "shapely",
    "geopandas",
)


def _get_dependency_versions() -> List[DependencyInfo]:
    """Resolve versions of the distributions this process actually imported.

    Keeps the installed distributions backing a top-level module already in
    sys.modules, so the manifest records what the run used rather than a fixed
    list. _CRITICAL_DISTRIBUTIONS are always included when installed.
    """
    import importlib.metadata as importlib_metadata

    versions: Dict[str, str] = {}

    # dist name -> version, for everything visible to this interpreter.
    installed: Dict[str, str] = {}
    try:
        for dist in importlib_metadata.distributions():
            name = (dist.metadata or {}).get("Name")
            if not name:
                continue
            installed[name] = dist.version or "unknown"
    except Exception:
        pass

    # top-level module name -> [distribution names].
    try:
        module_map = importlib_metadata.packages_distributions()
    except Exception:
        module_map = {}

    imported_tops = {name.split(".", 1)[0] for name in list(sys.modules)}
    for top, dist_names in module_map.items():
        if top not in imported_tops:
            continue
        for dist_name in dist_names:
            if dist_name in installed:
                versions[dist_name] = installed[dist_name]

    for dist_name in _CRITICAL_DISTRIBUTIONS:
        if dist_name in installed:
            versions[dist_name] = installed[dist_name]

    return [DependencyInfo(name=n, version=versions[n]) for n in sorted(versions)]


def _get_libc_version() -> str:
    """Return the C library identification, where the platform exposes one."""
    try:
        name, version = platform.libc_ver()
        if name:
            return f"{name} {version}".strip()
    except Exception:
        pass
    return "unknown"


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_environment_info(blender_version: Optional[str] = None) -> EnvironmentInfo:
    """Capture current environment information.

    blender_version comes from the Blender-side caller, for example "5.2.0".
    None reads it off the host interpreter instead.
    """
    import socket
    if blender_version is None:
        blender_version = _detect_blender_version()
    return EnvironmentInfo(
        platform=platform.system(),
        platform_version=platform.release(),
        python_version=sys.version.split()[0],
        blender_version=blender_version,
        scigraphs_version=_get_scigraphs_version(),
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        hostname=socket.gethostname(),
        machine=platform.machine(),
        libc_version=_get_libc_version(),
    )


def create_manifest(
    pipeline_hash: str,
    pipeline_title: str,
    seed: int,
    blender_version: Optional[str] = None,
) -> ProvenanceManifest:
    """Create a provenance manifest for a pipeline run.

    blender_version is supplied by the caller that has one. None falls back to
    _detect_blender_version.
    """
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return ProvenanceManifest(
        pipeline_hash=pipeline_hash,
        pipeline_title=pipeline_title,
        seed=seed,
        environment=get_environment_info(blender_version),
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
    """Record an input file or resource.

    source is 'file', 'network' or 'cache'. pinned means the data is frozen and
    so reproducible.
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
    """Record an output artifact, hashed and sized.

    artifact_type is free text such as 'render', 'export' or 'blend'.
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
    """Record one execution step, with its duration in milliseconds.

    operator is the Blender operator's bl_idname. status is 'success', 'error'
    or 'skipped'; error carries the message when it is 'error'.
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
    """Close out a manifest after execution: timings, success, dependencies."""
    # Re-read the dependencies. Layout and analysis backends load lazily, so the
    # list taken at create_manifest() time misses exactly the native libraries
    # whose build decides the result.
    manifest.dependencies = _get_dependency_versions()

    started = datetime.datetime.fromisoformat(manifest.started_at)
    completed = datetime.datetime.now(datetime.timezone.utc)

    manifest.completed_at = completed.isoformat()
    manifest.duration_ms = int((completed - started).total_seconds() * 1000)
    manifest.success = success


def save_manifest(manifest: ProvenanceManifest, filepath: str) -> None:
    """Write a manifest to a JSON file."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(manifest.to_dict(), f, indent=2)


def load_manifest(filepath: str) -> ProvenanceManifest:
    """Read a manifest back from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

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
        output_dir=data.get("output_dir", ""),
    )


def compare_manifests(
    manifest1: ProvenanceManifest,
    manifest2: ProvenanceManifest,
) -> Dict[str, Any]:
    """Compare two manifests and report whether the run reproduced.

    The report flags pipeline, seed and environment equality, lists per-path
    input and output differences, and sets "reproducible" when nothing differs.
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

    outputs1 = {o.path: o.hash for o in manifest1.outputs}
    outputs2 = {o.path: o.hash for o in manifest2.outputs}

    for path, hash1 in outputs1.items():
        hash2 = outputs2.get(os.path.basename(path))  # by filename
        if hash2 is None:
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
