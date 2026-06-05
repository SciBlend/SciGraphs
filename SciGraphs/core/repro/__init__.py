# Reproducible Pipelines for SciGraphs
#
# This package provides a declarative pipeline system for reproducible
# graph visualization and analysis workflows.

from .schema import PipelineSchema, validate_pipeline, get_default_pipeline
from .parser import parse_pipeline, canonicalize_pipeline, compute_pipeline_hash
from .executor import PipelineExecutor, ExecutionResult
from .registry import OperatorRegistry, get_registry
from .provenance import ProvenanceManifest, create_manifest, save_manifest
from .determinism import set_deterministic_seed, get_seed_context, SeedContext
from .reference import generate_reference_markdown, write_reference_markdown

__all__ = [
    # Schema
    'PipelineSchema',
    'validate_pipeline',
    'get_default_pipeline',
    # Parser
    'parse_pipeline',
    'canonicalize_pipeline',
    'compute_pipeline_hash',
    # Executor
    'PipelineExecutor',
    'ExecutionResult',
    # Registry
    'OperatorRegistry',
    'get_registry',
    # Provenance
    'ProvenanceManifest',
    'create_manifest',
    'save_manifest',
    # Determinism
    'set_deterministic_seed',
    'get_seed_context',
    'SeedContext',
    # Reference
    'generate_reference_markdown',
    'write_reference_markdown',
]
