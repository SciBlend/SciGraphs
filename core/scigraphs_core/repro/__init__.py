# Declarative pipeline system for reproducible graph visualization and analysis.
#
# Nothing is imported eagerly (see scigraphs_core/__init__.py). Eagerly pulling
# Blender-bound siblings would make schema/parser/determinism unusable outside
# Blender.
#
# Two tables: callers need re-exported names (PipelineSchema, …), and _LAZY must
# stay string-valued — tests/purity/test_core_purity.py follows strings only via
# ast.literal_eval; tuples would silently drop edges from the purity graph.

# Submodule -> relative import path (purity test reads these).
_LAZY = {
    'determinism': '.determinism',
    'digest': '.digest',
    'untrusted': '.untrusted',
    'parser': '.parser',
    'provenance': '.provenance',
    'schema': '.schema',
}

# Re-exported name -> (submodule, attribute).
# Executor/registry stay out: they need a live scene; import from
# SciGraphs.core.repro.executor instead.
_LAZY_ATTRS = {
    'PipelineSchema': ('.schema', 'PipelineSchema'),
    'validate_pipeline': ('.schema', 'validate_pipeline'),
    'get_default_pipeline': ('.schema', 'get_default_pipeline'),
    'parse_pipeline': ('.parser', 'parse_pipeline'),
    'canonicalize_pipeline': ('.parser', 'canonicalize_pipeline'),
    'compute_pipeline_hash': ('.parser', 'compute_pipeline_hash'),
    'ProvenanceManifest': ('.provenance', 'ProvenanceManifest'),
    'create_manifest': ('.provenance', 'create_manifest'),
    'save_manifest': ('.provenance', 'save_manifest'),
    'set_deterministic_seed': ('.determinism', 'set_deterministic_seed'),
    'get_seed_context': ('.determinism', 'get_seed_context'),
    'SeedContext': ('.determinism', 'SeedContext'),
    'generate_reference_markdown': ('.reference', 'generate_reference_markdown'),
    'write_reference_markdown': ('.reference', 'write_reference_markdown'),
}

__all__ = [
    'PipelineSchema',
    'validate_pipeline',
    'get_default_pipeline',
    'parse_pipeline',
    'canonicalize_pipeline',
    'compute_pipeline_hash',
    'ProvenanceManifest',
    'create_manifest',
    'save_manifest',
    'set_deterministic_seed',
    'get_seed_context',
    'SeedContext',
    'generate_reference_markdown',
    'write_reference_markdown',
]


def __getattr__(name):
    """Import a submodule or re-exported name on first access (PEP 562)."""
    import importlib

    target = _LAZY.get(name)
    if target is not None:
        module = importlib.import_module(target, __name__)
        globals()[name] = module
        return module

    pair = _LAZY_ATTRS.get(name)
    if pair is not None:
        module = importlib.import_module(pair[0], __name__)
        value = getattr(module, pair[1])
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY) | set(_LAZY_ATTRS))
