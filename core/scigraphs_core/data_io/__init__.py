# Reading graphs in and writing them out: files, databases, SuiteSparse.

_LAZY = {
    # Blender-bound modules stayed with the add-on, as SciGraphs.core.<name>.
    "db_connector": ".db_connector",
    "export_utils": ".export_utils",
    "sql_importer": ".sql_importer",
    "suitesparse_importer": ".suitesparse_importer",
}

__all__ = list(_LAZY)


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target, __name__)
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
