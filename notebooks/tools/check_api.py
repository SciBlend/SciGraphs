# Does every name the notebooks reach for actually exist?
#     python3 notebooks/tools/check_api.py
# Resolves `sg.<module>.<name>` and `nb.<name>` against the AST of SciGraphs/api
# and nb.py, so a typo fails here, not mid-notebook. No Blender, no kernel.

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
API = ROOT / "SciGraphs" / "api"
NB = ROOT / "notebooks" / "tools" / "nb.py"
SRC = ROOT / "notebooks" / "_src"

API_MODULES = ("context", "graphs", "preview", "render", "thin")


def top_level_names(path):
    """Every name a module binds at module scope."""
    names = set()
    for stmt in ast.parse(path.read_text(), filename=str(path)).body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            names |= {t.id for t in stmt.targets if isinstance(t, ast.Name)}
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            names |= {a.asname or a.name.split(".")[0]
                      for a in stmt.names if a.name != "*"}
    return names


def surfaces():
    found = {"nb": top_level_names(NB)}
    for module in API_MODULES:
        found[f"sg.{module}"] = top_level_names(API / f"{module}.py")
    return found


def references():
    """Every `sg.<module>.<name>` and `nb.<name>` written in the sources."""
    used = {}
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            inner = node.value
            if isinstance(inner, ast.Name) and inner.id == "nb":
                key = "nb"
            elif isinstance(inner, ast.Attribute) and isinstance(inner.value, ast.Name) \
                    and inner.value.id == "sg":
                key = f"sg.{inner.attr}"
            else:
                continue
            used.setdefault(key, {}).setdefault(node.attr, []).append(
                f"{path.name}:{node.lineno}")
    return used


def shadowed_calls():
    """Module functions called from a scope that shadows them with a local.
    `sg.preview.render` took a flag named after the function it called, resolved
    cleanly, and raised `TypeError: 'bool' object is not callable` at run time.
    """
    bad = []
    for path in sorted(API.glob("*.py")) + [NB]:
        tree = ast.parse(path.read_text(), filename=str(path))
        module_functions = {n.name for n in tree.body
                            if isinstance(n, ast.FunctionDef)}
        for fn in tree.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            local = {a.arg for a in fn.args.args}
            local |= {a.arg for a in fn.args.kwonlyargs}
            for node in ast.walk(fn):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    local.add(node.id)
            shadowed = local & module_functions
            if not shadowed:
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id in shadowed:
                    bad.append(f"{path.name}:{node.lineno}  {fn.name}() calls "
                               f"{node.func.id}() but shadows it with a local")
    return bad


def dynamic_module_paths():
    """Module paths the notebooks assemble as strings, checked against the tree.
    They name the add-on because a module-level cache in a second copy is a
    different dict, and the string escapes static checking: when 14 of 22 such
    modules moved into the scigraphs_core wheel, six notebooks died on cell one.
    """
    import re

    addon_core = ROOT / "SciGraphs" / "core"
    wheel = ROOT / "core" / "scigraphs_core"
    bad = []
    pattern = re.compile(r'INSTALLED\s*\+\s*\n?\s*"\.core\.([a-z_.]+)"')
    for path in sorted(SRC.glob("*.py")):
        # Blanked, not dropped: the notebooks discuss this, line numbers hold.
        text = "\n".join("" if line.lstrip().startswith("#") else line
                         for line in path.read_text().splitlines())
        for match in pattern.finditer(text):
            dotted = match.group(1)
            parts = dotted.split(".")
            in_addon = ((addon_core.joinpath(*parts).with_suffix(".py")).is_file()
                        or (addon_core.joinpath(*parts) / "__init__.py").is_file())
            in_wheel = ((wheel.joinpath(*parts).with_suffix(".py")).is_file()
                        or (wheel.joinpath(*parts) / "__init__.py").is_file())
            if in_addon:
                continue
            line = text[:match.start()].count("\n") + 1
            where = "moved to the wheel: use scigraphs_core." + dotted \
                if in_wheel else "names no module in either tree"
            bad.append(f"{path.name}:{line}  INSTALLED + '.core.{dotted}'  {where}")
    return bad


def stale_imports():
    """Nothing should import the layer that was deleted."""
    bad = []
    tools = [q for q in sorted((ROOT / "notebooks" / "tools").glob("*.py"))
             if q.name != pathlib.Path(__file__).name]
    for path in sorted(SRC.glob("*.py")) + tools:
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if "sgnb" in line or "nbsupport" in line:
                bad.append(f"{path.name}:{i}  {line.strip()[:80]}")
    return bad


def main():
    known = surfaces()
    for name in sorted(known):
        print(f"{name}: {len(known[name])} names")

    used = references()
    missing, total = [], 0
    for module in sorted(used):
        for attr, sites in sorted(used[module].items()):
            total += 1
            if module not in known:
                missing.append((f"{module} (no such module)", attr, sites))
            elif attr not in known[module]:
                missing.append((module, attr, sites))

    sources = list(SRC.glob("*.py"))
    print(f"\n{total} distinct attributes across {len(sources)} notebook sources")

    shadowed = shadowed_calls()
    if shadowed:
        print(f"\n{len(shadowed)} shadowed calls (these raise at runtime):")
        for line in shadowed:
            print(f"  {line}")

    dynamic = dynamic_module_paths()
    if dynamic:
        print(f"\n{len(dynamic)} run-time module path(s) that no longer resolve:")
        for line in dynamic:
            print(f"  {line}")

    stale = stale_imports()
    if stale:
        print(f"\n{len(stale)} references to the removed sgnb layer:")
        for line in stale:
            print(f"  {line}")

    if missing:
        print(f"\n{len(missing)} DO NOT RESOLVE:")
        for module, attr, sites in missing:
            print(f"  {module}.{attr}")
            for site in sites[:4]:
                print(f"      {site}")
    if missing or stale or shadowed or dynamic:
        return 1
    print("\nEVERY REFERENCE RESOLVES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
