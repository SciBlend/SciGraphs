# Shader library: name -> WGSL module plus the layout declared in a sibling
# .json (schema in README section 1). Layouts and bindings live there, never in
# Python. A missing shader returns None so the caller can degrade; a malformed
# one raises.

import json
import os

import wgpu

from ...mesh import ShaderRef

HERE = os.path.dirname(os.path.abspath(__file__))
WGSL_DIR = os.path.join(HERE, "wgsl")

# Checked against the array shape at upload; see buffers._column_array.
FORMAT_COLUMNS = {
    "float32": 1,
    "float32x2": 2,
    "float32x3": 3,
    "float32x4": 4,
}

VISIBILITY = {
    "vertex": wgpu.ShaderStage.VERTEX,
    "fragment": wgpu.ShaderStage.FRAGMENT,
    "compute": wgpu.ShaderStage.COMPUTE,
}

EXPAND_MODES = {"instanced_quad"}

# The two the renderer owns; any other must come via render(blocks=...).
CAMERA_BLOCK = "camera"
DRAW_BLOCK = "draw"

# Some .wgsl files spell the same two blocks with a `u_` prefix.
BINDING_ALIASES = {"u_camera": CAMERA_BLOCK, "u_draw": DRAW_BLOCK}

# The camera block is 224 bytes here and 208 in some declarations, identical
# over the first 208. Bind at the declared size; declare 208, get 224, fail.


class ShaderError(RuntimeError):
    """A shader pair exists and is wrong. Never raised for a missing one."""


def variant_name(ref):
    """File stem for a ShaderRef. This suffix table is duplicated in
    SciGraphs/ui/gpu_render/shaders.py and the two must not drift."""
    if isinstance(ref, str):
        return ref
    suffix = {(False, False): "", (True, False): "_f",
              (False, True): "_d", (True, True): "_fd"}[
        (bool(ref.filtered), bool(ref.animated))]
    return ref.base + suffix


class Attribute:
    __slots__ = ("name", "location", "format", "columns")

    def __init__(self, name, location, fmt):
        self.name = name
        self.location = int(location)
        self.format = fmt
        self.columns = FORMAT_COLUMNS[fmt]

    def __repr__(self):
        return f"<Attribute {self.name}@{self.location} {self.format}>"


class Binding:
    __slots__ = ("name", "group", "binding", "type", "visibility", "dynamic",
                 "size")

    def __init__(self, name, group, binding, btype, visibility, dynamic,
                 size=None):
        self.name = BINDING_ALIASES.get(name, name)
        self.group = int(group)
        self.binding = int(binding)
        self.type = btype
        self.visibility = visibility
        self.dynamic = bool(dynamic)
        self.size = None if size is None else int(size)

    def __repr__(self):
        return f"<Binding {self.name}@{self.group}.{self.binding}>"


class Expand:
    """The declared "one MeshSpec vertex becomes one primitive" rule, for the
    shaders standing in for a fixed-function size WebGPU lacks. README 3.2."""

    __slots__ = ("mode", "vertex_count", "topology")

    def __init__(self, mode, vertex_count, topology):
        self.mode = mode
        self.vertex_count = int(vertex_count)
        self.topology = topology


class Shader:
    __slots__ = ("name", "module", "vertex_entry", "fragment_entry",
                 "attributes", "bindings", "expand", "cull", "writes_depth",
                 "depth_compare", "declared_topology", "default_state",
                 "_by_name")

    def __init__(self, name, module, decl):
        self.name = name
        self.module = module
        self.vertex_entry = decl["vertex_entry"]
        self.fragment_entry = decl["fragment_entry"]
        self.attributes = decl["attributes"]
        self.bindings = decl["bindings"]
        self.expand = decl["expand"]
        self.cull = decl["cull"]
        self.writes_depth = decl["writes_depth"]
        self.depth_compare = decl["depth_compare"]
        self.declared_topology = decl["declared_topology"]
        # Beats the renderer's default, loses to the spec.
        self.default_state = decl["default_state"]
        self._by_name = {b.name: b for b in self.bindings}

    def binding(self, name):
        return self._by_name.get(name)

    @property
    def wants_camera(self):
        return CAMERA_BLOCK in self._by_name

    @property
    def wants_draw(self):
        return DRAW_BLOCK in self._by_name

    @property
    def layout_key(self):
        return tuple((a.name, a.location, a.format) for a in self.attributes)

    def __repr__(self):
        return f"<Shader {self.name} {len(self.attributes)} attrs>"


def _require(decl, key, name):
    if key not in decl:
        raise ShaderError(f"{name}.json: missing required key {key!r}")
    return decl[key]


def _normalize_schema1(decl, name):
    """Flatten the alternate declaration shape into this loader's, recognized
    by its "entry_points" key, which the native schema never has."""
    ep = decl.get("entry_points") or {}
    if "vertex" not in ep or "fragment" not in ep:
        raise ShaderError(
            f"{name}.json: 'entry_points' must name both 'vertex' and "
            f"'fragment'")

    bindings = []
    for group in decl.get("bind_groups", []) or []:
        gid = group.get("group", 0)
        for b in group.get("bindings", []) or []:
            bindings.append({
                "name": b.get("name", ""),
                "group": gid,
                "binding": b.get("binding", 0),
                "type": b.get("type", "uniform"),
                "visibility": b.get("visibility", ["vertex", "fragment"]),
                "dynamic": b.get("dynamic", False),
                "size": b.get("size"),
            })

    state = decl.get("state") or {}
    out = {
        "vertex_entry": ep["vertex"],
        "fragment_entry": ep["fragment"],
        "attributes": decl.get("attributes", []),
        "bindings": bindings,
        "expand": decl.get("expand"),
        "cull": state.get("cull", decl.get("cull", "none")),
        # Not @builtin(frag_depth); this is the depth buffer write.
        "writes_depth": bool(state.get("depth_write", True)),
        "depth_compare": state.get("depth_compare"),
        "declared_topology": decl.get("topology"),
        "state": {"blend": state.get("blend"),
                  "depth_test": state.get("depth_test"),
                  "depth_write": state.get("depth_write")},
    }
    return out


def _default_state(raw):
    """A DrawState from a declaration's 'state' object, or None."""
    if not raw:
        return None
    blend = raw.get("blend")
    if isinstance(blend, str):
        # Declarations spell them lowercase ('none'), DrawState uppercase.
        blend = blend.upper()
    if blend is None and raw.get("depth_test") is None \
            and raw.get("depth_write") is None:
        return None
    from ...mesh import DrawState
    return DrawState(blend=blend, depth_test=raw.get("depth_test"),
                     depth_write=raw.get("depth_write"))


def parse_declaration(decl, name):
    """Validate one .json into the objects above. Without these checks a bad
    declaration surfaces as a wgpu validation error that names no file."""
    if not isinstance(decl, dict):
        raise ShaderError(f"{name}.json: top level must be an object")

    if "entry_points" in decl:
        decl = _normalize_schema1(decl, name)

    vs = _require(decl, "vertex_entry", name)
    fs = _require(decl, "fragment_entry", name)
    if not isinstance(vs, str) or not isinstance(fs, str):
        raise ShaderError(f"{name}.json: entry points must be strings")

    raw_attrs = _require(decl, "attributes", name)
    if not isinstance(raw_attrs, list) or not raw_attrs:
        raise ShaderError(f"{name}.json: 'attributes' must be a non-empty list")

    attrs, seen_names, seen_locs = [], set(), set()
    for i, a in enumerate(raw_attrs):
        if not isinstance(a, dict):
            raise ShaderError(f"{name}.json: attributes[{i}] must be an object")
        for key in ("name", "location", "format"):
            if key not in a:
                raise ShaderError(
                    f"{name}.json: attributes[{i}] missing {key!r}")
        if a["format"] not in FORMAT_COLUMNS:
            raise ShaderError(
                f"{name}.json: attributes[{i}] format {a['format']!r} not one "
                f"of {sorted(FORMAT_COLUMNS)}")
        if a["name"] in seen_names:
            raise ShaderError(f"{name}.json: duplicate attribute {a['name']!r}")
        if a["location"] in seen_locs:
            raise ShaderError(
                f"{name}.json: duplicate location {a['location']}")
        seen_names.add(a["name"])
        seen_locs.add(int(a["location"]))
        attrs.append(Attribute(a["name"], a["location"], a["format"]))

    bindings, seen_slots = [], set()
    for i, b in enumerate(decl.get("bindings", []) or []):
        if not isinstance(b, dict):
            raise ShaderError(f"{name}.json: bindings[{i}] must be an object")
        for key in ("name", "group", "binding"):
            if key not in b:
                raise ShaderError(f"{name}.json: bindings[{i}] missing {key!r}")
        slot = (int(b["group"]), int(b["binding"]))
        if slot in seen_slots:
            raise ShaderError(
                f"{name}.json: two bindings at group {slot[0]} binding {slot[1]}")
        seen_slots.add(slot)
        btype = b.get("type", "uniform")
        if btype not in ("uniform", "read-only-storage", "storage"):
            raise ShaderError(f"{name}.json: bindings[{i}] type {btype!r}")
        vis = b.get("visibility", ["vertex", "fragment"])
        if isinstance(vis, str):
            vis = [vis]
        mask = 0
        for v in vis:
            if v not in VISIBILITY:
                raise ShaderError(
                    f"{name}.json: bindings[{i}] visibility {v!r} not one of "
                    f"{sorted(VISIBILITY)}")
            mask |= VISIBILITY[v]
        # Both "block_size" and "size" are in circulation; read one only and
        # the whole buffer gets bound, silently.
        bindings.append(Binding(b["name"], b["group"], b["binding"], btype,
                                mask, b.get("dynamic", False),
                                b.get("size", b.get("block_size"))))

    expand = None
    raw = decl.get("expand")
    if raw is not None:
        if not isinstance(raw, dict):
            raise ShaderError(f"{name}.json: 'expand' must be an object")
        mode = raw.get("mode")
        if mode not in EXPAND_MODES:
            raise ShaderError(
                f"{name}.json: expand.mode {mode!r} not one of "
                f"{sorted(EXPAND_MODES)}")
        expand = Expand(mode, raw.get("vertex_count", 4),
                        raw.get("topology", "triangle-strip"))

    cull = decl.get("cull", "none")
    if cull not in ("none", "front", "back"):
        raise ShaderError(f"{name}.json: cull {cull!r}")

    compare = decl.get("depth_compare") or "less-equal"
    if compare not in ("never", "less", "equal", "less-equal", "greater",
                       "not-equal", "greater-equal", "always"):
        raise ShaderError(f"{name}.json: depth_compare {compare!r}")

    return {
        "vertex_entry": vs,
        "fragment_entry": fs,
        "attributes": tuple(attrs),
        "bindings": tuple(bindings),
        "expand": expand,
        "cull": cull,
        "writes_depth": bool(decl.get("writes_depth", True)),
        "depth_compare": compare,
        "declared_topology": decl.get("declared_topology"),
        "default_state": _default_state(decl.get("state")),
    }


class ShaderLibrary:
    """Loads and caches the shaders in a directory, misses included. At 200k
    edges the repeated stat for a shader that is not there shows up."""

    def __init__(self, ctx, directory=WGSL_DIR):
        self.ctx = ctx
        self.directory = directory
        self._cache = {}

    def available(self):
        if not os.path.isdir(self.directory):
            return []
        out = []
        for entry in sorted(os.listdir(self.directory)):
            if not entry.endswith(".wgsl"):
                continue
            stem = entry[:-5]
            if os.path.exists(os.path.join(self.directory, stem + ".json")):
                out.append(stem)
        return out

    def get(self, ref):
        """The Shader for a ShaderRef or name. None means "not ported yet";
        a present but broken pair raises ShaderError."""
        name = variant_name(ref)
        if name in self._cache:
            return self._cache[name]
        shader = self._load(name)
        self._cache[name] = shader
        return shader

    def _load(self, name):
        wgsl_path = os.path.join(self.directory, name + ".wgsl")
        json_path = os.path.join(self.directory, name + ".json")
        if not os.path.exists(wgsl_path):
            return None
        if not os.path.exists(json_path):
            raise ShaderError(
                f"{name}.wgsl exists with no {name}.json; the JSON is the "
                f"layout declaration and there is no default for it")

        with open(json_path, "r", encoding="utf-8") as fh:
            try:
                raw = json.load(fh)
            except ValueError as exc:
                raise ShaderError(f"{name}.json is not valid JSON: {exc}") from exc
        decl = parse_declaration(raw, name)

        with open(wgsl_path, "r", encoding="utf-8") as fh:
            source = fh.read()
        for entry in (decl["vertex_entry"], decl["fragment_entry"]):
            if entry not in source:
                raise ShaderError(
                    f"{name}.json names entry point {entry!r}, which does not "
                    f"appear in {name}.wgsl")
        try:
            module = self.ctx.device.create_shader_module(label=name,
                                                          code=source)
        except Exception as exc:    # noqa: BLE001 - naga reports many types
            raise ShaderError(f"{name}.wgsl failed to compile: {exc}") from exc

        return Shader(name, module, decl)


def resolve(library, ref):
    """``library.get(ref)``, tolerating a library of None."""
    if library is None:
        return None
    return library.get(ref if isinstance(ref, (str, ShaderRef))
                       else ShaderRef(str(ref)))
