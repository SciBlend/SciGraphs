// SciGraphs: sphere impostor, filtered. WGSL port of
// shaders.py::_build_sphere(filtered=True). It is sphere.wgsl, the shared
// filter block, and one guard line at the top of vs_main. Nothing else
// differs, so the filter cannot reach the unfiltered path, which is what
// tests/test_filtered.py checks.

// Replicated verbatim; WGSL has no #include. Do not edit one copy alone.
struct Camera {
    view       : mat4x4<f32>,   // world -> view                       offset   0
    proj       : mat4x4<f32>,   // view  -> clip, z in [0, 1]          offset  64
    view_proj  : mat4x4<f32>,   // the product, precomputed on the CPU offset 128
    viewport   : vec4<f32>,     // (w, h, 1/w, 1/h)                    offset 192
    params     : vec4<f32>,     // (near, far, aspect, unused)         offset 208
};

// draw.py::compute_lighting collapses the lamps into these three terms, in
// view space. Byte-identical to the GLSL std140 block. README §2.
struct LightRig {
    key_dir  : vec4<f32>,   // xyz = direction TOWARD the key light  offset   0
    key_col  : vec4<f32>,   // rgb = key color, w = rim strength    offset  16
    fill_dir : vec4<f32>,   // xyz = direction toward combined fill  offset  32
    fill_col : vec4<f32>,   // rgb = fill color; w unused           offset  48
    ambient  : vec4<f32>,   // rgb = ambient term; w unused          offset  64
};

// Binding 1 is `draw`, unread here: slots mean the same thing folder-wide.
@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(2) var<uniform> lights : LightRig;

// >>> SCIG FILTER BLOCK <<<
// Replicated verbatim in all four *_f files; WGSL has no #include, so edit
// all four or none. README §5.

// filter_gpu.py::uniform_rows rewrites these 128 bytes every frame, which makes
// a threshold slider one buffer write, 0.04 ms, against a 1.2 s batch rebuild
// on a 2M-node graph. Four clauses, because 8 vec4 is the block Blender
// guarantees.
struct FilterStack {
    lo   : vec4<f32>,   // node clauses: low end of each kept range   offset   0
    hi   : vec4<f32>,   // node clauses: high end                     offset  16
    inv  : vec4<f32>,   // node clauses: 1 = keep what falls OUTSIDE  offset  32
    act  : vec4<f32>,   // node clauses: 1 = this slot is in use      offset  48
    elo  : vec4<f32>,   // edge clauses, the same four fields         offset  64
    ehi  : vec4<f32>,   //                                            offset  80
    einv : vec4<f32>,   //                                            offset  96
    eact : vec4<f32>,   //                                            offset 112
};

@group(0) @binding(3) var<uniform> fstack : FilterStack;

// One vec4 per node, then one per edge; the GLSL's 4096-wide texture exists
// only because Blender's `gpu` has no storage buffer.
@group(0) @binding(4) var<storage, read> fchan : array<vec4<f32>>;

// `act` raises an unused slot to 1, which is why its lo/hi may hold anything.
fn scig_span(v: vec4<f32>, lo: vec4<f32>, hi: vec4<f32>,
             inv: vec4<f32>, act: vec4<f32>) -> bool {
    let ins = step(lo, v) * step(v, hi);
    var keep = mix(ins, vec4<f32>(1.0) - ins, inv);
    keep = max(keep, vec4<f32>(1.0) - act);
    return min(min(keep.x, keep.y), min(keep.z, keep.w)) > 0.5;
}

// `rows` is `frow`: (own node row, other endpoint's row, own edge row).
// Testing both .x and .y makes a node clause take the edges that would dangle.
fn scig_keep(rows: vec3<f32>) -> bool {
    if (any(fstack.act != vec4<f32>(0.0))) {
        if (!scig_span(fchan[i32(rows.x)], fstack.lo, fstack.hi,
                       fstack.inv, fstack.act)) {
            return false;
        }
        // && short-circuits where & does not, so a node vertex skips this.
        if (rows.y != rows.x
            && !scig_span(fchan[i32(rows.y)], fstack.lo, fstack.hi,
                          fstack.inv, fstack.act)) {
            return false;
        }
    }
    if (rows.z >= 0.0 && any(fstack.eact != vec4<f32>(0.0))) {
        if (!scig_span(fchan[i32(rows.z)], fstack.elo, fstack.ehi,
                       fstack.einv, fstack.eact)) {
            return false;
        }
    }
    return true;
}

// x and y cull a rejected vertex under both APIs, not z. Collapsing it to a
// point is not reliably discarded, and mutating the constant is unobservable
// on this GPU (README §5.3).
const SCIG_CULL : vec4<f32> = vec4<f32>(2.0, 2.0, 2.0, 1.0);
// >>> END SCIG FILTER BLOCK <<<

// @location order is MeshSpec.attrs in mesh.py::sphere_spec.
struct VertexIn {
    @location(0) pos: vec3<f32>,     // node center, world space
    @location(1) corner: vec2<f32>,  // quad corner in [-1, 1]^2 == sphere xy
    @location(2) radius: f32,        // sphere radius, world/view units
    @location(3) color: vec4<f32>,
    @location(4) frow: vec3<f32>,    // (own row, other endpoint's row, edge row)
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) view_center: vec3<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,
}

struct FragmentOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

// clip -> window depth. w <= 0 puts the surface at or behind the eye, where
// the divide hands the depth test a NaN; the clamp keeps it driver-independent.
fn depth_from_clip(clip: vec4<f32>) -> f32 {
    if (clip.w <= 0.0) {
        return 0.0;
    }
    return clamp(clip.z / clip.w, 0.0, 1.0);
}

// The vertex stage returns a struct, so reject by value; the rest is zeroed.
fn scig_reject() -> VertexOut {
    var out: VertexOut;
    out.clip_pos = SCIG_CULL;
    return out;
}

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    // First statement, where shaders.py::_guard() puts it.
    if (!scig_keep(v.frow)) { return scig_reject(); }

    let view_center = (camera.view * vec4<f32>(v.pos, 1.0)).xyz;

    // A sphere's true silhouette under perspective is slightly wider than its
    // radius. The GLSL approximates it the same way, kept for parity.
    let view_pos = view_center + vec3<f32>(v.corner * v.radius, 0.0);

    var out: VertexOut;
    out.clip_pos = camera.proj * vec4<f32>(view_pos, 1.0);
    out.uv = v.corner;
    out.view_center = view_center;
    out.radius = v.radius;
    out.color = v.color;
    return out;
}

@fragment
fn fs_main(f: VertexOut) -> FragmentOut {
    let p = f.uv;
    let r2 = dot(p, p);
    if (r2 > 1.0) {
        discard;
    }

    // +z points at the camera, so the visible hemisphere is the +root.
    let z = sqrt(max(0.0, 1.0 - r2));
    let normal = vec3<f32>(p.x, p.y, z);

    let view_surface = f.view_center + f.radius * normal;
    let clip = camera.proj * vec4<f32>(view_surface, 1.0);

    let d_key = max(dot(normal, lights.key_dir.xyz), 0.0);
    let d_fill = max(dot(normal, lights.fill_dir.xyz), 0.0);

    // Brightest at the silhouette, so same-colored nodes stay apart.
    let rim = pow(1.0 - z, 3.0) * lights.key_col.w;

    // WGSL will not broadcast a scalar into a vector as GLSL did.
    let lit = f.color.rgb * (lights.ambient.rgb
                             + lights.key_col.rgb * d_key
                             + lights.fill_col.rgb * d_fill)
              + vec3<f32>(rim);

    var out: FragmentOut;
    // Alpha forced to 1. Blending is off, and a node's own alpha would pull
    // the background into an opaque object.
    out.color = vec4<f32>(min(lit, vec3<f32>(1.0)), 1.0);
    out.depth = depth_from_clip(clip);
    return out;
}
