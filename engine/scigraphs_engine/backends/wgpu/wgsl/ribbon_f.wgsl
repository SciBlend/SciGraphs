// Edge ribbon (cylinder impostor), filtered. WGSL port of
// shaders.py::_build_ribbon(filtered=True): ribbon.wgsl, the shared filter
// block, and one line at the top of vs_main.
//
// An edge vertex tests both endpoints against the node clauses and its own row
// against the edge clauses, so a node clause also takes the edges that would
// dangle off it. mesh.ribbon_spec repeats one frow across all four corners.

// Replicated verbatim; do not edit one copy. See sphere.wgsl.
struct Camera {
    view       : mat4x4<f32>,   // world -> view                       offset   0
    proj       : mat4x4<f32>,   // view  -> clip, z in [0, 1]          offset  64
    view_proj  : mat4x4<f32>,   // the product, precomputed on the CPU offset 128
    viewport   : vec4<f32>,     // (w, h, 1/w, 1/h)                    offset 192
    params     : vec4<f32>,     // (near, far, aspect, unused)         offset 208
};

struct LightRig {
    key_dir  : vec4<f32>,   // xyz = direction TOWARD the key light  offset   0
    key_col  : vec4<f32>,   // rgb = key color, w = rim strength    offset  16
    fill_dir : vec4<f32>,   // xyz = direction toward combined fill  offset  32
    fill_col : vec4<f32>,   // rgb = fill color; w unused           offset  48
    ambient  : vec4<f32>,   // rgb = ambient term; w unused          offset  64
};

// Binding 1 (`draw`) goes unread here; slot numbers match across the folder.
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

// @location order is MeshSpec.attrs in mesh.py::ribbon_spec. Both endpoints ride
// on all four vertices, or the corners pick different perpendiculars and shear.
struct VertexIn {
    @location(0) pos_a: vec3<f32>,   // segment start, world space
    @location(1) pos_b: vec3<f32>,   // segment end, world space
    @location(2) side: f32,          // -1 or +1: which side of the axis
    @location(3) endsel: f32,        // 0 at A, 1 at B
    @location(4) radius: f32,        // tube radius; per-endpoint, so tapers work
    @location(5) color: vec4<f32>,
    @location(6) frow: vec3<f32>,    // (own row, other endpoint's row, edge row)
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) side: f32,
    @location(1) color: vec4<f32>,
    // All four corners carry the same value, so interpolation is exact.
    @location(2) normal_basis: vec3<f32>,
    @location(3) view_pos: vec3<f32>,
    @location(4) radius: f32,
}

struct FragmentOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

// Identical to sphere.wgsl's; see the comment there.
fn depth_from_clip(clip: vec4<f32>) -> f32 {
    if (clip.w <= 0.0) {
        return 0.0;
    }
    return clamp(clip.z / clip.w, 0.0, 1.0);
}

// The vertex stage returns a struct, so a rejection needs one; only clip_pos is read.
fn scig_reject() -> VertexOut {
    var out: VertexOut;
    out.clip_pos = SCIG_CULL;
    return out;
}

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    // First statement, as shaders.py::_guard() emits it.
    if (!scig_keep(v.frow)) { return scig_reject(); }

    let a = (camera.view * vec4<f32>(v.pos_a, 1.0)).xyz;
    let b = (camera.view * vec4<f32>(v.pos_b, 1.0)).xyz;

    var center = mix(a, b, v.endsel);

    var dir = b - a;
    let seg_len = length(dir);
    if (seg_len > 1e-8) {
        dir = dir / seg_len;
    } else {
        // A zero-length segment has no direction; any unit vector keeps it finite.
        dir = vec3<f32>(1.0, 0.0, 0.0);
    }

    // Square cap: push each end outward by half a radius.
    center = center + dir * ((v.endsel * 2.0 - 1.0) * v.radius * 0.5);

    // "Sideways on screen": perpendicular to the segment and the view.
    let view_axis = vec3<f32>(0.0, 0.0, 1.0);
    var offset = cross(dir, view_axis);
    let ol = length(offset);
    if (ol > 1e-6) {
        offset = offset / ol;
    } else {
        // Aimed at the camera it has no screen direction; it reads as a disk.
        offset = normalize(cross(dir, vec3<f32>(0.0, 1.0, 0.0)));
    }

    let view_pos = center + offset * (v.side * v.radius);

    var out: VertexOut;
    out.clip_pos = camera.proj * vec4<f32>(view_pos, 1.0);
    out.side = v.side;
    out.color = v.color;
    out.normal_basis = offset;
    out.view_pos = view_pos;
    out.radius = v.radius;
    return out;
}

@fragment
fn fs_main(f: VertexOut) -> FragmentOut {
    let x = clamp(f.side, -1.0, 1.0);
    let z = sqrt(max(0.0, 1.0 - x * x));

    let normal = normalize(f.normal_basis * x + vec3<f32>(0.0, 0.0, 1.0) * z);

    // Depth comes from the front of the tube, not from the flat quad.
    let axis_pt = f.view_pos - f.normal_basis * (x * f.radius);
    let surface = axis_pt + normal * f.radius;
    let clip = camera.proj * vec4<f32>(surface, 1.0);

    let d_key = max(dot(normal, lights.key_dir.xyz), 0.0);
    let d_fill = max(dot(normal, lights.fill_dir.xyz), 0.0);
    let lit = f.color.rgb * (lights.ambient.rgb
                             + lights.key_col.rgb * d_key
                             + lights.fill_col.rgb * d_fill);

    var out: FragmentOut;
    // No rim term: draw.py passes rim = 0 for ribbons. Opaque, as in sphere.wgsl.
    out.color = vec4<f32>(min(lit, vec3<f32>(1.0)), 1.0);
    out.depth = depth_from_clip(clip);
    return out;
}
