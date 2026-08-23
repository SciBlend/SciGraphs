// round_point, filtered: round_point.wgsl plus the shared filter block plus one
// line at the top of vs_main, a WGSL port of
// shaders.py::_build_round_point(filtered=True). sphere_f.wgsl's header says why
// that shape matters. Only round_point declares `expand`, so its buffers step
// per instance and `frow` is per node, which makes the filter all-or-nothing
// without mesh.py repeating one value four times as it does for the impostors.

struct Camera {
    view       : mat4x4<f32>,   // world -> view                       offset   0
    proj       : mat4x4<f32>,   // view  -> clip, z in [0, 1]          offset  64
    view_proj  : mat4x4<f32>,   // the product, precomputed on the CPU offset 128
    viewport   : vec4<f32>,     // (w, h, 1/w, 1/h)                    offset 192
    params     : vec4<f32>,     // (near, far, aspect, unused)         offset 208
};

struct Draw {
    point_size : f32,   // pixels, diameter
    line_width : f32,
    weight     : f32,   // MeshGroup.weight, or -1
    flags      : f32,
    tint       : vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(1) var<uniform> draw   : Draw;

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

struct VertexOut {
    @builtin(position) clip_position : vec4<f32>,
    @location(0) color  : vec4<f32>,
    @location(1) corner : vec2<f32>,
};

// Triangle-strip corner order: (-1,-1), (1,-1), (-1,1), (1,1).
fn strip_corner(i : u32) -> vec2<f32> {
    let x = select(-1.0, 1.0, (i & 1u) == 1u);
    let y = select(-1.0, 1.0, (i & 2u) == 2u);
    return vec2<f32>(x, y);
}

// A value, not a bare `return`: the vertex stage has to return a struct.
fn scig_reject() -> VertexOut {
    var out: VertexOut;
    out.clip_position = SCIG_CULL;
    return out;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32,
           @location(0) pos   : vec3<f32>,
           @location(1) color : vec4<f32>,
           @location(2) frow  : vec3<f32>) -> VertexOut {
    // First statement, as shaders.py::_guard() emits it.
    if (!scig_keep(frow)) { return scig_reject(); }

    var out : VertexOut;
    let corner = strip_corner(vertex_index);
    var clip = camera.view_proj * vec4<f32>(pos, 1.0);

    // clip.w cancels the perspective divide, so the offset stays a pixel count.
    let radius_px = max(draw.point_size, 1.0) * 0.5;
    let offset = corner * radius_px * 2.0
                 * vec2<f32>(camera.viewport.z, camera.viewport.w);
    clip = vec4<f32>(clip.xy + offset * clip.w, clip.z, clip.w);

    out.clip_position = clip;
    out.color = color;
    out.corner = corner;
    return out;
}

@fragment
fn fs_main(in : VertexOut) -> @location(0) vec4<f32> {
    if (dot(in.corner, in.corner) > 1.0) {
        discard;
    }
    return in.color;
}
