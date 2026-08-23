// The flat filtered line. WGSL port of shaders.py::_build_line_filter.
//
// Separate from uniform_color.wgsl: `mesh.line_spec` is indexed, one vertex per
// node shared by every edge touching it, so a vertex cannot say which edge it
// belongs to or carry that edge's `frow`. `mesh.filtered_line_spec` is
// unindexed, two vertices per edge; `filter_gpu.eligible` refuses that
// duplication when it is not worth paying.
//
// Color is `draw.tint`. Line width is one pixel and not fixable here, so
// `draw.line_width` is bound and never read (../README.md 3.3); wide lines are
// two triangles, in `ribbon_f`.

// Replicated verbatim; do not edit one copy. See sphere.wgsl.
struct Camera {
    view       : mat4x4<f32>,   // world -> view                       offset   0
    proj       : mat4x4<f32>,   // view  -> clip, z in [0, 1]          offset  64
    view_proj  : mat4x4<f32>,   // the product, precomputed on the CPU offset 128
    viewport   : vec4<f32>,     // (w, h, 1/w, 1/h)                    offset 192
    params     : vec4<f32>,     // (near, far, aspect, unused)         offset 208
};

// The per-draw block, ../README.md 1.3. Replicated; do not edit one copy.
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

// Slot numbers are the GLSL's: _add_filter(attr_slot=1) puts frow at 1. Color
// is a uniform with no interpolants, so the output is a bare
// @builtin(position) and there is no scig_reject().
@vertex
fn vs_main(@location(0) pos  : vec3<f32>,
           @location(1) frow : vec3<f32>) -> @builtin(position) vec4<f32> {
    // First statement, as shaders.py::_guard() emits it.
    if (!scig_keep(frow)) { return SCIG_CULL; }

    return camera.view_proj * vec4<f32>(pos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    // Alpha passes through. Edge alpha (0.35 by default) is a density hint;
    // forced opaque, a legible hairball becomes a solid.
    return draw.tint;
}
