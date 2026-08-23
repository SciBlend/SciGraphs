// Hierarchical edge bundling in the vertex shader. WGSL port of
// shaders.py::_build_heb_line (+ _HEB_GLSL). An edge is a cubic B-spline over
// the path from u up to the lowest common ancestor of u and v and back down,
// so edges sharing a stretch of hierarchy share a stretch of curve.
//
// In numpy that is 8.5M vertices and 239 MB on a 213k-edge graph, rebuilt
// whenever the strength moves, though strength only straightens the curve.
// Uploading the control polygon instead is at most 2*depth+3 points per edge:
// 1196 ms -> 6.7 ms on the GLSL side.
//
// One instance per edge through `expand` (../README.md 3.2), sample parameter
// from @builtin(vertex_index) and no sample buffer, so SAMPLES below and
// `expand.vertex_count` in the .json must agree. Line-strip: strips do not join
// across instances. The control point count rides in the .w of an edge's first
// control point, which a position never uses.

// Replicated verbatim; do not edit one copy. See sphere.wgsl.
struct Camera {
    view       : mat4x4<f32>,   // world -> view                       offset   0
    proj       : mat4x4<f32>,   // view  -> clip, z in [0, 1]          offset  64
    view_proj  : mat4x4<f32>,   // the product, precomputed on the CPU offset 128
    viewport   : vec4<f32>,     // (w, h, 1/w, 1/h)                    offset 192
    params     : vec4<f32>,     // (near, far, aspect, unused)         offset 208
};

// The GLSL's u_beta and u_stride push constants. `stride` is an integer in a
// float, read as `i32(stride + 0.5)`: Renderer.block_buffer uploads float32.
struct Heb {
    beta   : f32,   // bundling strength in [0, 1]. 1 = the full curve.
    stride : f32,   // control points allocated per edge (the widest polygon)
    pad0   : f32,
    pad1   : f32,
};

@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(5) var<uniform> heb    : Heb;
// `stride` vec4 per edge. xyz is the point; .w of the first is the count.
@group(0) @binding(6) var<storage, read> ctrl : array<vec4<f32>>;

// Must match `expand.vertex_count` in heb_line.json.
const SAMPLES : u32 = 33u;

struct VertexIn {
    // Per instance, not per vertex: one row per edge.
    @location(0) col_a: vec4<f32>,   // color at t = 0
    @location(1) col_b: vec4<f32>,   // color at t = 1
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

// Knot i of an open uniform vector. The repeated end knots make the curve
// interpolate its first and last control point, so an edge meets its nodes.
fn heb_knot(i: i32, degree: i32, interior: i32) -> f32 {
    if (i <= degree) {
        return 0.0;
    }
    if (i >= degree + 1 + interior) {
        return 1.0;
    }
    return f32(i - degree) / f32(interior + 1);
}

// Control point i. Strength is applied here, not to the evaluated curve:
// equation 1 of the paper, one operation per control point instead of one per
// sample. Both are linear, so beta = 0 gives a straight line, not a near one.
fn heb_ctrl(base: i32, i: i32, n: i32, first: vec3<f32>,
            last: vec3<f32>) -> vec3<f32> {
    let c = ctrl[base + i].xyz;
    if (heb.beta >= 1.0) {
        return c;
    }
    var w = 0.0;
    if (n > 1) {
        w = f32(i) / f32(n - 1);
    }
    return heb.beta * c + (1.0 - heb.beta) * (first + w * (last - first));
}

// de Boor. `d` is 4 entries because a cubic spans four control points.
fn heb_eval(base: i32, n: i32, t: f32) -> vec3<f32> {
    // Too short for a cubic: three points is a quadratic, two a line.
    let degree = min(3, n - 1);
    let interior = n - degree - 1;
    let first = ctrl[base].xyz;
    let last = ctrl[base + n - 1].xyz;
    if (degree < 1) {
        return first;
    }

    // Uniform interior knots, so the span holding t is arithmetic.
    let span = degree + min(i32(t * f32(interior + 1)), interior);

    var d: array<vec3<f32>, 4>;
    for (var j = 0; j <= degree; j = j + 1) {
        d[j] = heb_ctrl(base, j + span - degree, n, first, last);
    }
    for (var r = 1; r <= degree; r = r + 1) {
        for (var j = degree; j >= r; j = j - 1) {
            let i = j + span - degree;
            let lo = heb_knot(i, degree, interior);
            let hi = heb_knot(i + degree - r + 1, degree, interior);
            let den = hi - lo;
            var f = 0.0;
            if (den > 0.0) {
                f = (t - lo) / den;
            }
            d[j] = mix(d[j - 1], d[j], f);
        }
    }
    return d[degree];
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32,
           @builtin(instance_index) instance_index: u32,
           v: VertexIn) -> VertexOut {
    let e = i32(instance_index);
    let stride = i32(heb.stride + 0.5);
    let base = e * stride;
    let n = i32(ctrl[base].w + 0.5);
    let t = f32(vertex_index) / f32(SAMPLES - 1u);

    var out: VertexOut;
    out.clip_pos = camera.view_proj * vec4<f32>(heb_eval(base, n, t), 1.0);
    // Linear between the two endpoint colors, per section 3.3 of the paper.
    out.color = mix(v.col_a, v.col_b, t);
    return out;
}

@fragment
fn fs_main(f: VertexOut) -> @location(0) vec4<f32> {
    // Alpha passes through: many faint curves accumulate where they agree.
    return f.color;
}
