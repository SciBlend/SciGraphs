// Flat line with a color at each end. Named only by `mesh.segment_line_spec`
// with `col_a` given: hierarchical bundling wants a gradient along each curve
// and a different opacity per edge, and one uniform can express neither. No
// `draw` binding, the color is the attribute, and it goes through unclamped so
// that alpha can carry the per-edge opacity. One pixel wide (../README.md 3.3),
// no `@builtin(frag_depth)`; uniform_color.wgsl says why.
//
// Omitting `@interpolate` gives WGSL's default (perspective, center), GLSL's
// `smooth`. `@interpolate(linear)` differs only under perspective, where w is
// not 1, so the one test that can tell them apart is in tests/test_flat_lines.py.

// Replicated verbatim; do not edit one copy. See sphere.wgsl.
struct Camera {
    view       : mat4x4<f32>,   // world -> view                       offset   0
    proj       : mat4x4<f32>,   // view  -> clip, z in [0, 1]          offset  64
    view_proj  : mat4x4<f32>,   // the product, precomputed on the CPU offset 128
    viewport   : vec4<f32>,     // (w, h, 1/w, 1/h)                    offset 192
    params     : vec4<f32>,     // (near, far, aspect, unused)         offset 208
};

@group(0) @binding(0) var<uniform> camera : Camera;

struct VertexOut {
    @builtin(position) clip_position : vec4<f32>,
    @location(0) color : vec4<f32>,
};

@vertex
fn vs_main(@location(0) pos   : vec3<f32>,
           @location(1) color : vec4<f32>) -> VertexOut {
    var out : VertexOut;
    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in : VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}
