// The plain flat line: one position attribute, one color for the whole draw.
// Named by mesh.line_spec and by mesh.segment_line_spec with col_a=None; not
// line_f, which needs a `frow` that mesh.line_spec does not provide.
//
// The color rides in `draw.tint` because MeshSpec has no uniform-color field,
// and an attribute would repeat four floats per node: 16 MB on a million-node
// graph. Alpha passes through: the 0.35 default edge alpha is a density hint,
// and 1 turns a legible hairball solid. One pixel wide, not fixable here
// (../README.md 3.3). `"cull": "none"` is inert, WebGPU culling triangles only.

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

// The .json declares `camera` vertex-only and `draw` fragment-only.
@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(1) var<uniform> draw   : Draw;

// A bare @builtin(position): the color is a uniform, so no interpolants.
@vertex
fn vs_main(@location(0) pos : vec3<f32>) -> @builtin(position) vec4<f32> {
    return camera.view_proj * vec4<f32>(pos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return draw.tint;
}
