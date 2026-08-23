// round_point: a flat disc per node.
//
// WGSL has no gl_PointCoord and no point size, so the disc is an instanced
// quad: `"expand"` in round_point.json (README 3.2), draw(4, instance_count=N).
// Expanding on the CPU would quadruple a million-row buffer to carry a sign,
// 48 MB of position data to draw 12 MB of graph. The corner runs [-1,1]
// centered, so the disc test is dot(c,c) > 1.0.

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

@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32,
           @location(0) pos   : vec3<f32>,
           @location(1) color : vec4<f32>) -> VertexOut {
    var out : VertexOut;
    let corner = strip_corner(vertex_index);
    var clip = camera.view_proj * vec4<f32>(pos, 1.0);

    // The clip.w factor keeps the offset a constant pixel count past the
    // perspective divide, the one thing gl_PointSize gave for free.
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
