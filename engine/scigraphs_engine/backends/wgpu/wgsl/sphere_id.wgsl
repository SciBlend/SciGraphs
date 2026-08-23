// Sphere impostor, ID pass. WGSL port of shaders.py::_build_sphere_id.
// sphere.wgsl minus the shading. ``color`` carries a 24-bit index from
// mesh.encode_ids_to_rgba, written through untouched; nodes and edges share
// one pass and one readback (bit 23 marks an edge, mesh.EDGE_ID_FLAG).
//
// Keep the depth write, or picking returns the wrong node where two impostors
// overlap. The target must be 8-bit UNORM and not sRGB: an -srgb target
// re-encodes the bytes, so indices decode wrong and it reads as a picking bug.

// Replicated verbatim; do not edit one copy. See sphere.wgsl. No LightRig: an
// ID pass has no lighting, so binding 2 is not declared.
struct Camera {
    view       : mat4x4<f32>,   // world -> view                       offset   0
    proj       : mat4x4<f32>,   // view  -> clip, z in [0, 1]          offset  64
    view_proj  : mat4x4<f32>,   // the product, precomputed on the CPU offset 128
    viewport   : vec4<f32>,     // (w, h, 1/w, 1/h)                    offset 192
    params     : vec4<f32>,     // (near, far, aspect, unused)         offset 208
};

@group(0) @binding(0) var<uniform> camera : Camera;

struct VertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) corner: vec2<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,  // encoded id, not a hue
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) view_center: vec3<f32>,
    @location(2) radius: f32,
    // Flat: an index has no in-between value for rounding to shift a byte of.
    @location(3) @interpolate(flat) color: vec4<f32>,
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

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    let view_center = (camera.view * vec4<f32>(v.pos, 1.0)).xyz;
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
    let z = sqrt(max(0.0, 1.0 - r2));
    let view_surface = f.view_center + f.radius * vec3<f32>(p.x, p.y, z);
    let clip = camera.proj * vec4<f32>(view_surface, 1.0);

    var out: FragmentOut;
    out.color = f.color;
    out.depth = depth_from_clip(clip);
    return out;
}
