// Edge ribbon (cylinder impostor). WGSL port of shaders.py::_build_ribbon.
// sphere.wgsl one dimension down, so see it for the absent OpenGL depth
// remap. A quad per segment, not a line, because WebGPU has no line width.
//
// Segments are drawn independently, so at a bend the end edges meet at an
// angle and leave a sub-pixel wedge: background through an opaque tube, dark
// ticks along a bundle. The square cap's half radius of overhang covers that
// corner, and both quads write true cylinder depth, so it resolves cleanly.

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

// Binding 1 is `draw`, unread here: slots mean the same thing folder-wide.
@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(2) var<uniform> lights : LightRig;

// @location order is MeshSpec.attrs in mesh.py::ribbon_spec. Both endpoints
// sit on all four vertices, or the corners get different perpendiculars.
struct VertexIn {
    @location(0) pos_a: vec3<f32>,   // segment start, world space
    @location(1) pos_b: vec3<f32>,   // segment end, world space
    @location(2) side: f32,          // -1 or +1: which side of the axis
    @location(3) endsel: f32,        // 0 at A, 1 at B
    @location(4) radius: f32,        // tube radius; per-endpoint, so tapers work
    @location(5) color: vec4<f32>,
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

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    let a = (camera.view * vec4<f32>(v.pos_a, 1.0)).xyz;
    let b = (camera.view * vec4<f32>(v.pos_b, 1.0)).xyz;

    var center = mix(a, b, v.endsel);

    var dir = b - a;
    let seg_len = length(dir);
    if (seg_len > 1e-8) {
        dir = dir / seg_len;
    } else {
        // No direction to take; any unit vector keeps the math finite.
        dir = vec3<f32>(1.0, 0.0, 0.0);
    }

    // Square cap; the half radius matters, see the header.
    center = center + dir * ((v.endsel * 2.0 - 1.0) * v.radius * 0.5);

    // "Sideways on screen": perpendicular to the segment and the view.
    let view_axis = vec3<f32>(0.0, 0.0, 1.0);
    var offset = cross(dir, view_axis);
    let ol = length(offset);
    if (ol > 1e-6) {
        offset = offset / ol;
    } else {
        // Points at or away from the camera; any perpendicular reads as a disk.
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

    // Onto the axis, then out to the front of the tube. That point, not the
    // flat quad, is what the depth buffer gets.
    let axis_pt = f.view_pos - f.normal_basis * (x * f.radius);
    let surface = axis_pt + normal * f.radius;
    let clip = camera.proj * vec4<f32>(surface, 1.0);

    let d_key = max(dot(normal, lights.key_dir.xyz), 0.0);
    let d_fill = max(dot(normal, lights.fill_dir.xyz), 0.0);
    let lit = f.color.rgb * (lights.ambient.rgb
                             + lights.key_col.rgb * d_key
                             + lights.fill_col.rgb * d_fill);

    var out: FragmentOut;
    // No rim term: draw.py passes rim = 0 here. Opaque, as in sphere.wgsl.
    out.color = vec4<f32>(min(lit, vec3<f32>(1.0)), 1.0);
    out.depth = depth_from_clip(clip);
    return out;
}
