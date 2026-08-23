// SciGraphs: edge ribbon, ID pass. WGSL port of shaders.py::_build_ribbon_id.
// Matches ribbon.wgsl down to the square caps and the cylinder-surface depth,
// so it wins exactly the fragments the beauty pass gives the edge. ``color``
// carries a 24-bit segment id (bit 23 set, see mesh.EDGE_ID_FLAG).

// Replicated verbatim; do not edit one copy. See sphere.wgsl.
// No LightRig: an ID pass has no lighting.
struct Camera {
    view       : mat4x4<f32>,   // world -> view                       offset   0
    proj       : mat4x4<f32>,   // view  -> clip, z in [0, 1]          offset  64
    view_proj  : mat4x4<f32>,   // the product, precomputed on the CPU offset 128
    viewport   : vec4<f32>,     // (w, h, 1/w, 1/h)                    offset 192
    params     : vec4<f32>,     // (near, far, aspect, unused)         offset 208
};

@group(0) @binding(0) var<uniform> camera : Camera;

struct VertexIn {
    @location(0) pos_a: vec3<f32>,
    @location(1) pos_b: vec3<f32>,
    @location(2) side: f32,
    @location(3) endsel: f32,
    @location(4) radius: f32,
    @location(5) color: vec4<f32>,  // encoded id, not a hue
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) side: f32,
    // Flat, for the reason given in sphere_id.wgsl.
    @location(1) @interpolate(flat) color: vec4<f32>,
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
        dir = vec3<f32>(1.0, 0.0, 0.0);
    }

    // Square cap, see ribbon.wgsl. Without it the ID and beauty passes disagree
    // by half a radius at every elbow and picking near a bend hits background.
    center = center + dir * ((v.endsel * 2.0 - 1.0) * v.radius * 0.5);

    let view_axis = vec3<f32>(0.0, 0.0, 1.0);
    var offset = cross(dir, view_axis);
    let ol = length(offset);
    if (ol > 1e-6) {
        offset = offset / ol;
    } else {
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
    let axis_pt = f.view_pos - f.normal_basis * (x * f.radius);
    let surface = axis_pt + normal * f.radius;
    let clip = camera.proj * vec4<f32>(surface, 1.0);

    var out: FragmentOut;
    out.color = f.color;
    out.depth = depth_from_clip(clip);
    return out;
}
