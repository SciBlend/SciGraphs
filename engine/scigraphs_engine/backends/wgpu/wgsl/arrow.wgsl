// SciGraphs: arrowhead. WGSL port of shaders.py::_build_arrow.
//
// Real geometry, not an impostor: a directed edge gets a cone at its target
// end, built with per-vertex normals in mesh.arrow_spec, so there is no
// @builtin(frag_depth) here. The LightRig is shared with the impostors so a
// head and its tube match. Shading is two-sided because a backface lit by its
// true normal is black, and the flip on @builtin(front_facing) needs culling
// off, or the head vanishes from half the viewing directions.

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

// @location order is MeshSpec.attrs in mesh.py::arrow_spec.
struct VertexIn {
    @location(0) pos: vec3<f32>,   // world space
    @location(1) nrm: vec3<f32>,   // world space
    @location(2) color: vec4<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal: vec3<f32>,   // view space
    @location(1) color: vec4<f32>,
}

// Separate struct for a builtin the vertex stage cannot produce; the
// @location numbers tie it to VertexOut.
struct FragmentIn {
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
}

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos = camera.proj * (camera.view * vec4<f32>(v.pos, 1.0));
    // w = 0 drops the translation; the rig is view space, so the normal must be.
    out.normal = normalize((camera.view * vec4<f32>(v.nrm, 0.0)).xyz);
    out.color = v.color;
    return out;
}

@fragment
fn fs_main(f: FragmentIn) -> @location(0) vec4<f32> {
    var n = normalize(f.normal);
    if (!f.front_facing) {
        n = -n;
    }

    let d_key = max(dot(n, lights.key_dir.xyz), 0.0);
    let d_fill = max(dot(n, lights.fill_dir.xyz), 0.0);
    let lit = f.color.rgb * (lights.ambient.rgb
                             + lights.key_col.rgb * d_key
                             + lights.fill_col.rgb * d_fill);

    return vec4<f32>(min(lit, vec3<f32>(1.0)), 1.0);
}
