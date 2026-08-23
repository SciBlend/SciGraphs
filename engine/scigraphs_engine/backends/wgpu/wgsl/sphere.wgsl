// Sphere impostor. WGSL port of shaders.py::_build_sphere. A quad per node,
// expanded in view space; the corner doubles as the sphere's own xy, so the
// fragment reconstructs the normal and writes true surface depth. Drop that
// depth and overlapping nodes resolve as flat cards, easy to miss since a
// billboard still draws a shaded disc.
//
// OpenGL clip space has z in [-w, w] and the GLSL wrote ``* 0.5 + 0.5``. WebGPU
// has z in [0, w] and drops the remap, so ``camera.proj`` must already be a
// WebGPU projection. An OpenGL one clips half the scene away, and putting the
// remap back looks plausible but sorts wrong.

// Replicated verbatim; WGSL has no #include. Do not edit one copy alone.
struct Camera {
    view       : mat4x4<f32>,   // world -> view                       offset   0
    proj       : mat4x4<f32>,   // view  -> clip, z in [0, 1]          offset  64
    view_proj  : mat4x4<f32>,   // the product, precomputed on the CPU offset 128
    viewport   : vec4<f32>,     // (w, h, 1/w, 1/h)                    offset 192
    params     : vec4<f32>,     // (near, far, aspect, unused)         offset 208
};

// draw.py::compute_lighting collapses the lamps into these three terms, in
// view space. Byte-identical to the GLSL std140 block. README §2.
struct LightRig {
    key_dir  : vec4<f32>,   // xyz = direction TOWARD the key light  offset   0
    key_col  : vec4<f32>,   // rgb = key color, w = rim strength    offset  16
    fill_dir : vec4<f32>,   // xyz = direction toward combined fill  offset  32
    fill_col : vec4<f32>,   // rgb = fill color; w unused           offset  48
    ambient  : vec4<f32>,   // rgb = ambient term; w unused          offset  64
};

// Binding 1 is `draw`, unread here: slot numbers match across this folder.
@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(2) var<uniform> lights : LightRig;

// @location order is MeshSpec.attrs in mesh.py::sphere_spec, and the GLSL slots.
struct VertexIn {
    @location(0) pos: vec3<f32>,     // node center, world space
    @location(1) corner: vec2<f32>,  // quad corner in [-1, 1]^2 == sphere xy
    @location(2) radius: f32,        // sphere radius, world/view units
    @location(3) color: vec4<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) view_center: vec3<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,
}

struct FragmentOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

// clip -> window depth. w <= 0 puts the surface point at or behind the eye,
// where the divide hands the depth test a NaN; the clamp keeps it defined.
fn depth_from_clip(clip: vec4<f32>) -> f32 {
    if (clip.w <= 0.0) {
        return 0.0;
    }
    return clamp(clip.z / clip.w, 0.0, 1.0);
}

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    let view_center = (camera.view * vec4<f32>(v.pos, 1.0)).xyz;

    // A sphere's true silhouette under perspective is slightly larger than its
    // radius. The GLSL approximated it the same way; kept for parity.
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

    // +z faces the camera in view space, so take the positive root.
    let z = sqrt(max(0.0, 1.0 - r2));
    let normal = vec3<f32>(p.x, p.y, z);

    let view_surface = f.view_center + f.radius * normal;
    let clip = camera.proj * vec4<f32>(view_surface, 1.0);

    let d_key = max(dot(normal, lights.key_dir.xyz), 0.0);
    let d_fill = max(dot(normal, lights.fill_dir.xyz), 0.0);

    // Brightest at the silhouette, so same-colored nodes stay separable.
    let rim = pow(1.0 - z, 3.0) * lights.key_col.w;

    // WGSL will not broadcast a scalar into a vector as GLSL did.
    let lit = f.color.rgb * (lights.ambient.rgb
                             + lights.key_col.rgb * d_key
                             + lights.fill_col.rgb * d_fill)
              + vec3<f32>(rim);

    var out: FragmentOut;
    // Alpha forced to 1. Blending is off for solid geometry, and a node's own
    // alpha would blend the background into an opaque object.
    out.color = vec4<f32>(min(lit, vec3<f32>(1.0)), 1.0);
    out.depth = depth_from_clip(clip);
    return out;
}
