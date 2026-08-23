// Flat billboard arrowhead, a port of shaders.py::_build_arrow_flat. The
// diagram head, against the object one in `arrow.wgsl`: three vertices, no
// normals, unlit, since at that size the cone's shading is mostly noise.
//
// All three vertices project at `b.z`, the tip's view-space depth, so the
// triangle cannot be foreshortened or seen edge-on. Give each its own z and the
// head becomes a flake that vanishes as the edge turns toward the camera, which
// is the cone's failure mode and why this variant exists.
//
// The direction comes from the endpoints' XY difference in view space, so on a
// nearly end-on edge the head points where the line visibly goes.

// Replicated verbatim; do not edit one copy. See sphere.wgsl.
struct Camera {
    view       : mat4x4<f32>,   // world -> view                       offset   0
    proj       : mat4x4<f32>,   // view  -> clip, z in [0, 1]          offset  64
    view_proj  : mat4x4<f32>,   // the product, precomputed on the CPU offset 128
    viewport   : vec4<f32>,     // (w, h, 1/w, 1/h)                    offset 192
    params     : vec4<f32>,     // (near, far, aspect, unused)         offset 208
};

// Culling is off on purpose. tests/test_arrow_flat.py draws all eight
// directions: back-face culling keeps every one, front-face drops every one, so
// the winding is constant. One triangle per arrow is not worth culling, and the
// wrong sense hides all the heads at once. No `lights` block, since the head is
// unlit and the renderer raises for a declared block the caller did not supply.
@group(0) @binding(0) var<uniform> camera : Camera;

// @location order = MeshSpec.attrs in mesh.py::arrow_flat_spec = GLSL slots.
struct VertexIn {
    @location(0) pos_a: vec3<f32>,   // the edge's far endpoint, world space
    @location(1) pos_b: vec3<f32>,   // the edge's target endpoint; the head sits here
    @location(2) corner: f32,        // 0 = tip, 1 and 2 = the two base corners
    @location(3) cone: vec3<f32>,    // (head length, half width, target node radius)
    @location(4) color: vec4<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;

    let a = (camera.view * vec4<f32>(v.pos_a, 1.0)).xyz;
    let b = (camera.view * vec4<f32>(v.pos_b, 1.0)).xyz;

    // The screen direction of the edge; z is dropped, see the header.
    let d = b.xy - a.xy;
    let len = length(d);
    if (len < 1e-9) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let dir = d / len;
    let perp = vec2<f32>(-dir.y, dir.x);

    // Back off by the target node's radius so the head sits on the sphere.
    let tip = b.xy - dir * v.cone.z;

    let k = i32(v.corner + 0.5);
    var p = tip;
    if (k == 1) {
        p = tip - dir * v.cone.x + perp * v.cone.y;
    } else if (k == 2) {
        p = tip - dir * v.cone.x - perp * v.cone.y;
    }

    // b.z for all three vertices, per the header.
    out.clip_pos = camera.proj * vec4<f32>(p, b.z, 1.0);
    out.color = v.color;
    return out;
}

@fragment
fn fs_main(f: VertexOut) -> @location(0) vec4<f32> {
    // Opaque: the inherited edge alpha (0.35 by default) is a density hint for
    // flat lines and would make heads see-through.
    return vec4<f32>(f.color.rgb, 1.0);
}
