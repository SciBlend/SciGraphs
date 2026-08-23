// One texel read plus one lerp: the flood's feature transform already found the
// nearest skeleton point. In place, since each invocation touches only its own
// texel. The smoothing needs two, so it ping-pongs instead.

void main()
{
  int j = int(gl_GlobalInvocationID.x);
  if (j >= u_np)
  {
    return;
  }
  int i = u_p0 + j;
  int k = i - (i / u_k) * u_k;
  if (k == 0 || k == u_k - 1)
  {
    return;   // the endpoints do not move
  }

  vec4 v = imageLoad(pts, scig_texel(i));
  ivec2 p = clamp(ivec2(v.xy * float(u_res)), ivec2(0), ivec2(u_res - 1));
  vec4 f = imageLoad(skel_ft, p);
  if (f.z < 0.5)
  {
    return;                // this cluster has no skeleton
  }

  vec2 target = (f.xy + vec2(0.5)) / float(u_res);
  v.xy = mix(v.xy, target, u_alpha);
  imageStore(pts, scig_texel(i), v);
}
