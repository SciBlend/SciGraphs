// One invocation per segment, walked with a DDA. The step count is the
// Chebyshev length, so consecutive samples land at most one texel apart.
// u_max_steps guards against absurd coordinates; it is not a quality knob.

void main()
{
  int j = int(gl_GlobalInvocationID.x);
  if (j >= u_nseg)
  {
    return;
  }
  int e = j / u_km1;
  int k = j - e * u_km1;
  int idx = (u_edge0 + e * u_estride) * u_k + k;

  vec2 a = imageLoad(pts, scig_texel(idx)).xy * float(u_res);
  vec2 b = imageLoad(pts, scig_texel(idx + 1)).xy * float(u_res);
  vec2 d = abs(b - a);
  int n = clamp(int(max(d.x, d.y)) + 1, 1, u_max_steps);
  for (int s = 0; s <= n; ++s)
  {
    vec2 p = mix(a, b, float(s) / float(n));
    ivec2 q = clamp(ivec2(p), ivec2(0), ivec2(u_res - 1));
    imageAtomicAdd(dens, q, 1u);
  }
}
