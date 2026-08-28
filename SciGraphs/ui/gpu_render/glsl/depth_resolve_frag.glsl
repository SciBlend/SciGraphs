void main()
{
  ivec2 base = ivec2(gl_FragCoord.xy) * u_s;
  float d = texelFetch(src, base, 0).r;
  for (int i = 0; i < u_s; i++)
  {
    for (int j = 0; j < u_s; j++)
    {
      d = min(d, texelFetch(src, base + ivec2(j, i), 0).r);
    }
  }
  float ndc = d * 2.0 - 1.0;
  float denom = ndc + u_a;
  float dist = (abs(denom) > 1e-12) ? abs(u_b / denom) : 1.0e10;
  if (d >= 1.0) { dist = 1.0e10; }
  fragColor = vec4(dist, d, 0.0, 0.0);
}
