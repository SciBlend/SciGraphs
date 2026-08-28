void main()
{
  ivec2 base = ivec2(gl_FragCoord.xy) * u_s;
  vec4 acc = vec4(0.0);
  for (int i = 0; i < u_s; i++)
{
    for (int j = 0; j < u_s; j++)
{
      acc += texelFetch(src, base + ivec2(j, i), 0);
    }
  }
  vec4 c = acc / float(u_s * u_s);
  if (u_premul != 0) {
    c = max(c, vec4(0.0));
    c = vec4(c.rgb * c.a, c.a);
  }
  fragColor = c;
}
