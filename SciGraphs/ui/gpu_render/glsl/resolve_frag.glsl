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
  fragColor = acc / float(u_s * u_s);
}
