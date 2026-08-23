void main()
{
  int idx = scig_index();
  if (idx >= u_count)
  {
    return;
  }
  int e = idx / u_kout;
  int m = idx - e * u_kout;
  int base = e * u_kin;

  if (m == 0)
  {
    imageStore(pos_out, scig_texel(idx), imageLoad(pos_in, scig_texel(base)));
    return;
  }
  if (m == u_kout - 1)
  {
    imageStore(pos_out, scig_texel(idx),
               imageLoad(pos_in, scig_texel(base + u_kin - 1)));
    return;
  }

  float total = 0.0;
  for (int s = 0; s < u_kin - 1; ++s)
  {
    total += distance(imageLoad(pos_in, scig_texel(base + s)).xyz,
                      imageLoad(pos_in, scig_texel(base + s + 1)).xyz);
  }
  float target = total * float(m) / float(u_kout - 1);

  float acc = 0.0;
  for (int s = 0; s < u_kin - 1; ++s)
  {
    vec3 A = imageLoad(pos_in, scig_texel(base + s)).xyz;
    vec3 B = imageLoad(pos_in, scig_texel(base + s + 1)).xyz;
    float seg = distance(A, B);
    if (acc + seg >= target || s == u_kin - 2)
    {
      float f = (seg > 1e-12) ? clamp((target - acc) / seg, 0.0, 1.0) : 0.0;
      imageStore(pos_out, scig_texel(idx), vec4(mix(A, B, f), 0.0));
      return;
    }
    acc += seg;
  }
}
