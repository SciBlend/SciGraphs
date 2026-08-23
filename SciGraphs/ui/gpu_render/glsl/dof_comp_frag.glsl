void main()
{
  vec4 sharp = texture(color, uv);
  vec4 blurc = texture(blur, uv);
  float center_dist = dist_of(texture(depth, uv).r);
  float cc = coc_of(center_dist);
  for (int ring = 1; ring <= 2; ring++)
{
    float rr = u_max_radius * (float(ring) * 0.33);
    for (int i = 0; i < 8; i++)
{
      float a = 0.785398 * float(i) + float(ring) * 0.4;
      vec2 tc = uv + vec2(cos(a), sin(a)) * u_texel * rr;
      float d = dist_of(texture(depth, tc).r);
      float c = coc_of(d);
      if (d < center_dist - 1e-4 && c >= rr) { cc = max(cc, c); }
    }
  }
  float t = smoothstep(0.5, 2.0, cc);
  vec4 sp = vec4(sharp.rgb * sharp.a, sharp.a);
  vec4 bp = vec4(blurc.rgb * blurc.a, blurc.a);
  vec4 o = mix(sp, bp, t);
  fragColor = vec4((o.a > 1e-4) ? o.rgb / o.a : blurc.rgb,
                   clamp(o.a, 0.0, 1.0));
}
