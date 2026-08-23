float shape_of(float theta)
{
  float s = 1.0;
  if (u_blades >= 2.5)
{
    float seg = 6.2831853 / u_blades;
    float a = mod(theta - u_blade_rot, seg) - 0.5 * seg;
    s = cos(3.14159265 / u_blades) / cos(a);
  }
  float ct = cos(theta);
  float st = sin(theta) / max(u_ratio, 1e-3);
  return s * inversesqrt(ct * ct + st * st);}void main()
{
  vec4 base = texture(color, uv);
  float center_dist = dist_of(texture(depth, uv).r);
  float center_coc = coc_of(center_dist);
  vec4 bpm = vec4(base.rgb * base.a, base.a);
  float bw = 1.0 + u_boost * dot(bpm.rgb, bpm.rgb);
  vec4 sum = bpm * bw;
  float wsum = bw;
  float ign = fract(52.9829189 * fract(
      dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))) + u_seed);
  float ang = ign * 6.2831853;
  float radius = u_rad_scale;
  for (int i = 0; i < 8192; i++)
{
    if (radius >= u_max_radius) { break; }
    vec2 tc = uv + vec2(cos(ang), sin(ang)) * u_texel * radius;
    vec4 cs = texture(color, tc);
    float s_dist = dist_of(texture(depth, tc).r);
    float s_coc = coc_of(s_dist);
    if (s_dist > center_dist) { s_coc = clamp(s_coc, 0.0, center_coc * 2.0); }
    float m = smoothstep(radius - 0.5, radius + 0.5, s_coc * shape_of(ang));
    vec4 spm = vec4(cs.rgb * cs.a, cs.a);
    float w = 1.0 + u_boost * dot(spm.rgb, spm.rgb);
    float cw = mix(1.0, w, m);
    sum += mix(sum / wsum, spm, m) * cw;
    wsum += cw;
    ang += 2.39996323;
    radius += u_rad_scale / radius;
  }
  vec4 acc = sum / wsum;
  fragColor = vec4((acc.a > 1e-4) ? acc.rgb / acc.a : base.rgb,
                   clamp(acc.a, 0.0, 1.0));
}
