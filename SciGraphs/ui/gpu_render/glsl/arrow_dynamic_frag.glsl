void main()
{
  vec3 n = normalize(v_normal);
  if (!gl_FrontFacing) { n = -n; }
  float d_key = max(dot(n, u_light.key_dir.xyz), 0.0);
  float d_fill = max(dot(n, u_light.fill_dir.xyz), 0.0);
  vec3 lit = v_color.rgb * (u_light.ambient.rgb + u_light.key_col.rgb * d_key + u_light.fill_col.rgb * d_fill);
  fragColor = vec4(min(lit, vec3(1.0)), 1.0);
}
