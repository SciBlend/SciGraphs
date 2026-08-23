void main()
{
  vec2 p = v_uv;
  float r2 = dot(p, p);
  if (r2 > 1.0) { discard; }
  float z = sqrt(1.0 - r2);
  vec3 normal = vec3(p.x, p.y, z);
  vec3 view_surface = v_view_center + v_radius * normal;
  vec4 clip = u_proj * vec4(view_surface, 1.0);
  gl_FragDepth = (clip.z / clip.w) * 0.5 + 0.5;
  float d_key = max(dot(normal, u_light.key_dir.xyz), 0.0);
  float d_fill = max(dot(normal, u_light.fill_dir.xyz), 0.0);
  float rim = pow(1.0 - z, 3.0) * u_light.key_col.w;
  vec3 lit = v_color.rgb * (u_light.ambient.rgb + u_light.key_col.rgb * d_key + u_light.fill_col.rgb * d_fill) + rim;
  fragColor = vec4(min(lit, vec3(1.0)), 1.0);
}
