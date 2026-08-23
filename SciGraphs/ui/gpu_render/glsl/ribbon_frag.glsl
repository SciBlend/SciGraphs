void main()
{
  float x = clamp(v_side, -1.0, 1.0);
  float z = sqrt(max(0.0, 1.0 - x * x));
  vec3 normal = normalize(v_normal_basis * x + vec3(0.0, 0.0, 1.0) * z);
  vec3 axis_pt = v_view_pos - v_normal_basis * (x * v_radius);
  vec3 surface = axis_pt + normal * v_radius;
  vec4 clip = u_proj * vec4(surface, 1.0);
  gl_FragDepth = (clip.z / clip.w) * 0.5 + 0.5;
  float d_key = max(dot(normal, u_light.key_dir.xyz), 0.0);
  float d_fill = max(dot(normal, u_light.fill_dir.xyz), 0.0);
  vec3 lit = v_color.rgb * (u_light.ambient.rgb + u_light.key_col.rgb * d_key + u_light.fill_col.rgb * d_fill);
  fragColor = vec4(min(lit, vec3(1.0)), 1.0);
}
