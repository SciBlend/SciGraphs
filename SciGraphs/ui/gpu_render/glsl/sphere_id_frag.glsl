void main()
{
  vec2 p = v_uv;
  float r2 = dot(p, p);
  if (r2 > 1.0) { discard; }
  float z = sqrt(1.0 - r2);
  vec3 view_surface = v_view_center + v_radius * vec3(p.x, p.y, z);
  vec4 clip = u_proj * vec4(view_surface, 1.0);
  gl_FragDepth = (clip.z / clip.w) * 0.5 + 0.5;
  fragColor = v_color;
}
