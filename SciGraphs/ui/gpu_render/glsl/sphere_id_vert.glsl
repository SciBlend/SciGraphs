void main()
{
  vec4 view_center = u_view * vec4(pos, 1.0);
  vec3 vc = view_center.xyz;
  vec3 view_pos = vc + vec3(corner * radius, 0.0);
  gl_Position = u_proj * vec4(view_pos, 1.0);
  v_uv = corner;
  v_view_center = vc;
  v_radius = radius;
  v_color = color;
}
