void main()
{
  vec3 a = (u_view * vec4(pos_a, 1.0)).xyz;
  vec3 b = (u_view * vec4(pos_b, 1.0)).xyz;
  vec3 center = mix(a, b, endsel);
  vec3 dir = b - a;
  float len = length(dir);
  dir = (len > 1e-8) ? dir / len : vec3(1.0, 0.0, 0.0);
  center += dir * ((endsel * 2.0 - 1.0) * radius * 0.5);
  vec3 view_axis = vec3(0.0, 0.0, 1.0);
  vec3 offset = cross(dir, view_axis);
  float ol = length(offset);
  offset = (ol > 1e-6) ? offset / ol : normalize(cross(dir, vec3(0.0, 1.0, 0.0)));
  vec3 view_pos = center + offset * (side * radius);
  gl_Position = u_proj * vec4(view_pos, 1.0);
  v_side = side;
  v_color = color;
  v_normal_basis = offset;
  v_view_pos = view_pos;
  v_radius = radius;
}
