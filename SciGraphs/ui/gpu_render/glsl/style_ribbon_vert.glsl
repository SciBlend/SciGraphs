void main()
{
  int e = gl_InstanceID;
  if (!scig_style_keep(e)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec3 ea, eb;
  float eid;
  scig_edge_ends(e, ea, eb, eid);
  vec3 q0 = scig_style_point(ea, eb, seg.x, eid);
  vec3 q1 = scig_style_point(ea, eb, seg.y, eid);
  if (distance(q0, q1) < 1e-9) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

  float endsel = seg.z;
  float side = seg.w;
  float t = mix(seg.x, seg.y, endsel);
  float radius = u_style.width.x * scig_edge(e, 1).y * scig_style_taper(t);
  int tmode = int(u_style.width.w + 0.5);
  if ((tmode == 1 && endsel < 0.5) || (tmode == 2 && endsel > 0.5))
  {
    radius *= u_style.width.z;
  }

  vec3 a = (u_view * vec4(q0, 1.0)).xyz;
  vec3 b = (u_view * vec4(q1, 1.0)).xyz;
  vec3 center = mix(a, b, endsel);
  vec3 dir = b - a;
  float len = length(dir);
  dir = (len > 1e-8) ? dir / len : vec3(1.0, 0.0, 0.0);
  center += dir * ((endsel * 2.0 - 1.0) * radius * 0.5);
  vec3 offset = cross(dir, vec3(0.0, 0.0, 1.0));
  float ol = length(offset);
  offset = (ol > 1e-6) ? offset / ol
                       : normalize(cross(dir, vec3(0.0, 1.0, 0.0)));
  vec3 view_pos = center + offset * (side * radius);
  gl_Position = u_proj * vec4(view_pos, 1.0);
  v_side = side;
  v_color = u_style.color;
  v_normal_basis = offset;
  v_view_pos = view_pos;
  v_radius = radius;
}
