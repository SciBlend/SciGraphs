void main()
{
  int e = gl_InstanceID;
  if (!scig_style_keep(e)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec3 a, b;
  float eid;
  scig_edge_ends(e, a, b, eid);
  f_color = u_style.color;
  gl_Position = u_mvp * vec4(scig_style_point(a, b, sample_t, eid), 1.0);
}
