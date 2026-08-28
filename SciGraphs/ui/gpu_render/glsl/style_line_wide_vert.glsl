void main()
{
  int e = gl_InstanceID;
  if (!scig_style_keep(e)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec3 a, b;
  float eid;
  scig_edge_ends(e, a, b, eid);
  vec3 q0 = scig_style_point(a, b, seg.x, eid);
  vec3 q1 = scig_style_point(a, b, seg.y, eid);
  if (distance(q0, q1) < 1e-9) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

  float t = (seg.z < 0.5) ? seg.x : seg.y;
  f_color = u_style.color;
  vec4 pa = u_mvp * vec4(q0, 1.0);
  vec4 pb = u_mvp * vec4(q1, 1.0);
  vec4 p = (seg.z < 0.5) ? pa : pb;
  if (pa.w <= 0.0 || pb.w <= 0.0) { gl_Position = p; return; }

  float w = mix(u_style.width.x, u_style.width.y, scig_edge(e, 1).x)
          * scig_style_taper(t);
  vec2 d = (pb.xy / pb.w - pa.xy / pa.w) * u_viewport;
  float len = length(d);
  vec2 dir = (len > 1e-6) ? (d / len) : vec2(1.0, 0.0);
  vec2 off = vec2(-dir.y, dir.x) * (seg.w * w) / u_viewport;
  gl_Position = vec4((p.xy / p.w + off) * p.w, p.z, p.w);
}
