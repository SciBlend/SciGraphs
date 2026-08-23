void main()
{
  vec4 pa = u_mvp * vec4(scig_pos(int(edge.x + 0.5)), 1.0);
  vec4 pb = u_mvp * vec4(scig_pos(int(edge.y + 0.5)), 1.0);
  vec4 base = (edge.z < 0.5) ? pa : pb;
  if (pa.w <= 0.0 || pb.w <= 0.0) { gl_Position = base; return; }
  vec2 na = pa.xy / pa.w;
  vec2 nb = pb.xy / pb.w;
  vec2 d = (nb - na) * u_viewport;
  float len = length(d);
  vec2 dir = (len > 1e-6) ? (d / len) : vec2(1.0, 0.0);
  vec2 perp = vec2(-dir.y, dir.x);
  vec2 off = perp * (edge.w * u_width) / u_viewport;
  gl_Position = vec4((base.xy / base.w + off) * base.w,
                     base.z, base.w);
}
