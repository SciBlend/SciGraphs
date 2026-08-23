void main()
{
  int e = gl_InstanceID;
  int base = e * u_stride;
  int n = int(fetch_texel(u_ctrl, base, u_ctrl_row).w + 0.5);
  float beta = heb_beta(e);
  float t = (seg.z < 0.5) ? seg.x : seg.y;
  vec4 pa = u_mvp * vec4(heb_eval(base, n, seg.x, beta), 1.0);
  vec4 pb = u_mvp * vec4(heb_eval(base, n, seg.y, beta), 1.0);
  vec4 ca = fetch_texel(u_meta, e * u_meta_stride, u_meta_row);
  vec4 cb = fetch_texel(u_meta, e * u_meta_stride + 1, u_meta_row);
  f_color = mix(ca, cb, t);
  vec4 p = (seg.z < 0.5) ? pa : pb;
  if (pa.w <= 0.0 || pb.w <= 0.0) { gl_Position = p; return; }
  float wn = fetch_texel(u_width, e, u_width_row).x;
  float w = mix(u_w_lo, u_w_hi, wn);
  vec2 d = (pb.xy / pb.w - pa.xy / pa.w) * u_viewport;
  float len = length(d);
  vec2 dir = (len > 1e-6) ? (d / len) : vec2(1.0, 0.0);
  vec2 off = vec2(-dir.y, dir.x) * (seg.w * w) / u_viewport;
  gl_Position = vec4((p.xy / p.w + off) * p.w, p.z, p.w);
}
