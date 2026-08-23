void main()
{
  int e = gl_InstanceID;
  int base = e * u_stride;
  int n = int(fetch_texel(u_ctrl, base, u_ctrl_row).w + 0.5);
  vec3 p = heb_eval(base, n, sample_t, heb_beta(e));
  vec4 ca = fetch_texel(u_meta, e * u_meta_stride, u_meta_row);
  vec4 cb = fetch_texel(u_meta, e * u_meta_stride + 1, u_meta_row);
  f_color = mix(ca, cb, sample_t);
  gl_Position = u_mvp * vec4(p, 1.0);
}
