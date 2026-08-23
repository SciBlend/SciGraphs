void main()
{
  int e = gl_InstanceID;
  int base = e * u_stride;
  int n = int(fetch_texel(u_ctrl, base, u_ctrl_row).w + 0.5);
  gl_Position = u_mvp * vec4(
      heb_eval(base, n, sample_t, heb_beta(e)), 1.0);
}
