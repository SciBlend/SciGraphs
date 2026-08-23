void main()
{
  vec2 ndc = mix(u_rect.xy, u_rect.zw, corner);
  v_ndc = ndc;
  gl_Position = vec4(ndc, u_slab.x, 1.0);
}
