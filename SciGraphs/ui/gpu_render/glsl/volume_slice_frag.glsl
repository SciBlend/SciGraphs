void main()
{
  vec4 h = u_clip_to_vol * vec4(v_ndc, u_slab.x, 1.0);
  vec3 t = h.xyz / h.w;
  if (any(lessThan(t, vec3(0.0))) || any(greaterThan(t, vec3(1.0))))
    { discard; }
  vec4 s = texture(field, t);
  float d = clamp(s.a, 0.0, 1.0);
  vec4 tf = texture(lut, vec2(d, 0.5));
  if (tf.a <= 0.0) { discard; }
  float w = tf.a;
  if (u_field.z >= 0.5)
{
    float width = max(u_field.w, 0.01);
    float band = d * u_field.z;
    float e = abs(band - floor(band + 0.5));
    float shell = 1.0 - smoothstep(0.0, width, e);
    if (shell <= 0.0) { discard; }
    w *= shell / width;
  }
  vec4 h2 = u_clip_to_vol * vec4(v_ndc, u_slab.x + u_slab.y, 1.0);
  vec3 t2 = h2.xyz / h2.w;
  float len = length((t2 - t) * u_shape);
  float alpha = 1.0 - exp(-u_field.x * w * len);
  vec3 rgb = (u_field.y > 0.5) ? s.rgb : tf.rgb;
  fragColor = vec4(rgb, clamp(alpha, 0.0, 1.0));
}
