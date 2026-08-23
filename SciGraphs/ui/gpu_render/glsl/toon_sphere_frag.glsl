// Notes in source order:
// Re-testing on the ungrown coordinate bounds the OUTER ring for any glyph.
// Ink on the plane through the node center joins the silhouette in depth.
// 1.0 - n.z is what the lit sphere cubes for its rim; same input, LIT matches.
void main()
{
  float grow = scig_toon_grow();
  vec2 p = v_uv * grow;
  int gid = scig_toon_glyph();
  vec3 n; float h; float edge;
  if (!scig_glyph_eval(p, gid, u_toon.screen.w, u_toon.misc.y, n, h, edge))
{
    if (u_toon.screen.z <= 0.5) { discard; }
    vec3 bn; float bh; float bedge;
    if (!scig_glyph_eval(v_uv, gid, u_toon.screen.w, u_toon.misc.y, bn, bh, bedge)) { discard; }
    vec4 bclip = u_proj * vec4(v_view_center + v_radius * vec3(p.x, p.y, 0.0), 1.0);
    gl_FragDepth = (bclip.z / bclip.w) * 0.5 + 0.5;
    fragColor = vec4(u_toon.outline.rgb, 1.0);
    return;
  }
  vec3 view_surface = v_view_center + v_radius * vec3(p.x, p.y, h);
  vec4 clip = u_proj * vec4(view_surface, 1.0);
  gl_FragDepth = (clip.z / clip.w) * 0.5 + 0.5;
  vec3 shaded = scig_toon_shade(n, v_color.rgb, 1.0 - n.z, gl_FragCoord.xy);
  fragColor = vec4(scig_toon_outline(shaded, edge / grow), 1.0);
}
