// Notes in source order:
// gl_PointCoord runs top-down, glyphs bottom-up; unflipped, the DISK
// triangle points down and the SPHERE one up.
// Alpha passes through: this tier's alpha is a density hint.
void main()
{
  vec2 p = vec2(gl_PointCoord.x - 0.5, 0.5 - gl_PointCoord.y) * 2.0;
  vec3 n; float h; float edge;
  if (!scig_glyph_eval(p, scig_toon_glyph(), u_toon.screen.w,
                       u_toon.misc.y, n, h, edge)) { discard; }
  fragColor = vec4(scig_toon_outline(f_color.rgb, edge), f_color.a);
}
