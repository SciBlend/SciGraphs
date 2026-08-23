// Notes in source order:
// A mesh normal can face away; unclamped, 1 - n.z cubes far too bright.
// No outline: v_normal gives no distance to the silhouette to ink.
void main()
{
  vec3 n = normalize(v_normal);
  if (!gl_FrontFacing) { n = -n; }
  float rim01 = clamp(1.0 - n.z, 0.0, 1.0);
  vec3 shaded = scig_toon_shade(n, v_color.rgb, rim01, gl_FragCoord.xy);
  fragColor = vec4(shaded, 1.0);
}
