const float SCIG_RIM_EDGE = 0.75;
const float SCIG_GOOCH_ALPHA = 0.2;
const float SCIG_GOOCH_BETA = 0.6;

int scig_toon_mode()
{
  return int(u_toon.warm.w + 0.5);
}

int scig_toon_glyph()
{
  return int(u_toon.cool.w + 0.5);
}

bool scig_toon_is_lit()
{
  return scig_toon_mode() == 0;
}

// The half vector between the key light and the view axis. key_dir may be the
// exact opposite of the view axis (a backlight), and normalize() of the zero
// vector is a NaN, which shows as a black or transparent hole in the node.
vec3 scig_toon_half()
{
  vec3 h = u_light.key_dir.xyz + vec3(0.0, 0.0, 1.0);
  float l = length(h);
  return (l > 1e-6) ? h / l : vec3(0.0, 0.0, 1.0);
}

// One diffuse term, quantized into `steps` bands with soft edges. smoothstep
// around each threshold rather than floor(), which aliases along the band
// boundary into a couple-of-pixels-wide curve crawling as the view turns.
// `softness` is the half-width of the transition in units of one band.
float scig_toon_ramp(float d)
{
  float steps = max(floor(u_toon.bands.x + 0.5), 1.0);
  float s = clamp(d, 0.0, 1.0) * steps;
  float i = floor(s);
  // smoothstep is undefined when its two edges coincide, and softness == 0 is
  // both a legal setting and the first one a user reaches for.
  float w = max(u_toon.bands.y, 1e-3);
  return min((i + smoothstep(0.5 - w, 0.5 + w, s - i)) / steps, 1.0);
}

// The hard specular: on or off, never a falloff. `spec_size` is measured from
// the top of the Blinn-Phong lobe so that the slider grows the highlight
// instead of moving it.
float scig_toon_spec(vec3 n)
{
  float gloss = max(u_toon.misc.x, 1.0);
  float lobe = pow(max(dot(n, scig_toon_half()), 0.0), gloss);
  return step(1.0 - u_toon.bands.w, lobe) * u_toon.bands.z;
}

// Rim as a band. The lit path's pow(rim01, 3) is a gradient, which a stylized
// node must not have at the silhouette; the band edge reuses `softness` so the
// rim and the ramp are equally crisp.
float scig_toon_rim(float rim01)
{
  float w = max(u_toon.bands.y, 1e-3);
  return smoothstep(SCIG_RIM_EDGE - w, SCIG_RIM_EDGE + w, rim01) * u_toon.shadow.w;
}

// The banded diffuse at band value `q` (key) and `qf` (fill).
//
// The darkest band, q == 0, is the albedo through the shadow tint; the
// brightest leaves it alone. Tinting with the band value rather than with a
// second threshold keeps the tint quantized with everything else. The fill goes
// through the same ramp as the key; leaving it smooth would put a continuous
// gradient back into a cel image, visible immediately on a sphere.
vec3 scig_toon_diffuse(vec3 albedo, float q, float qf)
{
  vec3 base = albedo * mix(u_toon.shadow.rgb, vec3(1.0), q);
  return base * (u_light.ambient.rgb
                 + u_light.key_col.rgb * q
                 + u_light.fill_col.rgb * qf);
}

// Screen-space halftone coverage: 1 = full ink, 0 = bare paper.
//
// The grid is in device pixels, not in quad space, so the dots stay the same
// size whatever the node's radius is and read as a print screen laid over the
// image rather than a texture stuck to each node; the pattern therefore does
// not follow a node when the view pans. Dot area tracks the tone, hence the
// sqrt: a radius linear in the band value reads far too dark in the midtones.
float scig_halftone(vec2 frag, float value)
{
  float cells = max(u_toon.screen.x, 1.0) * 0.01;
  float ca = cos(u_toon.screen.y);
  float sa = sin(u_toon.screen.y);
  vec2 g = vec2(frag.x * ca - frag.y * sa, frag.x * sa + frag.y * ca) * cells;
  vec2 c = fract(g) - 0.5;
  float d = length(c);
  float radius = 0.5 * sqrt(clamp(1.0 - value, 0.0, 1.0));
  // fwidth, not a fixed epsilon: the dot edge is a hard circle in a grid fixed
  // to the screen, so without a derivative-sized transition it crawls and
  // moires as soon as anything moves.
  float aa = max(fwidth(d), 1e-4);
  // The top band asks for no dot at all, and an anti-aliased circle of radius 0
  // still leaves a speck at every cell center.
  return (1.0 - smoothstep(radius - aa, radius + aa, d)) * step(1e-5, radius);
}

// The stylized shade of one fragment.
//   n      unit shading normal, view space (z toward the camera)
//   albedo the node/edge color
//   rim01  0 facing the camera, 1 at the silhouette; for the sphere impostor
//          and every glyph built on it that is 1.0 - n.z
//   frag   gl_FragCoord.xy, for the screen-space halftone screen
//
// The result is clamped here rather than at the call site, so every caller
// clamps the way _build_sphere does.
vec3 scig_toon_shade(vec3 n, vec3 albedo, float rim01, vec2 frag)
{
  int mode = scig_toon_mode();
  float d_key = max(dot(n, u_light.key_dir.xyz), 0.0);
  float d_fill = max(dot(n, u_light.fill_dir.xyz), 0.0);

  // LIT: term for term shaders._build_sphere. Do not "simplify" this.
  if (mode == 0)
  {
    float rim = pow(rim01, 3.0) * u_light.key_col.w;
    vec3 lit = albedo * (u_light.ambient.rgb + u_light.key_col.rgb * d_key + u_light.fill_col.rgb * d_fill) + rim;
    return min(lit, vec3(1.0));
  }

  // FLAT: the albedo, untouched. No ambient, key, specular or rim. This is the
  // mode that makes an outline read, since an outline against a shaded ball is
  // just a darker ball.
  if (mode == 3)
  {
    return min(albedo, vec3(1.0));
  }

  vec3 lit;
  if (mode == 2)
  {
    // GOOCH: the cool-to-warm ramp of Gooch et al. (1998). The signed n.l, so
    // the shadowed side is cool rather than absent and a technical illustration
    // keeps its shape where a lit render goes to ambient. The bands are
    // ignored; this ramp is already the abstraction.
    float ndl = clamp(dot(n, u_light.key_dir.xyz), -1.0, 1.0);
    vec3 cool = u_toon.cool.rgb + SCIG_GOOCH_ALPHA * albedo;
    vec3 warm = u_toon.warm.rgb + SCIG_GOOCH_BETA * albedo;
    lit = mix(cool, warm, (1.0 + ndl) * 0.5);
  }
  else if (mode == 4)
  {
    // HALFTONE: the TOON ramp, spent on dot area instead of on brightness. Both
    // ends of the ramp are evaluated as colors and the dot coverage picks
    // between them, so the ink is the shadow tint and the paper the lit color
    // -- a two-color screen, not a gray multiplied over albedo.
    float qf = scig_toon_ramp(d_fill);
    float ink = scig_halftone(frag, scig_toon_ramp(d_key));
    lit = mix(scig_toon_diffuse(albedo, 1.0, qf),
              scig_toon_diffuse(albedo, 0.0, qf), ink);
  }
  else
  {
    // TOON.
    lit = scig_toon_diffuse(albedo, scig_toon_ramp(d_key), scig_toon_ramp(d_fill));
  }

  lit += scig_toon_spec(n) + scig_toon_rim(rim01);
  return min(lit, vec3(1.0));
}

// Mixes the outline color in near the silhouette.
//   edge01  0 at the glyph center, exactly 1 at the silhouette
//
// INNER eats the outer `width` of the glyph, leaving the silhouette unchanged
// so the beauty pass still covers exactly the fragments the ID pass does. OUTER
// expects the caller to have grown the quad by (1 + width) in the vertex shader
// and to pass an edge01 measured in the grown quad, so the ink lands outside
// the glyph -- at the cost of a silhouette wider than the ID pass's, which is
// why INNER is the default.
vec3 scig_toon_outline(vec3 shaded, float edge01)
{
  float w = max(u_toon.outline.w, 0.0);
  if (w <= 0.0)
  {
    return shaded;
  }
  float lim = (u_toon.screen.z > 0.5) ? 1.0 / (1.0 + w) : 1.0 - w;
  float aa = max(fwidth(edge01), 1e-4);
  return mix(shaded, u_toon.outline.rgb,
             smoothstep(lim - aa, lim + aa, edge01));
}
