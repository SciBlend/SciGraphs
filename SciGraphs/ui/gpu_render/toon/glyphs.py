# Node glyphs, GLSL source only. Every glyph is the unit disc in disguise: each
# gives, for the fragment's direction, the center and half-width of the shape
# along it. Quad coordinate p maps to disc coordinate q = (p - dir * mid) / half
# and from there it is the sphere impostor again. The disc is mid 0.0 half 1.0,
# exact in IEEE, so ROUND is the sphere bit for bit. An SDF gradient would not.

# In GLYPH_IDS order, but not imported from toon/__init__: this module has to
# stay loadable without its package. A test checks the two agree.
GLYPH_ITEMS = (
    ('ROUND', "Round", "Sphere impostor: the default node, unchanged"),
    ('SQUARE', "Square", "Axis-aligned square tile"),
    ('DIAMOND', "Diamond", "Square on its point"),
    ('TRIANGLE', "Triangle", "Equilateral triangle, apex up"),
    ('HEXAGON', "Hexagon", "Regular hexagon, apex up"),
    ('STAR', "Star", "Five-pointed star"),
    ('RING', "Ring", "Annulus, shaded as a torus section"),
    ('CROSS', "Cross", "Plus sign with square arms"),
)


# Every shape is inscribed in the unit disc at the sphere's silhouette, so a
# glyph's fragments are a subset of the round node's and the quad never grows.
GLYPH_GLSL = """
const float SCIG_PI = 3.14159265;
// Ring: outer rim at 1.0, inner rim at 0.55, so the tube is 0.225 of the radius
// thick and sits on the circle at 0.775.
const float SCIG_RING_MID = 0.775;
const float SCIG_RING_HALF = 0.225;
// Star: the classic 5/2 pentagram waist. Any smaller and the points break up at
// the sizes a node is actually drawn at.
const float SCIG_STAR_INNER = 0.382;
// Cross: arm half-width, and the arm length that puts the arm corners -- not
// the tips -- on the unit circle, sqrt(1 - w*w).
const float SCIG_CROSS_W = 0.30;
const float SCIG_CROSS_L = 0.95393920;

// Distance from the center to the boundary of a regular n-gon with
// circumradius 1, in the direction u, with a vertex at angle `phase`.
//
// cos(k) cannot reach zero: |k| <= pi/n <= pi/3 for every n used here, so the
// division is safe and the guard only matters if a caller passes n < 3.
float scig_ngon_span(vec2 u, float sides, float phase)
{
  float half_sector = SCIG_PI / sides;
  float a = atan(u.y, u.x) - phase;
  float k = mod(a, 2.0 * half_sector) - half_sector;
  return cos(half_sector) / max(cos(k), 1e-4);
}

// Same, for a five-pointed star: fold the direction into one half-sector and
// take the radial distance to the straight segment joining an outer point
// (radius 1) to the neighboring inner vertex (radius SCIG_STAR_INNER).
//
// The denominator is a sum of two sines over [0, 36 degrees] which are never
// both zero, so it cannot vanish; guarded anyway.
float scig_star_span(vec2 u)
{
  float period = 2.0 * SCIG_PI / 5.0;
  float half_p = period * 0.5;
  float a = mod(atan(u.y, u.x) - SCIG_PI * 0.5, period);
  a = min(a, period - a);
  float d = sin(a) + SCIG_STAR_INNER * sin(half_p - a);
  return SCIG_STAR_INNER * sin(half_p) / max(d, 1e-4);
}

// A plus sign: the union of two rectangles, so the span is the larger of the
// two box spans. A box with half-extents (a, b) reaches min(a/|ux|, b/|uy|) in
// direction u; the epsilon on each component is what stands in for the infinity
// of an axis-aligned direction, and min() then picks the other axis anyway.
float scig_cross_span(vec2 u)
{
  float ax = max(abs(u.x), 1e-4);
  float ay = max(abs(u.y), 1e-4);
  return max(min(SCIG_CROSS_L / ax, SCIG_CROSS_W / ay),
             min(SCIG_CROSS_W / ax, SCIG_CROSS_L / ay));
}

// The shape itself: where the glyph's medial point sits along direction u, and
// how far the boundary is from it. ROUND is (0.0, 1.0) and falls out of the
// initialization, so the disc costs no branch and no arithmetic that could
// round.
//
// if/else rather than a switch because the id is uniform across the whole draw:
// every fragment takes the same path, so there is nothing to diverge.
void scig_glyph_span(int id, vec2 u, out float mid, out float half_w)
{
  mid = 0.0;
  half_w = 1.0;
  if (id == 1) { half_w = scig_ngon_span(u, 4.0, SCIG_PI * 0.25); }
  else if (id == 2) { half_w = scig_ngon_span(u, 4.0, 0.0); }
  else if (id == 3) { half_w = scig_ngon_span(u, 3.0, SCIG_PI * 0.5); }
  else if (id == 4) { half_w = scig_ngon_span(u, 6.0, SCIG_PI * 0.5); }
  else if (id == 5) { half_w = scig_star_span(u); }
  else if (id == 6) { mid = SCIG_RING_MID; half_w = SCIG_RING_HALF; }
  else if (id == 7) { half_w = scig_cross_span(u); }
}

// Evaluate the glyph over the impostor quad. Returns false when the fragment
// is outside the glyph, and the caller discards.
//   p      quad coordinate in [-1, 1]^2 (the `corner` attribute)
//   id     glyph id, GLYPH_IDS
//   rot    rotation in radians
//   bulge  0 = flat plate (n = +Z), 1 = full spherical relief
//   n      out: unit shading normal, view space (z toward the camera)
//   h      out: height above the quad plane, in units of the radius, so the
//          surface point is view_center + radius * vec3(p, h) -- which for
//          ROUND is exactly what _build_sphere writes gl_FragDepth from
//   edge   out: 0 at the deepest interior point, 1 exactly at the silhouette
//
// `bulge` flattens the dome rather than tilting the normal off it: n stays unit
// by construction at every bulge, and at bulge == 1 the multiply by 1.0 is
// exact, so the reduction to the sphere survives it.
bool scig_glyph_eval(vec2 p, int id, float rot, float bulge,
                     out vec3 n, out float h, out float edge)
{
  float rad = length(p);
  // The center of the quad has no direction, and any unit vector will do: for
  // every glyph whose span depends on the direction, mid is 0 there, so the
  // fallback scales a zero-length vector and cannot show; for the ring it lands
  // the fragment outside, which is where the center of a ring belongs.
  vec2 dir = (rad > 1e-6) ? p / rad : vec2(1.0, 0.0);
  float cs = cos(rot);
  float sn = sin(rot);
  vec2 u = vec2(dir.x * cs + dir.y * sn, dir.y * cs - dir.x * sn);

  float mid;
  float half_w;
  scig_glyph_span(id, u, mid, half_w);
  vec2 q = (p - dir * mid) / max(half_w, 1e-6);

  float e2 = dot(q, q);
  if (e2 > 1.0) {
    n = vec3(0.0, 0.0, 1.0);
    h = 0.0;
    edge = 1.0;
    return false;
  }
  // length(q), not sqrt(e2): for ROUND, q is p bit for bit, and length() is the
  // same call the lit path makes on the same bits.
  edge = length(q);
  h = sqrt(max(0.0, 1.0 - e2));
  vec2 qb = q * clamp(bulge, 0.0, 1.0);
  n = vec3(qb, sqrt(max(0.0, 1.0 - dot(qb, qb))));
  return true;
}
"""
