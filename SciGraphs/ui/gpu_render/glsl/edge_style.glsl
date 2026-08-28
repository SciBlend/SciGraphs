#define SCIG_STRAIGHT 0
#define SCIG_CURVED 1
#define SCIG_QUADRATIC 2
#define SCIG_ARC 3
#define SCIG_TAPERED 4
#define SCIG_ORTHOGONAL 5
#define SCIG_SELF_LOOP 6

vec4 scig_edge(int e, int slot)
{
  int i = e * 2 + slot;
  return texelFetch(u_edge, ivec2(i % 4096, i / 4096), 0);
}

vec3 scig_edge_frow(int e)
{
  return vec3(scig_edge(e, 0).xy, scig_edge(e, 1).z);
}

vec3 scig_unit(vec3 v)
{
  return v / max(length(v), 1e-12);
}

float scig_flip = 1.0;

vec3 scig_perp(vec3 d, float flip)
{
  vec3 p = cross(d * flip, vec3(0.0, 0.0, 1.0));
  if (length(p) < 1e-10)
  {
    p = cross(d * flip, vec3(1.0, 0.0, 0.0));
  }
  return scig_unit(p);
}

float scig_dir_sign(vec3 a, vec3 b, float eid, float flip)
{
  int m = int(u_style.shape.z + 0.5);
  if (m == 1) { return flip; }
  if (m == 2) { return -flip; }
  if (m == 3) { return (mod(eid, 2.0) < 0.5) ? 1.0 : -1.0; }
  return (a.x + a.y > b.x + b.y) ? 1.0 : -1.0;
}

void scig_edge_ends(int e, out vec3 a, out vec3 b, out float eid)
{
  vec4 t0 = scig_edge(e, 0);
  a = scig_pos(int(t0.x + 0.5));
  b = scig_pos(int(t0.y + 0.5));
  eid = t0.z;
  scig_flip = (t0.x <= t0.y) ? 1.0 : -1.0;
  if (t0.w != 0.0)
  {
    vec3 shift = scig_perp(scig_unit(b - a), scig_flip) * t0.w;
    a += shift;
    b += shift;
  }
}

vec3 scig_ortho_point(vec3 a, vec3 b, float t)
{
  int k = int(t * 3.0 + 0.5);
  if (k <= 0) { return a; }
  if (k >= 3) { return b; }
  vec3 mid = (a + b) * 0.5;
  vec3 pa, pb;
  int os = int(u_style.shape.w + 0.5);
  if (os == 1)
  {
    pa = vec3(b.x, a.y, a.z);
    pb = vec3(b.x, b.y, a.z);
  }
  else if (os == 2)
  {
    pa = vec3(a.x, b.y, a.z);
    pb = vec3(a.x, b.y, b.z);
  }
  else if (os == 3)
  {
    pa = (abs(b.x - a.x) > abs(b.y - a.y)) ? vec3(b.x, a.y, a.z)
                                           : vec3(a.x, b.y, a.z);
    pb = pa;
  }
  else
  {
    pa = vec3(mid.x, a.y, a.z);
    pb = vec3(mid.x, b.y, mid.z);
  }
  return (k == 1) ? pa : pb;
}

vec3 scig_style_point(vec3 a, vec3 b, float t, float eid)
{
  int style = int(u_style.shape.x + 0.5);
  float curv = u_style.shape.y;

  if (style == SCIG_SELF_LOOP)
  {
    float r = u_style.curve.x;
    float ang = 6.28318530717958647 * t;
    return a + vec3(r * 0.5, 0.0, r * 0.5) + r * vec3(cos(ang), sin(ang), 0.0);
  }
  if (style == SCIG_ORTHOGONAL)
  {
    return scig_ortho_point(a, b, t);
  }
  if (style == SCIG_STRAIGHT || curv < 1e-3)
  {
    return mix(a, b, t);
  }

  vec3 d = b - a;
  float len = length(d);
  float sgn = scig_dir_sign(a, b, eid, scig_flip);

  if (style == SCIG_ARC)
  {
    float sag = len * curv * 0.5;
    if (sag <= 1e-3) { return mix(a, b, t); }
    float rad = (len * len) / max(8.0 * sag, 1e-12) + sag * 0.5;
    vec3 c = (a + b) * 0.5
           - scig_perp(scig_unit(d), scig_flip) * sgn * (rad - sag);
    vec3 vi = mix(scig_unit(a - c), scig_unit(b - c), t);
    return c + scig_unit(vi) * rad;
  }

  vec3 perp = scig_perp(scig_unit(d), scig_flip) * (len * curv * 0.5 * sgn);
  float u = 1.0 - t;
  if (style == SCIG_CURVED)
  {
    vec3 c1 = a + d * 0.25 + perp * 0.5;
    vec3 c2 = a + d * 0.75 + perp * 0.5;
    return u * u * u * a + 3.0 * u * u * t * c1
         + 3.0 * u * t * t * c2 + t * t * t * b;
  }
  vec3 c = (a + b) * 0.5 + perp;
  return u * u * a + 2.0 * u * t * c + t * t * b;
}

float scig_style_taper(float t)
{
  if (int(u_style.shape.x + 0.5) != SCIG_TAPERED) { return 1.0; }
  return mix(u_style.curve.y, u_style.curve.z, t);
}
