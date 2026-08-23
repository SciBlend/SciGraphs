vec4 fetch_texel(sampler2D tex, int idx, int row_len)
{
  return texelFetch(tex, ivec2(idx % row_len, idx / row_len), 0);
}

/* Knot i of an open uniform vector: degree+1 zeros, `interior` evenly spaced
 * values, degree+1 ones. Computed rather than stored; it is two comparisons. */
float heb_knot(int i, int degree, int interior)
{
  if (i <= degree)
  {
    return 0.0;
  }
  if (i >= degree + 1 + interior)
  {
    return 1.0;
  }
  return float(i - degree) / float(interior + 1);
}

/* The strength this edge is drawn at.
 *
 * u_beta is the global dial. A third meta texel per edge, when the pack step
 * emitted one, scales it per edge, so long links can hug their bundle while
 * short ones stay straight. Edges whose meta stride is two never had that texel
 * and read the global value, so the shader serves both layouts without the
 * caller having to say which it uploaded.
 */
float heb_beta(int e)
{
  if (u_meta_stride < 3)
  {
    return u_beta;
  }
  return u_beta * fetch_texel(u_meta, e * u_meta_stride + 2, u_meta_row).x;
}

/* Control point i, with the bundling strength applied here rather than to the
 * evaluated curve: equation 1 of the paper, and the variant that costs one
 * operation per control point instead of one per drawn sample. */
vec3 heb_ctrl(int base, int i, int n, vec3 first, vec3 last, float beta)
{
  vec3 c = fetch_texel(u_ctrl, base + i, u_ctrl_row).xyz;
  if (beta >= 1.0)
  {
    return c;
  }
  float w = (n > 1) ? float(i) / float(n - 1) : 0.0;
  return beta * c + (1.0 - beta) * (first + w * (last - first));
}

vec3 heb_eval(int base, int n, float t, float beta)
{
  int degree = min(3, n - 1);
  int interior = n - degree - 1;
  vec3 first = fetch_texel(u_ctrl, base, u_ctrl_row).xyz;
  vec3 last = fetch_texel(u_ctrl, base + n - 1, u_ctrl_row).xyz;
  if (degree < 1)
  {
    return first;
  }

  /* Uniform interior knots, so the span containing t is arithmetic. */
  int span = degree + min(int(t * float(interior + 1)), interior);

  vec3 d[4];
  for (int j = 0; j <= degree; j++)
  {
    d[j] = heb_ctrl(base, j + span - degree, n, first, last, beta);
  }
  for (int r = 1; r <= degree; r++)
  {
    for (int j = degree; j >= r; j--)
    {
      int i = j + span - degree;
      float lo = heb_knot(i, degree, interior);
      float hi = heb_knot(i + degree - r + 1, degree, interior);
      float den = hi - lo;
      d[j] = mix(d[j - 1], d[j], den > 0.0 ? (t - lo) / den : 0.0);
    }
  }
  return d[degree];
}
