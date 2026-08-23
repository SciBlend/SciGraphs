float dist_of(float d)
{
  float ndc = d * 2.0 - 1.0;
  float denom = ndc + u_a;
  return (abs(denom) > 1e-6) ? abs(u_b / denom) : 1e6;
}

float coc_of(float dist)
{
  return clamp(0.5 * u_coc_scale * abs(dist - u_focus) / max(dist, 1e-6),
               0.0, u_max_radius);
}
