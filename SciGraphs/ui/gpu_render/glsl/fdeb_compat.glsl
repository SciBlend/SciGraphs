float scig_vis(vec3 pa, vec3 pu, float pl, vec3 qa, vec3 qu, float ql)
{
  float t0 = dot(qa - pa, pu);
  float t1 = dot(qa + qu * ql - pa, pu);
  float mid = 0.5 * (t0 + t1);
  float width = abs(t1 - t0);
  return max(1.0 - 2.0 * abs(pl * 0.5 - mid) / max(width, 1e-12), 0.0);
}

float scig_compat(vec3 pa, vec3 pu, float pl, vec3 qa, vec3 qu, float ql)
{
  float angle = abs(dot(pu, qu));
  float l_avg = 0.5 * (pl + ql);
  float scale = 2.0 / (l_avg / max(min(pl, ql), 1e-12)
                       + max(pl, ql) / max(l_avg, 1e-12));
  vec3 pm = pa + pu * (pl * 0.5);
  vec3 qm = qa + qu * (ql * 0.5);
  float pos = l_avg / (l_avg + distance(pm, qm));
  float vis = 1.0;
  if (u_vis != 0)
  {
    vis = min(scig_vis(pa, pu, pl, qa, qu, ql),
              scig_vis(qa, qu, ql, pa, pu, pl));
  }
  return angle * scale * pos * vis;
}
