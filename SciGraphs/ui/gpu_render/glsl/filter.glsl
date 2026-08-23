// The 4096 below is shaders.FILTER_TEX_ROW, which filter_gpu.py uses to lay the
// texture out. Nothing interpolates it, so the two are only equal by hand:
// change one and change the other, or every row after the first reads garbage.
vec4 scig_frow(int idx)
{
  return texelFetch(u_fchan, ivec2(idx % 4096, idx / 4096), 0);
}

bool scig_span(vec4 v, vec4 lo, vec4 hi, vec4 inv, vec4 act)
{
  vec4 ins = step(lo, v) * step(v, hi);
  vec4 keep = mix(ins, 1.0 - ins, inv);
  keep = max(keep, 1.0 - act);
  return min(min(keep.x, keep.y), min(keep.z, keep.w)) > 0.5;
}

bool scig_keep(vec3 rows)
{
  if (u_filter.act != vec4(0.0))
  {
    if (!scig_span(scig_frow(int(rows.x)), u_filter.lo, u_filter.hi,
                   u_filter.inv, u_filter.act))
    {
      return false;
    }
    if (rows.y != rows.x
        && !scig_span(scig_frow(int(rows.y)), u_filter.lo, u_filter.hi,
                      u_filter.inv, u_filter.act))
    {
      return false;
    }
  }
  if (rows.z >= 0.0 && u_filter.eact != vec4(0.0))
  {
    if (!scig_span(scig_frow(int(rows.z)), u_filter.elo, u_filter.ehi,
                   u_filter.einv, u_filter.eact))
    {
      return false;
    }
  }
  return true;
}
