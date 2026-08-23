void main()
{
  int e = int(gl_GlobalInvocationID.x);
  if (e >= u_edges)
  {
    return;
  }
  vec4 spec = imageLoad(edge_img, scig_texel(e));
  if (spec.w > 0.5)
  {                             // endpoints share a cell: nothing to route
    imageStore(len_img, scig_texel(e), vec4(0.0, 0.0, 0.0, 0.0));
    return;
  }
  int f = int(spec.x + 0.5);
  int src = int(spec.y + 0.5);
  int c = int(spec.z + 0.5);
  int base = f * u_cells;
  int len = 0;

  for (int s = 0; s < u_max_steps; ++s)
  {
    imageStore(route_img, scig_texel(e * u_max_steps + s),
               vec4(float(c), 0.0, 0.0, 0.0));
    imageAtomicAdd(usage, scig_texel(c), 1u);
    len = s + 1;
    if (c == src)
    {
      break;
    }

    ivec3 g = scig_coord(c);
    float own = imageLoad(cost_img, scig_texel(c)).x;
    float here = imageLoad(dist_img, scig_texel(base + c)).x;
    int pick = -1;
    precise float bestv = 1e30;   // same reason as the relaxation's `best`
    float bestd = 1e30;

    for (int dz = -1; dz <= 1; ++dz)
    {
      for (int dy = -1; dy <= 1; ++dy)
      {
        for (int dx = -1; dx <= 1; ++dx)
        {
          if (dx == 0 && dy == 0 && dz == 0)
          {
            continue;
          }
          ivec3 n = g + ivec3(dx, dy, dz);
          if (scig_outside(n))
          {
            continue;
          }
          int nc = scig_flat(n);
          float dn = imageLoad(dist_img, scig_texel(base + nc)).x;
          if (dn >= 1e29)
          {
            continue;
          }
          float w = 0.5 * (own + imageLoad(cost_img, scig_texel(nc)).x)
                  * scig_step(dx, dy, dz) * u_cell;
          if (dn + w < bestv)
          {
            bestv = dn + w;
            bestd = dn;
            pick = nc;
          }
        }
      }
    }
    // Strict descent or stop. Without this a plateau in an unconverged field is
    // a two-cell cycle that runs until u_max_steps and writes a route that is a
    // scribble in one place.
    if (pick < 0 || bestd >= here)
    {
      break;
    }
    c = pick;
  }
  imageStore(len_img, scig_texel(e), vec4(float(len), 0.0, 0.0, 0.0));
}
