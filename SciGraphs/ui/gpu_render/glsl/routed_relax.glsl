void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= u_total)
  {
    return;
  }
  int f = i / u_cells;
  int c = i - f * u_cells;
  int base = f * u_cells;

  ivec3 g = scig_coord(c);
  float cur = imageLoad(dist_in, scig_texel(i)).x;
  // `precise` forbids contracting `dn + w` into an fma. Fused and unfused differ
  // in the last bit, an unfused numpy reference cannot follow a fused kernel,
  // and a one-ULP disagreement over a hundred-odd rounds resolves a tied pair of
  // equal-cost routes the other way. Measured on the Vulkan/SPIR-V path: without
  // it the two backends differ by 9.5e-6 after one solve and by 0.51 after four
  // reinforcement rounds, with 151 of 592 edges taking a different road; with it
  // they are bit-identical, and a four-round solve at R = 128 costs 274 ms
  // against 277 ms, so the fma bought nothing. The OpenGL driver did not
  // contract and agreed either way, which is how this was found.
  precise float best = cur;
  float own = imageLoad(cost_img, scig_texel(c)).x;

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
        float dn = imageLoad(dist_in, scig_texel(base + nc)).x;
        if (dn >= 1e29)
        {
          continue;
        }
        float w = 0.5 * (own + imageLoad(cost_img, scig_texel(nc)).x)
                * scig_step(dx, dy, dz) * u_cell;
        best = min(best, dn + w);
      }
    }
  }

  // The convergence flag is a counter rather than a store, so the rounds between
  // readbacks accumulate instead of the last one overwriting the rest.
  if (best < cur - u_eps)
  {
    imageAtomicAdd(changed, ivec2(0, 0), 1u);
  }
  imageStore(dist_out, scig_texel(i), vec4(best, 0.0, 0.0, 0.0));
}
