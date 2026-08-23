// Taken from inside, so the boundary is a subset, as the skeleton assumes.

void main()
{
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= u_size.x || p.y >= u_size.y)
  {
    return;
  }
  uint here = imageLoad(mask_img, p).x;
  uint edge = 0u;
  if (here != 0u)
  {
    const ivec2 off[4] = ivec2[4](ivec2(1, 0), ivec2(-1, 0),
                                  ivec2(0, 1), ivec2(0, -1));
    for (int i = 0; i < 4; ++i)
    {
      ivec2 q = p + off[i];
      // The image border counts as outside. A shape running off the edge of the
      // raster would otherwise have no boundary there and its skeleton would
      // run out to the margin.
      if (q.x < 0 || q.y < 0 || q.x >= u_size.x || q.y >= u_size.y)
      {
        edge = 1u;
      }
      else if (imageLoad(mask_img, q).x == 0u)
      {
        edge = 1u;
      }
    }
  }
  imageStore(out_img, p, uvec4(edge, 0u, 0u, 0u));
}
