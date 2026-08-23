// Skeleton from the boundary's feature transform: how far a texel's feature sits
// from its neighbors'. The test is relative, so on the axis of a ribbon of
// half-width w the ratio is 2 whatever w is. u_min_dist drops texels near the
// boundary, where it degenerates and a convex corner passes anything.

void main()
{
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= u_size.x || p.y >= u_size.y)
  {
    return;
  }

  uint out_v = 0u;
  if (imageLoad(inside_img, p).x != 0u)
  {
    vec4 f = imageLoad(ft_img, p);
    float dp = (f.z > 0.5) ? distance(vec2(p), f.xy) : 0.0;
    if (dp >= u_min_dist)
    {
      float sep = 0.0;
      const ivec2 off[4] = ivec2[4](ivec2(1, 0), ivec2(-1, 0),
                                    ivec2(0, 1), ivec2(0, -1));
      for (int i = 0; i < 4; ++i)
      {
        ivec2 q = p + off[i];
        if (q.x < 0 || q.y < 0 || q.x >= u_size.x || q.y >= u_size.y)
        {
          continue;
        }
        vec4 g = imageLoad(ft_img, q);
        if (g.z < 0.5)
        {
          continue;
        }
        sep = max(sep, distance(f.xy, g.xy));
      }
      if (sep > u_tau * dp)
      {
        out_v = 1u;
      }
    }
  }
  imageStore(out_img, p, uvec4(out_v, 0u, 0u, 0u));
}
