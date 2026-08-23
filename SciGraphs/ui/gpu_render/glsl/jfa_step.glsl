// Strict comparison and scan order (dy outer, dx inner) match the reference.

void main()
{
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= u_size.x || p.y >= u_size.y)
  {
    return;
  }

  vec4 best = imageLoad(in_img, p);
  float bd = (best.z > 0.5) ? distance(vec2(p), best.xy) : 3.4e38;

  for (int dy = -1; dy <= 1; ++dy)
  {
    for (int dx = -1; dx <= 1; ++dx)
    {
      ivec2 q = p + ivec2(dx, dy) * u_step;
      if (q.x < 0 || q.y < 0 || q.x >= u_size.x || q.y >= u_size.y)
      {
        continue;
      }
      vec4 c = imageLoad(in_img, q);
      if (c.z < 0.5)
      {
        continue;
      }
      float d = distance(vec2(p), c.xy);
      if (d < bd)
      {
        bd = d;
        best = c;
      }
    }
  }
  imageStore(out_img, p, best);
}
