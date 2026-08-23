// Materialized only for cushion shading, which samples it per fragment.

void main()
{
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= u_size.x || p.y >= u_size.y)
  {
    return;
  }
  vec4 f = imageLoad(ft_img, p);
  float d = (f.z > 0.5) ? distance(vec2(p), f.xy) : u_empty;
  imageStore(out_img, p, vec4(d, 0.0, 0.0, 0.0));
}
