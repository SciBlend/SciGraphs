// Exact Euclidean dilation: a (2r+1)^2 scan at r=6 is 169 loads against 81.

void main()
{
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= u_size.x || p.y >= u_size.y)
  {
    return;
  }
  vec4 f = imageLoad(ft_img, p);
  bool in_set = (f.z > 0.5) && (distance(vec2(p), f.xy) <= u_radius);
  imageStore(out_img, p, uvec4(in_set ? 1u : 0u, 0u, 0u, 0u));
}
