// Takes the sharp turns off a curve pulled toward a skeleton with corners.

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= u_n)
  {
    return;
  }
  int k = i - (i / u_k) * u_k;
  vec4 v = imageLoad(src, scig_texel(i));
  if (k > 0 && k < u_k - 1)
  {
    vec2 a = imageLoad(src, scig_texel(i - 1)).xy;
    vec2 b = imageLoad(src, scig_texel(i + 1)).xy;
    v.xy = mix(v.xy, 0.5 * (a + b), u_lambda);
  }
  imageStore(dst, scig_texel(i), v);
}
