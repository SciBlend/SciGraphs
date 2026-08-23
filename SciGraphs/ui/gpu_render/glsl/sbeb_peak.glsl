// Into a 1x1 image, so the threshold is relative without returning to Python.
// Without the `v != 0` guard all quarter of a million invocations hit one atomic
// address; with it only the inked texels contend, a few percent.

void main()
{
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= u_res || p.y >= u_res)
  {
    return;
  }
  uint v = imageLoad(dens, p).x;
  if (v != 0u)
  {
    imageAtomicMax(peak, ivec2(0, 0), v);
  }
}
