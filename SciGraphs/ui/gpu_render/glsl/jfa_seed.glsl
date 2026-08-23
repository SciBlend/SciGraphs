void main()
{
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= u_size.x || p.y >= u_size.y)
  {
    return;
  }

  // The threshold is relative to a value another dispatch left in a 1x1 image,
  // not to a number that came back to Python. Reading it here keeps the
  // per-cluster pipeline free of a GPU sync: a readback between the density pass
  // and this one would stall a queue that is otherwise never waited on, and
  // there are three of these per cluster per iteration.
  uint peak = imageLoad(peak_img, ivec2(0, 0)).x;
  uint cut = max(u_floor, uint(u_fraction * float(peak)));
  uint v = imageLoad(mask_img, p).x;
  imageStore(out_img, p, (v >= cut) ? vec4(float(p.x), float(p.y), 1.0, 0.0)
                                    : vec4(-1.0, -1.0, 0.0, 0.0));
}
