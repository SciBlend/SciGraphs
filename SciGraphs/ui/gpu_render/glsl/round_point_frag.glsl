void main()
{
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) { discard; }
  fragColor = f_color;
}
