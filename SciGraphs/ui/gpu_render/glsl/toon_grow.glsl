float scig_toon_grow()
{
  return (u_toon.screen.z > 0.5) ? (1.0 + max(u_toon.outline.w, 0.0)) : 1.0;
}
