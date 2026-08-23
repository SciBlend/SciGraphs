ivec3 scig_cell(vec3 p)
{
  return clamp(ivec3(floor((p - u_lo) * u_inv)), ivec3(0), u_res - ivec3(1));
}

int scig_cell_id(ivec3 c)
{
  return (c.x * u_res.y + c.y) * u_res.z + c.z;
}

// The bucket a point belongs to: its cell, refined by where it sits along its
// own edge. Only one point per edge is a candidate for a given subdivision
// index, so the force kernel can fetch those and nothing else.
int scig_bucket(int i, vec3 p)
{
  return scig_cell_id(scig_cell(p)) * u_k + (i - (i / u_k) * u_k);
}
