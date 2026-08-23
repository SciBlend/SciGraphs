void main()
{
  int i = scig_index();
  if (i >= u_count)
  {
    return;
  }
  vec3 p = imageLoad(pos_img, scig_texel(i)).xyz;
  int c = scig_bucket(i, p);
  // The cursor starts at zero and counts up; adding the bucket's start offset
  // gives this point a slot no other point can get.
  uint slot = imageAtomicAdd(cell_cursor, scig_texel(c), 1u);
  int start = int(imageLoad(cell_start, scig_texel(c)).x + 0.5);
  imageStore(sorted_idx, scig_texel(start + int(slot)),
             uvec4(uint(i), 0u, 0u, 0u));
}
