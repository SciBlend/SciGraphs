void main()
{
  int i = scig_index();
  if (i >= u_count)
  {
    return;
  }
  vec3 p = imageLoad(pos_img, scig_texel(i)).xyz;
  imageAtomicAdd(cell_count, scig_texel(scig_bucket(i, p)), 1u);
}
