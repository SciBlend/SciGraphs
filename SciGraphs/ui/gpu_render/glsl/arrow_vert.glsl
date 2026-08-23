void main()
{
  gl_Position = u_proj * (u_view * vec4(pos, 1.0));
  v_normal = normalize((u_view * vec4(nrm, 0.0)).xyz);
  v_color = color;
}
