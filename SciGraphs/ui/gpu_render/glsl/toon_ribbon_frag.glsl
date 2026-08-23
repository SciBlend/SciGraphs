void main()
{
  float x = clamp(v_side, -1.0, 1.0);
  float z = sqrt(max(0.0, 1.0 - x * x));
  vec3 normal = normalize(v_normal_basis * x + vec3(0.0, 0.0, 1.0) * z);
  vec3 axis_pt = v_view_pos - v_normal_basis * (x * v_radius);
  vec3 surface = axis_pt + normal * v_radius;
  vec4 clip = u_proj * vec4(surface, 1.0);
  gl_FragDepth = (clip.z / clip.w) * 0.5 + 0.5;
  float rim01 = abs(x);
  vec3 shaded = scig_toon_shade(normal, v_color.rgb, rim01, gl_FragCoord.xy);
  fragColor = vec4(scig_toon_outline(shaded, rim01), 1.0);
}
