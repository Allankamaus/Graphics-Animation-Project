in vec3 vWorld;
in vec3 vNormal;
out vec4 FragColor;

void main() {
    vec2 p = vWorld.xz;
    float patches = fbm2(p * 0.18);
    float fine = valueNoise2(p * 3.1);
    vec3 lush = vec3(0.050, 0.100, 0.024);
    vec3 dry  = vec3(0.100, 0.098, 0.040);
    vec3 soil = vec3(0.095, 0.070, 0.045);
    vec3 albedo = mix(lush, dry, smoothstep(0.50, 0.85, patches)) * (0.8 + 0.4 * fine);

    // Worn footpath along the ridge crest, with a ragged edge.
    float edge = 0.30 + 0.16 * (valueNoise2(p * 1.7) - 0.5);
    float trail = 1.0 - smoothstep(edge, edge + 0.35, abs(p.y));
    albedo = mix(albedo, soil * (0.85 + 0.3 * fine), trail * 0.65);

    vec3 color = shadeSurface(albedo, normalize(vNormal), vWorld, 0.0, 0.9);
    FragColor = vec4(toDisplay(applyFog(color, vWorld), gl_FragCoord.xy), 1.0);
}
