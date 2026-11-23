in vec3 vWorld;
in vec3 vNormal;
in vec3 vAlbedo;
in float vT;
out vec4 FragColor;

void main() {
    // Blades are two-sided: light the side facing the viewer, softened toward "up".
    vec3 n = normalize(vNormal);
    if (dot(n, uEye - vWorld) < 0.0) n = -n;
    n = normalize(mix(n, vec3(0.0, 1.0, 0.0), 0.3));
    float ao = mix(0.30, 1.0, smoothstep(0.0, 0.8, vT));
    vec3 color = shadeSurface(vAlbedo, n, vWorld, 0.55, ao);
    FragColor = vec4(toDisplay(applyFog(color, vWorld), gl_FragCoord.xy), 1.0);
}
