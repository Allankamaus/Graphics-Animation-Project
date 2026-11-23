in vec3 vWorld;
in vec3 vNormal;
in vec3 vColor;
out vec4 FragColor;

uniform vec3 uTint;

void main() {
    vec3 V = normalize(uEye - vWorld);
    vec3 n = normalize(vNormal);
    if (dot(n, V) < 0.0) n = -n; // inside of open cloth (jacket hem, collar)
    vec3 color = shadeSurface(vColor * uTint, n, vWorld, 0.0, 1.0);

    // Sky-coloured rim light keeps the dark outfit readable against the night sky.
    float rim = pow(1.0 - max(dot(n, V), 0.0), 3.0);
    color += rim * skyGradient(normalize(n + vec3(0.0, 0.3, 0.0))) * 0.9;

    // Soft cloth sheen from the key lights.
    vec3 hs = normalize(V + uSunDir), hm = normalize(V + uMoonDir);
    color += (sunRadiance() * pow(max(dot(n, hs), 0.0), 48.0) + moonRadiance() * pow(max(dot(n, hm), 0.0), 48.0)) * 0.12;

    FragColor = vec4(toDisplay(applyFog(color, vWorld), gl_FragCoord.xy), 1.0);
}
