// Surface lighting shared by terrain, grass and the runner. Requires common.glsl.
// Two key lights (sun, moon), each with its own shadow map of the runner, plus
// sky ambient and aerial-perspective fog that both sample the same sky model.

uniform sampler2DShadow uShadowSun;
uniform sampler2DShadow uShadowMoon;
uniform mat4 uSunMatrix;
uniform mat4 uMoonMatrix;
uniform vec3 uEye;

float shadowLookup(sampler2DShadow map, mat4 lightMatrix, vec3 worldPos, vec3 n, vec3 L) {
    float ndl = max(dot(n, L), 0.0);
    vec4 clip = lightMatrix * vec4(worldPos + n * (0.02 + 0.04 * (1.0 - ndl)), 1.0);
    vec3 p = clip.xyz / clip.w * 0.5 + 0.5;
    if (p.x <= 0.0 || p.x >= 1.0 || p.y <= 0.0 || p.y >= 1.0 || p.z >= 1.0) return 1.0;
    vec2 texel = 1.5 / vec2(textureSize(map, 0));
    float lit = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            lit += texture(map, vec3(p.xy + vec2(x, y) * texel, p.z - 0.0015));
    return lit / 9.0;
}

vec3 sunRadiance()  { return sunTint() * 3.2 * smoothstep(-0.03, 0.10, uSunDir.y); }
vec3 moonRadiance() { return vec3(0.32, 0.42, 0.75) * 0.60 * nightAmount() * smoothstep(-0.05, 0.12, uMoonDir.y); }

vec3 ambientLight(vec3 n) {
    vec3 sky = mix(skyGradient(vec3(0.0, 1.0, 0.0)),
                   skyGradient(normalize(vec3(n.x, 0.25, n.z) + vec3(0.0, 1e-3, 0.0))), 0.5) * 1.1;
    vec3 ground = sky * vec3(0.30, 0.34, 0.22);
    return mix(ground, sky, 0.5 + 0.5 * n.y);
}

// translucency: light transmitted through thin surfaces (grass blades) when back-lit.
vec3 shadeSurface(vec3 albedo, vec3 n, vec3 worldPos, float translucency, float ao) {
    vec3 V = normalize(uEye - worldPos);
    vec3 sunL  = sunRadiance()  * shadowLookup(uShadowSun,  uSunMatrix,  worldPos, n, uSunDir);
    vec3 moonL = moonRadiance() * shadowLookup(uShadowMoon, uMoonMatrix, worldPos, n, uMoonDir);
    vec3 direct = sunL * max(dot(n, uSunDir), 0.0) + moonL * max(dot(n, uMoonDir), 0.0);
    vec3 back = (sunL * pow(max(dot(-V, uSunDir), 0.0), 3.0)
               + moonL * pow(max(dot(-V, uMoonDir), 0.0), 3.0)) * translucency;
    return albedo * (ambientLight(n) * ao + (direct + back) * mix(0.55, 1.0, ao));
}

vec3 applyFog(vec3 color, vec3 worldPos) {
    vec3 d = worldPos - uEye;
    float dist = length(d);
    float fog = 1.0 - exp(-pow(dist * 0.024, 2.0));
    return mix(color, skyGradient(d / max(dist, 1e-4)), clamp(fog, 0.0, 1.0));
}
