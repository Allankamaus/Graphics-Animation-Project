// Shared by every scene program: hashing/noise, the terrain height field
// (mirrors terrainHeight() in main.cpp exactly) and the analytic day/night sky.

const float PI = 3.14159265359;

uniform vec3 uSunDir;     // unit vector pointing toward the sun
uniform vec3 uMoonDir;    // unit vector pointing toward the moon
uniform float uTime;      // seconds since start: wind, twinkle
uniform float uStarAngle; // rotation of the celestial sphere, radians

// ---------------------------------------------------------------- noise
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float hash13(vec3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}
float valueNoise2(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x),
               mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y);
}
float fbm2(vec2 p) {
    float sum = 0.0, amp = 0.5;
    for (int i = 0; i < 4; ++i) { sum += amp * valueNoise2(p); p = p * 2.03 + vec2(17.1, 9.2); amp *= 0.5; }
    return sum;
}
float valueNoise3(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float a = mix(hash13(i + vec3(0, 0, 0)), hash13(i + vec3(1, 0, 0)), u.x);
    float b = mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), u.x);
    float c = mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), u.x);
    float d = mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), u.x);
    return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}
float fbm3(vec3 p) {
    float sum = 0.0, amp = 0.5;
    for (int i = 0; i < 4; ++i) { sum += amp * valueNoise3(p); p = p * 2.03 + vec3(5.2, 1.3, 8.7); amp *= 0.5; }
    return sum;
}

// ---------------------------------------------------------------- terrain
// A ridge running along +x that climbs steadily, with gentle rolling detail.
// Keep in lockstep with terrainHeight() in main.cpp (feet, camera and grass use both).
float terrainHeight(vec2 p) {
    float ridge = -0.34 * (sqrt(p.y * p.y + 1.5) - sqrt(1.5));
    float slope = 0.20 * p.x;
    float rolls = 0.35 * sin(0.11 * p.x + 0.6) * cos(0.07 * p.y)
                + 0.12 * sin(0.37 * p.x + 0.21 * p.y + 1.3)
                + 0.05 * sin(0.83 * p.x - 0.57 * p.y + 2.1);
    return slope + ridge + rolls;
}
vec3 terrainNormal(vec2 p) {
    const float e = 0.1;
    float hx = terrainHeight(p + vec2(e, 0)) - terrainHeight(p - vec2(e, 0));
    float hz = terrainHeight(p + vec2(0, e)) - terrainHeight(p - vec2(0, e));
    return normalize(vec3(-hx, 2.0 * e, -hz));
}

// ---------------------------------------------------------------- sky model
// Everything is driven by the sun's elevation so day, golden hour, blue hour
// and night blend continuously with no discrete switch anywhere.
float dayAmount()      { return smoothstep(-0.04, 0.30, uSunDir.y); }
float nightAmount()    { return 1.0 - smoothstep(-0.30, -0.05, uSunDir.y); }
float twilightAmount() { float e = uSunDir.y; return smoothstep(-0.32, -0.06, e) * (1.0 - smoothstep(0.0, 0.30, e)); }
vec3 sunTint()         { return mix(vec3(1.0, 0.36, 0.10), vec3(1.0, 0.92, 0.80), smoothstep(0.0, 0.35, uSunDir.y)); }

// Sky radiance without discrete objects (no stars / discs): used for the sky
// backdrop, aerial-perspective fog and ambient light, so they always agree.
vec3 skyGradient(vec3 dir) {
    float day = dayAmount(), night = nightAmount(), dusk = twilightAmount();
    vec3 zenith  = mix(vec3(0.010, 0.026, 0.090), vec3(0.14, 0.34, 0.78), day);
    vec3 horizon = mix(vec3(0.035, 0.075, 0.180), vec3(0.58, 0.72, 0.90), day);
    zenith  = mix(zenith,  vec3(0.09, 0.11, 0.30), dusk * 0.55);
    horizon = mix(horizon, vec3(0.50, 0.26, 0.30), dusk * 0.70);

    float up = clamp(dir.y, 0.0, 1.0);
    vec3 col = mix(zenith, horizon, pow(1.0 - up, 2.2));

    // Warm band on the sun's side of the horizon at dawn and dusk.
    float facing = 0.5;
    if (length(uSunDir.xz) > 1e-4 && length(dir.xz) > 1e-4)
        facing = 0.5 + 0.5 * dot(normalize(dir.xz), normalize(uSunDir.xz));
    col += vec3(1.25, 0.42, 0.10) * dusk * pow(facing, 4.0) * pow(1.0 - up, 3.0) * 1.2;

    // Mie-like glow around the sun and a soft halo around the moon.
    float cs = max(dot(dir, uSunDir), 0.0);
    float sunUp = smoothstep(-0.10, 0.02, uSunDir.y);
    col += sunTint() * (pow(cs, 10.0) * 0.35 + pow(cs, 120.0) * 0.9) * sunUp * (0.4 + 0.6 * day + dusk);
    float cm = max(dot(dir, uMoonDir), 0.0);
    col += vec3(0.35, 0.45, 0.70) * pow(cm, 40.0) * 0.25 * night * smoothstep(-0.05, 0.10, uMoonDir.y);

    // Below the horizon: hazy, darker continuation (mostly hidden by terrain).
    if (dir.y < 0.0) col *= mix(1.0, 0.6, smoothstep(0.0, -0.3, dir.y));
    return col;
}

// ---------------------------------------------------------------- display
// Eyes adapt: full exposure only once the sun is well below the horizon.
float sceneExposure() { return mix(2.6, 1.0, smoothstep(-0.20, 0.08, uSunDir.y)); }
vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
// Exposure + filmic tonemap + gamma, with a 1-LSB dither against gradient banding.
vec3 toDisplay(vec3 linearColor, vec2 fragCoord) {
    vec3 c = pow(aces(linearColor * sceneExposure()), vec3(1.0 / 2.2));
    return c + (hash12(fragCoord) - 0.5) / 255.0;
}
