// Sky backdrop: gradient + sun + moon + rotating star field and Milky Way.
in vec2 vNdc;
out vec4 FragColor;

uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamForward;
uniform float uTanHalfFov;
uniform float uAspect;

vec3 rotateAbout(vec3 v, vec3 axis, float a) {
    float c = cos(a), s = sin(a);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

// One star per (sparse) cell of a 3D grid intersected with the view sphere.
float starLayer(vec3 d, float scale, float density, float seed) {
    vec3 p = d * scale;
    vec3 cell = floor(p);
    float h = hash13(cell + seed);
    if (h > density) return 0.0;
    vec3 pos = vec3(hash13(cell + seed + 1.7), hash13(cell + seed + 4.1), hash13(cell + seed + 9.3)) * 0.5 + 0.25;
    vec3 f = p - cell - pos;
    float magnitude = pow(hash13(cell + seed + 13.7), 3.0);
    float twinkle = 0.75 + 0.25 * sin(uTime * (1.5 + 3.0 * hash13(cell + seed + 5.5)) + h * 40.0);
    return exp(-dot(f, f) / 0.03) * (0.3 + 2.5 * magnitude) * twinkle;
}

void main() {
    vec3 dir = normalize(uCamForward + vNdc.x * uTanHalfFov * uAspect * uCamRight + vNdc.y * uTanHalfFov * uCamUp);
    vec3 col = skyGradient(dir);
    float night = nightAmount();

    // Stars fade in through twilight and are extinguished near the horizon.
    float starVis = (1.0 - smoothstep(-0.22, -0.02, uSunDir.y)) * smoothstep(-0.02, 0.20, dir.y);
    vec3 sd = rotateAbout(dir, normalize(vec3(0.0, 0.82, -0.57)), uStarAngle);
    if (starVis > 0.0) {
        vec3 galaxyNormal = normalize(vec3(0.35, 0.55, 0.76));
        float band = exp(-pow(dot(sd, galaxyNormal) / 0.24, 2.0));
        float clouds = fbm3(sd * 4.5);
        float dust = fbm3(sd * 9.0 + 3.1);
        vec3 milky = vec3(0.30, 0.40, 0.75) * band * smoothstep(0.30, 0.80, clouds)
                   * (1.0 - 0.7 * smoothstep(0.45, 0.70, dust) * band);
        float stars = starLayer(sd, 170.0, 0.05 * (1.0 + 2.5 * band), 0.0)
                    + 1.6 * starLayer(sd, 70.0, 0.035, 31.0);
        vec3 starColor = mix(vec3(0.75, 0.85, 1.0), vec3(1.0, 0.88, 0.75), hash13(floor(sd * 170.0)));
        col += (milky * 0.06 + starColor * stars * 0.35) * starVis;

        // One bright star with diffraction spikes.
        vec3 star = normalize(vec3(0.20, 0.62, -0.76));
        float fd = dot(sd, star);
        if (fd > 0.99) {
            vec3 t1 = normalize(cross(star, vec3(0.0, 1.0, 0.0)));
            vec3 t2 = cross(t1, star);
            vec3 v = sd - star * fd;
            float a = dot(v, t1), b = dot(v, t2), r2 = a * a + b * b;
            float core = exp(-r2 / (0.0016 * 0.0016)) * 3.0 + 0.4 * 0.000015 / (r2 + 0.000015);
            float spikes = exp(-abs(a) / 0.0009) * exp(-abs(b) / 0.018) + exp(-abs(b) / 0.0009) * exp(-abs(a) / 0.018);
            col += vec3(0.85, 0.92, 1.0) * (core + spikes * 0.8) * (0.85 + 0.15 * sin(uTime * 3.1)) * starVis;
        }
    }

    // Moon: occludes stars at night, a faint pale disc by day.
    float moonVis = smoothstep(-0.03, 0.03, uMoonDir.y) * smoothstep(-0.01, 0.01, dir.y);
    float moonAngle = acos(clamp(dot(dir, uMoonDir), -1.0, 1.0));
    float moonDisk = (1.0 - smoothstep(0.016 * 0.90, 0.016, moonAngle)) * moonVis;
    if (moonDisk > 0.0) {
        float maria = fbm3(dir * 260.0);
        vec3 moon = vec3(0.95, 0.96, 1.0) * (0.65 + 0.45 * smoothstep(0.35, 0.70, maria)) * mix(0.25, 2.4, night);
        col = col * (1.0 - moonDisk * night) + moon * moonDisk;
    }

    // Sun disc.
    float sunAngle = acos(clamp(dot(dir, uSunDir), -1.0, 1.0));
    col += sunTint() * 30.0 * (1.0 - smoothstep(0.0085, 0.0100, sunAngle)) * smoothstep(-0.005, 0.01, dir.y);

    FragColor = vec4(toDisplay(col, gl_FragCoord.xy), 1.0);
}
