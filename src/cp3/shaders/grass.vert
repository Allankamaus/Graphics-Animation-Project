// Instanced grass blades. The field is a tile that wraps around the runner along x,
// so the meadow is endless; blades shrink to nothing near the tile edges.
layout (location = 0) in vec2 aBlade; // x: t along blade 0..1, y: side -1..1
layout (location = 1) in vec4 aShape; // x, z, height, width
layout (location = 2) in vec4 aStyle; // facing angle, lean, seed 0..1, kind (0 grass, 1 seed-grass, 2 flower)

uniform mat4 uView;
uniform mat4 uProjection;
uniform float uGrassCenterX;
uniform float uGrassCenterZ;
uniform vec2 uGrassTile;

out vec3 vWorld;
out vec3 vNormal;
out vec3 vAlbedo;
out float vT;

void main() {
    float t = aBlade.x;
    float side = aBlade.y;
    int kind = int(aStyle.w + 0.5);
    float seed = aStyle.z;

    float halfX = uGrassTile.x * 0.5;
    float rel = mod(aShape.x - uGrassCenterX + halfX, uGrassTile.x) - halfX;
    vec2 root = vec2(uGrassCenterX + rel, aShape.y);
    float edgeX = 1.0 - smoothstep(0.78, 1.0, abs(rel) / halfX);
    float edgeZ = 1.0 - smoothstep(0.78, 1.0, abs(aShape.y - uGrassCenterZ) / (uGrassTile.y * 0.5));
    float trail = mix(0.25, 1.0, smoothstep(0.20, 0.80, abs(root.y)));
    float h = aShape.z * edgeX * edgeZ * trail;

    // Bend = the blade's own lean plus wind (slow gusts travelling across the field + flutter).
    vec2 facing = vec2(cos(aStyle.x), sin(aStyle.x));
    vec2 across = vec2(-facing.y, facing.x);
    vec2 wind = normalize(vec2(1.0, -0.35));
    float gust = 0.5 + 0.5 * sin(uTime * 0.6 + dot(root, wind) * 0.12);
    gust *= gust;
    float flutter = sin(uTime * 2.6 + dot(root, wind) * 0.8 + seed * 6.2831);
    vec2 bend = facing * aStyle.y + wind * (0.12 + 0.55 * gust) * (0.7 + 0.3 * flutter);
    float bend2 = dot(bend, bend);
    vec2 offset = bend * t * t * h;
    float droop = 0.35 * bend2 * t * t * h; // keeps blade length roughly constant

    float width;
    if (kind == 2) width = t < 0.85 ? 0.004 : (t < 0.95 ? aShape.w * 2.0 : 0.0);
    else width = aShape.w * pow(1.0 - t, 0.65);

    vec3 acrossW = vec3(across.x, 0.0, across.y);
    vec3 base = vec3(root.x, terrainHeight(root), root.y);
    vWorld = base + vec3(offset.x, t * h - droop, offset.y) + acrossW * (side * width * 0.5);

    vec2 slope = bend * 2.0 * t * h;
    vec3 tangent = normalize(vec3(slope.x, h - 0.7 * bend2 * t * h + 1e-4, slope.y));
    vNormal = normalize(cross(acrossW, tangent) + acrossW * side * 0.35);

    vec3 rootColor = mix(vec3(0.018, 0.040, 0.010), vec3(0.030, 0.035, 0.012), seed);
    vec3 tipColor = mix(vec3(0.10, 0.20, 0.035), vec3(0.20, 0.19, 0.07), smoothstep(0.55, 1.0, seed));
    if (kind == 1) tipColor = vec3(0.30, 0.26, 0.14);
    vAlbedo = mix(rootColor, tipColor, t);
    if (kind == 2 && t > 0.85) {
        float pick = fract(seed * 7.0);
        vAlbedo = pick < 0.45 ? vec3(0.80, 0.80, 0.75) : (pick < 0.75 ? vec3(0.85, 0.65, 0.10) : vec3(0.35, 0.20, 0.60));
    }
    vT = t;
    gl_Position = uProjection * uView * vec4(vWorld, 1.0);
}
