// Terrain grid follows the runner; height and normal come from the shared height field.
layout (location = 0) in vec2 aGrid;

uniform mat4 uView;
uniform mat4 uProjection;
uniform vec2 uTerrainOffset; // snapped to whole grid cells so vertices never swim

out vec3 vWorld;
out vec3 vNormal;

void main() {
    vec2 p = aGrid + uTerrainOffset;
    vWorld = vec3(p.x, terrainHeight(p), p.y);
    vNormal = terrainNormal(p);
    gl_Position = uProjection * uView * vec4(vWorld, 1.0);
}
