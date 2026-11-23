// CP3/CP4: a runner sprinting up an endless grassy ridge under a full day/night sky.
//
// Rendering overview (all GLSL in src/cp3/shaders):
//   1. Shadow pass  - the runner is rendered into two depth maps (sun and moon).
//   2. Sky          - full-screen analytic sky: gradient, sun, moon, stars, Milky Way.
//   3. Terrain      - a grid that follows the runner; height computed on the GPU.
//   4. Runner       - procedural rig built from ellipsoids and cones each frame.
//   5. Grass        - ~110k instanced, wind-animated blades in a tile that wraps around the runner.
// Every lit surface uses the same sky model for ambient light and fog, so the
// whole frame stays coherent through sunrise, day, sunset and night.

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace {
constexpr float PI = 3.14159265358979323846f;
constexpr int WINDOW_W = 1280;
constexpr int WINDOW_H = 720;
constexpr float FIELD_OF_VIEW = PI / 3.3f;
constexpr int SHADOW_MAP_SIZE = 2048;

// ================================================================ math
struct Vec3 {
    float x{}, y{}, z{};
    Vec3 operator+(Vec3 b) const { return {x + b.x, y + b.y, z + b.z}; }
    Vec3 operator-(Vec3 b) const { return {x - b.x, y - b.y, z - b.z}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
};
float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
float length(Vec3 v) { return std::sqrt(dot(v, v)); }
Vec3 normalize(Vec3 v) { return v * (1.0f / std::sqrt(std::max(dot(v, v), 1e-12f))); }
float approach(float rate, float dt) { return 1.0f - std::exp(-rate * dt); } // frame-rate independent smoothing

struct Mat4 { std::array<float, 16> m{}; }; // column-major, as OpenGL expects
Mat4 identity() { Mat4 r{}; r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1; return r; }
Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row)
            for (int k = 0; k < 4; ++k) r.m[c * 4 + row] += a.m[k * 4 + row] * b.m[c * 4 + k];
    return r;
}
Mat4 perspective(float fov, float aspect, float nearZ, float farZ) {
    Mat4 r{};
    const float f = 1.0f / std::tan(fov * 0.5f);
    r.m[0] = f / aspect; r.m[5] = f; r.m[10] = (farZ + nearZ) / (nearZ - farZ);
    r.m[11] = -1; r.m[14] = (2 * farZ * nearZ) / (nearZ - farZ);
    return r;
}
Mat4 ortho(float l, float r, float b, float t, float n, float f) {
    Mat4 o = identity();
    o.m[0] = 2 / (r - l); o.m[5] = 2 / (t - b); o.m[10] = -2 / (f - n);
    o.m[12] = -(r + l) / (r - l); o.m[13] = -(t + b) / (t - b); o.m[14] = -(f + n) / (f - n);
    return o;
}
struct ViewBasis { Vec3 right, up, forward; };
ViewBasis viewBasis(Vec3 eye, Vec3 center, Vec3 up) {
    const Vec3 f = normalize(center - eye), s = normalize(cross(f, up));
    return {s, cross(s, f), f};
}
Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
    const ViewBasis b = viewBasis(eye, center, up);
    Mat4 r = identity();
    r.m[0] = b.right.x; r.m[4] = b.right.y; r.m[8] = b.right.z;
    r.m[1] = b.up.x; r.m[5] = b.up.y; r.m[9] = b.up.z;
    r.m[2] = -b.forward.x; r.m[6] = -b.forward.y; r.m[10] = -b.forward.z;
    r.m[12] = -dot(b.right, eye); r.m[13] = -dot(b.up, eye); r.m[14] = dot(b.forward, eye);
    return r;
}

// ================================================================ terrain (CPU mirror)
// Must match terrainHeight() in shaders/common.glsl: the GPU draws the ground,
// the CPU uses the same function to plant feet and keep the camera above it.
float terrainHeight(float x, float z) {
    const float ridge = -0.34f * (std::sqrt(z * z + 1.5f) - std::sqrt(1.5f));
    const float slope = 0.20f * x;
    const float rolls = 0.35f * std::sin(0.11f * x + 0.6f) * std::cos(0.07f * z)
                      + 0.12f * std::sin(0.37f * x + 0.21f * z + 1.3f)
                      + 0.05f * std::sin(0.83f * x - 0.57f * z + 2.1f);
    return slope + ridge + rolls;
}

// ================================================================ shaders
std::optional<std::string> readText(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

class Program {
public:
    GLuint id = 0;
    GLint location(const std::string& name) {
        const auto it = locations_.find(name);
        if (it != locations_.end()) return it->second;
        const GLint loc = glGetUniformLocation(id, name.c_str());
        locations_.emplace(name, loc);
        return loc; // -1 for uniforms a program does not use; glUniform ignores -1
    }
    void set(const std::string& n, float v) { glUniform1f(location(n), v); }
    void set(const std::string& n, int v) { glUniform1i(location(n), v); }
    void set(const std::string& n, Vec3 v) { glUniform3f(location(n), v.x, v.y, v.z); }
    void set(const std::string& n, float x, float y) { glUniform2f(location(n), x, y); }
    void set(const std::string& n, const Mat4& m) { glUniformMatrix4fv(location(n), 1, GL_FALSE, m.m.data()); }

private:
    std::unordered_map<std::string, GLint> locations_;
};

// Concatenates shader fragments (e.g. common.glsl + lighting.glsl + body) under one #version.
std::optional<std::string> assembleSource(const std::string& dir, const std::vector<std::string>& files) {
    std::string source = "#version 330 core\n";
    for (const std::string& file : files) {
        const auto text = readText(dir + "/" + file);
        if (!text) {
            std::cerr << "Missing shader file: " << dir << "/" << file << '\n';
            return std::nullopt;
        }
        source += "// ---- " + file + " ----\n" + *text + "\n";
    }
    return source;
}

GLuint compileShader(GLenum type, const std::string& source, const std::string& label) {
    const GLuint shader = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        std::array<char, 4096> log{};
        glGetShaderInfoLog(shader, static_cast<GLsizei>(log.size()), nullptr, log.data());
        std::cerr << "Shader compile error in " << label << ":\n" << log.data() << '\n';
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

std::optional<Program> loadProgram(const std::string& dir, const std::vector<std::string>& vertexFiles,
                                   const std::vector<std::string>& fragmentFiles) {
    const std::string label = vertexFiles.back() + " + " + fragmentFiles.back();
    const auto vs = assembleSource(dir, vertexFiles), fs = assembleSource(dir, fragmentFiles);
    if (!vs || !fs) return std::nullopt;
    const GLuint v = compileShader(GL_VERTEX_SHADER, *vs, vertexFiles.back());
    const GLuint f = compileShader(GL_FRAGMENT_SHADER, *fs, fragmentFiles.back());
    if (!v || !f) {
        if (v) glDeleteShader(v);
        if (f) glDeleteShader(f);
        return std::nullopt;
    }
    Program program;
    program.id = glCreateProgram();
    glAttachShader(program.id, v);
    glAttachShader(program.id, f);
    glLinkProgram(program.id);
    glDeleteShader(v);
    glDeleteShader(f);
    GLint ok = 0;
    glGetProgramiv(program.id, GL_LINK_STATUS, &ok);
    if (!ok) {
        std::array<char, 4096> log{};
        glGetProgramInfoLog(program.id, static_cast<GLsizei>(log.size()), nullptr, log.data());
        std::cerr << "Program link error in " << label << ":\n" << log.data() << '\n';
        glDeleteProgram(program.id);
        return std::nullopt;
    }
    return program;
}

// ================================================================ runner body mesh
struct Vertex { float px, py, pz, nx, ny, nz, r, g, b; };

// The runner is rebuilt every frame as world-space triangles made of smooth
// primitives: tapered capsules for limbs (so joints blend instead of looking
// like strung beads), ellipsoids, open tubes for cloth and cones for hair.
class BodyBuilder {
public:
    struct Ring { Vec3 centre; float rx, rz, axial; }; // axial: normal's component along the axis

    void begin(Vec3 lateral) { vertices_.clear(); lateral_ = lateral; }
    const std::vector<Vertex>& vertices() const { return vertices_; }

    // Sphere-swept segment: radius ra around a, rb around b, tapering in between.
    void capsule(Vec3 a, Vec3 b, float ra, float rb, Vec3 color) {
        const Frame f = frameAlong(b - a);
        std::vector<Ring> rings;
        for (int k = 0; k <= CAP_STACKS; ++k) {
            const float phi = -PI / 2 + (PI / 2) * static_cast<float>(k) / CAP_STACKS;
            rings.push_back({a + f.axis * (ra * std::sin(phi)), ra * std::cos(phi), ra * std::cos(phi), std::sin(phi)});
        }
        for (int k = 0; k <= CAP_STACKS; ++k) {
            const float phi = (PI / 2) * static_cast<float>(k) / CAP_STACKS;
            rings.push_back({b + f.axis * (rb * std::sin(phi)), rb * std::cos(phi), rb * std::cos(phi), std::sin(phi)});
        }
        stitch(rings, f, color);
    }

    // Ellipsoid with explicit (mutually perpendicular) semi-axes.
    void ellipsoid(Vec3 centre, Vec3 ax, Vec3 ay, Vec3 az, Vec3 color) {
        auto point = [&](float lat, float lon) {
            const float sx = std::cos(lat) * std::cos(lon), sy = std::sin(lat), sz = std::cos(lat) * std::sin(lon);
            const Vec3 p = centre + ax * sx + ay * sy + az * sz;
            // Normal of a linearly transformed sphere = inverse-transpose applied to the sphere normal.
            const Vec3 n = normalize(ax * (sx / dot(ax, ax)) + ay * (sy / dot(ay, ay)) + az * (sz / dot(az, az)));
            return Vertex{p.x, p.y, p.z, n.x, n.y, n.z, color.x, color.y, color.z};
        };
        for (int j = 0; j < ELLIPSOID_STACKS; ++j) {
            const float a0 = -PI / 2 + PI * j / ELLIPSOID_STACKS, a1 = -PI / 2 + PI * (j + 1) / ELLIPSOID_STACKS;
            for (int i = 0; i < SLICES; ++i) {
                const float b0 = 2 * PI * i / SLICES, b1 = 2 * PI * (i + 1) / SLICES;
                const Vertex p00 = point(a0, b0), p10 = point(a0, b1), p01 = point(a1, b0), p11 = point(a1, b1);
                vertices_.insert(vertices_.end(), {p00, p01, p10, p10, p01, p11});
            }
        }
    }

    void cone(Vec3 base, Vec3 tip, float radius, Vec3 color) {
        const Frame f = frameAlong(tip - base);
        const float len = length(tip - base), axial = radius / std::sqrt(radius * radius + len * len);
        stitch({{base, radius, radius, axial}, {tip, 0.0f, 0.0f, axial}}, f, color);
    }

    // Open tube through elliptical rings sharing one cross-section frame (cloth: hems, collars).
    void tube(const std::vector<Ring>& rings, Vec3 xAxis, Vec3 zAxis, Vec3 color) {
        stitch(rings, Frame{normalize(xAxis), normalize(cross(zAxis, xAxis)), normalize(zAxis)}, color);
    }

private:
    static constexpr int SLICES = 16, CAP_STACKS = 4, ELLIPSOID_STACKS = 12;
    struct Frame { Vec3 x, axis, z; };

    Frame frameAlong(Vec3 d) const {
        const Vec3 axis = length(d) > 1e-6f ? normalize(d) : Vec3{0, 1, 0};
        Vec3 x = cross(axis, lateral_);
        if (length(x) < 1e-4f) x = cross(axis, Vec3{1, 0, 0});
        x = normalize(x);
        return {x, axis, cross(x, axis)};
    }

    void stitch(const std::vector<Ring>& rings, const Frame& f, Vec3 color) {
        auto at = [&](const Ring& r, int j) {
            const float t = 2 * PI * static_cast<float>(j) / SLICES, c = std::cos(t), s = std::sin(t);
            const Vec3 p = r.centre + f.x * (r.rx * c) + f.z * (r.rz * s);
            // Outward normal of an ellipse is proportional to (cos/rx, sin/rz).
            const Vec3 radial = normalize(f.x * (c * std::max(r.rz, 1e-4f)) + f.z * (s * std::max(r.rx, 1e-4f)));
            const Vec3 n = normalize(radial * std::sqrt(std::max(0.0f, 1.0f - r.axial * r.axial)) + f.axis * r.axial);
            return Vertex{p.x, p.y, p.z, n.x, n.y, n.z, color.x, color.y, color.z};
        };
        for (std::size_t i = 0; i + 1 < rings.size(); ++i)
            for (int j = 0; j < SLICES; ++j) {
                const Vertex p00 = at(rings[i], j), p01 = at(rings[i + 1], j);
                const Vertex p10 = at(rings[i], j + 1), p11 = at(rings[i + 1], j + 1);
                vertices_.insert(vertices_.end(), {p00, p01, p10, p10, p01, p11});
            }
    }

    std::vector<Vertex> vertices_;
    Vec3 lateral_{0, 0, 1};
};

// GPU side: one streamed vertex buffer, re-filled each frame.
struct BodyMesh { GLuint vao{}, vbo{}; GLsizei count{}; };

BodyMesh makeBodyMesh() {
    BodyMesh mesh;
    glGenVertexArrays(1, &mesh.vao);
    glGenBuffers(1, &mesh.vbo);
    glBindVertexArray(mesh.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    for (GLuint attribute = 0; attribute < 3; ++attribute) {
        glEnableVertexAttribArray(attribute);
        glVertexAttribPointer(attribute, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                              reinterpret_cast<void*>(static_cast<std::uintptr_t>(attribute * 3 * sizeof(float))));
    }
    glBindVertexArray(0);
    return mesh;
}
void uploadBody(BodyMesh& mesh, const std::vector<Vertex>& vertices) {
    mesh.count = static_cast<GLsizei>(vertices.size());
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)), vertices.data(), GL_STREAM_DRAW);
}
void destroy(BodyMesh& mesh) { glDeleteBuffers(1, &mesh.vbo); glDeleteVertexArrays(1, &mesh.vao); }

// ---------------------------------------------------------------- terrain grid
struct Terrain { GLuint vao{}, vbo{}, ebo{}; GLsizei indexCount{}; };
constexpr float TERRAIN_STEP = 0.5f;

// Sample positions along one axis: uniform spacing out to fineHalf, then growing
// geometrically out to farHalf so the far edge sits deep inside the fog.
std::vector<float> axisSamples(float fineHalf, float farHalf, float growth) {
    std::vector<float> positive{0.0f};
    float p = 0.0f, step = TERRAIN_STEP;
    while (p < farHalf) {
        if (p >= fineHalf) step *= growth;
        p += step;
        positive.push_back(p);
    }
    std::vector<float> all;
    for (auto it = positive.rbegin(); it != positive.rend(); ++it) if (*it > 0) all.push_back(-*it);
    all.insert(all.end(), positive.begin(), positive.end());
    return all;
}

Terrain makeTerrain() {
    const std::vector<float> xs = axisSamples(90.0f, 90.0f, 1.0f);
    const std::vector<float> zs = axisSamples(18.0f, 130.0f, 1.07f);
    std::vector<float> positions;
    positions.reserve(xs.size() * zs.size() * 2);
    for (float z : zs)
        for (float x : xs) { positions.push_back(x); positions.push_back(z); }
    std::vector<GLuint> indices;
    const auto columns = static_cast<GLuint>(xs.size());
    for (GLuint row = 0; row + 1 < zs.size(); ++row)
        for (GLuint col = 0; col + 1 < columns; ++col) {
            const GLuint i00 = row * columns + col, i10 = i00 + 1, i01 = i00 + columns, i11 = i01 + 1;
            for (GLuint i : {i00, i01, i10, i10, i01, i11}) indices.push_back(i);
        }
    Terrain t;
    t.indexCount = static_cast<GLsizei>(indices.size());
    glGenVertexArrays(1, &t.vao);
    glGenBuffers(1, &t.vbo);
    glGenBuffers(1, &t.ebo);
    glBindVertexArray(t.vao);
    glBindBuffer(GL_ARRAY_BUFFER, t.vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(positions.size() * sizeof(float)), positions.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, t.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(indices.size() * sizeof(GLuint)), indices.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);
    return t;
}
void destroy(Terrain& t) { glDeleteBuffers(1, &t.ebo); glDeleteBuffers(1, &t.vbo); glDeleteVertexArrays(1, &t.vao); }

// ---------------------------------------------------------------- grass field
enum class BladeKind { Grass = 0, SeedGrass = 1, Flower = 2 };
struct GrassInstance { float x, z, height, width, facing, lean, seed, kind; };
struct GrassField { GLuint vao{}, bladeVbo{}, instanceVbo{}; GLsizei instanceCount{}, bladeVertexCount{}; };
constexpr float GRASS_TILE_X = 64.0f, GRASS_Z_MIN = -12.0f, GRASS_Z_MAX = 16.0f;

std::vector<GrassInstance> scatterGrass() {
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    auto U = [&] { return unit(rng); };
    auto kindOf = [](BladeKind k) { return static_cast<float>(static_cast<int>(k)); };
    const float depth = GRASS_Z_MAX - GRASS_Z_MIN, area = GRASS_TILE_X * depth;
    std::vector<GrassInstance> blades;
    blades.reserve(static_cast<std::size_t>(area * 110.0f));

    // Tufts: blades splay outward from a shared root, like real bunch grass.
    const int tufts = static_cast<int>(area * 2.8f);
    for (int c = 0; c < tufts; ++c) {
        const float cx = -GRASS_TILE_X * 0.5f + U() * GRASS_TILE_X, cz = GRASS_Z_MIN + U() * depth;
        const bool seedGrass = U() < 0.05f;
        const float tuftHeight = seedGrass ? 0.75f + 0.45f * U() : 0.22f + 0.40f * U() * U() + 0.08f * U();
        const float radius = 0.12f + 0.30f * U(), tuftLean = 0.15f + 0.35f * U();
        const int count = 14 + static_cast<int>(28.0f * U());
        for (int k = 0; k < count; ++k) {
            const float r = radius * std::sqrt(U()), a = U() * 2 * PI, outer = r / radius;
            blades.push_back({cx + r * std::cos(a), cz + r * std::sin(a),
                              tuftHeight * (0.55f + 0.6f * U()) * (1.0f - 0.35f * outer),
                              seedGrass ? 0.012f + 0.008f * U() : 0.024f + 0.020f * U(),
                              a + (U() - 0.5f) * 0.8f, tuftLean * (0.5f + U()) + 0.25f * outer, U(),
                              kindOf(seedGrass ? BladeKind::SeedGrass : BladeKind::Grass)});
        }
    }
    // Loose blades filling the gaps between tufts.
    const int loose = static_cast<int>(area * 20.0f);
    for (int i = 0; i < loose; ++i)
        blades.push_back({-GRASS_TILE_X * 0.5f + U() * GRASS_TILE_X, GRASS_Z_MIN + U() * depth, 0.12f + 0.30f * U(),
                          0.020f + 0.016f * U(), U() * 2 * PI, 0.10f + 0.40f * U(), U(), kindOf(BladeKind::Grass)});
    // Sparse wildflowers.
    const int flowers = static_cast<int>(area * 0.5f);
    for (int i = 0; i < flowers; ++i)
        blades.push_back({-GRASS_TILE_X * 0.5f + U() * GRASS_TILE_X, GRASS_Z_MIN + U() * depth, 0.22f + 0.25f * U(),
                          0.025f, U() * 2 * PI, 0.05f + 0.15f * U(), U(), kindOf(BladeKind::Flower)});
    return blades;
}

GrassField makeGrassField() {
    // One blade = a triangle strip of (t, side) pairs closing in a single tip vertex.
    std::vector<float> blade;
    for (float t : {0.0f, 0.22f, 0.44f, 0.64f, 0.80f, 0.90f}) { blade.insert(blade.end(), {t, -1.0f, t, 1.0f}); }
    blade.insert(blade.end(), {1.0f, 0.0f});
    const std::vector<GrassInstance> instances = scatterGrass();

    GrassField g;
    g.bladeVertexCount = static_cast<GLsizei>(blade.size() / 2);
    g.instanceCount = static_cast<GLsizei>(instances.size());
    glGenVertexArrays(1, &g.vao);
    glGenBuffers(1, &g.bladeVbo);
    glGenBuffers(1, &g.instanceVbo);
    glBindVertexArray(g.vao);
    glBindBuffer(GL_ARRAY_BUFFER, g.bladeVbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(blade.size() * sizeof(float)), blade.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, g.instanceVbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(instances.size() * sizeof(GrassInstance)), instances.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(GrassInstance), nullptr);
    glVertexAttribDivisor(1, 1);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(GrassInstance), reinterpret_cast<void*>(4 * sizeof(float)));
    glVertexAttribDivisor(2, 1);
    glBindVertexArray(0);
    return g;
}
void destroy(GrassField& g) { glDeleteBuffers(1, &g.instanceVbo); glDeleteBuffers(1, &g.bladeVbo); glDeleteVertexArrays(1, &g.vao); }

// ---------------------------------------------------------------- shadow maps
struct ShadowMap { GLuint fbo{}, texture{}; };

std::optional<ShadowMap> makeShadowMap() {
    ShadowMap s;
    glGenTextures(1, &s.texture);
    glBindTexture(GL_TEXTURE_2D, s.texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glGenFramebuffers(1, &s.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, s.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, s.texture, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    const bool complete = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (!complete) {
        std::cerr << "Shadow framebuffer is incomplete on this driver.\n";
        glDeleteFramebuffers(1, &s.fbo);
        glDeleteTextures(1, &s.texture);
        return std::nullopt;
    }
    return s;
}
void destroy(ShadowMap& s) { glDeleteFramebuffers(1, &s.fbo); glDeleteTextures(1, &s.texture); }

Mat4 lightMatrix(Vec3 towardLight, Vec3 focus) {
    const Vec3 dir = normalize(towardLight);
    const Vec3 up = std::abs(dir.y) > 0.95f ? Vec3{0, 0, 1} : Vec3{0, 1, 0};
    return multiply(ortho(-2.4f, 2.4f, -2.4f, 2.4f, 1.0f, 45.0f), lookAt(focus + dir * 20.0f, focus, up));
}

// ================================================================ day / night
struct DayCycle {
    float hours = 21.0f;          // 0..24
    float secondsPerDay = 180.0f; // one full day in three minutes
    bool paused = false;

    void advance(float dt, float scrubDirection) {
        if (!paused) hours += dt * 24.0f / secondsPerDay;
        hours += scrubDirection * dt * 3.0f; // [ and ] scrub three hours per second
        hours = std::fmod(std::fmod(hours, 24.0f) + 24.0f, 24.0f);
    }
    // Rises behind-left of the default view at 06:00 and sets ahead-right at 18:00.
    Vec3 sunDir() const {
        const float a = (hours - 6.0f) / 24.0f * 2.0f * PI;
        return normalize({-std::cos(a), std::sin(a), -0.45f});
    }
    Vec3 moonDir() const {
        const Vec3 s = sunDir();
        return normalize({-s.x, -s.y, -0.55f});
    }
    float starAngle() const { return hours / 24.0f * 2.0f * PI; }
    const char* phaseName() const {
        const float e = sunDir().y;
        if (e > 0.25f) return "Day";
        if (e > 0.0f) return hours < 12.0f ? "Morning golden hour" : "Golden hour";
        if (e > -0.18f) return hours < 12.0f ? "Dawn twilight" : "Dusk twilight";
        return "Night";
    }
};

// ================================================================ runner
constexpr float RUN_SPEED = 3.6f;        // m/s
constexpr float STANCE_FRACTION = 0.36f; // share of each leg's cycle spent on the ground
constexpr float THIGH = 0.47f, SHIN = 0.47f;
constexpr float ANKLE_HEIGHT = 0.09f;    // ankle joint above the ground when the foot is planted

// Metres covered per full gait cycle (two steps); shorter strides when slower.
float strideLength(float motion) { return 2.6f * (0.35f + 0.65f * motion); }

struct RunnerState {
    float x = 0.0f;        // position along the ridge
    float velocity = 0.0f; // signed, m/s along +x
    float yaw = 0.0f;      // 0 faces +x, PI faces -x
    float phase = 0.0f;    // gait cycle, radians
    float motion = 0.0f;   // 0 standing .. 1 full sprint (blends the pose)
};

void updateRunner(RunnerState& r, float dt, int input, bool autoRun) {
    const float desired = input != 0 ? static_cast<float>(input) * RUN_SPEED : (autoRun ? RUN_SPEED : 0.0f);
    r.velocity += (desired - r.velocity) * approach(4.0f, dt);
    r.x += r.velocity * dt;
    if (std::abs(r.velocity) > 0.1f) r.yaw += ((r.velocity > 0 ? 0.0f : PI) - r.yaw) * approach(8.0f, dt);
    r.motion += (std::clamp(std::abs(r.velocity) / RUN_SPEED, 0.0f, 1.0f) - r.motion) * approach(6.0f, dt);
    // Phase advances with distance travelled, so a planted foot moves back relative
    // to the hips exactly as fast as the ground does: no foot sliding at any speed.
    r.phase = std::fmod(r.phase + std::abs(r.velocity) * dt * 2.0f * PI / strideLength(r.motion), 2.0f * PI);
}

namespace outfit {
constexpr Vec3 JACKET{0.018f, 0.020f, 0.028f};
constexpr Vec3 TROUSERS{0.028f, 0.028f, 0.034f};
constexpr Vec3 SKIN{0.62f, 0.42f, 0.32f};
constexpr Vec3 HAIR{0.012f, 0.012f, 0.022f};
constexpr Vec3 SHOE{0.62f, 0.035f, 0.030f};
constexpr Vec3 SOLE{0.75f, 0.74f, 0.70f};
}

float smooth01(float x) { x = std::clamp(x, 0.0f, 1.0f); return x * x * (3.0f - 2.0f * x); }

// Cubic Hermite: C1-continuous curve from p0 to p1 with end tangents m0, m1.
float hermite(float p0, float p1, float m0, float m1, float s) {
    const float s2 = s * s, s3 = s2 * s;
    return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * m1;
}

// Where one foot is, relative to the hips, at a point in its gait cycle.
struct FootPlan { float forward; float lift; float pitch; };

// t in [0,1): stance first (foot planted, sliding back under the body at ground
// speed), then swing (foot carried forward on a Hermite curve whose end speeds
// match the stance, so lift-off and touch-down have no velocity jumps).
FootPlan planFoot(float t, float halfStance) {
    if (t < STANCE_FRACTION) {
        const float s = t / STANCE_FRACTION;
        return {halfStance * (1.0f - 2.0f * s), 0.0f,
                0.15f * (1.0f - smooth01(s / 0.25f)) - 0.5f * smooth01((s - 0.65f) / 0.35f)}; // heel strike -> toe-off roll
    }
    const float s = (t - STANCE_FRACTION) / (1.0f - STANCE_FRACTION);
    const float groundSpeed = -2.0f * halfStance * (1.0f - STANCE_FRACTION) / STANCE_FRACTION; // d(forward)/ds
    const float k = std::sin(PI * std::pow(s, 0.8f)); // peaks early: the heel kicks up behind
    return {hermite(-halfStance, halfStance, groundSpeed, groundSpeed, s), 0.42f * k * k,
            -0.5f + 0.65f * smooth01(s) - 0.7f * std::sin(PI * s)};
}

// Two-bone IK in the sagittal (x, y) plane. Returns the knee; pulls `ankle` in
// if it is out of reach so the leg straightens instead of stretching.
Vec3 solveKnee(Vec3 hip, Vec3& ankle) {
    float dx = ankle.x - hip.x, dy = ankle.y - hip.y;
    float d = std::sqrt(dx * dx + dy * dy);
    if (d < 1e-5f) { dx = 0; dy = -1; d = 1; }
    const float ux = dx / d, uy = dy / d;
    const float reach = std::clamp(d, std::abs(THIGH - SHIN) + 1e-3f, THIGH + SHIN - 1e-3f);
    ankle.x = hip.x + ux * reach;
    ankle.y = hip.y + uy * reach;
    const float cosA = (THIGH * THIGH + reach * reach - SHIN * SHIN) / (2.0f * THIGH * reach);
    const float sinA = std::sqrt(std::max(0.0f, 1.0f - cosA * cosA));
    const float nx = -uy, ny = ux; // perpendicular on the forward side: knees bend forward
    const float along = THIGH / (THIGH + SHIN);
    return {hip.x + THIGH * (ux * cosA + nx * sinA), hip.y + THIGH * (uy * cosA + ny * sinA),
            hip.z + (ankle.z - hip.z) * along};
}

// Poses the runner for this frame and writes its triangles into `body`.
// The pose is authored in a local frame (x forward, y up, z to the runner's
// side) whose origin sits on the ground under the hips. Feet are planned in
// that frame, dropped onto the real terrain, and the legs reach them with IK.
// Returns the chest position for the camera and shadow frustum.
Vec3 poseRunner(const RunnerState& r, float time, BodyBuilder& body) {
    using namespace outfit;
    const float m = r.motion, ph = r.phase;
    // Angle measured from straight down in the sagittal plane; positive swings forward.
    auto swingDir = [](float a) { return Vec3{std::sin(a), -std::cos(a), 0.0f}; };

    const Vec3 fwd{std::cos(r.yaw), 0, std::sin(r.yaw)}, up{0, 1, 0};
    const Vec3 lat = cross(fwd, up);
    const Vec3 origin{r.x, terrainHeight(r.x, 0.0f), 0};
    auto P = [&](Vec3 p) { return origin + fwd * p.x + up * p.y + lat * p.z; }; // point
    auto D = [&](Vec3 v) { return fwd * v.x + up * v.y + lat * v.z; };          // direction

    // Hips ride lowest at each mid-stance and highest in flight (twice per cycle).
    const float bounce = -std::cos(2.0f * (ph - PI * STANCE_FRACTION));
    const float lean = 0.06f + m * (0.50f + 0.03f * bounce);
    const Vec3 spine{std::sin(lean), std::cos(lean), 0};
    const Vec3 spinePerp{std::cos(lean), -std::sin(lean), 0}; // forward, square to the spine
    const Vec3 pelvis{0, 0.93f - 0.07f * m + 0.035f * m * bounce, 0};
    const Vec3 chest = pelvis + spine * (0.40f + 0.01f * std::sin(time * 2.0f) * (1.0f - m));
    const Vec3 neck = pelvis + spine * 0.58f;
    const float headLean = lean * 0.45f; // head stays up, eyes on the ridge ahead
    const Vec3 headUp{std::sin(headLean), std::cos(headLean), 0};
    const Vec3 headFwd{std::cos(headLean), -std::sin(headLean), 0};
    const Vec3 headC = neck + headUp * 0.15f;
    const Vec3 sideZ{0, 0, 1};

    struct Leg { Vec3 hip, knee, ankle, heel, toe, footDir, footDown; };
    struct Arm { Vec3 shoulder, elbow, hand, foreDir; };
    std::array<Leg, 2> legs{};
    std::array<Arm, 2> arms{};
    const float halfStance = 0.5f * STANCE_FRACTION * strideLength(m);
    constexpr float FOOT_BIAS = 0.05f; // feet land slightly ahead of the hips
    for (int i = 0; i < 2; ++i) {
        const float side = i == 0 ? -1.0f : 1.0f;
        const float t = std::fmod(ph + (i == 0 ? 0.0f : PI), 2.0f * PI) / (2.0f * PI);
        const FootPlan plan = planFoot(t, halfStance);
        // Blend toward a relaxed standing stance as the runner slows down.
        const float forward = 0.02f + (plan.forward + FOOT_BIAS - 0.02f) * m;

        Leg& L = legs[i];
        L.hip = pelvis + Vec3{0, -0.04f, side * 0.095f};
        const Vec3 footGround = P({forward, 0, side * 0.11f});
        const float ground = terrainHeight(footGround.x, footGround.z) - origin.y;
        const Vec3 ahead = footGround + fwd * 0.1f, behind = footGround - fwd * 0.1f;
        const float slope = std::atan2(terrainHeight(ahead.x, ahead.z) - terrainHeight(behind.x, behind.z), 0.2f);
        L.ankle = {forward, ground + ANKLE_HEIGHT + plan.lift * m, side * 0.11f};
        L.knee = solveKnee(L.hip, L.ankle);
        const float pitch = slope + plan.pitch * m;
        L.footDir = {std::cos(pitch), std::sin(pitch), 0};
        L.footDown = {std::sin(pitch), -std::cos(pitch), 0};
        L.heel = L.ankle - L.footDir * 0.04f + L.footDown * 0.02f;
        L.toe = L.ankle + L.footDir * 0.17f + L.footDown * 0.03f;

        // Arms swing opposite the same-side leg, driven by the foot plan so they stay in sync.
        const float legSwing = halfStance > 1e-4f ? plan.forward / halfStance : 0.0f;
        Arm& A = arms[i];
        A.shoulder = neck - spine * 0.08f + Vec3{0, 0, side * 0.18f};
        const float upper = -m * 0.75f * legSwing + lean * 0.30f;
        A.elbow = A.shoulder + swingDir(upper) * 0.29f + Vec3{0, 0, side * 0.03f};
        const float fore = upper + m * (1.30f - 0.25f * legSwing) + (1 - m) * 0.20f;
        A.foreDir = swingDir(fore);
        A.hand = A.elbow + A.foreDir * 0.25f + Vec3{0, 0, -side * 0.025f};
    }

    body.begin(lat);
    // Jacket: abdomen, chest, broad shoulders, then a hem that flares and trails with speed.
    body.ellipsoid(P(pelvis), D(spinePerp) * 0.120f, D(spine) * 0.110f, D(sideZ) * 0.165f, TROUSERS);
    body.ellipsoid(P(pelvis + spine * 0.17f), D(spinePerp) * 0.120f, D(spine) * 0.18f, D(sideZ) * 0.155f, JACKET);
    body.ellipsoid(P(pelvis + spine * 0.40f), D(spinePerp) * 0.145f, D(spine) * 0.20f, D(sideZ) * 0.200f, JACKET);
    body.ellipsoid(P(neck - spine * 0.08f), D(spinePerp) * 0.110f, D(spine) * 0.090f, D(sideZ) * 0.230f, JACKET);
    const float flutter = 0.03f * m * std::sin(2.0f * ph + 0.7f);
    body.tube({{P(pelvis + spine * 0.14f), 0.125f, 0.160f, 0.0f},
               {P(pelvis - spine * 0.02f), 0.137f, 0.174f, 0.0f},
               {P(pelvis - spine * 0.14f - spinePerp * (0.05f * m + flutter)), 0.152f, 0.192f, 0.0f}},
              D(spinePerp), D(sideZ), JACKET);
    body.tube({{P(neck - spine * 0.04f), 0.078f, 0.088f, 0.0f}, {P(neck + headUp * 0.07f), 0.068f, 0.076f, 0.0f}},
              D(headFwd), D(sideZ), JACKET); // standing collar
    body.capsule(P(neck - spine * 0.02f), P(headC - headUp * 0.05f), 0.046f, 0.044f, SKIN);

    for (const Leg& L : legs) {
        body.capsule(P(L.hip), P(L.knee), 0.100f, 0.068f, TROUSERS);
        body.capsule(P(L.knee), P(L.ankle), 0.066f, 0.046f, TROUSERS);
        body.capsule(P(L.heel), P(L.toe), 0.052f, 0.046f, SHOE);
        body.ellipsoid(P((L.heel + L.toe) * 0.5f + L.footDown * 0.04f), D(L.footDir) * 0.15f, D(L.footDown) * 0.022f,
                       D(sideZ) * 0.056f, SOLE);
    }
    for (const Arm& A : arms) {
        body.capsule(P(A.shoulder), P(A.elbow), 0.068f, 0.056f, JACKET);
        body.capsule(P(A.elbow), P(A.hand), 0.054f, 0.042f, JACKET);
        const Vec3 fistSide = normalize(cross(A.foreDir, sideZ));
        body.ellipsoid(P(A.hand + A.foreDir * 0.045f), D(A.foreDir) * 0.055f, D(fistSide) * 0.042f, D(sideZ) * 0.036f, SKIN);
    }

    // Head, ears and a full head of spiky, swept-back hair.
    body.ellipsoid(P(headC), D(headFwd) * 0.105f, D(headUp) * 0.122f, D(sideZ) * 0.095f, SKIN);
    for (float side : {-1.0f, 1.0f})
        body.ellipsoid(P(headC + sideZ * (side * 0.092f) - headFwd * 0.01f), D(headFwd) * 0.022f, D(headUp) * 0.034f,
                       D(sideZ) * 0.012f, SKIN);
    const Vec3 hairC = headC + headUp * 0.035f - headFwd * 0.03f;
    body.ellipsoid(P(hairC), D(headFwd) * 0.112f, D(headUp) * 0.118f, D(sideZ) * 0.104f, HAIR);
    body.ellipsoid(P(headC - headFwd * 0.07f - headUp * 0.035f), D(headFwd) * 0.060f, D(headUp) * 0.085f, D(sideZ) * 0.090f, HAIR);
    struct Spike { float f, u, l, len; };
    constexpr std::array<Spike, 20> spikes{{
        {-0.95f, 0.25f, 0.00f, 0.21f}, {-0.85f, 0.45f, 0.40f, 0.18f}, {-0.85f, 0.45f, -0.40f, 0.18f},
        {-0.70f, 0.70f, 0.10f, 0.19f}, {-0.55f, 0.80f, -0.30f, 0.16f}, {-0.20f, 1.00f, 0.25f, 0.15f},
        {0.10f, 1.00f, -0.15f, 0.14f}, {0.45f, 0.85f, 0.30f, 0.12f}, {0.60f, 0.75f, -0.30f, 0.11f},
        {0.80f, 0.45f, 0.10f, 0.10f}, {-0.90f, -0.15f, 0.45f, 0.16f}, {-0.90f, -0.15f, -0.45f, 0.16f},
        {-0.30f, 0.40f, 0.95f, 0.12f}, {-0.30f, 0.40f, -0.95f, 0.12f}, {-0.95f, -0.35f, 0.00f, 0.15f},
        {-0.75f, 0.05f, 0.65f, 0.14f}, {-0.75f, 0.05f, -0.65f, 0.14f}, {0.25f, 0.70f, 0.70f, 0.11f},
        {0.25f, 0.70f, -0.70f, 0.11f}, {-0.40f, 0.95f, 0.00f, 0.17f},
    }};
    for (std::size_t i = 0; i < spikes.size(); ++i) {
        const Spike& s = spikes[i];
        const Vec3 dir = normalize(headFwd * s.f + headUp * s.u + sideZ * s.l);
        const float sway = 0.08f * m * std::sin(time * 9.0f + static_cast<float>(i) * 1.7f);
        const Vec3 base = hairC + dir * 0.07f;
        const Vec3 flow = normalize(dir - headFwd * (0.45f * m) + sideZ * sway);
        body.cone(P(base), P(base + flow * s.len), 0.048f, HAIR);
    }
    return P(chest);
}

// ================================================================ camera & input
struct OrbitCamera {
    static constexpr float DEFAULT_YAW = -0.55f, DEFAULT_PITCH = -0.18f, DEFAULT_DISTANCE = 4.2f;
    float yaw = DEFAULT_YAW;     // behind-left of the runner
    float pitch = DEFAULT_PITCH; // negative: looking up at the runner from low in the grass
    float distance = DEFAULT_DISTANCE;
    Vec3 target{};
    bool hasTarget = false;
    void reset() { yaw = DEFAULT_YAW; pitch = DEFAULT_PITCH; distance = DEFAULT_DISTANCE; }
};

struct App {
    OrbitCamera camera;
    RunnerState runner;
    DayCycle day;
    bool autoRun = true;
    std::array<bool, GLFW_KEY_LAST + 1> keys{};
    bool mouseDown = false, dragging = false;
    double lastX = 0, lastY = 0;
    bool held(int key) const { return keys[static_cast<std::size_t>(key)]; }
};
App g_app;

void keyCallback(GLFWwindow* window, int key, int, int action, int) {
    if (key < 0 || key > GLFW_KEY_LAST) return;
    g_app.keys[static_cast<std::size_t>(key)] = action != GLFW_RELEASE;
    if (action != GLFW_PRESS) return;
    switch (key) {
    case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GLFW_TRUE); break;
    case GLFW_KEY_R: g_app.runner = RunnerState{}; break;
    case GLFW_KEY_C: g_app.camera.reset(); break;
    case GLFW_KEY_SPACE: g_app.autoRun = !g_app.autoRun; break;
    case GLFW_KEY_T: g_app.day.paused = !g_app.day.paused; break;
    case GLFW_KEY_1: g_app.day.hours = 6.2f; break;  // sunrise
    case GLFW_KEY_2: g_app.day.hours = 12.0f; break; // noon
    case GLFW_KEY_3: g_app.day.hours = 17.8f; break; // sunset
    case GLFW_KEY_4: g_app.day.hours = 21.5f; break; // night
    default: break;
    }
}
void mouseButton(GLFWwindow*, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) g_app.mouseDown = action == GLFW_PRESS;
}
void cursorPos(GLFWwindow*, double x, double y) {
    OrbitCamera& c = g_app.camera;
    if (g_app.mouseDown) {
        if (!g_app.dragging) { g_app.lastX = x; g_app.lastY = y; g_app.dragging = true; }
        c.yaw += static_cast<float>(x - g_app.lastX) * 0.006f;
        c.pitch = std::clamp(c.pitch - static_cast<float>(y - g_app.lastY) * 0.006f, -0.45f, 1.30f);
    } else {
        g_app.dragging = false;
    }
    g_app.lastX = x;
    g_app.lastY = y;
}
void scroll(GLFWwindow*, double, double y) {
    g_app.camera.distance = std::clamp(g_app.camera.distance - static_cast<float>(y) * 0.5f, 2.0f, 25.0f);
}

// ================================================================ frame
struct FrameUniforms {
    Vec3 sun, moon, eye;
    float time, starAngle;
    Mat4 view, projection, sunMatrix, moonMatrix;
};
void applyFrameUniforms(Program& p, const FrameUniforms& f) {
    p.set("uSunDir", f.sun);
    p.set("uMoonDir", f.moon);
    p.set("uEye", f.eye);
    p.set("uTime", f.time);
    p.set("uStarAngle", f.starAngle);
    p.set("uView", f.view);
    p.set("uProjection", f.projection);
    p.set("uSunMatrix", f.sunMatrix);
    p.set("uMoonMatrix", f.moonMatrix);
}

// How the program runs. Capture and record both use a fixed 60 Hz time step so
// their output is deterministic and smooth regardless of how fast frames render.
struct InteractiveMode {};
struct CaptureMode { std::string path; int frames; }; // save frame N as a PPM, then exit
struct RecordMode { std::string path; };              // scripted day/night film, piped to ffmpeg
using RunMode = std::variant<InteractiveMode, CaptureMode, RecordMode>;

struct Options {
    float hour = 21.0f;
    RunMode mode = InteractiveMode{};
};

std::optional<Options> parseOptions(int argc, char** argv) {
    constexpr const char* usage =
        "Usage: graphics_demo [--hour 0-24] [--capture out.ppm [--frames N] | --record out.mp4]\n";
    Options o;
    std::optional<std::string> capturePath, recordPath;
    std::optional<int> frames;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--hour" && hasValue) o.hour = std::clamp(std::strtof(argv[++i], nullptr), 0.0f, 24.0f);
        else if (arg == "--capture" && hasValue) capturePath = argv[++i];
        else if (arg == "--frames" && hasValue) frames = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--record" && hasValue) recordPath = argv[++i];
        else { std::cerr << usage; return std::nullopt; }
    }
    if ((capturePath && recordPath) || (frames && !capturePath)) { std::cerr << usage; return std::nullopt; }
    if (capturePath) o.mode = CaptureMode{*capturePath, frames.value_or(90)};
    else if (recordPath) o.mode = RecordMode{*recordPath};
    return o;
}

// Reads the back buffer as tightly packed RGB rows, bottom row first.
std::vector<unsigned char> readPixels(int width, int height) {
    std::vector<unsigned char> pixels(static_cast<std::size_t>(width) * height * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    return pixels;
}

bool writePpm(const std::string& path, int width, int height) {
    const std::vector<unsigned char> pixels = readPixels(width, height);
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "P6\n" << width << ' ' << height << "\n255\n";
    for (int row = height - 1; row >= 0; --row)
        out.write(reinterpret_cast<const char*>(pixels.data() + static_cast<std::size_t>(row) * width * 3), width * 3);
    return static_cast<bool>(out);
}

// ---------------------------------------------------------------- recording
// Film script: wall-clock seconds -> hour of day. Hours keep increasing past 24
// so the night segment can run through midnight. Each segment lasts 7 s, so
// every phase of the day gets equal screen time however long it really is.
struct TimeKey { float seconds, hours; };
constexpr std::array<TimeKey, 7> FILM_SCRIPT{{
    {0.0f, 10.5f},  // day
    {7.0f, 16.4f}, // afternoon into golden hour
    {14.0f, 18.2f}, // sunset
    {21.0f, 19.6f}, // dusk twilight into night
    {28.0f, 28.6f}, // night through midnight to pre-dawn (04:36)
    {35.0f, 30.4f}, // dawn twilight and sunrise (06:24)
    {42.0f, 34.0f}, // morning back to day (10:00)
}};

// Catmull-Rom through the keys: C1-continuous, so the sky and stars never
// change speed abruptly at a segment boundary.
float scriptedHour(float seconds) {
    const auto& k = FILM_SCRIPT;
    const float t = std::clamp(seconds, k.front().seconds, k.back().seconds);
    std::size_t i = 0;
    while (i + 2 < k.size() && t > k[i + 1].seconds) ++i;
    auto slope = [&](std::size_t j) {
        const std::size_t a = j == 0 ? 0 : j - 1, b = std::min(j + 1, k.size() - 1);
        return (k[b].hours - k[a].hours) / (k[b].seconds - k[a].seconds);
    };
    const float span = k[i + 1].seconds - k[i].seconds;
    const float h = hermite(k[i].hours, k[i + 1].hours, slope(i) * span, slope(i + 1) * span, (t - k[i].seconds) / span);
    return std::fmod(h, 24.0f);
}

#ifdef _WIN32
#define OPEN_PIPE _popen
#define CLOSE_PIPE _pclose
#else
#define OPEN_PIPE popen
#define CLOSE_PIPE pclose
#endif

// Streams raw frames into an ffmpeg process that encodes H.264, so no
// intermediate images ever touch the disk.
class VideoEncoder {
public:
    static std::optional<VideoEncoder> open(const std::string& path, int width, int height, int fps) {
        const std::string command = "ffmpeg -hide_banner -loglevel error -y -f rawvideo -pixel_format rgb24 -video_size " +
                                    std::to_string(width) + "x" + std::to_string(height) + " -framerate " +
                                    std::to_string(fps) + " -i - -vf vflip -c:v libx264 -preset slow -crf 20 " +
                                    "-pix_fmt yuv420p -movflags +faststart \"" + path + "\"";
        FILE* pipe = OPEN_PIPE(command.c_str(), "wb");
        if (!pipe) { std::cerr << "Could not start ffmpeg. Is it installed and on PATH?\n"; return std::nullopt; }
        return VideoEncoder(pipe, width, height);
    }
    VideoEncoder(VideoEncoder&& o) noexcept : pipe_(std::exchange(o.pipe_, nullptr)), width_(o.width_), height_(o.height_) {}
    VideoEncoder(const VideoEncoder&) = delete;
    VideoEncoder& operator=(const VideoEncoder&) = delete;
    VideoEncoder& operator=(VideoEncoder&&) = delete;
    ~VideoEncoder() { if (pipe_) CLOSE_PIPE(pipe_); }

    bool write(const std::vector<unsigned char>& frame, int width, int height) {
        if (width != width_ || height != height_) { std::cerr << "Window size changed during recording.\n"; return false; }
        return std::fwrite(frame.data(), 1, frame.size(), pipe_) == frame.size();
    }
    // Flushes and waits for ffmpeg; true only if it encoded successfully.
    bool finish() { const int status = CLOSE_PIPE(std::exchange(pipe_, nullptr)); return status == 0; }

private:
    VideoEncoder(FILE* pipe, int width, int height) : pipe_(pipe), width_(width), height_(height) {}
    FILE* pipe_;
    int width_, height_;
};

struct Scene {
    Program sky, terrain, grass, character, shadow;
    BodyMesh body;
    Terrain ground;
    GrassField field;
    ShadowMap sunShadow, moonShadow;
    GLuint emptyVao = 0;
};

std::optional<Scene> loadScene(const std::string& dir) {
    auto sky = loadProgram(dir, {"sky.vert"}, {"common.glsl", "sky.frag"});
    auto terrain = loadProgram(dir, {"common.glsl", "terrain.vert"}, {"common.glsl", "lighting.glsl", "terrain.frag"});
    auto grass = loadProgram(dir, {"common.glsl", "grass.vert"}, {"common.glsl", "lighting.glsl", "grass.frag"});
    auto character = loadProgram(dir, {"scene.vert"}, {"common.glsl", "lighting.glsl", "scene.frag"});
    auto shadow = loadProgram(dir, {"shadow.vert"}, {"shadow.frag"});
    if (!sky || !terrain || !grass || !character || !shadow) {
        std::cerr << "Shader directory: " << dir << '\n';
        return std::nullopt;
    }
    auto sunShadow = makeShadowMap(), moonShadow = makeShadowMap();
    if (!sunShadow || !moonShadow) return std::nullopt;
    Scene s{*sky, *terrain, *grass, *character, *shadow, makeBodyMesh(), makeTerrain(), makeGrassField(),
            *sunShadow, *moonShadow, 0};
    glGenVertexArrays(1, &s.emptyVao);
    for (Program* p : {&s.terrain, &s.grass, &s.character}) {
        glUseProgram(p->id);
        p->set("uShadowSun", 1);
        p->set("uShadowMoon", 2);
    }
    return s;
}

void destroyScene(Scene& s) {
    for (Program* p : {&s.sky, &s.terrain, &s.grass, &s.character, &s.shadow}) glDeleteProgram(p->id);
    destroy(s.body);
    destroy(s.ground);
    destroy(s.field);
    destroy(s.sunShadow);
    destroy(s.moonShadow);
    glDeleteVertexArrays(1, &s.emptyVao);
}

void drawBody(Program& p, const BodyMesh& body) {
    p.set("uModel", identity());
    p.set("uTint", Vec3{1, 1, 1});
    glBindVertexArray(body.vao);
    glDrawArrays(GL_TRIANGLES, 0, body.count);
}

void renderShadow(Scene& s, const ShadowMap& map, const Mat4& matrix) {
    glBindFramebuffer(GL_FRAMEBUFFER, map.fbo);
    glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
    glClear(GL_DEPTH_BUFFER_BIT);
    glUseProgram(s.shadow.id);
    s.shadow.set("uLightMatrix", matrix);
    drawBody(s.shadow, s.body);
}

void renderFrame(Scene& s, const FrameUniforms& f, const ViewBasis& cam, int width, int height) {
    // 1. Shadow maps (runner only; it is the only thing tall enough to cast a visible shadow).
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(2.0f, 4.0f);
    renderShadow(s, s.sunShadow, f.sunMatrix);
    renderShadow(s, s.moonShadow, f.moonMatrix);
    glDisable(GL_POLYGON_OFFSET_FILL);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, s.sunShadow.texture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, s.moonShadow.texture);
    glActiveTexture(GL_TEXTURE0);

    // 2. Sky.
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glUseProgram(s.sky.id);
    applyFrameUniforms(s.sky, f);
    s.sky.set("uCamRight", cam.right);
    s.sky.set("uCamUp", cam.up);
    s.sky.set("uCamForward", cam.forward);
    s.sky.set("uTanHalfFov", std::tan(FIELD_OF_VIEW * 0.5f));
    s.sky.set("uAspect", static_cast<float>(width) / static_cast<float>(height));
    glBindVertexArray(s.emptyVao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    // 3. Terrain, snapped to whole cells under the runner.
    glUseProgram(s.terrain.id);
    applyFrameUniforms(s.terrain, f);
    s.terrain.set("uTerrainOffset", std::floor(g_app.runner.x / TERRAIN_STEP) * TERRAIN_STEP, 0.0f);
    glBindVertexArray(s.ground.vao);
    glDrawElements(GL_TRIANGLES, s.ground.indexCount, GL_UNSIGNED_INT, nullptr);

    // 4. Runner and 5. grass are both drawn two-sided (open cloth tubes, thin blades).
    glDisable(GL_CULL_FACE);
    glUseProgram(s.character.id);
    applyFrameUniforms(s.character, f);
    drawBody(s.character, s.body);

    glUseProgram(s.grass.id);
    applyFrameUniforms(s.grass, f);
    s.grass.set("uGrassCenterX", g_app.runner.x);
    s.grass.set("uGrassCenterZ", (GRASS_Z_MIN + GRASS_Z_MAX) * 0.5f);
    s.grass.set("uGrassTile", GRASS_TILE_X, GRASS_Z_MAX - GRASS_Z_MIN);
    glBindVertexArray(s.field.vao);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, s.field.bladeVertexCount, s.field.instanceCount);
    glEnable(GL_CULL_FACE);
    glBindVertexArray(0);
}
} // namespace

int main(int argc, char** argv) {
    const auto options = parseOptions(argc, argv);
    if (!options) return 2;
    const auto* capture = std::get_if<CaptureMode>(&options->mode);
    const auto* record = std::get_if<RecordMode>(&options->mode);
    const bool fixedStep = !std::holds_alternative<InteractiveMode>(options->mode);
    g_app.day.hours = record ? scriptedHour(0.0f) : options->hour;
    g_app.day.paused = fixedStep;
    constexpr int OUTPUT_FPS = 60;

    if (!glfwInit()) { std::cerr << "Could not initialize GLFW.\n"; return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    if (record) glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // the video's frame size is fixed
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow* window = glfwCreateWindow(WINDOW_W, WINDOW_H, "Field Run", nullptr, nullptr);
    if (!window) { std::cerr << "Could not create an OpenGL 3.3 window. Check graphics drivers.\n"; glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(fixedStep ? 0 : 1); // offline modes render as fast as they can
    if (!gladLoadGL(glfwGetProcAddress)) {
        std::cerr << "Could not load OpenGL functions. Update the graphics driver.\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButton);
    glfwSetCursorPosCallback(window, cursorPos);
    glfwSetScrollCallback(window, scroll);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_MULTISAMPLE);

    auto scene = loadScene(SHADER_DIR);
    if (!scene) { glfwDestroyWindow(window); glfwTerminate(); return 1; }

    std::optional<VideoEncoder> encoder = [&]() -> std::optional<VideoEncoder> {
        if (!record) return std::nullopt;
        int fbWidth = 0, fbHeight = 0;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        return VideoEncoder::open(record->path, fbWidth, fbHeight, OUTPUT_FPS);
    }();
    if (record && !encoder) { destroyScene(*scene); glfwDestroyWindow(window); glfwTerminate(); return 1; }

    BodyBuilder builder;
    float previous = static_cast<float>(glfwGetTime()), simTime = 0.0f, titleTimer = 0.0f, fpsAccum = 0.0f;
    int fpsFrames = 0, frame = 0, exitCode = 0;
    while (!glfwWindowShouldClose(window)) {
        const float now = static_cast<float>(glfwGetTime());
        const float realDt = now - previous;
        previous = now;
        const float dt = fixedStep ? 1.0f / OUTPUT_FPS : std::min(realDt, 0.05f);
        simTime += dt;

        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        if (width == 0 || height == 0) { glfwWaitEvents(); continue; } // minimised

        // Simulation.
        const int input = (g_app.held(GLFW_KEY_D) || g_app.held(GLFW_KEY_RIGHT) ? 1 : 0)
                        - (g_app.held(GLFW_KEY_A) || g_app.held(GLFW_KEY_LEFT) ? 1 : 0);
        updateRunner(g_app.runner, dt, input, g_app.autoRun);
        const float scrub = (g_app.held(GLFW_KEY_RIGHT_BRACKET) ? 1.0f : 0.0f) - (g_app.held(GLFW_KEY_LEFT_BRACKET) ? 1.0f : 0.0f);
        g_app.day.advance(dt, scrub);
        if (record) g_app.day.hours = scriptedHour(simTime);
        OrbitCamera& cam = g_app.camera;
        if (g_app.held(GLFW_KEY_W) || g_app.held(GLFW_KEY_UP)) cam.pitch = std::min(1.30f, cam.pitch + dt);
        if (g_app.held(GLFW_KEY_S) || g_app.held(GLFW_KEY_DOWN)) cam.pitch = std::max(-0.45f, cam.pitch - dt);

        const Vec3 chest = poseRunner(g_app.runner, simTime, builder);
        uploadBody(scene->body, builder.vertices());

        // Camera: smoothed orbit around the chest, kept above the ground, with the
        // runner framed right of centre like a low cinematic tracking shot.
        if (!cam.hasTarget) { cam.target = chest; cam.hasTarget = true; }
        cam.target = cam.target + (chest - cam.target) * approach(6.0f, dt);
        const Vec3 offset{std::sin(cam.yaw) * std::cos(cam.pitch), std::sin(cam.pitch), std::cos(cam.yaw) * std::cos(cam.pitch)};
        Vec3 eye = cam.target + offset * cam.distance;
        eye.y = std::max(eye.y, terrainHeight(eye.x, eye.z) + 0.25f);
        const Vec3 worldUp{0, 1, 0};
        const Vec3 right = viewBasis(eye, cam.target, worldUp).right;
        const Vec3 aim = cam.target - right * 0.55f + Vec3{0, 0.35f, 0};
        const ViewBasis basisNow = viewBasis(eye, aim, worldUp);

        FrameUniforms f{};
        f.sun = g_app.day.sunDir();
        f.moon = g_app.day.moonDir();
        f.eye = eye;
        f.time = simTime;
        f.starAngle = g_app.day.starAngle();
        f.view = lookAt(eye, aim, worldUp);
        f.projection = perspective(FIELD_OF_VIEW, static_cast<float>(width) / static_cast<float>(height), 0.05f, 400.0f);
        f.sunMatrix = lightMatrix(f.sun, chest);
        f.moonMatrix = lightMatrix(f.moon, chest);

        renderFrame(*scene, f, basisNow, width, height);

        ++frame;
        if (capture && frame >= capture->frames) {
            if (!writePpm(capture->path, width, height)) {
                std::cerr << "Could not write " << capture->path << '\n';
                exitCode = 1;
            }
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        if (record) {
            if (!encoder->write(readPixels(width, height), width, height)) {
                std::cerr << "Writing to ffmpeg failed.\n";
                exitCode = 1;
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
            if (frame % OUTPUT_FPS == 0)
                std::cout << "\rRecorded " << frame / OUTPUT_FPS << " / " << FILM_SCRIPT.back().seconds << " s" << std::flush;
            if (simTime >= FILM_SCRIPT.back().seconds) glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        glfwSwapBuffers(window);
        glfwPollEvents();

        fpsAccum += realDt;
        ++fpsFrames;
        titleTimer += realDt;
        if (titleTimer > 0.25f) {
            const float h = g_app.day.hours;
            const int hh = static_cast<int>(h), mm = static_cast<int>((h - static_cast<float>(hh)) * 60.0f);
            std::array<char, 256> title{};
            std::snprintf(title.data(), title.size(),
                          "Field Run | %02d:%02d %s%s | %.0f fps | Space auto-run  A/D run  T pause time  [ ] scrub  1-4 presets  C camera",
                          hh, mm, g_app.day.phaseName(), g_app.day.paused ? " (paused)" : "",
                          fpsAccum > 0 ? static_cast<float>(fpsFrames) / fpsAccum : 0.0f);
            glfwSetWindowTitle(window, title.data());
            titleTimer = 0.0f;
            fpsAccum = 0.0f;
            fpsFrames = 0;
        }
    }
    if (encoder) {
        std::cout << "\nFinishing encode...\n";
        if (!encoder->finish()) { std::cerr << "ffmpeg reported an encoding error.\n"; exitCode = 1; }
        else if (exitCode == 0) std::cout << "Saved " << record->path << '\n';
    }
    destroyScene(*scene);
    glfwDestroyWindow(window);
    glfwTerminate();
    return exitCode;
}
