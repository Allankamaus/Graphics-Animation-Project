// CP2 - Transformation & Animation
// Pure C++ software rasterizer: 2D affine matrices, a hierarchical skeleton
// and a procedural run cycle. Writes one PPM per frame into ./frames/.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

constexpr double PI = 3.14159265358979323846;

// ---------------------------------------------------------------- math ----

struct Vec2 { double x, y; };

// 3x3 affine matrix in homogeneous 2D coordinates, row-major, applied to
// column vectors: p' = M * [x y 1]^T. Compose with operator*: (A * B) applies B first.
struct Mat3 {
	double m[3][3];

	static Mat3 identity() {
		return {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
	}
	static Mat3 translate(double tx, double ty) {
		return {{{1, 0, tx}, {0, 1, ty}, {0, 0, 1}}};
	}
	static Mat3 scale(double sx, double sy) {
		return {{{sx, 0, 0}, {0, sy, 0}, {0, 0, 1}}};
	}
	// Counter-clockwise rotation (radians) in a y-up coordinate system.
	static Mat3 rotate(double a) {
		double c = std::cos(a), s = std::sin(a);
		return {{{c, -s, 0}, {s, c, 0}, {0, 0, 1}}};
	}

	Mat3 operator*(const Mat3& o) const {
		Mat3 r{};
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				for (int k = 0; k < 3; ++k)
					r.m[i][j] += m[i][k] * o.m[k][j];
		return r;
	}
	Vec2 apply(Vec2 p) const {
		return { m[0][0] * p.x + m[0][1] * p.y + m[0][2],
				 m[1][0] * p.x + m[1][1] * p.y + m[1][2] };
	}
};

// ------------------------------------------------------------- raster ----

struct Color { unsigned char r, g, b; };

class Canvas {
public:
	int width, height;
	std::vector<Color> buffer;

	Canvas(int w, int h, Color bg = {20, 24, 35})
		: width(w), height(h), buffer(w * h, bg) {}

	void setPixel(int x, int y, Color c) {
		if (x >= 0 && x < width && y >= 0 && y < height)
			buffer[y * width + x] = c;
	}

	// Bresenham's line algorithm
	void drawLine(int x0, int y0, int x1, int y1, Color c) {
		int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
		int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
		int error = dx + dy;
		while (true) {
			setPixel(x0, y0, c);
			if (x0 == x1 && y0 == y1) break;
			int e2 = 2 * error;
			if (e2 >= dy) { error += dy; x0 += sx; }
			if (e2 <= dx) { error += dx; y0 += sy; }
		}
	}

	// Midpoint circle algorithm (outline)
	void drawCircle(int cx, int cy, int r, Color c) {
		int x = r, y = 0, err = 1 - r;
		while (x >= y) {
			setPixel(cx + x, cy + y, c); setPixel(cx - x, cy + y, c);
			setPixel(cx + x, cy - y, c); setPixel(cx - x, cy - y, c);
			setPixel(cx + y, cy + x, c); setPixel(cx - y, cy + x, c);
			setPixel(cx + y, cy - x, c); setPixel(cx - y, cy - x, c);
			++y;
			if (err < 0) err += 2 * y + 1;
			else { --x; err += 2 * (y - x) + 1; }
		}
	}

	bool savePPM(const std::string& filename) const {
		std::ofstream ofs(filename, std::ios::binary);
		if (!ofs.is_open()) return false;
		ofs << "P6\n" << width << " " << height << "\n255\n";
		for (const auto& p : buffer) ofs.put(p.r).put(p.g).put(p.b);
		return true;
	}
};

// ----------------------------------------------------------- skeleton ----

// One joint in a tree. Its frame is the parent's frame, moved by `offset`
// and then rotated by `angle`. A bone is drawn from the parent joint to this one.
struct Joint {
	int parent;      // -1 for the root
	Vec2 offset;     // position in the parent's frame (model space, y up)
	double angle;    // rotation of this joint's frame, radians
	Mat3 world;      // filled in by updateWorld()
};

enum JointId {
	HIP, CHEST, HEAD,
	SHOULDER_L, ELBOW_L, HAND_L,
	SHOULDER_R, ELBOW_R, HAND_R,
	THIGH_L, KNEE_L, FOOT_L,
	THIGH_R, KNEE_R, FOOT_R,
	JOINT_COUNT
};

// Segment lengths in model units (pixels at scale 1)
constexpr double TORSO = 80, NECK = 22, UPPER_ARM = 36, FOREARM = 34,
				 THIGH = 48, SHIN = 46;
constexpr double LEG = THIGH + SHIN;

std::vector<Joint> buildSkeleton() {
	std::vector<Joint> j(JOINT_COUNT);
	auto set = [&](int id, int parent, Vec2 off) {
		j[id] = { parent, off, 0.0, Mat3::identity() };
	};
	set(HIP, -1, {0, 0});
	set(CHEST, HIP, {0, TORSO});
	set(HEAD, CHEST, {0, NECK});

	// Limbs hang down (-y) from their parent joint at angle 0.
	set(SHOULDER_L, CHEST, {0, -6});  set(ELBOW_L, SHOULDER_L, {0, -UPPER_ARM});  set(HAND_L, ELBOW_L, {0, -FOREARM});
	set(SHOULDER_R, CHEST, {0, -6});  set(ELBOW_R, SHOULDER_R, {0, -UPPER_ARM});  set(HAND_R, ELBOW_R, {0, -FOREARM});
	set(THIGH_L, HIP, {0, 0});        set(KNEE_L, THIGH_L, {0, -THIGH});          set(FOOT_L, KNEE_L, {0, -SHIN});
	set(THIGH_R, HIP, {0, 0});        set(KNEE_R, THIGH_R, {0, -THIGH});          set(FOOT_R, KNEE_R, {0, -SHIN});
	return j;
}

// Joints are stored parent-before-child, so one forward pass is enough.
// `root` is the model->screen-space base transform applied above the hip.
void updateWorld(std::vector<Joint>& j, const Mat3& root) {
	for (auto& jt : j) {
		Mat3 local = Mat3::translate(jt.offset.x, jt.offset.y) * Mat3::rotate(jt.angle);
		jt.world = (jt.parent < 0 ? root : j[jt.parent].world) * local;
	}
}

Vec2 position(const Joint& j) { return j.world.apply({0, 0}); }

// Procedural run cycle. phase is in [0, 2*pi). Positive angle swings a limb
// forward (+x). Left and right are half a cycle apart.
void poseRun(std::vector<Joint>& j, double phase) {
	const double legSwing = 0.75, armSwing = 0.8;

	auto leg = [&](int thigh, int knee, double p) {
		j[thigh].angle = legSwing * std::sin(p);
		// Knee folds backward (negative) most while the leg swings forward.
		j[knee].angle = -(0.25 + 1.0 * std::max(0.0, std::cos(p)));
	};
	auto arm = [&](int shoulder, int elbow, double p) {
		j[shoulder].angle = armSwing * std::sin(p);
		j[elbow].angle = 1.1;  // forearm held forward
	};

	leg(THIGH_L, KNEE_L, phase);
	leg(THIGH_R, KNEE_R, phase + PI);
	arm(SHOULDER_L, ELBOW_L, phase + PI);  // arms oppose the same-side leg
	arm(SHOULDER_R, ELBOW_R, phase);

	j[CHEST].angle = -0.22;  // lean forward
	j[HEAD].angle = 0.12;    // keep the head a little more upright
}

// -------------------------------------------------------------- scene ----

constexpr int W = 800, H = 600;
constexpr int GROUND_Y = 420;       // screen row of the ground line
constexpr int FRAMES_PER_CYCLE = 24;
constexpr int CYCLES = 2;
constexpr double LEG_SWING = 0.75;

// Ground speed that roughly matches the planted foot, so feet don't skate.
constexpr double SPEED_PER_FRAME = LEG * LEG_SWING * (2 * PI / FRAMES_PER_CYCLE) * 0.8;

void drawBone(Canvas& c, const Mat3& view, Vec2 a, Vec2 b, Color col) {
	Vec2 p = view.apply(a), q = view.apply(b);
	c.drawLine((int)std::lround(p.x), (int)std::lround(p.y),
			   (int)std::lround(q.x), (int)std::lround(q.y), col);
}

void renderFrame(Canvas& canvas, std::vector<Joint>& skel, int frame) {
	const Color green = {46, 139, 87}, tick = {30, 90, 60}, white = {240, 240, 240};
	double phase = 2 * PI * (frame % FRAMES_PER_CYCLE) / FRAMES_PER_CYCLE;

	// View: model space is y-up with the ground at y=0; flip to screen space
	// (y-down) and put the ground at GROUND_Y. The camera follows the runner,
	// so the runner stays centred and the ground scrolls past.
	Mat3 view = Mat3::translate(W / 2.0, GROUND_Y) * Mat3::scale(1, -1);

	// Pose, then lift the hip so the lowest foot rests exactly on the ground.
	poseRun(skel, phase);
	updateWorld(skel, Mat3::identity());
	double lowest = std::min(position(skel[FOOT_L]).y, position(skel[FOOT_R]).y);
	// Small bounce on top: body is lowest at mid-stance, twice per cycle.
	double bounce = 3.0 * std::cos(2 * phase);
	updateWorld(skel, Mat3::translate(0, -lowest + bounce));

	std::fill(canvas.buffer.begin(), canvas.buffer.end(), Color{20, 24, 35});

	// Ground line plus scrolling tick marks so motion is visible.
	canvas.drawLine(0, GROUND_Y, W - 1, GROUND_Y, green);
	double scroll = std::fmod(frame * SPEED_PER_FRAME, 80.0);
	for (double x = -scroll; x < W; x += 80.0)
		canvas.drawLine((int)x, GROUND_Y + 1, (int)x - 14, GROUND_Y + 14, tick);

	// Bones: parent joint -> joint.
	for (const auto& jt : skel) {
		if (jt.parent < 0) continue;
		// Skip joints that only exist to anchor limb frames at zero length.
		if (jt.offset.x == 0 && jt.offset.y == 0) continue;
		drawBone(canvas, view, position(skel[jt.parent]), position(jt), white);
	}
	// Head: circle centred just above the head joint.
	Vec2 hp = view.apply(position(skel[HEAD]));
	canvas.drawCircle((int)std::lround(hp.x), (int)std::lround(hp.y) - 10, 10, white);
}

int main() {
	namespace fs = std::filesystem;
	fs::create_directories("frames");

	Canvas canvas(W, H);
	auto skel = buildSkeleton();

	int total = FRAMES_PER_CYCLE * CYCLES;
	for (int f = 0; f < total; ++f) {
		renderFrame(canvas, skel, f);
		char name[64];
		std::snprintf(name, sizeof name, "frames/frame_%03d.ppm", f);
		if (!canvas.savePPM(name)) {
			std::cerr << "Error writing " << name << std::endl;
			return 1;
		}
	}
	std::cout << "Wrote " << total << " frames to frames/" << std::endl;
	return 0;
}
