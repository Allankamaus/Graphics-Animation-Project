#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

struct Color { unsigned char r, g, b; };

class Canvas {
public:
	int width, height;
	std::vector<Color> buffer;

	Canvas(int w, int h, Color bg = {20, 24, 35})
		: width(w), height(h), buffer(w * h, bg) {}

	void setPixel(int x, int y, Color c) {
		if (x >= 0 && x < width && y >= 0 && y < height) {
			buffer[y * width + x] = c;
		}
	}

	// Bresenham's Line Algorithm
	void drawLine(int x0, int y0, int x1, int y1, Color c) {
		int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
		int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
		int error = dx + dy, e2;

		while (true) {
			setPixel(x0, y0, c);
			if (x0 == x1 && y0 == y1) break;
			e2 = 2 * error;

			if (e2 >= dy) { error += dy; x0 += sx; }
			if (e2 <= dx) { error += dx; y0 += sy; }
		}
	}

	bool savePPM(const std::string& filename) const {
		std::ofstream ofs(filename, std::ios::binary);
		if (!ofs.is_open()) return false;
		ofs << "P6\n" << width << " " << height << "\n255\n";
		for (const auto& p : buffer) {
			ofs.put(p.r).put(p.g).put(p.b);
		}
		return true;
	}
};

int main() {
	Canvas canvas(800, 600);
	Color green = { 46, 139, 87 };
	Color white = { 240, 240, 240 };

	// 1. Draw field horizontal line
	canvas.drawLine(0, 420, 800, 420, green);

	// 2. Draw the pose of the runner
	canvas.drawLine(400, 260, 400, 340, white); //Torso
	canvas.drawLine(400, 340, 370, 420, white); // Left Leg
	canvas.drawLine(400, 340, 430, 410, white); // Right leg
	canvas.drawLine(400, 280, 365, 310, white); // Left arm
	canvas.drawLine(400, 280, 435, 300, white); // Right arm

	if (canvas.savePPM("cp1_scene.ppm")) {
		std::cout << "Generated cp1_scene.ppm" << std::endl;
	}
	else {
	std::cerr << "Error generating image" << std::endl;
	}

	return 0;
}