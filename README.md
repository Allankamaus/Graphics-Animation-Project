# Graphics Project : Guy Running in a Field

## Group Project
- **Topic:** Character Animation and Procedural Field
- **Language:** C++ (CP1/CP2 Pure C++, CP3/CP4 OpenGL)

## Progess
- [x] **CP1 - Technical Foundation:** Pure C++ Software rasterizer, Bresenham line algorithm, PPM exporter.
- [x] **CP2 - Transformation & Animation:** Pure C++ 2D/3D affine matrix transformations & run cycle
- [ ] **CP3 - OpenGL Integration:** Transition to hardware-accelerated pipeline, shaders, camera controls
- [ ] **CP4 - Render & Final Demo:** Terrain shading, procedural vegetation and final presentation


## Setup

**How to compile and run**

*Using GCC or MinGW*
```bash
g++ -std=c++17 src/cp1/main.cpp -o cp1_scene
```

Output: Generates `cp1_scene.ppm`. I have also committed the converted `cp1_scene.png` in the repo to preview what the output looks so far

**CP2 (run cycle)**
```bash
g++ -std=c++17 src/cp2/main.cpp -o cp2_scene
./cp2_scene
ffmpeg -framerate 12 -i frames/frame_%03d.ppm -pix_fmt yuv420p cp2_run.mp4
```

Output: 48 frames (two run cycles) in `frames/`. `cp2_run.mp4` is the stitched preview. CP2 implements 2D affine transforms (`Mat3`), a joint hierarchy and a sine-driven run cycle. 3D (`Mat4`) is left for CP3.

- Convert a .ppm image to a .png on [this website](https://jumpshare.com/)
