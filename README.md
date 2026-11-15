# Graphics Project : Guy Running in a Field

## Group Project
- **Topic:** Character Animation and Procedural Field
- **Language:** C++ (CP1/CP2 Pure C++, CP3/CP4 OpenGL)

## Progess
- [x] **CP1 - Technical Foundation:** Pure C++ Software rasterizer, Bresenham line algorithm, PPM exporter.
- [ ] **CP2 - Transformation & Animation:** Pure C++ 2D/3D affine matrix transformations & run cycle
- [ ] **CP3 - OpenGL Integration:** Transition to hardware-accelerated pipeline, shaders, camera controls
- [ ] **CP4 - Render & Final Demo:** Terrain shading, procedural vegetation and final presentation


## Setup

**How to compile and run**

*Using GCC or MinGW*
```bash
g++ -std=c++17 src/cp1/main.cpp -o cp1_scene
```

Output: Generates `cp1_scene.ppm`. I have also committed the converted `cp1_scene.png` in the repo to preview what the output looks so far

- Convert a .ppm image to a .png on [this website](https://jumpshare.com/)
