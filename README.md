# Graphics Project : Guy Running in a Field

## Group Project
- **Topic:** Character Animation and Procedural Field
- **Language:** C++ (CP1/CP2 Pure C++, CP3/CP4 OpenGL)

## Progess
- [x] **CP1 - Technical Foundation:** Pure C++ Software rasterizer, Bresenham line algorithm, PPM exporter.
- [x] **CP2 - Transformation & Animation:** Pure C++ 2D/3D affine matrix transformations & run cycle
- [x] **CP3 - OpenGL Integration:** Transition to hardware-accelerated pipeline, shaders, camera controls
- [x] **CP4 - Render & Final Demo:** Terrain shading, procedural vegetation and final presentation


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

## Build and run the OpenGL demo (Windows)

Requirements: Visual Studio 2022 with the Desktop development with C++ workload, CMake 3.20 or newer, and an internet connection on the first configure (GLFW and GLAD are fetched by CMake).

From the repository root in PowerShell or the VS Code terminal, run:

cmake --preset vs2022-x64
cmake --build --preset vs2022-x64-debug
.\out\build\vs2022-x64\Debug\graphics_demo.exe

The OpenGL CP3/CP4 implementation and GLSL shaders are in src/cp3/. Build output is kept under out/ and ignored by Git.

**Alternative: MSYS2 / MinGW (no Visual Studio).** GLAD's generator needs Python with `jinja2`.

```bash
pacman -S mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja
cmake -S . -B out/build/mingw -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build out/build/mingw
./out/build/mingw/graphics_demo.exe
```

The shader path is compiled in as an absolute path, so the exe can be launched from any folder.

### The scene

A runner in a dark jacket and red sneakers sprints up an endless grassy ridge, seen in a low tracking shot from the grass.

- **Runner:** a procedural rig rebuilt every frame from tapered capsules, ellipsoids and cones: a run cycle with knee kick-back and opposing arm swing, a flared jacket hem, and spiky hair that streams back with speed. Feet are planted on the terrain wherever the runner is.
- **Vegetation:** about 200k instanced, wind-animated grass blades: tufts, tall seed grass and wildflowers, with translucency when lit from behind. The grass tile wraps around the runner, so the meadow never ends.
- **Day/night cycle:** one analytic sky drives the backdrop, ambient light, fog and exposure. It covers sunrise, day, golden hour, blue hour and a starry night with a Milky Way and the moon. The sun and moon each cast a shadow of the runner.

### Controls

| Key | Action |
| --- | --- |
| Space | Toggle auto-run |
| A / D or ← / → | Run back / forward |
| Mouse drag, scroll, W / S | Orbit, zoom, tilt camera |
| C | Reset camera to the low hero angle |
| T | Pause / resume the day cycle (a full day takes 3 minutes) |
| [ / ] | Scrub time backward / forward |
| 1 / 2 / 3 / 4 | Sunrise / noon / sunset / night |
| R | Reset runner |

Command line: `graphics_demo --hour 18` starts at a given time of day. `--capture frame.ppm --frames 120` renders 120 frames, saves the last one and exits. `--record film.mp4` renders a 42 s, 60 fps film with 7 s each of day, golden hour, dusk, night, dawn and morning, piped straight into ffmpeg (which must be on `PATH`).
