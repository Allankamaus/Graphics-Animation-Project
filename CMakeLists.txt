cmake_minimum_required(VERSION 3.20)
project(GraphicsAnimationProject LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(FetchContent)
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)
FetchContent_Declare(glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG v2.0.6
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR cmake
)
FetchContent_MakeAvailable(glad)
glad_add_library(glad_gl_core_33 STATIC REPRODUCIBLE API gl:core=3.3)
add_executable(graphics_demo src/cp3/main.cpp)
target_link_libraries(graphics_demo PRIVATE glfw glad_gl_core_33)
target_compile_definitions(graphics_demo PRIVATE SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/cp3/shaders")
if(MSVC)
    target_compile_options(graphics_demo PRIVATE /W4)
else()
    target_compile_options(graphics_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(MINGW)
    # Bundle libstdc++/libgcc so the exe never picks up a mismatched runtime DLL from PATH.
    target_link_options(graphics_demo PRIVATE -static)
endif()
