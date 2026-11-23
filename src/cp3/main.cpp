#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr float PI = 3.14159265358979323846f;
constexpr int WINDOW_W = 1280;
constexpr int WINDOW_H = 720;

struct Vec3 {
    float x{}, y{}, z{};
    Vec3 operator+(Vec3 b) const { return {x+b.x,y+b.y,z+b.z}; }
    Vec3 operator-(Vec3 b) const { return {x-b.x,y-b.y,z-b.z}; }
    Vec3 operator*(float s) const { return {x*s,y*s,z*s}; }
};
float dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
Vec3 cross(Vec3 a, Vec3 b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
Vec3 normalize(Vec3 v) {
    const float n=std::sqrt(std::max(dot(v,v),1e-8f));
    return v*(1.0f/n);
}

struct Mat4 { std::array<float,16> m{}; };
Mat4 identity() { Mat4 r{}; r.m[0]=r.m[5]=r.m[10]=r.m[15]=1; return r; }
Mat4 multiply(const Mat4& a,const Mat4& b) {
    Mat4 r{};
    for(int c=0;c<4;++c) for(int row=0;row<4;++row)
        for(int k=0;k<4;++k) r.m[c*4+row]+=a.m[k*4+row]*b.m[c*4+k];
    return r;
}
Mat4 translate(Vec3 p) { Mat4 r=identity(); r.m[12]=p.x;r.m[13]=p.y;r.m[14]=p.z;return r; }
Mat4 scale(Vec3 s) { Mat4 r=identity();r.m[0]=s.x;r.m[5]=s.y;r.m[10]=s.z;return r; }
Mat4 perspective(float fov,float aspect,float nearZ,float farZ) {
    Mat4 r{}; const float f=1.0f/std::tan(fov*0.5f);
    r.m[0]=f/aspect;r.m[5]=f;r.m[10]=(farZ+nearZ)/(nearZ-farZ);
    r.m[11]=-1;r.m[14]=(2*farZ*nearZ)/(nearZ-farZ);return r;
}
Mat4 lookAt(Vec3 eye,Vec3 center,Vec3 up) {
    Vec3 f=normalize(center-eye),s=normalize(cross(f,up)),u=cross(s,f);
    Mat4 r=identity();
    r.m[0]=s.x;r.m[4]=s.y;r.m[8]=s.z;
    r.m[1]=u.x;r.m[5]=u.y;r.m[9]=u.z;
    r.m[2]=-f.x;r.m[6]=-f.y;r.m[10]=-f.z;
    r.m[12]=-dot(s,eye);r.m[13]=-dot(u,eye);r.m[14]=dot(f,eye);return r;
}

struct Vertex { float px,py,pz,nx,ny,nz,r,g,b; };
struct Mesh { GLuint vao{},vbo{}; GLsizei count{}; GLenum mode{GL_TRIANGLES}; };
Mesh upload(const std::vector<Vertex>& vertices,GLenum mode=GL_TRIANGLES) {
    Mesh mesh;mesh.count=static_cast<GLsizei>(vertices.size());mesh.mode=mode;
    glGenVertexArrays(1,&mesh.vao);glGenBuffers(1,&mesh.vbo);
    glBindVertexArray(mesh.vao);glBindBuffer(GL_ARRAY_BUFFER,mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER,static_cast<GLsizeiptr>(vertices.size()*sizeof(Vertex)),vertices.data(),GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),reinterpret_cast<void*>(3*sizeof(float)));
    glEnableVertexAttribArray(2);glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),reinterpret_cast<void*>(6*sizeof(float)));
    glBindVertexArray(0);return mesh;
}
Vertex vertex(Vec3 p,Vec3 n,Vec3 c) { return {p.x,p.y,p.z,n.x,n.y,n.z,c.x,c.y,c.z}; }

GLuint compileShader(GLenum type,const std::string& source) {
    GLuint shader=glCreateShader(type);const char* src=source.c_str();glShaderSource(shader,1,&src,nullptr);glCompileShader(shader);
    GLint ok=0;glGetShaderiv(shader,GL_COMPILE_STATUS,&ok);
    if(!ok){char log[2048]{};glGetShaderInfoLog(shader,sizeof(log),nullptr,log);std::cerr<<"Shader compile error: "<<log<<'\n';glDeleteShader(shader);return 0;}
    return shader;
}
GLuint makeProgram(const std::string& vs,const std::string& fs) {
    GLuint v=compileShader(GL_VERTEX_SHADER,vs),f=compileShader(GL_FRAGMENT_SHADER,fs);
    if(!v||!f)return 0;GLuint p=glCreateProgram();glAttachShader(p,v);glAttachShader(p,f);glLinkProgram(p);
    glDeleteShader(v);glDeleteShader(f);GLint ok=0;glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){char log[2048]{};glGetProgramInfoLog(p,sizeof(log),nullptr,log);std::cerr<<"Program link error: "<<log<<'\n';glDeleteProgram(p);return 0;}return p;
}
std::string readText(const std::string& file) { std::ifstream in(file);std::ostringstream out;out<<in.rdbuf();return out.str(); }

float terrainHeight(float x,float z) {
    return 0.30f*std::sin(x*0.24f)*std::cos(z*0.19f)+0.15f*std::sin(z*0.47f+x*0.12f);
}
Vec3 terrainNormal(float x,float z) {
    constexpr float e=0.05f;float dx=(terrainHeight(x+e,z)-terrainHeight(x-e,z))/(2*e);
    float dz=(terrainHeight(x,z+e)-terrainHeight(x,z-e))/(2*e);return normalize({-dx,1,-dz});
}
Mesh makeTerrain() {
    std::vector<Vertex> v;constexpr int N=140;constexpr float step=0.20f;
    v.reserve(N*N*6);
    for(int z=0;z<N;++z)for(int x=0;x<N;++x){
        float x0=(x-N/2)*step,z0=(z-N/2)*step,x1=x0+step,z1=z0+step;
        Vec3 p00{x0,terrainHeight(x0,z0),z0},p10{x1,terrainHeight(x1,z0),z0};
        Vec3 p01{x0,terrainHeight(x0,z1),z1},p11{x1,terrainHeight(x1,z1),z1};
        auto color=[](float h){float t=std::clamp((h+0.5f),0.0f,1.0f);return Vec3{0.12f+0.10f*t,0.29f+0.28f*t,0.10f+0.09f*t};};
        for(auto tri: {std::array<Vec3,3>{p00,p01,p10},std::array<Vec3,3>{p10,p01,p11}})
            for(Vec3 p:tri)v.push_back(vertex(p,terrainNormal(p.x,p.z),color(p.y)));
    }return upload(v);
}
Mesh makeGrass() {
    std::vector<Vertex> v;std::mt19937 rng(42);std::uniform_real_distribution<float> pos(-13.5f,13.5f),height(0.18f,0.58f),lean(-0.14f,0.14f);
    for(int i=0;i<1500;++i){float x=pos(rng),z=pos(rng);if(std::abs(x)<1.1f&&std::abs(z)<1.1f)continue;
        float y=terrainHeight(x,z),h=height(rng),l=lean(rng);Vec3 n{0,1,0};
        Vec3 c{0.16f+0.12f*(h-0.18f),0.38f+0.30f*(h-0.18f),0.11f};
        Vec3 a{x-0.035f,y,z},b{x+0.035f,y,z},tip{x+l,y+h,z};
        v.push_back(vertex(a,n,c));v.push_back(vertex(b,n,c));v.push_back(vertex(tip,n,c*1.35f));
        Vec3 d{x,y,z-0.035f},e{x,y,z+0.035f},tip2{x+l*0.6f,y+h*0.82f,z+0.03f};
        v.push_back(vertex(d,n,c));v.push_back(vertex(e,n,c));v.push_back(vertex(tip2,n,c*1.25f));
    }return upload(v);
}
Mesh makeSphere(int slices=16,int stacks=10) {
    std::vector<Vertex> v;
    for(int j=0;j<stacks;++j){float a0=-PI/2+PI*j/stacks,a1=-PI/2+PI*(j+1)/stacks;
        for(int i=0;i<slices;++i){float b0=2*PI*i/slices,b1=2*PI*(i+1)/slices;
            auto point=[](float a,float b){return Vec3{std::cos(a)*std::cos(b),std::sin(a),std::cos(a)*std::sin(b)};};
            Vec3 p00=point(a0,b0),p10=point(a0,b1),p01=point(a1,b0),p11=point(a1,b1),white{1,1,1};
            for(Vec3 p:{p00,p01,p10,p10,p01,p11})v.push_back(vertex(p,normalize(p),white));
        }
    }return upload(v);
}
Mesh makeCylinder() {
    std::vector<Vertex> v;constexpr int sides=12;
    for(int i=0;i<sides;++i){float a=2*PI*i/sides,b=2*PI*(i+1)/sides;
        Vec3 p0{std::cos(a),0,std::sin(a)},p1{std::cos(b),0,std::sin(b)},q0{p0.x,1,p0.z},q1{p1.x,1,p1.z};
        for(Vec3 p:{p0,q0,p1,p1,q0,q1})v.push_back(vertex(p,normalize({p.x,0,p.z}),{1,1,1}));
    }return upload(v);
}
Mesh makeTrees() {
    std::vector<Vertex> v;std::mt19937 rng(98);std::uniform_real_distribution<float> pos(-13,13),size(0.75f,1.35f);
    auto tri=[&](Vec3 a,Vec3 b,Vec3 c,Vec3 color){Vec3 n=normalize(cross(b-a,c-a));v.push_back(vertex(a,n,color));v.push_back(vertex(b,n,color));v.push_back(vertex(c,n,color));};
    for(int t=0;t<44;++t){float x=pos(rng),z=pos(rng);if(std::abs(x)<3.3f&&std::abs(z)<3.3f)continue;
        float h=size(rng),y=terrainHeight(x,z);Vec3 top{x,y+1.7f*h,z},base{x,y,z},b1{x+0.17f*h,y+1.0f*h,z},b2{x,y+1.0f*h,z+0.17f*h};
        Vec3 trunk{0.30f,0.18f,0.09f};tri(base,b1,b2,trunk);tri(base,{b1.x,y,b1.z},{b1.x,y+1.0f*h,b1.z},trunk);
        Vec3 left{x-0.8f*h,y+0.7f*h,z},right{x+0.8f*h,y+0.7f*h,z},front{x,y+0.7f*h,z-0.8f*h},back{x,y+0.7f*h,z+0.8f*h};
        Vec3 foliage{0.10f,0.32f,0.12f};tri(top,left,front,foliage);tri(top,front,right,foliage);tri(top,right,back,foliage);tri(top,back,left,foliage);
        Vec3 bottom{x,y+0.15f*h,z};tri(bottom,front,left,foliage*0.82f);tri(bottom,right,front,foliage*0.82f);tri(bottom,back,right,foliage*0.82f);tri(bottom,left,back,foliage*0.82f);
    }return upload(v);
}

struct Camera { float yaw{0.55f},pitch{0.32f},distance{10.0f};double lastX{},lastY{};bool dragging{}; };
Camera camera;bool keys[1024]{};bool mouseDown=false;float runnerX=0;float runTime=0;
void keyCallback(GLFWwindow* w,int key,int,int action,int){
    if(key>=0&&key<1024)keys[key]=action!=GLFW_RELEASE;
    if(action==GLFW_PRESS&&key==GLFW_KEY_ESCAPE)glfwSetWindowShouldClose(w,GLFW_TRUE);
    if(action==GLFW_PRESS&&key==GLFW_KEY_R){runnerX=0;runTime=0;}
}
void mouseButton(GLFWwindow*,int button,int action,int){if(button==GLFW_MOUSE_BUTTON_LEFT)mouseDown=action==GLFW_PRESS;}
void cursorPos(GLFWwindow*,double x,double y){
    if(mouseDown){if(!camera.dragging){camera.lastX=x;camera.lastY=y;camera.dragging=true;}
        camera.yaw+=static_cast<float>(x-camera.lastX)*0.006f;camera.pitch=std::clamp(camera.pitch-static_cast<float>(y-camera.lastY)*0.006f,0.08f,1.25f);
    }else camera.dragging=false;camera.lastX=x;camera.lastY=y;
}
void scroll(GLFWwindow*,double,double y){camera.distance=std::clamp(camera.distance-static_cast<float>(y)*0.8f,4.0f,25.0f);}
Vec3 limbPoint(Vec3 origin,float length,float angle,float z=0){return {origin.x+std::sin(angle)*length,origin.y-std::cos(angle)*length,origin.z+z};}
void drawMesh(GLuint program,const Mesh& mesh,const Mat4& model,Vec3 color,GLint modelLoc,GLint colorLoc){
    glUniformMatrix4fv(modelLoc,1,GL_FALSE,model.m.data());glUniform3f(colorLoc,color.x,color.y,color.z);
    glBindVertexArray(mesh.vao);glDrawArrays(mesh.mode,0,mesh.count);
}
void ellipsoid(GLuint p,const Mesh& sphere,Vec3 a,Vec3 b,float width,Vec3 color,GLint modelLoc,GLint colorLoc){
    Vec3 d=b-a,mid=(a+b)*0.5f;float len=std::sqrt(dot(d,d));
    // Limb capsules are approximated with oriented ellipsoids; the orientation is in the X/Y plane.
    float angle=std::atan2(d.x,d.y);Mat4 rot=identity();rot.m[0]=std::cos(angle);rot.m[1]=-std::sin(angle);rot.m[4]=std::sin(angle);rot.m[5]=std::cos(angle);
    Mat4 model=multiply(translate(mid),multiply(rot,scale({width,len*0.62f,width})));
    drawMesh(p,sphere,model,color,modelLoc,colorLoc);
}
void drawRunner(GLuint p,const Mesh& sphere,float x,float phase,GLint modelLoc,GLint colorLoc){
    float bounce=0.08f+0.10f*(0.5f+0.5f*std::cos(2*phase));
    Vec3 hip{x,1.05f+bounce,0},chest{x-0.08f,1.95f+bounce,0},head{x-0.12f,2.32f+bounce,0};
    Vec3 dark{0.10f,0.18f,0.28f},shirt{0.90f,0.35f,0.12f},skin{0.78f,0.54f,0.35f},shoe{0.08f,0.08f,0.10f};
    ellipsoid(p,sphere,hip,chest,0.24f,shirt,modelLoc,colorLoc);
    Mat4 headModel=multiply(translate(head),scale({0.22f,0.27f,0.22f}));drawMesh(p,sphere,headModel,skin,modelLoc,colorLoc);
    for(int side=0;side<2;++side){float offset=side?PI:0;float legA=0.72f*std::sin(phase+offset);float kneeA=-0.35f-0.55f*std::max(0.0f,std::cos(phase+offset));
        Vec3 knee=limbPoint(hip,0.57f,legA,side?0.12f:-0.12f);Vec3 foot=limbPoint(knee,0.52f,legA+kneeA,side?0.12f:-0.12f);
        ellipsoid(p,sphere,hip,knee,0.13f,dark,modelLoc,colorLoc);ellipsoid(p,sphere,knee,foot,0.10f,dark,modelLoc,colorLoc);
        Vec3 toe=foot+Vec3{0.24f,0,0};ellipsoid(p,sphere,foot,toe,0.10f,shoe,modelLoc,colorLoc);
        float armA=0.68f*std::sin(phase+offset+PI);Vec3 shoulder{x-0.08f,1.88f+bounce,side?0.20f:-0.20f};
        Vec3 elbow=limbPoint(shoulder,0.43f,armA,0);Vec3 hand=limbPoint(elbow,0.37f,armA+0.7f,0);
        ellipsoid(p,sphere,shoulder,elbow,0.09f,shirt,modelLoc,colorLoc);ellipsoid(p,sphere,elbow,hand,0.075f,skin,modelLoc,colorLoc);
    }
}

void processMovement(float dt){float speed=3.2f*dt;if(keys[GLFW_KEY_A]||keys[GLFW_KEY_LEFT])runnerX-=speed;if(keys[GLFW_KEY_D]||keys[GLFW_KEY_RIGHT])runnerX+=speed;runnerX=std::clamp(runnerX,-10.0f,10.0f);if(keys[GLFW_KEY_W]||keys[GLFW_KEY_UP])camera.pitch=std::min(1.15f,camera.pitch+dt);if(keys[GLFW_KEY_S]||keys[GLFW_KEY_DOWN])camera.pitch=std::max(0.10f,camera.pitch-dt);}
}

int main(){
    if(!glfwInit()){std::cerr<<"Could not initialize GLFW.\n";return 1;}
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,GL_TRUE);
#endif
    GLFWwindow* window=glfwCreateWindow(WINDOW_W,WINDOW_H,"Field Run | Computer Graphics Project",nullptr,nullptr);
    if(!window){std::cerr<<"Could not create an OpenGL 3.3 window. Check graphics drivers.\n";glfwTerminate();return 1;}
    glfwMakeContextCurrent(window);glfwSwapInterval(1);
    if(!gladLoadGL(glfwGetProcAddress)){std::cerr<<"Could not load OpenGL functions. Update the graphics driver.\n";glfwDestroyWindow(window);glfwTerminate();return 1;}
    glfwSetKeyCallback(window,keyCallback);glfwSetMouseButtonCallback(window,mouseButton);glfwSetCursorPosCallback(window,cursorPos);glfwSetScrollCallback(window,scroll);
    glEnable(GL_DEPTH_TEST);glEnable(GL_CULL_FACE);glCullFace(GL_BACK);glClearColor(0.53f,0.76f,0.91f,1);

    const std::string shaderDir=SHADER_DIR;
    GLuint program=makeProgram(readText(shaderDir+"/scene.vert"),readText(shaderDir+"/scene.frag"));
    if(!program){glfwDestroyWindow(window);glfwTerminate();return 1;}
    Mesh terrain=makeTerrain(),grass=makeGrass(),trees=makeTrees(),sphere=makeSphere(),trunk=makeCylinder();
    GLint modelLoc=glGetUniformLocation(program,"uModel"),viewLoc=glGetUniformLocation(program,"uView"),projectionLoc=glGetUniformLocation(program,"uProjection"),colorLoc=glGetUniformLocation(program,"uTint"),lightLoc=glGetUniformLocation(program,"uLightDirection"),eyeLoc=glGetUniformLocation(program,"uEyePosition");
    float previous=static_cast<float>(glfwGetTime());
    while(!glfwWindowShouldClose(window)){
        float now=static_cast<float>(glfwGetTime()),dt=std::min(now-previous,0.04f);previous=now;processMovement(dt);runTime+=dt;
        if(keys[GLFW_KEY_A]||keys[GLFW_KEY_D]||keys[GLFW_KEY_LEFT]||keys[GLFW_KEY_RIGHT])runTime+=dt*2.0f;
        int width,height;glfwGetFramebufferSize(window,&width,&height);glViewport(0,0,width,height);glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        Vec3 target{runnerX,1.0f,0};Vec3 eye=target+Vec3{std::sin(camera.yaw)*std::cos(camera.pitch)*camera.distance,std::sin(camera.pitch)*camera.distance,std::cos(camera.yaw)*std::cos(camera.pitch)*camera.distance};
        Mat4 view=lookAt(eye,target,{0,1,0}),projection=perspective(PI/3.2f,static_cast<float>(width)/std::max(height,1),0.1f,100.0f);
        glUseProgram(program);glUniformMatrix4fv(viewLoc,1,GL_FALSE,view.m.data());glUniformMatrix4fv(projectionLoc,1,GL_FALSE,projection.m.data());glUniform3f(lightLoc,-0.45f,-1.0f,-0.30f);glUniform3f(eyeLoc,eye.x,eye.y,eye.z);
        drawMesh(program,terrain,identity(),{1,1,1},modelLoc,colorLoc);drawMesh(program,grass,identity(),{1,1,1},modelLoc,colorLoc);drawMesh(program,trees,identity(),{1,1,1},modelLoc,colorLoc);
        (void)trunk;
        const float phase=runTime*7.0f;drawRunner(program,sphere,runnerX,phase,modelLoc,colorLoc);
        glfwSwapBuffers(window);glfwPollEvents();
    }
    for(Mesh* m:{&terrain,&grass,&trees,&sphere,&trunk}){glDeleteBuffers(1,&m->vbo);glDeleteVertexArrays(1,&m->vao);}
    glDeleteProgram(program);glfwDestroyWindow(window);glfwTerminate();return 0;
}
