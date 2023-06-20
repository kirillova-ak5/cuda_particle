#pragma once
#include "INCLUDE/glatter/glatter.h"
#include "INCLUDE/GL/freeglut.h"
#include "physics.cuh"

#include "cuda_runtime.h" // for cuda functions on host
#include "cuda_gl_interop.h" // for cuda function about mapping OpenGL resources

enum physics_type;

enum part_type
{
  PART_DEAD,
  PART_FIRST,
  PART_SECOND,
  PART_THIRD,
};

struct particle
{
  float x, y;   // position
  float vx, vy; // velocity
  part_type type;
  float remainingAliveTime; // remain time in miliseconds
  float originAliveTime;	// ms
  physics_type phType;

};

struct spawner
{
  float x, y;
  float vx, vy;
  part_type type;
  float spread;
  float intensity;
  int directionsCount;
  float particleAliveTime; //in ms
  physics_type phType;
};
struct spawner_cbuf
{
  int nSpawners;
  spawner spawners[20];
};

enum shape_type
{
  SHAPE_CIRCLE,
  SHAPE_SQUARE,
  SHAPE_SEGMENT
};

struct shape
{
  shape_type type;
  float params[4]; // x1 y1 x2 y2 for square and segment, (x1>x2, y1>y2)
                   //cx, cy, radius, 0 for circle
};

struct shapes_cbuf
{
  int nShapes;
  shape shapes[20 + 3]; // last 3 reservet for basket
};

struct basket
{
  int x1, y1, x2, y2;
};


class part_mgr
{
  particle* partPoolCur;
  spawner_cbuf spawnersHost;
  shapes_cbuf shapesHost;
  basket basketHost;
  int numInbasket;

public:
  static const int MAX_PARTICLES = 2000;
  static const int MAX_SPAWNERS = 20;
  static const int MAX_SHAPES = 20;
  static const int NUM_INBASKET_PARTS_TO_WIN = 30;

  void Init(void);
  void Compute(cudaSurfaceObject_t s, dim3 texSize, double timeDelta);
  void Kill(void);
  void AddCircle(float cx, float cy, float radius);
  void AddSquare(float x1, float y1, float x2, float y2);
  void AddSegment(float x1, float y1, float x2, float y2);
  const shapes_cbuf& GetShapes(void);
  int SelectShape(int x, int y); // returns shapeHandle
  void MoveShape(int shapeHandle, int dx, int dy);
  int GetInbasket(void) { return numInbasket; }
};

__device__ __constant__ spawner_cbuf spawnersDevice;
__device__ __constant__ shapes_cbuf shapesDevice;
__device__ __constant__ basket basketDevice;
__device__ int inbasketParticlesCount;

__global__ void Fill(cudaSurfaceObject_t s, dim3 texDim);
__device__ void SquareCollision(shape* shape, particle* part, float shiftX, float shiftY);
__device__ void CircleCollision(shape* shape, particle* part, float shiftX, float shiftY);
__device__ void SegmentCollision(shape* shape, particle* part, float shiftX, float shiftY);
__device__ void ShapesCollisionCheck(particle* part, double timeDelta);
