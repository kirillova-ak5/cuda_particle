#include "INCLUDE/glatter/glatter.h"
#include "INCLUDE/GL/freeglut.h"

#include "cuda_runtime.h" // for cuda functions on host
#include "cuda_gl_interop.h" // for cuda function about mapping OpenGL resources


enum part_type
{
  PART_DEAD,
  PART_FIRST,
  PART_SECOND
};
struct particle
{
  float x, y;   // position
  float vx, vy; // velocity
  part_type type;
  float aliveTime; // remain time
};

struct spawner
{
  float x, y;
  float vx, vy;
  part_type type;
  float spread;
  float intensity;
};
struct spawner_cbuf
{
  int nSpawners;
  spawner spawners[20];
};

class part_mgr
{
  particle* partPoolCur;
  particle* partPoolPrev;
  spawner_cbuf spawnersHost;

public:
  static const int MAX_PARTICLES = 2000;
  static const int MAX_SPAWNERS = 20;

  void Init(void);
  void Compute(cudaSurfaceObject_t s, dim3 texSize);
  void Kill(void);
};

__global__ void Fill(cudaSurfaceObject_t s, dim3 texDim);