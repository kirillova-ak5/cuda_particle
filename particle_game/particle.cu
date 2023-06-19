#include "particle.h"
#include "physics.h"
#include "math.h"

#include "cudaGL.h" // for kernel function surf2Dwrite
#include "device_launch_parameters.h" // for kernel vars blockIdx and etc.

__device__ __constant__ spawner_cbuf spawnersDevice;
__device__ __constant__ shapes_cbuf shapesDevice;
//__device__ __constant__ physics_manager phManager;


__global__ void Fill(cudaSurfaceObject_t s, dim3 texDim)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= texDim.x || y >= texDim.y)
  {
    return;
  }

//  uchar4 data = make_uchar4(255.0f * x / texDim.x, 255.0f * y / texDim.y, 0x00, 0xff);
  uchar4 data = make_uchar4(0x88, 0xBB, 0xBB, 0xff);
  surf2Dwrite(data, s, x * sizeof(uchar4), y);
}

__global__ void DrawShapes(cudaSurfaceObject_t s)
{
  unsigned int i = blockIdx.x;
  if (i > shapesDevice.nShapes)
    return;
  shape shp = shapesDevice.shapes[i];
}

__global__ void DrawParticles(cudaSurfaceObject_t s, particle* poolCur, dim3 texDim)
{
  unsigned int i = blockIdx.x;
  particle part = poolCur[i];
  unsigned int x = part.x;
  unsigned int y = part.y;

  if (x + 1 >= texDim.x || y + 1 >= texDim.y || part.type == PART_DEAD)
  {
    return;
  }
  float r = part.type == PART_FIRST ? 255 : 128;
  float g = part.type == PART_FIRST ? 255 : 64;
  float b = 0;
  uchar4 data = make_uchar4(r, g, b, 0xff);
  surf2Dwrite(data, s, x * sizeof(uchar4), y);
  surf2Dwrite(data, s, (x + 1) * sizeof(uchar4), y);
  surf2Dwrite(data, s, x * sizeof(uchar4), y + 1);
  surf2Dwrite(data, s, (x + 1) * sizeof(uchar4), y + 1);
}

__device__ float dist(particle x, particle y) 
{
    //return float(sqrt(pow((x.x - y.x), 2) + pow((x.y - y.y), 2)));
    return float(fabs(x.x - y.x) + fabs(x.y - y.y)); //trying to accelerate app. Manhattan's metrics
}

__device__ void CollisionCheck(particle* poolPrev, particle* poolCur, int maxParticles)
{
    unsigned int i = blockIdx.x;
    float t;
    for (int j = i+1; j < maxParticles; j++)
    {
        if (poolPrev[i].type != PART_DEAD && poolPrev[j].type != PART_DEAD)
        {
            if (poolPrev[i].type != poolPrev[j].type)
            {
                if (dist(poolPrev[i], poolPrev[j]) < 2) 
                {
                    //first method
                    /*t = poolPrev[i].vx;
                    poolPrev[i].vx = 0.5 * (poolPrev[j].vx + t);
                    poolPrev[j].vx = 0.5 * (poolPrev[j].vx + t);

                    t = poolPrev[i].vy;
                    poolPrev[i].vy = 0.5 * (poolPrev[j].vy + t);
                    poolPrev[j].vy = 0.5 * (poolPrev[j].vy + t);*/

                    //second method
                    t = poolPrev[i].vx;
                    poolPrev[i].vx = poolPrev[j].vx;
                    poolPrev[j].vx = t;

                    t = poolPrev[i].vy;
                    poolPrev[i].vy = poolPrev[j].vy;
                    poolPrev[j].vy = t;

                    //third method
                    /*poolPrev[i].vx += poolPrev[j].vx;
                    poolPrev[i].vy += poolPrev[j].vy;
                    poolPrev[j].vx = 2 * poolPrev[j].vx + poolPrev[i].vx;
                    poolPrev[j].vy = 2 * poolPrev[j].vy + poolPrev[i].vy;*/

                }
            }
        }
        
    }
}

__global__ void Update(particle* poolPrev, particle* poolCur, double timeDelta, int maxParticles)
{
    unsigned int i = blockIdx.x;
    //phManager.physicsMakeAction(&poolCur[i]);
    if (poolCur[i].type == PART_DEAD)
      return;
    poolCur[i].vy -= 0.00015 * timeDelta;   //уберу константу потом, когда решится вопрос с физикой
    poolCur[i].x = poolPrev[i].x + poolPrev[i].vx * timeDelta;
    poolCur[i].y = poolPrev[i].y + poolPrev[i].vy * timeDelta;
    CollisionCheck(poolPrev, poolCur, maxParticles);
    poolCur[i].remainingAliveTime = max(poolPrev[i].remainingAliveTime - timeDelta, 0.f);
    poolCur[i].type = poolPrev[i].remainingAliveTime > 0 ? poolPrev[i].type : PART_DEAD;
}

__device__ unsigned seed = 123456789;
__device__ unsigned random(void)
{
  unsigned a = 1103515245;
  unsigned c = 12345;
  unsigned m = 1 << 31;
  seed = (a * seed + c) % m;
  return seed;
}
__global__ void Spawn(particle* poolCur, int maxParticles)
{
  int startSlot = 0;
  for (int i = 0; i < spawnersDevice.nSpawners; i++)
  {
    spawner sp = spawnersDevice.spawners[i];
    int numToSpawn = sp.intensity;
    for (int j = 0; j < numToSpawn; j++)
      for (int k = startSlot; k < maxParticles; k++) // max particles here
        if (poolCur[k].type == PART_DEAD)
        {

          particle p = { sp.x, sp.y, sp.vx + (random() % sp.directionsCount) * sp.spread, sp.vy + (random() % sp.directionsCount) * sp.spread, 
                            sp.type, sp.particleAliveTime, sp.particleAliveTime, sp.phType };

          poolCur[k] = p;
          startSlot = k + 1;
          break;
        }
  }
}

void part_mgr::Compute(cudaSurfaceObject_t s, dim3 texSize, double timeDelta)
{
  cudaMemcpyToSymbol(shapesDevice, &shapesHost, sizeof(shapes_cbuf));

  dim3 thread(1);
  dim3 block(MAX_PARTICLES);
  dim3 oneBlock(1);

  Spawn<<< oneBlock, thread >>>(partPoolCur, MAX_PARTICLES);
  Update<<< block, thread >>>(partPoolCur, partPoolCur, timeDelta, MAX_PARTICLES);
  DrawParticles <<< block, thread >>>(s, partPoolCur, texSize);

}

void part_mgr::Init(void)
{
    cudaError_t cudaStatus = cudaSuccess;
    particle tmp[MAX_PARTICLES];
    for (int i = 0; i < MAX_PARTICLES; i++)
    {
        particle p = { 0, 0, 0, 0, PART_DEAD, 0, 0, SPACE };
        tmp[i] = p;
    }
    cudaStatus = cudaMalloc(&partPoolCur, sizeof(particle) * MAX_PARTICLES);
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "failed!");
    //cudaStatus = cudaMalloc(&partPoolCur, sizeof(particle) * MAX_PARTICLES);
    //if (cudaStatus != cudaSuccess)
    //    fprintf(stderr, "failed!");

    cudaMemcpy(partPoolCur, tmp, sizeof(particle) * MAX_PARTICLES, cudaMemcpyHostToDevice);
    //cudaMemcpy(partPoolCur, tmp, sizeof(particle) * MAX_PARTICLES, cudaMemcpyHostToDevice);

    spawnersHost.nSpawners = 3;
    spawnersHost.spawners[0] = { 600, 500, -0.35, 0.35, PART_FIRST, 0.005, 1, 8, 3000, EARTH_PHYSICS };
    spawnersHost.spawners[1] = {700, 700, -0.25, -0.15, PART_SECOND, -0.008, 2, 10, 3000, SPACE};
    //spawnersHost.spawners[2] = {500, 500, -0.00015, -0.00015, PART_SECOND, -0.05, 3, 10, 1500, SPACE};
    cudaMemcpyToSymbol(spawnersDevice, &spawnersHost, sizeof(spawner_cbuf));
}

void part_mgr::Kill(void)
{
  cudaError_t cudaStatus = cudaSuccess;

  cudaStatus = cudaFree(partPoolCur);
  if (cudaStatus != cudaSuccess)
    fprintf(stderr, "failed!");
  //cudaStatus = cudaFree(partPoolCur);
  //if (cudaStatus != cudaSuccess)
   // fprintf(stderr, "failed!");

}

void part_mgr::AddCircle(float cx, float cy, float radius)
{
  if (shapesHost.nShapes == MAX_SHAPES)
    return;
  shapesHost.shapes[shapesHost.nShapes] = { SHAPE_CIRCLE, cx, cy, radius, 0 };
  shapesHost.nShapes++;
}

void part_mgr::AddSquare(float x1, float y1, float x2, float y2)
{
  if (shapesHost.nShapes == MAX_SHAPES)
    return;
  shapesHost.shapes[shapesHost.nShapes] = { SHAPE_SQUARE, x1, y1, x2, y2 };
  shapesHost.nShapes++;
}

void part_mgr::AddSegment(float x1, float y1, float x2, float y2)
{
  if (shapesHost.nShapes == MAX_SHAPES)
    return;
  shapesHost.shapes[shapesHost.nShapes] = { SHAPE_SEGMENT, x1, y1, x2, y2 };
  shapesHost.nShapes++;
}

const shapes_cbuf& part_mgr::GetShapes(void)
{
  return shapesHost;
}

inline int sqr(int x)
{
  return x * x;
}

inline bool btw(int x, int x1, int x2)
{
  return x1 < x2 ? (x >= x1 && x <= x2) : (x <= x1 && x >= x2);
}

inline float dist(int x0, int y0, int x1, int y1, int x2, int y2)
{
  return (float)abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / sqrt(sqr(y2 - y1) + sqr(x2 - x1));
}

int part_mgr::SelectShape(int x, int y)
{
  for (int i = 0; i < shapesHost.nShapes; i++)
  {
    shape& shp = shapesHost.shapes[i];
    switch (shp.type)
    {
    case SHAPE_CIRCLE:
      if (sqr(x - shp.params[0]) + sqr(y - shp.params[1]) <= sqr(shp.params[2]))
        return i;
      break;
    case SHAPE_SQUARE:
      if (btw(x, shp.params[0], shp.params[2]) && btw(y, shp.params[1], shp.params[3]))
        return i;
    case SHAPE_SEGMENT:
      if (btw(x, shp.params[0], shp.params[2]) && btw(y, shp.params[1], shp.params[3]) &&
        dist(x, y, shp.params[0], shp.params[1], shp.params[2], shp.params[3]) < 5)
        return i;
        break;
    default:
      break;
    }
  }
  return -1;
}

void part_mgr::MoveShape(int shapeHandle, int dx, int dy)
{
  if (shapeHandle < 0 || shapeHandle >= shapesHost.nShapes)
    return;
  shape& shp = shapesHost.shapes[shapeHandle];
  switch (shp.type)
  {
  case SHAPE_CIRCLE:
    shp.params[0] += dx;
    shp.params[1] += dy;
    break;
  case SHAPE_SQUARE:
  case SHAPE_SEGMENT:
    shp.params[0] += dx;
    shp.params[1] += dy;
    shp.params[2] += dx;
    shp.params[3] += dy;
    break;
  default:
    break;
  }
}

