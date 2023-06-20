#include "particle.cuh"
#include "physics.cuh"
#include "math.h"

#include "cudaGL.h" // for kernel function surf2Dwrite
#include "device_launch_parameters.h" // for kernel vars blockIdx and etc.

//
//  basics kernels for particles
//
__global__ void Fill(cudaSurfaceObject_t s, dim3 texDim)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= texDim.x || y >= texDim.y)
  {
    return;
  }

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
    return;

  float r = 0, g = 0, b = 0;
  switch (part.type)
  {
  case PART_FIRST: {r = 255; g = 255; break; }
  case PART_SECOND: {r = 128; g = 64; break; }
  case PART_THIRD: {g = 128; break; }
  }

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

__device__ void CollisionCheck(particle* poolCur, int maxParticles)
{
  unsigned int i = blockIdx.x;
  float t;
  for (int j = i + 1; j < maxParticles; j++)
    if (poolCur[i].type != PART_DEAD && poolCur[j].type != PART_DEAD)
      if (poolCur[i].type != poolCur[j].type)
        if (dist(poolCur[i], poolCur[j]) < 2)
        {
          t = poolCur[i].vx;
          poolCur[i].vx = poolCur[j].vx;
          poolCur[j].vx = t;

          t = poolCur[i].vy;
          poolCur[i].vy = poolCur[j].vy;
          poolCur[j].vy = t;
        }
}

__global__ void Update(particle* poolCur, double timeDelta, int maxParticles, dim3 texDim)
{
  unsigned int i = blockIdx.x;
  //phManager.physicsMakeAction(&poolCur[i]);
  if (poolCur[i].x + 1 >= texDim.x || poolCur[i].y + 1 >= texDim.y)
    poolCur[i].type = PART_DEAD;
  if (poolCur[i].type == PART_DEAD)
    return;

  ShapesCollisionCheck(&poolCur[i], timeDelta);
  poolCur[i].vy -= 0.00015 * timeDelta;   // IDD: I will remove const soon. After physics discussion
  poolCur[i].x = poolCur[i].x + poolCur[i].vx * timeDelta;
  poolCur[i].y = poolCur[i].y + poolCur[i].vy * timeDelta;

  CollisionCheck(poolCur, maxParticles);
  poolCur[i].remainingAliveTime = max(poolCur[i].remainingAliveTime - timeDelta, 0.f);
  poolCur[i].type = poolCur[i].remainingAliveTime > 0 ? poolCur[i].type : PART_DEAD;
}

__device__ bool btwDev(int x, int x1, int x2)
{
  return x1 < x2 ? (x >= x1 && x <= x2) : (x <= x1 && x >= x2);
}
__global__ void CheckBasket(particle* poolCur, int maxParticles)
{
  __shared__ int locNum;
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0)
    locNum = 0;
  __syncthreads();

  if (x >= maxParticles)
    return;
  particle p = poolCur[x];
  if (p.type == PART_DEAD)
    return;

  if (btwDev(p.x, basketDevice.x1, basketDevice.x2) && btwDev(p.y, basketDevice.y1, basketDevice.y2))
    atomicAdd(&locNum, 1);
  __syncthreads();

  if (threadIdx.x == 0)
    atomicAdd(&inbasketParticlesCount, locNum);
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

//
// part_mgr methods - system lifecycle
//
void part_mgr::Compute(cudaSurfaceObject_t s, dim3 texSize, double timeDelta)
{
  cudaMemcpyToSymbol(shapesDevice, &shapesHost, sizeof(shapes_cbuf));
  numInbasket = 0;
  cudaMemcpyToSymbol(inbasketParticlesCount, &numInbasket, sizeof(int));

  dim3 thread(1);
  dim3 block(MAX_PARTICLES);
  dim3 oneBlock(1);

  Spawn << < oneBlock, thread >> > (partPoolCur, MAX_PARTICLES);
  Update << < block, thread >> > (partPoolCur, timeDelta, MAX_PARTICLES, texSize);
  CheckBasket << <dim3(MAX_PARTICLES / 32 + 1), dim3(32) >> > (partPoolCur, MAX_PARTICLES);
  DrawParticles << < block, thread >> > (s, partPoolCur, texSize);
  cudaMemcpyFromSymbol(&numInbasket, inbasketParticlesCount, sizeof(int));
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

  cudaMemcpy(partPoolCur, tmp, sizeof(particle) * MAX_PARTICLES, cudaMemcpyHostToDevice);

  spawnersHost.nSpawners = 3;
  spawnersHost.spawners[0] = { 600, 500, 0.35, 0.35, PART_FIRST, 0.005, 2, 8, 3000, EARTH_PHYSICS };
  spawnersHost.spawners[1] = { 700, 700, 0.25, -0.15, PART_SECOND, -0.008, 3, 10, 3000, SPACE };
  spawnersHost.spawners[2] = { 1000, 300, -0.15, 0.45, PART_THIRD, -0.005, 1, 10, 1500, SPACE };
  cudaMemcpyToSymbol(spawnersDevice, &spawnersHost, sizeof(spawner_cbuf));

  basketHost = { 300, 5, 400, 150 };
  cudaMemcpyToSymbol(basketDevice, &basketHost, sizeof(basket));
  float x1 = basketHost.x1, x2 = basketHost.x2, y1 = basketHost.y1, y2 = basketHost.y2;
  shapesHost.shapes[MAX_SHAPES] = { SHAPE_SEGMENT, x1, y2, x1, y1 };
  shapesHost.shapes[MAX_SHAPES + 1] = { SHAPE_SEGMENT, x2, y1, x1, y1 };
  shapesHost.shapes[MAX_SHAPES + 2] = { SHAPE_SEGMENT, x2, y2, x2, y1 };

}

void part_mgr::Kill(void)
{
  cudaError_t cudaStatus = cudaSuccess;

  cudaStatus = cudaFree(partPoolCur);
  if (cudaStatus != cudaSuccess)
    fprintf(stderr, "failed!");
}

//
// shapes handling
//
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

//
// collision handling
//
struct pt
{
  float x, y;
};

__device__ void ShapesCollisionCheck(particle* part, double timeDelta)
{
  float shiftX, shiftY;
  shiftX = part->vx * timeDelta;
  shiftY = part->vy * timeDelta;
  shape sh;

  for (int i = 0; i < shapesDevice.nShapes; i++)
  {
    sh = shapesDevice.shapes[i];
    switch (sh.type)
    {
    case SHAPE_SQUARE: {SquareCollision(&sh, part, shiftX, shiftY); break; }
    case SHAPE_SEGMENT: {SegmentCollision(&sh, part, shiftX, shiftY); break; }
    case SHAPE_CIRCLE: {CircleCollision(&sh, part, shiftX, shiftY); break; }
    }
  }
  for (int i = 20; i < 23; i++)
  {
    sh = shapesDevice.shapes[i];
    switch (sh.type)
    {
    case SHAPE_SEGMENT: {SegmentCollision(&sh, part, shiftX, shiftY); break; }
    }
  }
}

__device__ void CircleCollision(shape* shape, particle* part, float shiftX, float shiftY)
{
  if (pow(part->y + shiftY - shape->params[1], 2) + pow(part->x + shiftX - shape->params[0], 2) <= shape->params[2] * shape->params[2])
  {
    pt prtcl = { part->vx, part->vy };
    pt norm = { shape->params[0] - part->x, shape->params[1] - part->y };
    float len = sqrt(pow(norm.x, 2) + pow(norm.y, 2));
    norm = { norm.x / len, norm.y / len };
    float t = 2 * (prtcl.x * norm.x + prtcl.y * norm.y);
    norm = { t * norm.x, t * norm.y };
    pt res = { prtcl.x - norm.x, prtcl.y - norm.y };
    part->vx = res.x;
    part->vy = res.y;
  }
}

__device__ void SquareCollision(shape* shape, particle* part, float shiftX, float shiftY)
{
  float newX = part->vx, newY = part->vy;
  if (part->x + shiftX <= shape->params[0] && part->x + shiftX >= shape->params[2]
    && part->y <= shape->params[1] && part->y >= shape->params[3])
  {
    newX *= -0.8;   //slowdown after hit
  }

  if (part->y + shiftY <= shape->params[1] && part->y + shiftY >= shape->params[3]
    && part->x <= shape->params[0] && part->x >= shape->params[2])
  {
    newY *= -0.8;    //slowdown after hit
  }

  part->vx = newX;
  part->vy = newY;
}

__device__  int area(pt a, pt b, pt c)
{
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

__device__  bool intersect_1(float a, float b, float c, float d)
{
  float t;
  if (a > b)
  {
    t = a;
    a = b;
    b = t;
  }
  if (c > d)
  {
    t = c;
    c = d;
    d = t;
  }
  return max(a, c) <= min(b, d);
}

__device__ int sign(float x)
{
  float eps = 1e-1;
  return x > eps ? 1 : (x < eps ? -1 : 0);
}

__device__ bool intersect(pt a, pt b, pt c, pt d)
{

  return intersect_1(a.x, b.x, c.x, d.x)
    && intersect_1(a.y, b.y, c.y, d.y)
    && sign(area(a, b, c)) * sign(area(a, b, d)) <= 0
    && sign(area(c, d, a)) * sign(area(c, d, b)) <= 0;
}

__device__ void SegmentCollision(shape* shape, particle* part, float shiftX, float shiftY)
{
  if (intersect(pt{ shape->params[0], shape->params[1] }, pt{ shape->params[2], shape->params[3] },
    pt{ part->x, part->y }, pt{ part->x + shiftX, part->y + shiftY }))
  {
    pt prtcl = { part->vx, part->vy };
    pt norm = { shape->params[1] - shape->params[3], -shape->params[0] + shape->params[2] };
    float side = sign(area(pt{ shape->params[0], shape->params[1] },
      pt{ shape->params[2], shape->params[3] }, pt{ part->x, part->y }));
    norm = { norm.x * side, norm.y * side };

    float len = sqrt(pow(norm.x, 2) + pow(norm.y, 2));
    norm = { norm.x / len, norm.y / len };
    float t = 2 * (prtcl.x * norm.x + prtcl.y * norm.y);
    norm = { t * norm.x, t * norm.y };
    pt res = { prtcl.x - norm.x, prtcl.y - norm.y };
    part->vx = res.x * 0.8;
    part->vy = res.y * 0.8;
  }
}




