#include "particle.h"

#include "cudaGL.h" // for kernel function surf2Dwrite
#include "device_launch_parameters.h" // for kernel vars blockIdx and etc.

__device__ __constant__ spawner_cbuf spawnersDevice;


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

__global__ void Update(particle* poolPrev, particle* poolCur, double timeDelta)
{
  unsigned int i = blockIdx.x;
  poolCur[i].x = poolPrev[i].x + poolPrev[i].vx * timeDelta;
  poolCur[i].y = poolPrev[i].y + poolPrev[i].vy * timeDelta;
  poolCur[i].aliveTime = max(poolPrev[i].aliveTime - timeDelta, 0.f);
  poolCur[i].type = poolPrev[i].aliveTime > 0 ? poolPrev[i].type : PART_DEAD;
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
__global__ void Spawn(particle* poolCur)
{
  int startSlot = 0;
  for (int i = 0; i < spawnersDevice.nSpawners; i++)
  {
    spawner sp = spawnersDevice.spawners[i];
    int numToSpawn = sp.intensity;
    for (int j = 0; j < numToSpawn; j++)
      for (int k = startSlot; k < 2000; k++) // max particles here
        if (poolCur[k].type == PART_DEAD)
        {
          particle p = { sp.x, sp.y, sp.vx + (random() % 10) * sp.spread, sp.vy + (random() % 10) * sp.spread, sp.type, 2000 };
          poolCur[k] = p;
          startSlot = k + 1;
          break;
        }
  }
}

void part_mgr::Compute(cudaSurfaceObject_t s, dim3 texSize, double timeDelta)
{
  dim3 thread(1);
  dim3 block(MAX_PARTICLES);
  dim3 oneBlock(1);
  Spawn<<< oneBlock, thread >>>(partPoolPrev);
  Update<<< block, thread >>>(partPoolPrev, partPoolCur, timeDelta);
  DrawParticles <<< block, thread >>>(s, partPoolCur, texSize);

  particle* tmp = partPoolPrev;
  partPoolPrev = partPoolCur;
  partPoolCur = tmp;
}

void part_mgr::Init(void)
{
  cudaError_t cudaStatus = cudaSuccess;
  particle tmp[MAX_PARTICLES];
  for (int i = 0; i < MAX_PARTICLES; i++)
  {
    particle p = { 0, 0, 0, 0, PART_DEAD, 0 };
    tmp[i] = p;
  }
  cudaStatus = cudaMalloc(&partPoolCur, sizeof(particle) * MAX_PARTICLES);
  if (cudaStatus != cudaSuccess)
    fprintf(stderr, "failed!");
  cudaStatus = cudaMalloc(&partPoolPrev, sizeof(particle) * MAX_PARTICLES);
  if (cudaStatus != cudaSuccess)
    fprintf(stderr, "failed!");

  cudaMemcpy(partPoolCur, tmp, sizeof(particle) * MAX_PARTICLES, cudaMemcpyHostToDevice);
  cudaMemcpy(partPoolPrev, tmp, sizeof(particle) * MAX_PARTICLES, cudaMemcpyHostToDevice);

  spawnersHost.nSpawners = 2;
  spawnersHost.spawners[0] = { 100, 100, 0.001, 0.001, PART_FIRST, 0.02, 2 };
  spawnersHost.spawners[1] = { 500, 500, -0.001, -0.001, PART_SECOND, -0.08, 3 };
  cudaMemcpyToSymbol(spawnersDevice, &spawnersHost, sizeof(spawner_cbuf));
}

void part_mgr::Kill(void)
{
  cudaError_t cudaStatus = cudaSuccess;

  cudaStatus = cudaFree(partPoolCur);
  if (cudaStatus != cudaSuccess)
    fprintf(stderr, "failed!");
  cudaStatus = cudaFree(partPoolPrev);
  if (cudaStatus != cudaSuccess)
    fprintf(stderr, "failed!");

}


