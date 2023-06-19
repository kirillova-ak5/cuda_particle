#include "physics.cuh"
#include "particle.cuh"


__device__ void erath_physics::affect(particle* p)
{
	p->vy += (p->originAliveTime - p->remainingAliveTime) * g;
}

__device__ void physics_manager::physicsMakeAction(particle* p)
{
	switch (p->phType) {
		//case SPACE: {space_physics.affect(p); break; }
		//case EARTH_PHYSICS: {erath_physics.affect(p); break; }
	default: {}
	}
}