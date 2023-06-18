#ifndef __PHYSICS_H_
#define __PHYSICS_H_

#include "device_launch_parameters.h" // for kernel vars blockIdx and etc.

struct particle;

enum physics_type {
	SPACE,
	EARTH_PHYSICS
};


struct space_physics {
public:
	__device__ void affect(particle* p) {};
};

struct erath_physics {
public:
	__device__	void affect(particle* p);

private:
	float g = 9.8;
};

class physics_manager {
public:
	__device__ physics_manager() {};
	__device__ ~physics_manager() {};
	__device__	void physicsMakeAction(particle* p);

private:
	//space_physics space_physics;
	//erath_physics erath_physics;
};









#endif __PHYSICS_H_