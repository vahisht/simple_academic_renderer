
#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "primitives.h"
#include "kd_tree.h"



struct find_struct {

	int object;
	float dist;

};

__host__ find_struct FindKDIntersectionGPU(RayLinear ray, Voxel V, kdNodeLinear *kd_tree, Triangle* scene_triangles_array);

__host__ void IlluminateKernel(int x, int y, float* ray_start, float *invMatrix, int width, int height, int index, Voxel V, kdNodeLinear *kd_tree, Triangle* scene_triangles_array, Material* materials, PointLight* lights, int lights_len, float* color_buffer);

