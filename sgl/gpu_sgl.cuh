
#pragma once

#define BLOCK_DIMENSION 16
#define FLOAT_MAX 300000000000000000000.0f

#define EPSILON 0.00000001f
#define CUDA_DEBUG false

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "primitives.cuh"
#include "kd_tree.h"



struct find_struct {

	int object;
	float dist;

};


struct  traversal_structure_linear
{
	int node;
	float mint;
	float maxt;

	__host__ __device__ traversal_structure_linear() {
	}

	__host__ __device__ traversal_structure_linear(int node, float mint, float maxt) {
		this->node = node;
		this->mint = mint;
		this->maxt = maxt;
	}
};




struct gpu_data {
	float* bitmap;
	float* invMatrix;
	float* rayStart;
	Triangle* scene_triangles;
	Material* materials;
	PointLight* lights;
	kdNodeLinear* kdtree;
	int* kd_node_triangles;
};

gpu_data cudaInit(int resolution, int triangles_num, int materials_num, int lights_num, int nodes_num, float* invMatrix, kdNodeLinear *kd_tree, Triangle* scene_triangles_array, Material* materials, PointLight* lights, float* ray_start, int triangles_kd, int* kd_indices_array);

void cudaDelete( gpu_data data, float* bitmap, int resolution, int nodes_num, kdNodeLinear* kd_tree);

__device__ find_struct FindKDIntersectionGPU(RayLinear ray, Voxel V, kdNodeLinear *kd_tree, Triangle* scene_triangles_array, int depth);

__host__ void IlluminateKernelCPU(int x, int y, float* ray_start, float *invMatrix, int width, int height, int index, Voxel V, kdNodeLinear *kd_tree, Triangle* scene_triangles_array, Material* materials, PointLight* lights, int lights_len, float* color_buffer, int depth);

//__global__ void IlluminateKernel(float* ray_start, float *invMatrix, int width, int height, Voxel V, kdNodeLinear *kd_tree, Triangle* scene_triangles_array, Material* materials, PointLight* lights, int lights_len, float* color_buffer);

void KernelStart(float* ray_start, float *invMatrix, int width, int height, Voxel V, kdNodeLinear *kd_tree, Triangle* scene_triangles_array, Material* materials, PointLight* lights, int lights_len, float* color_buffer, int depth, int triangles_n, int* kd_node_triangles);