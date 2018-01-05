

#include "gpu_sgl.cuh"

gpu_data cudaInit(int resolution, int triangles_num, int materials_num, int lights_num, int nodes_num, float* invMatrix, kdNodeLinear *kd_tree, Triangle* scene_triangles_array, Material* materials, PointLight* lights, float* ray_start, int triangles_kd, int* kd_indices_array) {
	gpu_data result;
	cudaError_t cudaStatus;
	long total = 0;
	size_t free, total_mem;
	int offset = 0;

	// FINAL BITMAP
	cudaStatus = cudaMalloc((void**)&(result.bitmap), 3*resolution*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc bitmap failed!: %s\n", cudaGetErrorString(cudaStatus));
	}
	total += 3 * resolution*sizeof(float);


	// INVERSE PROJECTION MATRIX
	cudaStatus = cudaMalloc((void**)&(result.invMatrix), 16 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc invmatrix failed!: %s\n", cudaGetErrorString(cudaStatus));
	} total += 16 * sizeof(float);
	cudaStatus = cudaMemcpy(result.invMatrix, invMatrix, 16*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy invmatrix failed!: %s\n", cudaGetErrorString(cudaStatus));
	}


	// RAY START COORDINATES
	cudaStatus = cudaMalloc((void**)&(result.rayStart), 4 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc raystart failed!: %s\n", cudaGetErrorString(cudaStatus));
	} total += 4 * sizeof(float);
	cudaStatus = cudaMemcpy(result.rayStart, ray_start, 4 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy raystart failed!: %s\n", cudaGetErrorString(cudaStatus));
	}


	// TRIANGLES ARRAY
	cudaStatus = cudaMalloc((void**)&(result.scene_triangles), triangles_num * sizeof(Triangle));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc triangles failed!: %s\n", cudaGetErrorString(cudaStatus));
	} total += triangles_num * sizeof(Triangle);
	//cout << "Triangles: " << triangles_num  << " : " << sizeof(Triangle) << endl;
	cudaStatus = cudaMemcpy(result.scene_triangles, scene_triangles_array, triangles_num * sizeof(Triangle), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy triangles failed!: %s\n", cudaGetErrorString(cudaStatus));
	}


	// MATERIALS ARRAY
	cudaStatus = cudaMalloc((void**)&(result.materials), materials_num * sizeof(Material));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc materials failed!: %s\n", cudaGetErrorString(cudaStatus));
	} total += materials_num * sizeof(Material);
	cudaStatus = cudaMemcpy(result.materials, materials, materials_num * sizeof(Material), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy materials failed!: %s\n", cudaGetErrorString(cudaStatus));
	}


	// SCENE LIGHTS ARRAY
	cudaStatus = cudaMalloc((void**)&(result.lights), lights_num * sizeof(PointLight));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc lights failed!: %s\n", cudaGetErrorString(cudaStatus));
	} total += lights_num * sizeof(PointLight);
	cudaStatus = cudaMemcpy(result.lights, lights, lights_num * sizeof(PointLight), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy lights failed!: %s\n", cudaGetErrorString(cudaStatus));
	}


	// KD TREE NODE TRIANGLES ARRAYS
	cudaStatus = cudaMalloc((void**)&(result.kd_node_triangles), triangles_kd * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc kdtree nodes failed!: %s\n", cudaGetErrorString(cudaStatus));
	} total += triangles_kd * sizeof(int);
	cudaStatus = cudaMemcpy(result.kd_node_triangles, kd_indices_array, triangles_kd * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy kdtree failed!: %s\n", cudaGetErrorString(cudaStatus));
	}

	/*for (int i = 0; i < nodes_num; i++)
	{
		if (kd_tree[i].triangles == NULL) continue;
		
		cudaStatus = cudaMemcpy(result.kd_node_triangles + offset, kd_tree[i].triangles, kd_tree[i].len * sizeof(int), cudaMemcpyHostToDevice);
		kd_tree[i].gpu_offset = offset;
		offset += kd_tree[i].len;

		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy kdtree %d failed!: %s\n", i, cudaGetErrorString(cudaStatus));
		}
	}*/


	// KD TREE ARRAY
	cudaStatus = cudaMalloc((void**)&(result.kdtree), nodes_num * sizeof(kdNodeLinear));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc kdtree failed!: %s\n", cudaGetErrorString(cudaStatus));
	} total += nodes_num * sizeof(kdNodeLinear);
	cudaStatus = cudaMemcpy(result.kdtree, kd_tree, nodes_num * sizeof(kdNodeLinear), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy kdtree failed!: %s\n", cudaGetErrorString(cudaStatus));
	}

	cout << "[CUDA]\t\tAllocated total of " << total / 1000000 << "MB" << endl;

	/*cudaMemGetInfo(&free, &total_mem);

	cout << "[CUDA]\t\tGPU reports " << free / 1000000 << "MB free of total " << total_mem / 1000000 << "MB" << endl;*/

	return result;
}
void cudaDelete(gpu_data data, float* bitmap, int resolution, int nodes_num, kdNodeLinear* kd_tree) {
	cudaError cudaStatus;

	cudaStatus = cudaMemcpy(bitmap, data.bitmap ,resolution * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy result to host failed!: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaFree(data.bitmap);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree bitmap failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaFree(data.invMatrix);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree invMatrix failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaFree(data.rayStart);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree rayStart failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaFree(data.scene_triangles);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree triangles failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaFree(data.materials);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree materials failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaFree(data.lights);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree lights failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	for (int i = 0; i < nodes_num; i++)
	{
		cudaStatus = cudaFree(kd_tree[i].triangles_device);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaFree kdtree node %d failed!: %s\n", i, cudaGetErrorString(cudaStatus));
		} 
	}

	cudaStatus = cudaFree(data.kdtree);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree kdtree failed: %s\n", cudaGetErrorString(cudaStatus));
	}

};

find_struct FindKDIntersection(RayLinear ray, Voxel V, kdNodeLinear *kd_tree, Triangle* scene_triangles_array, int depth) {
	find_struct result;

	int top = -1;
	traversal_structure_linear* stacktrav = new traversal_structure_linear[depth + 2];
	traversal_structure_linear actual;

	bool found = false;

	result.dist = FLOAT_MAX;
	result.object = -1;

	float tmp = 0.0f;
	float t = FLOAT_MAX;

	int axis;
	float tmp1;
	float tmp2;
	float value;
	float ray_origin_axis = 1.0;
	float ray_dir_axis = 1.0;
	int near_child, far_child;


	if (!(fabs(ray.direction.x) < EPSILON)) {
		tmp1 = (V.position.x - ray.origin.x) / ray.direction.x;
		tmp2 = (V.position.x + V.dX - ray.origin.x) / ray.direction.x;

		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

	}

	if (!(fabs(ray.direction.y) < EPSILON)) {
		tmp1 = (V.position.y - ray.origin.y) / ray.direction.y;
		tmp2 = (V.position.y + V.dY - ray.origin.y) / ray.direction.y;

		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

	}

	if (!(fabs(ray.direction.z) < EPSILON)) {
		tmp1 = (V.position.z - ray.origin.z) / ray.direction.z;
		tmp2 = (V.position.z + V.dZ - ray.origin.z) / ray.direction.z;

		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

	}

	//cout << "Final voxel bounds are are: " << tmp << " " << t << endl;

	stacktrav[++top] = traversal_structure_linear(0, tmp, t);
	//std::cout << "Pushed in " << this->root << " " << tmp << " " << t << std::endl;

	while (top >= 0) {
		actual = stacktrav[top];
		//std::cout << "Popped out " << actual.w << " " << actual.mint << " " << actual.maxt << std::endl;
		top--;

		while (kd_tree[actual.node].splits) {
			//trav_steps++;
			axis = kd_tree[actual.node].p.dimension;
			value = kd_tree[actual.node].p.position;
			//std::cout << "Split plane " << value << " (" << axis << ")" << std::endl;

			switch (axis) {
			case 0:
				ray_origin_axis = ray.origin.x;
				ray_dir_axis = ray.direction.x;
				break;
			case 1:
				ray_origin_axis = ray.origin.y;
				ray_dir_axis = ray.direction.y;
				break;
			case 2:
				ray_origin_axis = ray.origin.z;
				ray_dir_axis = ray.direction.z;
				break;
			}


			if (value < ray_origin_axis) {
				near_child = kd_tree[actual.node].right;
				far_child = kd_tree[actual.node].left;
				//nearChild = actual.w->right;
				//farChild = actual.w->left;
			}
			else {
				near_child = kd_tree[actual.node].left;
				far_child = kd_tree[actual.node].right;
				//nearChild = actual.w->left;
				//farChild = actual.w->right;
			}


			if (!(fabs(ray_dir_axis) < EPSILON)) {
				t = (value - ray_origin_axis) / ray_dir_axis;
			}
			else {
				t = FLOAT_MAX;
			}

			//std::cout << t << endl;

			if (t < 0 || t > actual.maxt) { // near
				actual.node = near_child;
			}
			else {
				if (t < actual.mint) { // far
					actual.node = far_child;
				}
				else { // wherever you are
					stacktrav[++top] = traversal_structure_linear(far_child, t, actual.maxt);
					actual.node = near_child;
					actual.maxt = t;
				}
			}

		} // while not leaf

		if (kd_tree[actual.node].triangles != NULL) {
			for (int i = 0; i < kd_tree[actual.node].len; i++)
			{
				//inters_steps++; 
				if (scene_triangles_array[kd_tree[actual.node].triangles[i]].FindIntersection(ray, tmp) && tmp < result.dist) { // most definitely not ok
					result.dist = tmp;
					result.object = kd_tree[actual.node].triangles[i];
					found = true;
				}
			}



		}

	} //while stack not empty

	delete[] stacktrav;

	return result;
}

__device__ find_struct FindKDIntersectionGPU(RayLinear ray, Voxel V, kdNodeLinear *kd_tree, Triangle* scene_triangles_array, int depth, int* kd_node_triangles) {
	find_struct result;

	int top = -1;
	traversal_structure_linear* stacktrav = (traversal_structure_linear*)malloc(sizeof(traversal_structure_linear)*(depth + 2));
	traversal_structure_linear actual;
	
	bool found = false;

	result.dist = FLOAT_MAX;
	result.object = -1;

	float tmp = 0.0f;
	float t = FLOAT_MAX;

	int axis;
	float tmp1;
	float tmp2;
	float value;
	float ray_origin_axis = 1.0;
	float ray_dir_axis = 1.0;
	int near_child, far_child;


	if (!(fabs(ray.direction.x) < EPSILON)) {
		tmp1 = (V.position.x - ray.origin.x) / ray.direction.x;
		tmp2 = (V.position.x + V.dX - ray.origin.x) / ray.direction.x;

		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

	}

	if (!(fabs(ray.direction.y) < EPSILON)) {
		tmp1 = (V.position.y - ray.origin.y) / ray.direction.y;
		tmp2 = (V.position.y + V.dY - ray.origin.y) / ray.direction.y;

		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

	}

	if (!(fabs(ray.direction.z) < EPSILON)) {
		tmp1 = (V.position.z - ray.origin.z) / ray.direction.z;
		tmp2 = (V.position.z + V.dZ - ray.origin.z) / ray.direction.z;

		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

	}

	//cout << "Final voxel bounds are are: " << tmp << " " << t << endl;

	stacktrav[++top] = traversal_structure_linear(0, tmp, t);
	//std::cout << "Pushed in " << this->root << " " << tmp << " " << t << std::endl;

	while ( top >= 0 ) {
		actual = stacktrav[top];
		//std::cout << "Popped out " << actual.w << " " << actual.mint << " " << actual.maxt << std::endl;
		top--;

		while (kd_tree[actual.node].splits) {
			//trav_steps++;
			axis = kd_tree[actual.node].p.dimension;
			value = kd_tree[actual.node].p.position;
			//std::cout << "Split plane " << value << " (" << axis << ")" << std::endl;

			switch (axis) {
			case 0:
				ray_origin_axis = ray.origin.x;
				ray_dir_axis = ray.direction.x;
				break;
			case 1:
				ray_origin_axis = ray.origin.y;
				ray_dir_axis = ray.direction.y;
				break;
			case 2:
				ray_origin_axis = ray.origin.z;
				ray_dir_axis = ray.direction.z;
				break;
			}


			if (value < ray_origin_axis) {
				near_child = kd_tree[actual.node].right;
				far_child = kd_tree[actual.node].left;
				//nearChild = actual.w->right;
				//farChild = actual.w->left;
			}
			else {
				near_child = kd_tree[actual.node].left;
				far_child = kd_tree[actual.node].right;
				//nearChild = actual.w->left;
				//farChild = actual.w->right;
			}


			if (!(fabs(ray_dir_axis) < EPSILON)) {
				t = (value - ray_origin_axis) / ray_dir_axis;
			}
			else {
				t = FLOAT_MAX;
			}

			//std::cout << t << endl;

			if (t < 0 || t > actual.maxt) { // near
				actual.node = near_child;
			}
			else {
				if (t < actual.mint) { // far
					actual.node = far_child;
				}
				else { // wherever you are
					stacktrav[++top] = traversal_structure_linear(far_child, t, actual.maxt);
					actual.node = near_child;
					actual.maxt = t;
				}
			}

		} // while not leaf

		if (kd_tree[actual.node].len > 0) {


			for (int i = kd_tree[actual.node].gpu_offset; i < kd_tree[actual.node].len + kd_tree[actual.node].gpu_offset; i++)
			{
				tmp = scene_triangles_array[kd_node_triangles[i]].FindIntersection(ray);
				//tmp = scene_triangles_array[kd_tree[actual.node].triangles_device[i]].FindIntersection(ray);
				if (tmp >= 0.0f && tmp < result.dist) { // most definitely not ok
					result.dist = tmp;
					result.object = kd_node_triangles[i];
					found = true;
				}
			}



		}

	} //while stack not empty

	free(stacktrav);

	return result;
}


__host__ void IlluminateKernelCPU(int x, int y, float* ray_start, float *invMatrix, int width, int height, int index, Voxel V, kdNodeLinear *kd_tree, Triangle* scene_triangles_array, Material* materials, PointLight* lights, int lights_len, float* color_buffer, int depth, int triangles_n)
{
	int obj = -1;
	int level = 0;
	RayLinear ray;

	//stack<traversal_structure_linear> stacktrav;
	int top = -1;
	//traversal_structure_linear* stacktrav = new traversal_structure_linear[2];

	traversal_structure_linear* stacktrav = (traversal_structure_linear*)malloc(sizeof(traversal_structure_linear)*(depth + 2));


	traversal_structure_linear actual;
	int object;
	bool found = false;
	float dist = FLOAT_MAX;
	float tmp = 0.0f;
	float t = FLOAT_MAX;


	float ray_end[4] = { x, y, -1, 1 };

	float a_orig[4];
	for (size_t i = 0; i < 4; i++) {
		a_orig[i] = ray_end[i];
		ray_end[i] = 0;
	}
	for (size_t i = 0; i < 4; i++) {
		for (size_t j = 0; j < 4; j++) {
			ray_end[i] += a_orig[j] * invMatrix[(i % 4) + 4 * j];
		}
	}

	for (size_t i = 0; i < 3; i++) {
		ray_end[i] = ray_end[i] / ray_end[3];
	}

	ray.adjust(ray_start[0], ray_start[1], ray_start[2], ray_end[0], ray_end[1], ray_end[2]);


	// Start
	int axis;
	float tmp1;
	float tmp2;
	float value;
	float ray_origin_axis = 1.0;
	float ray_dir_axis = 1.0;
	int near_child, far_child;

	int base = x + y*width;
	base *= 3;

	/*base *= 3;

	color_buffer[base] = 1.0f;
	color_buffer[base + 1] = 0.0f;
	color_buffer[base + 2] = 0.0f;

	return;*/

	if (!(fabs(ray.direction.x) < EPSILON)) {
		tmp1 = (V.position.x - ray.origin.x) / ray.direction.x;
		tmp2 = (V.position.x + V.dX - ray.origin.x) / ray.direction.x;

		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

	}

	if (!(fabs(ray.direction.y) < EPSILON)) {
		tmp1 = (V.position.y - ray.origin.y) / ray.direction.y;
		tmp2 = (V.position.y + V.dY - ray.origin.y) / ray.direction.y;

		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

	}

	if (!(fabs(ray.direction.z) < EPSILON)) {
		tmp1 = (V.position.z - ray.origin.z) / ray.direction.z;
		tmp2 = (V.position.z + V.dZ - ray.origin.z) / ray.direction.z;

		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

	}

	/*base *= 3;

	color_buffer[base] = 1.0f;
	color_buffer[base + 1] = 0.0f;
	color_buffer[base + 2] = 0.0f;

	free(stacktrav);
	return;*/

	//cout << "Final voxel bounds are are: " << tmp << " " << t << endl;

	//stacktrav.push(traversal_structure_linear(0, tmp, t));
	/*stacktrav[++top] = traversal_structure_linear(0, tmp, t);
	//std::cout << "Pushed in " << this->root << " " << tmp << " " << t << std::endl;

	while ( top >= 0 ) {
	actual = stacktrav[top];
	//std::cout << "Popped out " << actual.w << " " << actual.mint << " " << actual.maxt << std::endl;
	top--;

	while (kd_tree[actual.node].splits) {
	//trav_steps++;
	axis = kd_tree[actual.node].p.dimension;
	value = kd_tree[actual.node].p.position;
	//std::cout << "Split plane " << value << " (" << axis << ")" << std::endl;

	switch (axis) {
	case 0:
	ray_origin_axis = ray.origin.x;
	ray_dir_axis = ray.direction.x;
	break;
	case 1:
	ray_origin_axis = ray.origin.y;
	ray_dir_axis = ray.direction.y;
	break;
	case 2:
	ray_origin_axis = ray.origin.z;
	ray_dir_axis = ray.direction.z;
	break;
	}


	if (value < ray_origin_axis) {
	near_child = kd_tree[actual.node].right;
	far_child = kd_tree[actual.node].left;
	//nearChild = actual.w->right;
	//farChild = actual.w->left;
	}
	else {
	near_child = kd_tree[actual.node].left;
	far_child = kd_tree[actual.node].right;
	//nearChild = actual.w->left;
	//farChild = actual.w->right;
	}


	if (!(fabs(ray_dir_axis) < EPSILON)) {
	t = (value - ray_origin_axis) / ray_dir_axis;
	}
	else {
	t = FLOAT_MAX;
	}

	//std::cout << t << endl;

	if (t < 0 || t > actual.maxt) { // near
	actual.node = near_child;
	}
	else {
	if (t < actual.mint) { // far
	actual.node = far_child;
	}
	else { // wherever you are
	//stacktrav.push(traversal_structure_linear(far_child, t, actual.maxt));

	if (top >= depth) {

	base *= 3;

	color_buffer[base] = 0.0f;
	color_buffer[base + 1] = 1.0f;
	color_buffer[base + 2] = 0.0f;

	free(stacktrav);
	return;
	}

	stacktrav[++top] = traversal_structure_linear(far_child, t, actual.maxt);
	actual.node = near_child;
	actual.maxt = t;
	}
	}

	} // while not leaf

	if (kd_tree[actual.node].triangles != NULL) {
	for (int i = 0; i < kd_tree[actual.node].len; i++)
	{
	//inters_steps++;
	if (obj == kd_tree[actual.node].triangles[i]) continue;
	if (scene_triangles_array[kd_tree[actual.node].triangles[i]].FindIntersection(ray, tmp) && tmp < dist) { // most definitely not ok
	dist = tmp;
	object = kd_tree[actual.node].triangles[i];
	found = true;
	}
	}



	}

	} //while stack not empty
	*/
	//delete[] stacktrav;
	free(stacktrav);

	color_buffer[base] = 1.0f;
	color_buffer[base + 1] = 0.0f;
	color_buffer[base + 2] = 0.0f;

	return;



	for (size_t i = 0; i < triangles_n; i++)
	{
		if (scene_triangles_array[i].FindIntersection(ray, tmp) && tmp < dist) { // most definitely not ok
			dist = tmp;
			object = i;
			found = true;
		}
	}

	if (!found) {
		color_buffer[base] = 0.0f;
		color_buffer[base + 1] = 0.0f;
		color_buffer[base + 2] = 0.0f;
		return/* Vector3f(0.0f, 0.0f, 0.0f)*/;
	}


	color_buffer[base] = 1.0f;
	color_buffer[base + 1] = 0.0f;
	color_buffer[base + 2] = 0.0f;

	return;

	/*if (object->GetMaterial()->Emissive()) {
	MaterialEmissive* obj_emis_material = (MaterialEmissive*)(object->GetMaterial());
	return obj_emis_material->GetIntensity();
	}*/

	//cout << scene_triangles_array[object].GetMaterial() << endl;
	Material obj_material = materials[scene_triangles_array[object].material]; // must not be emissive object
																			   //return Vector3f(obj_material->getT(), obj_material->getT(), obj_material->getT());
																			   //DrawObject* recursion_object;
																			   //Vector3f mirroring;
																			   //Vector3f refracted;
	float cmp;
	float light_dst;
	int occluder;
	float recurs_distance;

	Vector3f color(0, 0, 0);

	Vertex hit = (ray.origin) + ((ray.direction) * dist);
	Vertex normal = scene_triangles_array[object].getNormal(hit);
	//return normal;
	normal.normalizeThis();





	// LIGHTS
	/*for (int i = 0; i < lights_len; i++)
	{
	find_struct shadow_data;
	Vertex dir_to_light = lights[i].position - hit;

	RayLinear shadowRay(hit.x, hit.y, hit.z, lights[i].position.x, lights[i].position.y, lights[i].position.z);


	shadow_data = FindKDIntersectionGPU(shadowRay, V, kd_tree, scene_triangles_array, depth);

	light_dst = sqrt(dir_to_light.x*dir_to_light.x + dir_to_light.y*dir_to_light.y + dir_to_light.z*dir_to_light.z);

	//cout << cmp << endl;

	// something's blocking the light
	if (shadow_data.dist < (light_dst - 0.001f) && shadow_data.dist > 0.001f) {
	//cout << "shadow!" << endl;
	continue;
	}

	float ppower = obj_material.getShine();
	dir_to_light.normalizeThis();

	float diffuse_dot_product = dotProduct(normal, dir_to_light);
	Vector3f diffuse = obj_material.GetDiffuse() * (lights[i].intensity) * max(diffuse_dot_product, 0.0f);
	Vector3f specular = obj_material.GetSpecular() * (lights[i].intensity) * pow(max(dotProduct(reflected(normal, dir_to_light), (ray.direction)), 0.0f), ppower);

	color = color + diffuse + specular;
	}*/


	//color_buffer[base] = color.x;
	//color_buffer[base + 1] = color.y;
	//color_buffer[base + 2] = color.z;

	color_buffer[base] = 1.0f;
	color_buffer[base + 1] = 0.0f;
	color_buffer[base + 2] = 0.0f;


	// for DPG there is no need for mirror rays
	return/* color*/;



}

__global__ void IlluminateKernel(float* ray_start, float *invMatrix, int width, int height, Voxel V, kdNodeLinear *kd_tree, Triangle* scene_triangles_array, Material* materials, PointLight* lights, int lights_len, float* color_buffer, int depth, int triangles_n, int* kd_node_triangles)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;


	int obj = -1;
	int level = 0;
	RayLinear ray;

	//stack<traversal_structure_linear> stacktrav;
	int top = -1;
	//traversal_structure_linear* stacktrav = new traversal_structure_linear[2];

	traversal_structure_linear* stacktrav = (traversal_structure_linear*) malloc( sizeof(traversal_structure_linear)*(depth+2) );
	//free(stacktrav);

	traversal_structure_linear actual;
	int object = -1;
	bool found = false;
	float dist = FLOAT_MAX;
	float tmp = 0.0f;
	float t = FLOAT_MAX;


	float ray_end[4] = { x, y, -1, 1 };

	float a_orig[4];
	for (size_t i = 0; i < 4; i++) {
		a_orig[i] = ray_end[i];
		ray_end[i] = 0;
	}
	for (size_t i = 0; i < 4; i++) {
		for (size_t j = 0; j < 4; j++) {
			ray_end[i] += a_orig[j] * invMatrix[(i % 4) + 4 * j];
		}
	}

	for (size_t i = 0; i < 3; i++) {
		ray_end[i] = ray_end[i] / ray_end[3];
	}

	ray.adjust(ray_start[0], ray_start[1], ray_start[2], ray_end[0], ray_end[1], ray_end[2]);


	// Start
	int axis;
	float tmp1;
	float tmp2;
	float value;
	float ray_origin_axis = 1.0;
	float ray_dir_axis = 1.0;
	int near_child, far_child;

	int base = x + y*width;
	base *= 3;

	color_buffer[base] = 0.0f;
	color_buffer[base + 1] = 0.0f;
	color_buffer[base + 2] = 1.0f;
	
	if (!(fabs(ray.direction.x) < EPSILON)) {
		tmp1 = (V.position.x - ray.origin.x) / ray.direction.x;
		tmp2 = (V.position.x + V.dX - ray.origin.x) / ray.direction.x;

		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

	}

	if (!(fabs(ray.direction.y) < EPSILON)) {
		tmp1 = (V.position.y - ray.origin.y) / ray.direction.y;
		tmp2 = (V.position.y + V.dY - ray.origin.y) / ray.direction.y;

		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

	}

	if (!(fabs(ray.direction.z) < EPSILON)) {
		tmp1 = (V.position.z - ray.origin.z) / ray.direction.z;
		tmp2 = (V.position.z + V.dZ - ray.origin.z) / ray.direction.z;

		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

	}


	//cout << "Final voxel bounds are are: " << tmp << " " << t << endl;

	//stacktrav.push(traversal_structure_linear(0, tmp, t));
	stacktrav[++top] = traversal_structure_linear(0, tmp, t);
	//std::cout << "Pushed in " << this->root << " " << tmp << " " << t << std::endl;

	while ( top >= 0 ) {
		actual = stacktrav[top];
		//std::cout << "Popped out " << actual.w << " " << actual.mint << " " << actual.maxt << std::endl;
		top--;

		while (kd_tree[actual.node].splits) {
			//trav_steps++;
			axis = kd_tree[actual.node].p.dimension;
			value = kd_tree[actual.node].p.position;
			//std::cout << "Split plane " << value << " (" << axis << ")" << std::endl;

			switch (axis) {
			case 0:
				ray_origin_axis = ray.origin.x;
				ray_dir_axis = ray.direction.x;
				break;
			case 1:
				ray_origin_axis = ray.origin.y;
				ray_dir_axis = ray.direction.y;
				break;
			case 2:
				ray_origin_axis = ray.origin.z;
				ray_dir_axis = ray.direction.z;
				break;
			}


			if (value < ray_origin_axis) {
				near_child = kd_tree[actual.node].right;
				far_child = kd_tree[actual.node].left;
				//nearChild = actual.w->right;
				//farChild = actual.w->left;
			}
			else {
				near_child = kd_tree[actual.node].left;
				far_child = kd_tree[actual.node].right;
				//nearChild = actual.w->left;
				//farChild = actual.w->right;
			}


			if (!(fabs(ray_dir_axis) < EPSILON)) {
				t = (value - ray_origin_axis) / ray_dir_axis;
			}
			else {
				t = FLOAT_MAX;
			}

			//std::cout << t << endl;

			if (t < 0 || t > actual.maxt) { // near
				actual.node = near_child;
			}
			else {
				if (t < actual.mint) { // far
					actual.node = far_child;
				}
				else { // wherever you are
					//stacktrav.push(traversal_structure_linear(far_child, t, actual.maxt));

					if (top >= depth) {

						color_buffer[base] = 0.0f;
						color_buffer[base + 1] = 1.0f;
						color_buffer[base + 2] = 0.0f;

						free(stacktrav);
						return;
					}

					stacktrav[++top] = traversal_structure_linear(far_child, t, actual.maxt);
					actual.node = near_child;
					actual.maxt = t;
				}
			}

		} // while not leaf

		if (kd_tree[actual.node].len > 0) {
			for (int i = kd_tree[actual.node].gpu_offset; i < kd_tree[actual.node].len + kd_tree[actual.node].gpu_offset; i++)
			{
				//inters_steps++; 
				if (obj == kd_node_triangles[i]) continue;
				//if (scene_triangles_array[kd_tree[actual.node].triangles[i]].FindIntersection(ray, tmp) && tmp < dist) { // most definitely not ok
				//	dist = tmp;
				//	object = kd_tree[actual.node].triangles[i];
				//	found = true;
				//}
				tmp = scene_triangles_array[kd_node_triangles[i]].FindIntersection(ray);
				if (tmp >= 0.0f && tmp < dist) { // most definitely not ok
					dist = tmp;
					object = kd_node_triangles[i];
					found = true;
				}
			}



		}

	} //while stack not empty

	free(stacktrav);
	
	color_buffer[base] = 0.0f;
	color_buffer[base + 1] = 0.0f;
	color_buffer[base + 2] = 0.0f;


	if ( !found ) {
		return;
	}


	

	//return;

	/*if (object->GetMaterial()->Emissive()) {
	MaterialEmissive* obj_emis_material = (MaterialEmissive*)(object->GetMaterial());
	return obj_emis_material->GetIntensity();
	}*/

	//cout << scene_triangles_array[object].GetMaterial() << endl;
	Material obj_material = materials[scene_triangles_array[object].material]; // must not be emissive object
																					//return Vector3f(obj_material->getT(), obj_material->getT(), obj_material->getT());
	//DrawObject* recursion_object;
	//Vector3f mirroring;
	//Vector3f refracted;
	float cmp;
	float light_dst;
	int occluder;
	float recurs_distance;

	Vector3f color(0, 0, 0);

	Vertex hit = (ray.origin) + ((ray.direction) * dist);
	Vertex normal = scene_triangles_array[object].getNormal(hit);
	//return normal;
	normal.normalizeThis();





	// LIGHTS
	for (int i = 0; i < lights_len; i++)
	{
		find_struct shadow_data;
		Vertex dir_to_light = lights[i].position - hit;

		RayLinear shadowRay(hit.x, hit.y, hit.z, lights[i].position.x, lights[i].position.y, lights[i].position.z);


		shadow_data = FindKDIntersectionGPU(shadowRay, V, kd_tree, scene_triangles_array, depth, kd_node_triangles);

		light_dst = sqrt(dir_to_light.x*dir_to_light.x + dir_to_light.y*dir_to_light.y + dir_to_light.z*dir_to_light.z);

		//cout << cmp << endl;

		// something's blocking the light
		if (shadow_data.dist < (light_dst - 0.001f) && shadow_data.dist > 0.001f) {
			//cout << "shadow!" << endl;
			continue;
		}

		float ppower = obj_material.getShine();
		dir_to_light.normalizeThis();

		float diffuse_dot_product = dotProduct(normal, dir_to_light);
		Vector3f diffuse = obj_material.GetDiffuse() * (lights[i].intensity) * max(diffuse_dot_product, 0.0f);
		Vector3f specular = obj_material.GetSpecular() * (lights[i].intensity) * pow(max(dotProduct(reflected(normal, dir_to_light), (ray.direction)), 0.0f), ppower);

		color = color + diffuse + specular;
	}


	color_buffer[base] = color.x;
	color_buffer[base + 1] = color.y;
	color_buffer[base + 2] = color.z;

	//color_buffer[base] = 1.0f;
	//color_buffer[base + 1] = 0.0f;
	//color_buffer[base + 2] = 0.0f;


	// for DPG there is no need for mirror rays
	return/* color*/;


}

void KernelStart(float* ray_start, float *invMatrix, int width, int height, Voxel V, kdNodeLinear *kd_tree, Triangle* scene_triangles_array, Material* materials, PointLight* lights, int lights_len, float* color_buffer, int depth, int triangles_n, int* kd_node_triangles) {
	
	dim3 blocks(width / BLOCK_DIMENSION, height / BLOCK_DIMENSION);
	dim3 threads(BLOCK_DIMENSION, BLOCK_DIMENSION);

	IlluminateKernel<<< blocks, threads >>>(ray_start, invMatrix, width, height, V, kd_tree, scene_triangles_array, materials, lights, lights_len, color_buffer, depth, triangles_n, kd_node_triangles);
	//addKernel<<<1, size >>>(dev_c, dev_a, dev_b);
	
	cudaError cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "IlluminateKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	else {
		if ( CUDA_DEBUG ) cout << "Kernel call successful" << endl;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error after launching IlluminateKernel: %s\n", cudaGetErrorString(cudaStatus));
	}
};

