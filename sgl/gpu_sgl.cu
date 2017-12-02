

#include "gpu_sgl.cuh"



__host__ find_struct FindKDIntersectionGPU(RayLinear ray, Voxel V, kdNodeLinear *kd_tree, Triangle* scene_triangles_array) {
	find_struct result;
	stack<traversal_structure_linear> stacktrav;
	traversal_structure_linear actual;
	bool found = false;

	result.dist = numeric_limits<float>::max();
	result.object = -1;

	float tmp = numeric_limits<float>::min();
	float t = numeric_limits<float>::max();

	int axis;
	float tmp1;
	float tmp2;
	float value;
	float ray_origin_axis = 1.0;
	float ray_dir_axis = 1.0;
	int near_child, far_child;


	if (!(fabs(ray.direction.getX()) < config::epsilon)) {
		tmp1 = (V.position.x - ray.origin.getX()) / ray.direction.getX();
		tmp2 = (V.position.x + V.dX - ray.origin.getX()) / ray.direction.getX();

		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

	}

	if (!(fabs(ray.direction.getY()) < config::epsilon)) {
		tmp1 = (V.position.y - ray.origin.getY()) / ray.direction.getY();
		tmp2 = (V.position.y + V.dY - ray.origin.getY()) / ray.direction.getY();

		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

	}

	if (!(fabs(ray.direction.getZ()) < config::epsilon)) {
		tmp1 = (V.position.z - ray.origin.getZ()) / ray.direction.getZ();
		tmp2 = (V.position.z + V.dZ - ray.origin.getZ()) / ray.direction.getZ();

		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

	}

	//cout << "Final voxel bounds are are: " << tmp << " " << t << endl;

	stacktrav.push(traversal_structure_linear(0, tmp, t));
	//std::cout << "Pushed in " << this->root << " " << tmp << " " << t << std::endl;

	while (!stacktrav.empty()) {
		actual = stacktrav.top();
		//std::cout << "Popped out " << actual.w << " " << actual.mint << " " << actual.maxt << std::endl;
		stacktrav.pop();

		while (kd_tree[actual.node].splits) {
			//trav_steps++;
			axis = kd_tree[actual.node].p.dimension;
			value = kd_tree[actual.node].p.position;
			//std::cout << "Split plane " << value << " (" << axis << ")" << std::endl;

			switch (axis) {
			case 0:
				ray_origin_axis = ray.origin.getX();
				ray_dir_axis = ray.direction.getX();
				break;
			case 1:
				ray_origin_axis = ray.origin.getY();
				ray_dir_axis = ray.direction.getY();
				break;
			case 2:
				ray_origin_axis = ray.origin.getZ();
				ray_dir_axis = ray.direction.getZ();
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


			if (!(fabs(ray_dir_axis) < config::epsilon)) {
				t = (value - ray_origin_axis) / ray_dir_axis;
			}
			else {
				t = numeric_limits<float>::max();
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
					stacktrav.push(traversal_structure_linear(far_child, t, actual.maxt));
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


	return result;
}


__host__ void IlluminateKernel(int x, int y, float* ray_start, float *invMatrix, int width, int height, int index, Voxel V, kdNodeLinear *kd_tree, Triangle* scene_triangles_array, Material* materials, PointLight* lights, int lights_len, float* color_buffer)
{
	int base = x + y*height;

	int obj = -1;
	int level = 0;
	RayLinear ray;

	stack<traversal_structure_linear> stacktrav;
	traversal_structure_linear actual;
	int object;
	bool found = false;
	float dist = numeric_limits<float>::max();
	float tmp = numeric_limits<float>::min();
	float t = numeric_limits<float>::max();

	int axis;
	float tmp1;
	float tmp2;
	float value;
	float ray_origin_axis = 1.0;
	float ray_dir_axis = 1.0;
	int near_child, far_child;

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


	if (!(fabs(ray.direction.getX()) < config::epsilon)) {
		tmp1 = (V.position.x - ray.origin.getX()) / ray.direction.getX();
		tmp2 = (V.position.x + V.dX - ray.origin.getX()) / ray.direction.getX();

		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

	}

	if (!(fabs(ray.direction.getY()) < config::epsilon)) {
		tmp1 = (V.position.y - ray.origin.getY()) / ray.direction.getY();
		tmp2 = (V.position.y + V.dY - ray.origin.getY()) / ray.direction.getY();

		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

	}

	if (!(fabs(ray.direction.getZ()) < config::epsilon)) {
		tmp1 = (V.position.z - ray.origin.getZ()) / ray.direction.getZ();
		tmp2 = (V.position.z + V.dZ - ray.origin.getZ()) / ray.direction.getZ();

		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

	}

	//cout << "Final voxel bounds are are: " << tmp << " " << t << endl;

	stacktrav.push(traversal_structure_linear(0, tmp, t));
	//std::cout << "Pushed in " << this->root << " " << tmp << " " << t << std::endl;

	while (!stacktrav.empty()) {
		actual = stacktrav.top();
		//std::cout << "Popped out " << actual.w << " " << actual.mint << " " << actual.maxt << std::endl;
		stacktrav.pop();

		while (kd_tree[actual.node].splits) {
			//trav_steps++;
			axis = kd_tree[actual.node].p.dimension;
			value = kd_tree[actual.node].p.position;
			//std::cout << "Split plane " << value << " (" << axis << ")" << std::endl;

			switch (axis) {
			case 0:
				ray_origin_axis = ray.origin.getX();
				ray_dir_axis = ray.direction.getX();
				break;
			case 1:
				ray_origin_axis = ray.origin.getY();
				ray_dir_axis = ray.direction.getY();
				break;
			case 2:
				ray_origin_axis = ray.origin.getZ();
				ray_dir_axis = ray.direction.getZ();
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


			if (!(fabs(ray_dir_axis) < config::epsilon)) {
				t = (value - ray_origin_axis) / ray_dir_axis;
			}
			else {
				t = numeric_limits<float>::max();
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
					stacktrav.push(traversal_structure_linear(far_child, t, actual.maxt));
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


	if (!found) { 
		base *= 3;
		color_buffer[base] = 0.0f;
		color_buffer[base + 1] = 0.0f;
		color_buffer[base + 2] = 0.0f;
		return/* Vector3f(0.0f, 0.0f, 0.0f)*/; 
	}

	/*if (object->GetMaterial()->Emissive()) {
	MaterialEmissive* obj_emis_material = (MaterialEmissive*)(object->GetMaterial());
	return obj_emis_material->GetIntensity();
	}*/

	//cout << scene_triangles_array[object].GetMaterial() << endl;
	Material obj_material = materials[scene_triangles_array[object].GetMaterial()]; // must not be emissive object
																					//return Vector3f(obj_material->getT(), obj_material->getT(), obj_material->getT());
	DrawObject* recursion_object;
	Vector3f mirroring;
	Vector3f refracted;
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
		Vertex dir_to_light = lights[i].GetPosition() - hit;

		RayLinear shadowRay(hit.getX(), hit.getY(), hit.getZ(), lights[i].GetPosition().getX(), lights[i].GetPosition().getY(), lights[i].GetPosition().getZ());


		shadow_data = FindKDIntersectionGPU(shadowRay, V, kd_tree, scene_triangles_array);

		light_dst = sqrt(dir_to_light.getX()*dir_to_light.getX() + dir_to_light.getY()*dir_to_light.getY() + dir_to_light.getZ()*dir_to_light.getZ());

		//cout << cmp << endl;

		// something's blocking the light
		if (shadow_data.dist < (light_dst - 0.001f) && shadow_data.dist > 0.001f) {
			//cout << "shadow!" << endl;
			continue;
		}

		float ppower = obj_material.getShine();
		dir_to_light.normalizeThis();

		float diffuse_dot_product = dotProduct(normal, dir_to_light);
		Vector3f diffuse = obj_material.GetDiffuse() * (lights[i].GetIntensity()) * max(diffuse_dot_product, 0.0f);
		Vector3f specular = obj_material.GetSpecular() * (lights[i].GetIntensity()) * pow(max(dotProduct(reflected(normal, dir_to_light), (ray.direction)), 0.0f), ppower);

		color = color + diffuse + specular;
	}


	base *= 3;
	color_buffer[base] = color.x;
	color_buffer[base + 1] = color.y;
	color_buffer[base + 2] = color.z;



	// for DPG there is no need for mirror rays
	return/* color*/;


}

