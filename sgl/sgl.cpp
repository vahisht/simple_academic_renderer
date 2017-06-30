

#include "sgl.h"


#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <typeinfo>
#include <algorithm>
#include <stack>


using namespace std;

#define INVALID_OPERATION_CHECK if (drawing || !active_context || !activeStack) {setErrCode(SGL_INVALID_OPERATION); return; }
#define INVALID_OPERATION_CHECK_WITHOUT_STACK if (drawing || !active_context) {setErrCode(SGL_INVALID_OPERATION); return; }



void MultVector(float *a, float *b);
void MultMatrix(float *a, const  float *b);
void setPixel(int x, int y, float z);
class ImageStructure;
//class Vertex;
//class Vector3f;
class TransformationStack;
class MaterialBase;
class SceneStructure;
const int CONTEXTS_COUNT = 32;
void setScale();
float dotProduct(Vector3f a, Vertex b);


bool drawing;									// flag - drawing in progress
float clear_r, clear_g, clear_b;				// clear color by sglClearColor
float draw_r, draw_g, draw_b;					// draw color by sglColor3f
float pointSize;								// size of point in pixels
float scale;									// overall scale[x, y] of the concatenated tranformation
bool depth_test_enabled;						// flag if depth test id enabled
sglEAreaMode fill_method;						// method used for filling drawed objects
sglEElementType drawing_object;					// drawing element
ImageStructure* active_context;					// currently active context
ImageStructure** structureList;					// list of contexts
float* viewPortTransformation;					// viewport Transformation
TransformationStack* modelViewStack = NULL;
TransformationStack* projectionStack = NULL;
TransformationStack* activeStack = NULL;		// Transformation stacks
bool drawing_scene;								// scena in progress
MaterialBase* used_material;					// current material
SceneStructure* scene;							// current scene object

int env_width;
int env_height;
float *env_texels;

long rays = 0;
long trav_steps = 0;
long inters_steps = 0;

KD_Tree *kd_scene;

// ==========================================================================
//  Ray-tracer
// ==========================================================================


class PointLight
{
public:
	PointLight(float x, float y, float z, float _r, float _g, float _b)	{
		position = new Vertex(x, y, z, 1);
		intensity = new Vector3f(_r, _g, _b);
	}
	~PointLight() {
		delete position;
		delete intensity;
	}
	Vertex* GetPosition()	{ return position; }
	Vector3f* GetIntensity(){ return intensity; }
private:
	Vertex* position;
	Vector3f* intensity;
	float r, g, b; // intensity
};

Vector3f reflected(Vector3f normal, Vertex to_light) {
	//Vertex light = to_light.normalize();
	return ((normal * 2 * (dotProduct(normal, to_light))) - to_light)*(-1);
}
void getEnvPixel(float x, float y, float* triplet) {
	int tex_x, tex_y;

	y = 1 - y;
	tex_x = (int)(x*((float)env_width));
	tex_y = (int)(y*((float)env_height));

	int step = 3 * (tex_x + tex_y*env_width);
	
	triplet[0] = env_texels[step];
	triplet[1] = env_texels[step + 1];
	triplet[2] = env_texels[step + 2];
}
Vector3f setEnvMapColor(float x, float y, float z) {
	float r, u, v;
	float c = sqrt(x*x + y*y);

	if (c > 0) {
		r = acos(z) / (2 * c*M_PI);
	}
	else {
		r = 0.0f;
	}

	u = (1.0f / 2.0f) + r*x;
	v = (1.0f / 2.0f) + r*y;

	float env_color[3] = { 0,0,0 }; 
	getEnvPixel(u, v, env_color);

	return Vector3f(env_color[0], env_color[1], env_color[2]);
}

class SceneStructure {
public:
	//SceneStructure() {}
	~SceneStructure() {
		for (vector<DrawObject*>::iterator it = scene_primitives.begin(); it != scene_primitives.end(); ++it)
			delete (*it);
		for (vector<PointLight*>::iterator it = scene_lights.begin(); it != scene_lights.end(); ++it)
			delete (*it);
		for (vector<Triangle*>::iterator it = scene_triangles.begin(); it != scene_triangles.end(); ++it)
			delete (*it);
	}
	void AddPrimitive(float x, float y, float z, float _radius) {
		scene_primitives.push_back(new Sphere(x, y, z, _radius, used_material));
	}
	void AddPrimitive(sglEElementType _object_type, vector<Vertex*>* _vertices, bool emissive) {
		
		if (emissive) {
			scene_emissives.push_back(new Triangle(_object_type, _vertices, used_material));
			scene_triangles.push_back(new Triangle(_object_type, _vertices, used_material));
		} else {
			scene_primitives.push_back(new Triangle(_object_type, _vertices, used_material));
			scene_triangles.push_back(new Triangle(_object_type, _vertices, used_material));
		}
	}
	void AddPrimitive(sglEElementType _object_type, vector<Vertex*>* _vertices, vector<Vertex*>* _normals, bool emissive) {
		//cout << "Processing " << _normals->size() << endl;
		if (emissive) {
			scene_emissives.push_back(new Triangle(_object_type, _vertices, _normals, used_material));
			scene_triangles.push_back(new Triangle(_object_type, _vertices, _normals, used_material));
		}
		else {
			scene_primitives.push_back(new Triangle(_object_type, _vertices, _normals, used_material));
			//cout << "--" << endl;
			//cout << _normals->size() << endl;
			scene_triangles.push_back(new Triangle(_object_type, _vertices, _normals, used_material));
			//scene_triangles[scene_triangles.size() - 1]->normals_print();
		}
	}
	void AddLight(float x, float y, float z, float r, float g, float b) {
		scene_lights.push_back(new PointLight(x, y, z, r, g, b));
	}

	DrawObject* FindIntersection(Ray &ray, float & dist) {
		DrawObject* object = NULL;
		dist = numeric_limits<float>::max();		// tady by asi bylo lepší mít far plane scény spíše než nekonečno. Věci co jsou za far plane už totiž nekreslíme
		float tmp;
		for (vector<Triangle*>::iterator it = scene_triangles.begin(); it != scene_triangles.end(); it++){
			//cout << typeid(*it).name() << endl;
			if ((*it)->FindIntersection(ray, tmp) && tmp < dist) {
				dist = tmp;
				object = (*it);
			}
		}
		/*for (vector<DrawObject*>::iterator it = scene_emissives.begin(); it != scene_emissives.end(); it++) {
			if ((*it)->FindIntersection(ray, tmp) && tmp < dist) {
				dist = tmp;
				object = (*it);
			}
		}*/
		return object;
	}
	DrawObject* FindShadowIntersection(Ray &ray, float & dist, DrawObject* obj) {
		dist = numeric_limits<float>::max();		// tady by asi bylo lepší mít far plane scény spíše než nekonečno. Věci co jsou za far plane už totiž nekreslíme
		float tmp;
		DrawObject* closest = NULL;
		for (vector<Triangle*>::iterator it = scene_triangles.begin(); it != scene_triangles.end(); it++) {
			if ((*it) == obj) continue;
			if ((*it)->FindIntersection(ray, tmp) && tmp < dist) {
				dist = tmp;
				closest = (*it);
			}
		}
		for (vector<DrawObject*>::iterator it = scene_emissives.begin(); it != scene_emissives.end(); it++) {
			if ((*it) == obj) continue;
			if ((*it)->FindIntersection(ray, tmp) && tmp < dist) {
				dist = tmp;
				closest = (*it);
			}
		}
		return closest;
	}

	Vector3f Illuminate(DrawObject* object, Ray &ray, float distance, int level) {
		//return Vector3f(1, 1, 1);
		if (object->GetMaterial()->Emissive()) {
			MaterialEmissive* obj_emis_material = (MaterialEmissive*)(object->GetMaterial());
			return obj_emis_material->GetIntensity();
		}

		Material* obj_material = (Material*)(object->GetMaterial()); // must not be emissive object
		DrawObject* recursion_object;
		Vector3f mirroring;
		Vector3f refracted;
		float cmp;
		float light_dst;
		float recurs_distance;

		Vector3f color(0, 0, 0);

		Vertex hit = (*ray.origin) + ((*ray.direction) * distance);
		Vector3f normal = object->getNormal(hit);
		//return normal;
		normal.Normalize();

		// set light color addition and shadows
		for (vector<PointLight*>::iterator it = scene_lights.begin(); it != scene_lights.end(); it++) {
			//cout << (*it)->GetPosition()->getX() << " " << (*it)->GetPosition()->getY() << " " << (*it)->GetPosition()->getZ() << endl;

			Vertex dir_to_light = *(*it)->GetPosition() - hit;

			Ray shadowRay(hit.getX(), hit.getY(), hit.getZ(), (*it)->GetPosition()->getX(), (*it)->GetPosition()->getY(), (*it)->GetPosition()->getZ());
			
			//kd_scene->FindKDIntersection(shadowRay, cmp, object);
			scene->FindShadowIntersection(shadowRay, cmp, object);
			light_dst = sqrt(dir_to_light.getX()*dir_to_light.getX() + dir_to_light.getY()*dir_to_light.getY() + dir_to_light.getZ()*dir_to_light.getZ());

			// something's blocking the light
			if (cmp < (light_dst - 0.001f) && cmp > 0.001f)
				continue;

			float ppower = obj_material->getShine();
			dir_to_light.normalizeThis();

			float diffuse_dot_product = dotProduct(normal, dir_to_light);
			Vector3f diffuse = obj_material->GetDiffuse() * (*(*it)->GetIntensity()) * max(diffuse_dot_product, 0.0f);
			Vector3f specular = obj_material->GetSpecular() * (*(*it)->GetIntensity()) * pow(max(dotProduct(reflected(normal, dir_to_light), (*ray.direction)), 0.0f), ppower);

			color = color + diffuse + specular;
		}

		// for DPG there is no need for mirror rays
		//return color;

		if (level > 8) return color;

		//set reflected
		if (obj_material->GetSpecular().x > 0.0f) {
			Vector3f reflected_dir = reflected(normal, *ray.direction);
			Ray mirrorRay(hit.getX(), hit.getY(), hit.getZ(), reflected_dir.x + hit.getX(), reflected_dir.y + hit.getY(), reflected_dir.z + hit.getZ());

			recursion_object = FindShadowIntersection(mirrorRay, recurs_distance, object);

			if (recursion_object) {
				mirroring = Illuminate(recursion_object, mirrorRay, recurs_distance, level + 1);
				mirroring = mirroring * obj_material->GetSpecular();

				color = color + mirroring;
			}
			else {
				if (env_texels) {
					mirroring = setEnvMapColor(mirrorRay.direction->getX(), mirrorRay.direction->getY(), mirrorRay.direction->getZ());
					mirroring = mirroring * obj_material->GetSpecular();
					color = color + mirroring;
				}
			}
		}

		//set refracted
		if (obj_material->getT() > 0.0f) {
			float ior = obj_material->getIor();
			float gamma, sqrterm;

			float dot = dotProduct(normal, *ray.direction);

			if (dot < 0.0) {
				// from outside into the inside of object
				gamma = 1.0 / ior;
			}
			else {
				// from the inside to outside of object
				gamma = ior;
				dot = -dot;
				normal = normal * (-1);
			}
			sqrterm = 1.0 - gamma * gamma * (1.0 - dot * dot);

			Vector3f refracted_dir;
			if (sqrterm > 0.0) {
				sqrterm = dot * gamma + sqrt(sqrterm);
				refracted_dir = normal * -sqrterm + (*ray.direction * gamma);
			}

			Ray refractedRay(hit.getX(), hit.getY(), hit.getZ(), refracted_dir.x + hit.getX(), refracted_dir.y + hit.getY(), refracted_dir.z + hit.getZ());
			recursion_object = FindIntersection(refractedRay, recurs_distance);

			if (recursion_object) {
				refracted = Illuminate(recursion_object, refractedRay, recurs_distance, level + 1);
				refracted = refracted * obj_material->getT();
			}
			else {
				if (env_texels) {
					refracted = setEnvMapColor(refractedRay.direction->getX(), refractedRay.direction->getY(), refractedRay.direction->getZ());
					refracted = refracted * obj_material->getT();
				}
			}

			color = color + refracted;
		}

		// indirect light
		for (vector<DrawObject*>::iterator it = scene_emissives.begin(); it != scene_emissives.end(); it++) {
			MaterialEmissive* emis_material = (MaterialEmissive*)((*it)->GetMaterial());
			Vector3f attentuationCons = emis_material->GetAttentuationCons();
			float area = ((Triangle*)(*it))->areaSize();

			int sample_size = 16;
			for (int i = 0; i < sample_size; i++){			
				Vertex sample = ((Triangle*)(*it))->randomSample();
				Vertex dir_to_light = sample - hit;
				Ray shadowRay(hit.getX(), hit.getY(), hit.getZ(), sample.getX(), sample.getY(), sample.getZ());

				// check if something's blocking the light
				FindShadowIntersection(shadowRay, cmp, object);
				light_dst = sqrt(dir_to_light.getX()*dir_to_light.getX() + dir_to_light.getY()*dir_to_light.getY() + dir_to_light.getZ()*dir_to_light.getZ());
				if (cmp < (light_dst - 0.01f) && cmp > 0.01f)
					continue;

				Vector3f light_triangle_normal = (*it)->getNormal(sample); // still the same (triangle)

				// attentuation
				float sample_distanc = Vertex::Distance(sample, hit);
				float denominator = attentuationCons.x + attentuationCons.y * sample_distanc + attentuationCons.z * sample_distanc * sample_distanc;
				float cos_fi = dotProduct(light_triangle_normal, (*shadowRay.direction * -1));	
				Vector3f sampleIntensity = emis_material->GetIntensity() * cos_fi;
				float c = (area / sample_size) / denominator;
				sampleIntensity = sampleIntensity * c;
				
				// object material
				float ppower = obj_material->getShine();
				dir_to_light.normalizeThis();
				float diffuse_dot_product = dotProduct(normal, dir_to_light);
				Vector3f diffuse = obj_material->GetDiffuse() * sampleIntensity * max(diffuse_dot_product, 0.0f);
				Vector3f specular = obj_material->GetSpecular() * sampleIntensity * pow(max(dotProduct(reflected(normal, dir_to_light), (*ray.direction)), 0.0f), ppower);

				color = color + diffuse + specular;
			}
		}
		return color;
	}

	vector<Triangle*> scene_triangles;
private:
	vector<DrawObject*> scene_primitives;
	vector<DrawObject*> scene_emissives;
	vector<PointLight*> scene_lights;
};


void vertexTransformation(float &x, float &y, float &z, float &w) {
	// vertex position
	float vertexPosition[4] = { x, y, z, w };

	//modelview o projection
	MultVector(vertexPosition, modelViewStack->GetCurrent());
	MultVector(vertexPosition, projectionStack->GetCurrent()); // normalized

	// apply viewport
	if (viewPortTransformation) {
		MultVector(vertexPosition, viewPortTransformation);
	}

	x = vertexPosition[0] / vertexPosition[3];
	y = vertexPosition[1] / vertexPosition[3];
	z = vertexPosition[2] / vertexPosition[3];
	w = vertexPosition[3] / vertexPosition[3];
}

void setScale() {
	//overall scale[x, y] of the concatenated (modelview o projection o viewport) tranformation.

	float concatenated[16];
	float concatenatedTmp[16];
	float* projectionM = projectionStack->GetCurrent();
	float* modelViewM = modelViewStack->GetCurrent();
	for (size_t i = 0; i < 16; i++) {
		concatenated[i] = 0;
	}

	// projectionM * modelViewM
	for (size_t i = 0; i < 16; i++) {
		for (size_t j = 0; j < 4; j++) {
			concatenated[i] += projectionM[i - i % 4 + j] * modelViewM[(i % 4) + 4 * j];
		}
	}

	// * viewPortTransformation
	for (size_t i = 0; i < 16; i++) {
		concatenatedTmp[i] = concatenated[i];
		concatenated[i] = 0;
	}
	for (size_t i = 0; i < 16; i++) {
		for (size_t j = 0; j < 4; j++) {
			concatenated[i] += concatenatedTmp[i - i % 4 + j] * viewPortTransformation[(i % 4) + 4 * j];
		}
	}

	// get scale from concatenated matrix
	scale = (concatenated[0] * concatenated[5]) - (concatenated[1] * concatenated[4]);
	scale = sqrt(scale);
	TransformationStack::scaleActualized = true;
}

void MultVector(float *a, float *b) {
	// a = b*a
	float a_orig[4];
	for (size_t i = 0; i < 4; i++) {
		a_orig[i] = a[i];
		a[i] = 0;
	}
	for (size_t i = 0; i < 4; i++) {
		for (size_t j = 0; j < 4; j++) {
			a[i] += a_orig[j] * b[(i % 4) + 4 * j];
		}
	}
}
void MultMatrix(float *a, const float *b) {
	// a = b*a
	float a_orig[16];
	for (size_t i = 0; i < 16; i++) {
		a_orig[i] = a[i];
		a[i] = 0;
	}

	for (size_t i = 0; i < 16; i++) {
		for (size_t j = 0; j < 4; j++) {
			a[i] += a_orig[(i % 4) + 4 * j] * b[i - i % 4 + j];
		}
	}
}
int InvertMatrix(float * x) {
	// inverse matrix computation using gauss_jacobi method
	// source: N.R. in C
	// if matrix is regular = computatation successfull = returns 0
	// in case of singular matrix returns 1

	int indxc[4], indxr[4], ipiv[4];
	int i, icol, irow, j, k, l, ll, n;
	float big, dum, pivinv, temp;
	// satisfy the compiler
	icol = irow = 0;

	// the size of the matrix
	n = 4;

	for (j = 0; j < n; j++) /* zero pivots */
		ipiv[j] = 0;

	for (i = 0; i < n; i++)
	{
		big = 0.0;
		for (j = 0; j < n; j++)
			if (ipiv[j] != 1)
				for (k = 0; k<n; k++)
				{
			if (ipiv[k] == 0)
			{
				if (fabs(x[k*n + j]) >= big)
				{
					big = fabs(x[k*n + j]);
					irow = j;
					icol = k;
				}
			}
			else
				if (ipiv[k] > 1)
					return 1; /* singular matrix */
				}
		++(ipiv[icol]);
		if (irow != icol)
		{
			for (l = 0; l<n; l++)
			{
				temp = x[l*n + icol];
				x[l*n + icol] = x[l*n + irow];
				x[l*n + irow] = temp;
			}
		}
		indxr[i] = irow;
		indxc[i] = icol;
		if (x[icol*n + icol] == 0.0)
			return 1; /* singular matrix */

		pivinv = 1.0 / x[icol*n + icol];
		x[icol*n + icol] = 1.0;
		for (l = 0; l<n; l++)
			x[l*n + icol] = x[l*n + icol] * pivinv;

		for (ll = 0; ll < n; ll++)
			if (ll != icol)
			{
			dum = x[icol*n + ll];
			x[icol*n + ll] = 0.0;
			for (l = 0; l<n; l++)
				x[l*n + ll] = x[l*n + ll] - x[l*n + icol] * dum;
			}
	}
	for (l = n; l--;)
	{
		if (indxr[l] != indxc[l])
			for (k = 0; k<n; k++)
			{
			temp = x[indxr[l] * n + k];
			x[indxr[l] * n + k] = x[indxc[l] * n + k];
			x[indxc[l] * n + k] = temp;
			}
	}

	return 0; // matrix is regular .. inversion has been succesfull
}


/// Current error code.
static sglEErrorCode _libStatus = SGL_NO_ERROR;

static inline void setErrCode(sglEErrorCode c)
{
	if (_libStatus == SGL_NO_ERROR)
		_libStatus = c;
	if (c != SGL_NO_ERROR)	// TODO: delete these two lines
		cout << "error: " << c << endl;
}

//---------------------------------------------------------------------------
// sglGetError()
//---------------------------------------------------------------------------
sglEErrorCode sglGetError(void)
{
	sglEErrorCode ret = _libStatus;
	_libStatus = SGL_NO_ERROR;
	return ret;
}

//---------------------------------------------------------------------------
// sglGetErrorString()
//---------------------------------------------------------------------------
const char* sglGetErrorString(sglEErrorCode error)
{
	static const char *errStrigTable[] =
	{
		"Operation succeeded",
		"Invalid argument(s) to a call",
		"Invalid enumeration argument(s) to a call",
		"Invalid call",
		"Quota of internal resources exceeded",
		"Internal library error",
		"Matrix stack overflow",
		"Matrix stack underflow",
		"Insufficient memory to finish the requested operation"
	};

	if ((int)error<(int)SGL_NO_ERROR || (int)error>(int)SGL_OUT_OF_MEMORY) {
		return "Invalid value passed to sglGetErrorString()";
	}

	return errStrigTable[(int)error];
}

//---------------------------------------------------------------------------
// Initialization functions
//---------------------------------------------------------------------------

void sglInit(void) {
	drawing = false;
	clear_r = clear_g = clear_b = 0;
	pointSize = 1;
	scale = 0;
	TransformationStack::scaleActualized = false;
	depth_test_enabled = false;
	fill_method = SGL_FILL;
	viewPortTransformation = new float[16];
	active_context = NULL;
	activeStack = NULL;
	modelViewStack = new TransformationStack();
	projectionStack = new TransformationStack();
	drawing_scene = false;
	used_material = NULL;
	scene = NULL;
	kd_scene = NULL;
	try {
		structureList = new ImageStructure*[CONTEXTS_COUNT];
		for (size_t i = 0; i < CONTEXTS_COUNT; i++)
			structureList[i] = NULL;
	}
	catch (std::bad_alloc&) {
		setErrCode(SGL_OUT_OF_MEMORY);
	}
	setErrCode(SGL_NO_ERROR);
}

void sglFinish(void) {
	for (size_t i = 0; i < CONTEXTS_COUNT; i++) {
		if (structureList[i])
			delete structureList[i];
	}
	delete[] structureList;
	delete[] viewPortTransformation;
	// smazani transformation stack
	delete modelViewStack;
	delete projectionStack;
	if (used_material)
		delete used_material;
	if (scene)
		delete scene;
	if (kd_scene)
		delete kd_scene;

	setErrCode(SGL_NO_ERROR);
}

int sglCreateContext(int width, int height) {
	int i = 0;
	while (i < CONTEXTS_COUNT && structureList[i]) { i++; }
	if (i >= CONTEXTS_COUNT) {
		setErrCode(SGL_OUT_OF_RESOURCES);
		return -1;
	}

	ImageStructure* newStructure;
	try {
		newStructure = new ImageStructure(i, width, height);
		structureList[i] = newStructure;
	}
	catch (std::bad_alloc&) {
		setErrCode(SGL_OUT_OF_MEMORY);
		return -1;
	}
	setErrCode(SGL_NO_ERROR);
	return i;
}

void sglDestroyContext(int id) {
	if (id >= CONTEXTS_COUNT || !structureList[id]) {
		setErrCode(SGL_INVALID_VALUE);
		return;
	}
	if (active_context->GetContextID() == id)
		active_context = NULL;
	delete structureList[id];
	setErrCode(SGL_NO_ERROR);
}

void sglSetContext(int id) {
	if (id >= CONTEXTS_COUNT || !structureList[id]) {
		setErrCode(SGL_INVALID_VALUE);
		return;
	}
	active_context = structureList[id];
	setErrCode(SGL_NO_ERROR);
}

int sglGetContext(void) {
	if (!active_context) { // return active context or any existing context?
		setErrCode(SGL_INVALID_OPERATION);
		return -1;
	}
	setErrCode(SGL_NO_ERROR);
	return active_context->GetContextID();
}

float *sglGetColorBufferPointer(void) {
	if (!active_context)
		return 0;
	return active_context->GetColorBuffer();
}

//---------------------------------------------------------------------------
// Drawing functions
//---------------------------------------------------------------------------

void swap(float &a, float &b) {
	float tmp = a;
	a = b;
	b = tmp;
}
void setPixel(int x, int y, float z) {
	if (x < 0 || y < 0 || x >= active_context->GetWidth() || y >= active_context->GetHeight())
		return;

	int base = x + y*(active_context->GetWidth());
	float* z_buffer = active_context->GetZBuffer();

	if (depth_test_enabled) {
		if (z_buffer[base] <= z)
		{
			return;
		} // further than existing pixel
		else
			z_buffer[base] = z;
	}

	base += base + base;
	float* color_buffer = sglGetColorBufferPointer();
	color_buffer[base] = draw_r;
	color_buffer[base + 1] = draw_g;
	color_buffer[base + 2] = draw_b;
}
void setPixelHorizontalLine(int x1, int x2, int y, float z1, float z2) {
	if (x1 > x2) {
		swap(x1, x2);
		swap(z1, z2);
	}
	float delta_x = abs(x1 - x2);
	float delta_z = (z2 - z1) / ((delta_x > 0) ? delta_x : 1);
	float z = z1;

	for (int i = x1; i <= x2; i++) {
		setPixel(i, y, z);
		z += delta_z;
	}
}
void SetPoint(float x, float y, float z) {
	for (int y2 = y - ceil(pointSize / 2) + 1; y2 <= y + floor(pointSize / 2); y2++) {
		for (int x2 = x - ceil(pointSize / 2) + 1; x2 <= x + floor(pointSize / 2); x2++) {
			setPixel(x2, y2, z);
		}
	}
	return;
}

// Finds the slope and calls the right method ( in hor. x1 is left to the x2, in vert y1 is below y2 )
void sglLine(float x1, float y1, float z1, float x2, float y2, float z2) {
	x1 = (int)x1; x2 = (int)x2; y1 = (int)y1; y2 = (int)y2;
	float x, y, z;
	bool main_x;
	int main_axis_position, minor_axis_position;
	int main_axis_step, minor_axis_step;
	float z_step;
	int main_length;
	int k1, k2;
	float p;

	if (abs(x2 - x1) >= abs(y2 - y1)) {
		// x as main axis
		main_x = true;
		main_length = abs(x1 - x2);
		x = (y1 >= y2) ? x1 : x2;
		main_axis_position = x;
		if (y1 >= y2)
			main_axis_step = (x1 >= x2) ? -1 : 1;
		else
			main_axis_step = (x2 >= x1) ? -1 : 1;

		// y as minor axis
		y = max(y1, y2);
		minor_axis_position = y;
		minor_axis_step = -1;

		// variables
		k1 = 2 * (abs(y1 - y2));
		k2 = 2 * (abs(y1 - y2) - abs(x1 - x2));
		p = 2 * abs(y1 - y2) - abs(x1 - x2);
	}
	else {
		// y as main axis
		main_x = false;
		main_length = abs(y1 - y2);
		y = max(y1, y2);
		main_axis_position = y;
		main_axis_step = -1;

		// x as minor axis
		x = (y1 >= y2) ? x1 : x2;
		minor_axis_position = x;
		if (y1 >= y2)
			minor_axis_step = (x1 >= x2) ? -1 : 1;
		else
			minor_axis_step = (x2 >= x1) ? -1 : 1;

		// variables
		k1 = 2 * (abs(x1 - x2));
		k2 = 2 * (abs(x1 - x2) - abs(y1 - y2));
		p = 2 * abs(x1 - x2) - abs(y1 - y2);
	}
	z = (y1 >= y2) ? z1 : z2;
	if (y1 >= y2)
		z_step = (z2 - z1) / ((main_length > 0) ? main_length : 1);
	else
		z_step = (z1 - z2) / ((main_length > 0) ? main_length : 1);


	setPixel(x, y, z);
	while (main_length > 0)
	{
		main_length--;
		main_axis_position += main_axis_step;
		if (p > 0) {
			minor_axis_position += minor_axis_step;
			p += k2;
		}
		else
			p += k1;
		x = (main_x) ? main_axis_position : minor_axis_position;
		y = (main_x) ? minor_axis_position : main_axis_position;
		z += z_step;
		setPixel(x, y, z);
	}
}

void sglClear(unsigned what) {
	if (!(what & SGL_COLOR_BUFFER_BIT) && !(what & SGL_DEPTH_BUFFER_BIT)) {
		setErrCode(SGL_INVALID_VALUE);
		return;
	}
	if (drawing || !active_context) {
		setErrCode(SGL_INVALID_OPERATION);
		return;
	}
	if (what & SGL_COLOR_BUFFER_BIT) {
		float *index_r, *index_g, *index_b;
		int canvas_size = active_context->GetCanvasSize();
		float* canvas = active_context->GetColorBuffer();
		index_r = &canvas[0];
		index_g = &canvas[1];
		index_b = &canvas[2];
		for (int i = 0; i < canvas_size; i++)
		{
			*index_r = clear_r;
			*index_g = clear_g;
			*index_b = clear_b;
			index_r += 3;
			index_g += 3;
			index_b += 3;
		}
	}
	if (what & SGL_DEPTH_BUFFER_BIT) {
		int canvas_size = active_context->GetCanvasSize();
		float* z_buffer = active_context->GetZBuffer();
		float* index = &z_buffer[0];
		for (int i = 0; i < canvas_size; i++)
		{
			*index = numeric_limits<float>::max();
			index++;
		}
	}
	setErrCode(SGL_NO_ERROR);
}

void sglBegin(sglEElementType mode) {
	if (drawing) {
		setErrCode(SGL_INVALID_OPERATION);
		return;
	}
	if (mode < SGL_POINTS || mode > SGL_LAST_ELEMENT_TYPE) { // include last_ele_type as valid?
		setErrCode(SGL_INVALID_ENUM);
		return;
	}
	drawing = true;
	if (mode != SGL_LAST_ELEMENT_TYPE) drawing_object = mode;
	setErrCode(SGL_NO_ERROR);
}

void sglEnd(void) {
	float x, y, z, x2, y2, z2, x3, y3, z3, w, w2, w3;
	int lenght;
	if (!drawing) {
		setErrCode(SGL_INVALID_OPERATION);
		return;
	}
	drawing = false;

	if (drawing_scene) {
		// add primitive to scene

		vector < Vertex* >* vertex_buffer = active_context->GetVertexBuffer();
		vector < Vertex* >* normal_buffer = active_context->GetNormalBuffer();

		switch (drawing_object){
		case SGL_POLYGON:
			//cout << "----" << endl;
			if (active_context->normalsOK()) {
				//cout << "Returned true" << endl;
				//cout << "Given " << normal_buffer->size() << endl;
				//cout << "A" << endl;
				scene->AddPrimitive(SGL_POLYGON, vertex_buffer, normal_buffer, used_material->Emissive());
			}
			else {
				//cout << "returned false" << endl;
				//cout << "B" << endl;
				scene->AddPrimitive(SGL_POLYGON, vertex_buffer, used_material->Emissive());
			}
			vertex_buffer = new vector < Vertex* >; // deletion of vertex buffer moved to the scene primitive
			active_context->SetVertexBuffer(vertex_buffer);
			normal_buffer = new vector < Vertex* >; // deletion of vertex buffer moved to the scene primitive
			active_context->SetNormalBuffer(normal_buffer);
			//cout << "____" << endl;
			break;
		default:
			break;
		}

		active_context->ClearVertexBuffer();
		setErrCode(SGL_NO_ERROR);
		return;
	}

	// do the drawing
	lenght = active_context->GetVertexBuffer()->size();

	switch (drawing_object) {
	case SGL_POINTS:
		for (int i = 0; i < lenght; i++) {
			x = active_context->GetVertexBuffer()->at(i)->getX();
			y = active_context->GetVertexBuffer()->at(i)->getY();
			z = active_context->GetVertexBuffer()->at(i)->getZ();
			w = active_context->GetVertexBuffer()->at(i)->getW();

			vertexTransformation(x, y, z, w);

			SetPoint(x, y, z);
		}
		break;
	case SGL_LINES:
		for (int i = 0; i < lenght; i++) {
			if (i + 1 == lenght) continue;
			if ((i % 2) == 1) continue;
			x = active_context->GetVertexBuffer()->at(i)->getX();
			y = active_context->GetVertexBuffer()->at(i)->getY();
			z = active_context->GetVertexBuffer()->at(i)->getZ();
			w = active_context->GetVertexBuffer()->at(i)->getW();
			x2 = active_context->GetVertexBuffer()->at(i + 1)->getX();
			y2 = active_context->GetVertexBuffer()->at(i + 1)->getY();
			z2 = active_context->GetVertexBuffer()->at(i + 1)->getZ();
			w2 = active_context->GetVertexBuffer()->at(i + 1)->getW();
			vertexTransformation(x, y, z, w);
			vertexTransformation(x2, y2, z2, w2);

			sglLine(x, y, z, x2, y2, z2);
		}
		break;
	case SGL_LINE_STRIP:
		for (int i = 0; i < lenght; i++) {
			if (i + 1 == lenght) continue;
			x = active_context->GetVertexBuffer()->at(i)->getX();
			y = active_context->GetVertexBuffer()->at(i)->getY();
			z = active_context->GetVertexBuffer()->at(i)->getZ();
			w = active_context->GetVertexBuffer()->at(i)->getW();
			x2 = active_context->GetVertexBuffer()->at(i + 1)->getX();
			y2 = active_context->GetVertexBuffer()->at(i + 1)->getY();
			z2 = active_context->GetVertexBuffer()->at(i + 1)->getZ();
			w2 = active_context->GetVertexBuffer()->at(i + 1)->getW();
			vertexTransformation(x, y, z, w);
			vertexTransformation(x2, y2, z2, w2);

			sglLine(x, y, z, x2, y2, z2);
		}
		break;
	case SGL_POLYGON:
		// podle area mode vyplnit
		if (fill_method == SGL_POINT) {
			for (int i = 0; i < lenght; i++) {
				x = active_context->GetVertexBuffer()->at(i)->getX();
				y = active_context->GetVertexBuffer()->at(i)->getY();
				z = active_context->GetVertexBuffer()->at(i)->getZ();
				w = active_context->GetVertexBuffer()->at(i)->getW();

				vertexTransformation(x, y, z, w);
				SetPoint(x, y, z);
			}
			break;
		}
		if (fill_method == SGL_FILL) {
			vector<Edge> edges;
			for (int i = 0; i < lenght; i++) {
				x = active_context->GetVertexBuffer()->at(i)->getX();
				y = active_context->GetVertexBuffer()->at(i)->getY();
				z = active_context->GetVertexBuffer()->at(i)->getZ();
				w = active_context->GetVertexBuffer()->at(i)->getW();
				if (i + 1 == lenght) {
					x2 = active_context->GetVertexBuffer()->at(0)->getX();
					y2 = active_context->GetVertexBuffer()->at(0)->getY();
					z2 = active_context->GetVertexBuffer()->at(0)->getZ();
					w2 = active_context->GetVertexBuffer()->at(0)->getW();
				}
				else {
					x2 = active_context->GetVertexBuffer()->at(i + 1)->getX();
					y2 = active_context->GetVertexBuffer()->at(i + 1)->getY();
					z2 = active_context->GetVertexBuffer()->at(i + 1)->getZ();
					w2 = active_context->GetVertexBuffer()->at(i + 1)->getW();
				}
				vertexTransformation(x, y, z, w);
				vertexTransformation(x2, y2, z2, w2);

				// fill edges of the area
				Edge newEdge = Edge(x, y, z, x2, y2, z2);
				if (newEdge.GetDeltaY() != 0)
					edges.push_back(newEdge);
			}

			// sort by y
			std::sort(edges.begin(), edges.end(), EdgeSort); // decreasing sort by start_y

			// fill area
			vector<pair<float, float>> border_positions; // boardes positions on current row
			while (!edges.empty()) {
				int current_y = edges[0].GetPositionY(); // most upper y
				border_positions.clear();

				// get border positions x
				unsigned int index = 0;
				while (index < edges.size()) {
					border_positions.push_back(pair<float, float>(edges[index].GetPositionX(), edges[index].GetPositionZ()));
					if (!edges[index].NextRow()) {
						// end of this line reached
						edges.erase(edges.begin() + index);
					}
					else
						index++;

					if (index < edges.size() && (int)edges[index].GetPositionY() < current_y) {
						break;
					}
				}

				// draw the area
				std::sort(border_positions.begin(), border_positions.end(), PairSort);
				for (size_t i = 0; i + 1 < border_positions.size(); i += 2){
					setPixelHorizontalLine(
						border_positions[i].first, border_positions[i + 1].first,
						current_y,
						int(border_positions[i].second), int(border_positions[i + 1].second));
				}
			}
			// - draw the border line => no break
		}
		// SGL_LINE as SGL_LINE_LOOP => no break
	case SGL_LINE_LOOP:
		// Assumes at least two vertices
		for (int i = 0; i < lenght; i++) {
			x = active_context->GetVertexBuffer()->at(i)->getX();
			y = active_context->GetVertexBuffer()->at(i)->getY();
			z = active_context->GetVertexBuffer()->at(i)->getZ();
			w = active_context->GetVertexBuffer()->at(i)->getW();
			if (i + 1 == lenght) {
				x2 = active_context->GetVertexBuffer()->at(0)->getX();
				y2 = active_context->GetVertexBuffer()->at(0)->getY();
				z2 = active_context->GetVertexBuffer()->at(0)->getZ();
				w2 = active_context->GetVertexBuffer()->at(0)->getW();
			}
			else {
				x2 = active_context->GetVertexBuffer()->at(i + 1)->getX();
				y2 = active_context->GetVertexBuffer()->at(i + 1)->getY();
				z2 = active_context->GetVertexBuffer()->at(i + 1)->getZ();
				w2 = active_context->GetVertexBuffer()->at(i + 1)->getW();
			}
			vertexTransformation(x, y, z, w);
			vertexTransformation(x2, y2, z2, w2);

			sglLine(x, y, z, x2, y2, z2);
		}
		break;
	case SGL_TRIANGLES:
		for (int i = 0; i < lenght; i = i + 3) {
			if (lenght - i<3) continue;
			x = active_context->GetVertexBuffer()->at(i)->getX();
			y = active_context->GetVertexBuffer()->at(i)->getY();
			z = active_context->GetVertexBuffer()->at(i)->getZ();
			w = active_context->GetVertexBuffer()->at(i)->getW();
			x2 = active_context->GetVertexBuffer()->at(i + 1)->getX();
			y2 = active_context->GetVertexBuffer()->at(i + 1)->getY();
			z2 = active_context->GetVertexBuffer()->at(i + 1)->getZ();
			w2 = active_context->GetVertexBuffer()->at(i + 1)->getW();
			x3 = active_context->GetVertexBuffer()->at(i + 2)->getX();
			y3 = active_context->GetVertexBuffer()->at(i + 2)->getY();
			z3 = active_context->GetVertexBuffer()->at(i + 2)->getZ();
			w3 = active_context->GetVertexBuffer()->at(i + 2)->getW();
			vertexTransformation(x, y, z, w);
			vertexTransformation(x2, y2, z2, w2);
			vertexTransformation(x3, y3, z3, w3);

			sglLine(x, y, z, x2, y2, z2);
			sglLine(x, y, z, x3, y3, z3);
			sglLine(x3, y3, z3, x2, y2, z2);
		}
		break;
	case SGL_AREA_LIGHT:

		break;
	case SGL_LAST_ELEMENT_TYPE:
		// handled above
		break;

	}
	active_context->ClearVertexBuffer();
	setErrCode(SGL_NO_ERROR);
}

void sglVertex4f(float x, float y, float z, float w) {
	if (!drawing || !active_context) return;
	active_context->AddVertex(x, y, z, w);
}
void sglVertex3f(float x, float y, float z) {
	if (!drawing || !active_context) return;
	active_context->AddVertex(x, y, z, 1);
}
void sglVertex3f(float vx, float vy, float vz, float nx, float ny, float nz) {
	if (!drawing || !active_context) return;
	active_context->AddVertex(vx, vy, vz, 1);
	active_context->AddNormal(nx, ny, nz, 1);
}
void sglVertex2f(float x, float y) {
	/// Input of a vertex. Assumes z=0, w=1.
	if (!drawing || !active_context) return;
	active_context->AddVertex(x, y, 0, 1);
}

void sglCircle(float x, float y, float z, float radius) {
	/// Circle drawing
	float w = 1;
	if (radius < 0) {
		setErrCode(SGL_INVALID_VALUE);
		return;
	}
	INVALID_OPERATION_CHECK;

	vertexTransformation(x, y, z, w);

	if (!TransformationStack::scaleActualized)
		setScale();
	radius *= scale;
	radius = (float)(int)radius;

	if (fill_method == SGL_POINT) {
		SetPoint(x, y, z);
		return;
	}

	int x_0 = int(x);
	int y_0 = int(y);

	float dvex = 3;
	float dvey = radius + radius - 2;
	float p = 1 - radius;

	x = 0;
	y = (float)(int)radius;
	while (x <= y) {
		// draw in 8 directions
		if (fill_method == SGL_FILL) {
			// fill area
			setPixelHorizontalLine(x_0, x_0 + int(x), y_0 + int(y), z, z);
			setPixelHorizontalLine(x_0, x_0 + int(x), y_0 - int(y), z, z);
			setPixelHorizontalLine(x_0 - int(x), x_0, y_0 + int(y), z, z);
			setPixelHorizontalLine(x_0 - int(x), x_0, y_0 - int(y), z, z);

			setPixelHorizontalLine(x_0, x_0 + int(y), y_0 + int(x), z, z);
			setPixelHorizontalLine(x_0, x_0 + int(y), y_0 - int(x), z, z);
			setPixelHorizontalLine(x_0 - int(y), x_0, y_0 + int(x), z, z);
			setPixelHorizontalLine(x_0 - int(y), x_0, y_0 - int(x), z, z);
		}
		else {
			// borders
			setPixel(x_0 + int(x), y_0 + int(y), z);
			setPixel(x_0 + int(x), y_0 - int(y), z);
			setPixel(x_0 - int(x), y_0 + int(y), z);
			setPixel(x_0 - int(x), y_0 - int(y), z);

			setPixel(x_0 + int(y), y_0 + int(x), z);
			setPixel(x_0 + int(y), y_0 - int(x), z);
			setPixel(x_0 - int(y), y_0 + int(x), z);
			setPixel(x_0 - int(y), y_0 - int(x), z);
		}

		if (p >= 0) {
			p -= dvey;
			dvey -= 2;
			y--;
		}
		p += dvex;
		dvex += 2;
		x++;
	}

	setErrCode(SGL_NO_ERROR);
}

void sglEllipse(float x, float y, float z, float a, float b) {
	vector<Vertex*> prvni, druhy, treti, ctvrty;
	Vertex* vert;

	if ((a < 0) || (b < 0)) {
		setErrCode(SGL_INVALID_VALUE);
		return;
	}
	INVALID_OPERATION_CHECK;

	float a2 = a*a;
	float b2 = b*b;
	float a2b2 = a2*b2;
	float x_0 = x;
	float y_0 = y;
	float x_swap = (a*a) / sqrt(a*a + b*b);
	float y_swap = (b*b) / sqrt(a*a + b*b);
	float x_step = x_swap / 5;
	float y_step = y_swap / 5;

	switch (fill_method) {
	case SGL_FILL:
		sglBegin(SGL_POLYGON);
		break;
	case SGL_POINT:
		sglBegin(SGL_POINTS);
		sglVertex3f(x, y, z);
		sglEnd();
		return;
		break;
	case SGL_LINE:
		sglBegin(SGL_LINE_LOOP);
		break;
	}

	x = 0;
	y = b;

	for (size_t i = 0; i < 5; i++) {
		// draw in 4 directions
		vert = new Vertex(x_0 + x, y_0 + y, z, 1);
		prvni.push_back(vert);
		vert = new Vertex(x_0 + x, y_0 - y, z, 1);
		druhy.push_back(vert);
		vert = new Vertex(x_0 - x, y_0 + y, z, 1);
		treti.push_back(vert);
		vert = new Vertex(x_0 - x, y_0 - y, z, 1);
		ctvrty.push_back(vert);

		x = x + x_step;
		y = sqrt(((a2b2)-(b2*x*x)) / (a2));

	}

	y = y_swap;
	x = x_swap;

	for (size_t i = 0; i < 5; i++) {
		// draw in 4 directions
		vert = new Vertex(x_0 + x, y_0 + y, z, 1);
		prvni.push_back(vert);
		vert = new Vertex(x_0 + x, y_0 - y, z, 1);
		druhy.push_back(vert);
		vert = new Vertex(x_0 - x, y_0 + y, z, 1);
		treti.push_back(vert);
		vert = new Vertex(x_0 - x, y_0 - y, z, 1);
		ctvrty.push_back(vert);

		y = y - y_step;
		x = sqrt(((a2b2)-(a2*y*y)) / (b2));

	}

	vert = new Vertex(x_0 + x, y_0 + y, z, 1);
	prvni.push_back(vert);
	vert = new Vertex(x_0 + x, y_0 - y, z, 1);
	druhy.push_back(vert);
	vert = new Vertex(x_0 - x, y_0 + y, z, 1);
	treti.push_back(vert);
	vert = new Vertex(x_0 - x, y_0 - y, z, 1);
	ctvrty.push_back(vert);

	for (unsigned int i = 0; i < prvni.size(); i++)
	{
		sglVertex3f(prvni[i]->getX(), prvni[i]->getY(), prvni[i]->getZ());
		delete prvni[i];
	}
	for (int i = druhy.size() - 1; i >= 0; i--)
	{
		sglVertex3f(druhy[i]->getX(), druhy[i]->getY(), druhy[i]->getZ());
		delete druhy[i];
	}
	for (unsigned int i = 0; i < ctvrty.size(); i++)
	{
		sglVertex3f(ctvrty[i]->getX(), ctvrty[i]->getY(), ctvrty[i]->getZ());
		delete ctvrty[i];
	}
	for (int i = treti.size() - 1; i >= 0; i--)
	{
		sglVertex3f(treti[i]->getX(), treti[i]->getY(), treti[i]->getZ());
		delete treti[i];
	}

	sglEnd();

	setErrCode(SGL_NO_ERROR);
}

void sglArc(float x, float y, float z, float radius, float from, float to) {
	/// Circular arc drawing.
	/**
	@param x   [in] circle center x
	@param y   [in] circle center y
	@param z   [in] circle center z
	@param radius [in] circle radius
	@param from   [in] starting angle (measured counterclockwise from the x-axis)
	@param to     [in] ending angle (measured counterclockwise from the x-axis)

	Use approximation by a linestrip / polygon of exactly 40*(to-from)/(2*PI) vertices
	(so that the speed measurements are fair).

	See sglEllipse() for more info.
	*/
	if (radius < 0) {
		setErrCode(SGL_INVALID_VALUE);
		return;
	}
	INVALID_OPERATION_CHECK;

	float x_act, y_act;
	float angle = from;
	float x_from = x + cos(from)*radius;
	float y_from = y + sin(from)*radius;
	float x_to = x + cos(to)*radius;
	float y_to = y + sin(to)*radius;
	float step = (float)((2 * M_PI) / 40.0f);

	switch (fill_method) {
	case SGL_FILL:
		sglBegin(SGL_POLYGON);
		sglVertex3f(x, y, z);
		break;
	case SGL_POINT:
		sglBegin(SGL_POINTS);
		sglVertex3f(x, y, z);
		sglEnd();
		//sglBegin(SGL_LINE_STRIP);
		return;
		break;
	case SGL_LINE:
		sglBegin(SGL_LINE_STRIP);
		break;
	}


	sglVertex3f(x_from, y_from, z);

	for (size_t i = 0; i < (40 * (to - from) / (2 * M_PI)) - 2; i++)
	{
		angle = angle + step;

		x_act = x + cos(angle)*radius;
		y_act = y + sin(angle)*radius;

		sglVertex3f(x_act, y_act, z);
	}

	sglVertex3f(x_to, y_to, z);

	sglEnd();

	setErrCode(SGL_NO_ERROR);
}

//---------------------------------------------------------------------------
// Transform functions
//---------------------------------------------------------------------------

void sglMatrixMode(sglEMatrixMode mode) {
	// SGL_INVALID_OPERATION

	switch (mode) {
	case SGL_MODELVIEW:
		activeStack = modelViewStack;
		break;
	case SGL_PROJECTION:
		activeStack = projectionStack;
		break;
	default:
		setErrCode(SGL_INVALID_ENUM);
		return;
	}

	setErrCode(SGL_NO_ERROR);
}

void sglPushMatrix(void) {
	/*
	ERRORS:
	- SGL_STACK_OVERFLOW
	is generated if sglPushMatrix is called while the current
	matrix stack is full. */
	if (drawing || !active_context) {
		setErrCode(SGL_INVALID_OPERATION);
		return;
	}

	//sglPushMatrix duplicates the current matrix on the stack.
	float* newMatrix = new float[16]();
	float* currentMatrix = activeStack->GetCurrent();
	activeStack->push();
	for (size_t i = 0; i < 16; i++) {
		newMatrix[i] = currentMatrix[i];
	}
	activeStack->setCurrent(newMatrix);
	setErrCode(SGL_NO_ERROR);
}

void sglPopMatrix(void) {
	INVALID_OPERATION_CHECK;
	if (activeStack->checkEmptyStack()) {
		setErrCode(SGL_STACK_UNDERFLOW);
		return;
	}

	activeStack->pop();
	setErrCode(SGL_NO_ERROR);
}

void sglLoadIdentity(void) {
	INVALID_OPERATION_CHECK;
	// possible removing the allocation if current!=NULL
	float* identity = new float[16]();
	for (size_t i = 0; i < 16; i++) {
		identity[i] = 0;
	}
	identity[0] = identity[5] = identity[10] = identity[15] = 1;

	activeStack->setCurrent(identity);
	setErrCode(SGL_NO_ERROR);
}

void sglLoadMatrix(const float *matrix) {
	INVALID_OPERATION_CHECK;
	float* newMatrix = new float[16]();

	for (size_t i = 0; i < 16; i++){
		newMatrix[i] = matrix[i];
	}

	activeStack->setCurrent(newMatrix);
	setErrCode(SGL_NO_ERROR);
}

void sglMultMatrix(const float *matrix) {
	INVALID_OPERATION_CHECK;

	MultMatrix(activeStack->GetCurrent(), matrix);

	TransformationStack::scaleActualized = false;
	setErrCode(SGL_NO_ERROR);
}

void sglTranslate(float x, float y, float z) {
	INVALID_OPERATION_CHECK;
	float translate[16] = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		x, y, z, 1
	};

	// multiply
	sglMultMatrix(translate);
	setErrCode(SGL_NO_ERROR);
}

void sglScale(float scalex, float scaley, float scalez) {
	INVALID_OPERATION_CHECK;
	float scale[16] = {
		scalex, 0, 0, 0,
		0, scaley, 0, 0,
		0, 0, scalez, 0,
		0, 0, 0, 1
	};

	// multiply
	sglMultMatrix(scale);
	setErrCode(SGL_NO_ERROR);
}

void sglRotateY(float angle) {
	INVALID_OPERATION_CHECK;
	float rotate[16] = {
		cos(angle), 0, sin(angle), 0,
		0, 1, 0, 0,
		(-1)*sin(angle), 0, cos(angle), 0,
		0, 0, 0, 1
	};

	// multiply
	sglMultMatrix(rotate);
	setErrCode(SGL_NO_ERROR);
}

void sglRotate2D(float angle, float centerx, float centery) {
	INVALID_OPERATION_CHECK;
	float rotate[16] = {
		cos(angle), sin(angle), 0, 0,
		(-1)*sin(angle), cos(angle), 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	};
	// posun
	// otoceni
	// posun zpet
	sglTranslate(centerx, centery, 0);
	sglMultMatrix(rotate);
	sglTranslate(-centerx, -centery, 0);

	setErrCode(SGL_NO_ERROR);
}

void sglOrtho(float left, float right, float bottom, float top, float near, float far) {
	INVALID_OPERATION_CHECK;

	float orthographic[16] = {
		2 / (right - left), 0, 0, 0,
		0, 2 / (top - bottom), 0, 0,
		0, 0, -2 / (far - near), 0,
		-((right + left) / (right - left)), -((top + bottom) / (top - bottom)), -((far + near) / (far - near)), 1
	};

	// multiply
	sglMultMatrix(orthographic);
	setErrCode(SGL_NO_ERROR);
}

void sglFrustum(float left, float right, float bottom, float top, float near, float far) {
	INVALID_OPERATION_CHECK;
	float perspective[16] = {
		near, 0, 0, 0,
		0, near, 0, 0,
		0, 0, near + far, -1,
		0, 0, far*near, 0
	};

	/*float perspective[16] = {
	2*near/(right-left), 0, 0, 0,
	0, 2*near/(top-bottom), 0, 0,
	(right+left)/(right-left), (top+bottom)/(top - bottom), (-1)*(near + far)/(far-near), -1,
	0, 0, (-2)*far*near/(far-near), 0
	};*/

	sglOrtho(left, right, bottom, top, near, far);

	// multiply
	sglMultMatrix(perspective);
	setErrCode(SGL_NO_ERROR);
}

void sglViewport(int x, int y, int width, int height) {
	if (width < 0 || height < 0) {
		setErrCode(SGL_INVALID_VALUE);
		return;
	}
	INVALID_OPERATION_CHECK_WITHOUT_STACK;

	for (size_t i = 0; i < 16; i++)
		viewPortTransformation[i] = 0;
	viewPortTransformation[0] = width / 2.0f;
	viewPortTransformation[5] = height / 2.0f;
	viewPortTransformation[10] = 1.0f; // depth / 2;
	viewPortTransformation[12] = x + width / 2.0f;
	viewPortTransformation[13] = y + height / 2.0f;
	viewPortTransformation[14] = 0.0f; // z + depth / 2;
	viewPortTransformation[15] = 1.0f;

	TransformationStack::scaleActualized = false;
	setErrCode(SGL_NO_ERROR);
}

//---------------------------------------------------------------------------
// Attribute functions
//---------------------------------------------------------------------------

void sglClearColor(float r, float g, float b, float alpha) {
	INVALID_OPERATION_CHECK_WITHOUT_STACK;

	clear_r = r;
	clear_g = g;
	clear_b = b;
	// Aplpha is ignored.
}

void sglColor3f(float r, float g, float b) {
	draw_r = r;
	draw_g = g;
	draw_b = b;

	if (used_material)
		delete used_material;
	used_material = NULL;
}

void sglAreaMode(sglEAreaMode mode) {
	/// Drawing mode of closed areas.
	INVALID_OPERATION_CHECK_WITHOUT_STACK;
	if (mode != SGL_POINT && mode != SGL_LINE && mode != SGL_FILL) {
		setErrCode(SGL_INVALID_ENUM);
		return;
	}

	fill_method = mode;
	setErrCode(SGL_NO_ERROR);
}

void sglPointSize(float size) {
	if (size <= 0) {
		setErrCode(SGL_INVALID_VALUE);
		return;
	}
	INVALID_OPERATION_CHECK_WITHOUT_STACK;

	pointSize = size;

	setErrCode(SGL_NO_ERROR);
}

void sglEnable(sglEEnableFlags cap) {
	INVALID_OPERATION_CHECK_WITHOUT_STACK;
	if (cap != SGL_DEPTH_TEST){
		setErrCode(SGL_INVALID_ENUM);
		return;
	}

	//SGL_DEPTH_TEST == 1;
	depth_test_enabled = true;
}

void sglDisable(sglEEnableFlags cap) {
	INVALID_OPERATION_CHECK_WITHOUT_STACK;
	if (cap != SGL_DEPTH_TEST){
		setErrCode(SGL_INVALID_ENUM);
		return;
	}
	depth_test_enabled = false;
}

//---------------------------------------------------------------------------
// RayTracing oriented functions
//---------------------------------------------------------------------------

void sglBeginScene() {
	/// Starts a definition of the scene.
	/// Any primitves specified by GL_POLYGON (in begin/end) or sglSphere
	/// will be added to the scene primitive lists
	/// If the scene is not empty it is cleared.
	/// During the scene definition no transformations applied to the
	/// primitive vertices!

	INVALID_OPERATION_CHECK_WITHOUT_STACK;

	drawing_scene = true;
	if (scene)
		delete scene;
	scene = new SceneStructure();
	if (kd_scene)
		delete scene;
	kd_scene = new KD_Tree();
	setErrCode(SGL_NO_ERROR);
}

void sglEndScene() {
	/// Ends a definition of the scene.

	INVALID_OPERATION_CHECK_WITHOUT_STACK;
	if (!drawing_scene) {
		setErrCode(SGL_INVALID_OPERATION);
		return;
	}

	drawing_scene = false;
	setErrCode(SGL_NO_ERROR);
}

void sglSphere(const float x,
	const float y,
	const float z,
	const float radius) {
	/// Definition of the sphere primitve.
	/// This function can only be used with scene definition, where it adds
	/// the sphere to the primitive list

	INVALID_OPERATION_CHECK_WITHOUT_STACK;
	if (!drawing_scene) {
		setErrCode(SGL_INVALID_OPERATION);
		return;
	}
	//if (DrawObject::number > 40) return; // test .. remove one sphere
	scene->AddPrimitive(x, y, z, radius);
	setErrCode(SGL_NO_ERROR);
}

void sglMaterial(const float r,
	const float g,
	const float b,
	const float kd,
	const float ks,
	const float shine,
	const float T,
	const float ior) {
	/// Input of a material using a Phong model.
	INVALID_OPERATION_CHECK_WITHOUT_STACK;

	if (used_material)
		delete used_material;
	used_material = new Material(r, g, b, kd, ks, shine, T, ior);
}

void sglPointLight(const float x,
	const float y,
	const float z,
	const float r,
	const float g,
	const float b) {
	/// Input of a point light.

	INVALID_OPERATION_CHECK_WITHOUT_STACK;
	if (!drawing_scene) {
		setErrCode(SGL_INVALID_OPERATION);
		return;
	}

	scene->AddLight(x, y, z, r, g, b);
	setErrCode(SGL_NO_ERROR);
}



void trace() {
	int x = 0, y = 0;
	float z = 0;

	setPixel(int(x), int(y), z);

}

void sglBuildKdTree() {
	cout << "Amount of triangles: " << scene->scene_triangles.size() << endl;
	kd_scene->doTheBuild(scene->scene_triangles);

}

void sglRayTraceScene() {
	DrawObject* draw_obj;
	//DrawObject *jedna, *druhy;

	//kd_scene->printOutTree();

	depth_test_enabled = false;

	float inv_matrix[16] = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	};
	if (viewPortTransformation)
		MultMatrix(inv_matrix, viewPortTransformation);
	MultMatrix(inv_matrix, projectionStack->GetCurrent());
	MultMatrix(inv_matrix, modelViewStack->GetCurrent());

	InvertMatrix(inv_matrix);


	float inv_modelview_matrix[16] = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	};	// invert matrix from ray position
	MultMatrix(inv_modelview_matrix, modelViewStack->GetCurrent());
	InvertMatrix(inv_modelview_matrix);

	float ray_start[4] = { 0, 0, 0, 1 };
	MultVector(ray_start, inv_modelview_matrix);

	for (float y = 0; y < active_context->GetHeight(); y++){
		for (float x = 0; x < active_context->GetWidth(); x++)
		{
			float dist;
			float ray_end[4] = { x, y, -1, 1 };
			MultVector(ray_end, inv_matrix);

			for (size_t i = 0; i < 3; i++) {
				ray_end[i] = ray_end[i] / ray_end[3];
			}

			Ray r(ray_start[0], ray_start[1], ray_start[2], ray_end[0], ray_end[1], ray_end[2]);
			/*if (x == 650 && y == 400) {
				kd_scene->FindKDIntersection(r, dist);
				draw_r = 1.0;
				draw_g = 0.0;
				draw_b = 0.0;
				setPixel(x, y, 1);
				continue;
			} else*/ {
			
			/*if (jedna || druhy) {
				cout << "--------" << endl;
				cout << r.direction->getX() << "-" << r.direction->getY() << "-" << r.direction->getZ() << endl;
				cout << "Normal: " << jedna << endl;
				cout << "kd: " << druhy << endl;
			}*/

			 draw_obj = scene->FindIntersection(r, dist);
			 //draw_obj = kd_scene->FindKDIntersection(r, dist);

			//if (draw_obj) cout << draw_obj << endl;
			}

			if (draw_obj) {
				Vector3f color = scene->Illuminate(draw_obj, r, dist, 0);
				draw_r = color.x; 
				draw_g = color.y;
				draw_b = color.z;
				setPixel(int(x), int(y), 1);
			}
			else {
				if (env_texels) {
					Vector3f color = setEnvMapColor(r.direction->getX(), r.direction->getY(), r.direction->getZ());
					draw_r = color.x;
					draw_g = color.y;
					draw_b = color.z;
					setPixel(int(x), int(y), 1);
				}
			}
		}
	}

	/*cout << "Total rays: " << rays << endl;
	cout << "Total trav. steps: " << trav_steps << endl;
	cout << "Total inters. queries: " << inters_steps << endl;
	cout << "Average trav. steps: " << float(trav_steps)/float(rays) << endl;
	cout << "Average inters. queries: " << float(inters_steps) / float(rays) << endl;*/


}

void sglRasterizeScene() {}

void sglEnvironmentMap(const int width,
	const int height,
	float *texels)
{
	env_width = width;
	env_height = height;
	env_texels = texels;
}

void sglEmissiveMaterial(
	const float r,
	const float g,
	const float b,
	const float c0,
	const float c1,
	const float c2
	)
{
	INVALID_OPERATION_CHECK_WITHOUT_STACK;

	if (used_material)
		delete used_material;
	used_material = new MaterialEmissive(r, g, b, c0, c1, c2);
}