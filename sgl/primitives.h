//---------------------------------------------------------------------------
// primitives.h
// Header file for primitives definition
// Date:  2017/06
// Author: Michal Kuèera, CTU Prague
//---------------------------------------------------------------------------

#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <algorithm>
#include <stack>


#define NULL 0

using namespace std;

class Vertex;
class Vector3f;

										/// Types of graphic elements which are specified using the sglVertexXf command
enum sglEElementType {
	/// Points
	SGL_POINTS = 1,
	/// Lines
	SGL_LINES,
	/// Line strip
	SGL_LINE_STRIP,
	/// Closed line strip
	SGL_LINE_LOOP,
	/// Triangle list
	SGL_TRIANGLES,
	/// General, non-convex polygon
	SGL_POLYGON,
	/// Area light - restricted to a quad for simplicity
	SGL_AREA_LIGHT,
	SGL_LAST_ELEMENT_TYPE
};

// ==========================================================================
//  Primitives and structures definitions
// ==========================================================================

class Vertex
{
public:
	Vertex() { x = 0; y = 0; z = 0; };
	Vertex(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {};
	float getX() { return x; }
	float getY() { return y; }
	float getZ() { return z; }
	float getW() { return w; }
	Vertex normalize() {
		float length = sqrt(x*x + y*y + z*z);
		return Vertex(x / length, y / length, z / length, 1);
	}
	void divideByW() {
		x = x / w;
		y = y / w;
		z = z / w;
	}
	void normalizeThis() {
		float length = sqrt(x*x + y*y + z*z);

		this->x = this->x / length;
		this->y = this->y / length;
		this->z = this->z / length;
	}
	Vertex operator-(Vertex v2) { return Vertex(x - v2.getX(), y - v2.getY(), z - v2.getZ(), 1); }
	Vertex operator+(Vertex v2) { return Vertex(x + v2.getX(), y + v2.getY(), z + v2.getZ(), 1); }
	Vertex operator*(float v) { return Vertex(v*x, v*y, v*z, 1); }
	static float DotProduct(Vertex & v1, Vertex v2) { return (v1.getX()*v2.getX() + v1.getY()*v2.getY() + v1.getZ()*v2.getZ()); }
	static float Distance(Vertex v1, Vertex v2) {
		float x = v2.getX() - v1.getX();
		float y = v2.getY() - v1.getY();
		float z = v2.getZ() - v1.getZ();
		return sqrtf(x*x + y*y + z*z);
	}
private:
	float x, y, z, w;
};

class Vector3f
{
public:
	Vector3f() { x = 0; y = 0; z = 0; };
	Vector3f(Vertex* old) { x = old->getX(); y = old->getY(); z = old->getZ(); };
	Vector3f(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {};
	Vector3f operator-(Vector3f v2) { return Vector3f(x - v2.x, y - v2.y, z - v2.z); }
	Vector3f operator-(Vertex* v2) { return Vector3f(x - v2->getX(), y - v2->getY(), z - v2->getZ()); }
	Vector3f operator-(Vertex v2) { return Vector3f(x - v2.getX(), y - v2.getY(), z - v2.getZ()); }
	Vector3f operator+(Vertex v2) { return Vector3f(x + v2.getX(), y + v2.getY(), z + v2.getZ()); }
	Vector3f operator+(Vector3f v2) { return Vector3f(x + v2.x, y + v2.y, z + v2.z); }
	Vector3f operator*(float v) { return Vector3f(v*x, v*y, v*z); }
	Vector3f operator*(Vertex v2) { return Vector3f(x * v2.getX(), y * v2.getY(), z * v2.getZ()); }
	Vector3f operator*(Vector3f v2) { return Vector3f(x * v2.x, y * v2.y, z * v2.z); }
	Vector3f operator/(float v) { return Vector3f(v / x, v / y, v / z); }
	void operator=(Vertex v2) { x = v2.getX(); y = v2.getY(); z = v2.getZ(); }
	void operator=(Vector3f v2) { x = v2.x; y = v2.y; z = v2.z; }
	static float DotProduct(Vector3f & v1, Vector3f & v2) { return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z); }
	void Normalize() {
		float length = sqrt(Vector3f::DotProduct(*this, *this));
		if (length != 0) {
			x /= length;
			y /= length;
			z /= length;
		}
	}
	float x, y, z;
};


float dotProduct(Vector3f a, Vertex b);

float dotProduct(Vertex a, Vertex b);

class Edge {
public:
	Edge(float x1, float y1, float z1, float x2, float y2, float z2) {
		x1 = (int)x1; x2 = (int)x2; y1 = (int)y1; y2 = (int)y2;
		int main_length;

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
		delta_y = abs(y2 - y1);

		end_position = min(y1, y2) + 1;
	}
	float GetPositionX() { return x; }
	float GetPositionY() { return y; }
	float GetPositionZ() { return z; }
	int GetDeltaY() { return delta_y; }
	bool NextRow() {
		int old_position = (main_x) ? minor_axis_position : main_axis_position;
		int new_position = old_position;
		while (old_position == new_position)
		{
			main_axis_position += main_axis_step;
			if (p > 0) {
				minor_axis_position += minor_axis_step;
				p += k2;
			}
			else
				p += k1;
			z += z_step;
			new_position = (main_x) ? minor_axis_position : main_axis_position;
		}

		x = (main_x) ? main_axis_position : minor_axis_position;
		y = (main_x) ? minor_axis_position : main_axis_position;
		return (y >= end_position);
	}
private:
	float x, y, z; // starting position (upper vertex)
	int delta_y; // amount of rows the edge is in
	int end_position;
	bool main_x;
	int main_axis_position, minor_axis_position;
	int main_axis_step, minor_axis_step;
	float z_step;
	int k1, k2;
	float p;
};

bool EdgeSort(Edge e1, Edge e2);

bool PairSort(pair<float, float> p1, pair<float, float> p2);

class ImageStructure
{
public:
	ImageStructure(int _id, int _width, int _height) : id(_id), width(_width), height(_height) {
		canvas_size = width * height;
		canvas = new float[3 * canvas_size];
		z_buffer = new float[canvas_size];
		vertex_buffer = new vector < Vertex* >;
		normal_buffer = new vector < Vertex* >;

		// init
		float *index, *index_r, *index_g, *index_b;
		index_r = &canvas[0];
		index_g = &canvas[1];
		index_b = &canvas[2];
		index = &z_buffer[0];
		for (unsigned int i = 0; i < canvas_size; i++)
		{
			*index_r = 0;
			*index_g = 0;
			*index_b = 0;
			*index = numeric_limits<float>::max();
			index_r += 3;
			index_g += 3;
			index_b += 3;
			index++;
		}
	}
	~ImageStructure() {
		delete[] canvas;
		delete[] z_buffer;
		ClearVertexBuffer();
		delete vertex_buffer;
		delete normal_buffer;
	}
	int GetContextID() { return id; }
	int GetWidth() { return width; }
	int GetHeight() { return height; }
	int GetCanvasSize() { return canvas_size; }
	float* GetColorBuffer() { return canvas; }
	float* GetZBuffer() { return z_buffer; }
	vector<Vertex*>* GetVertexBuffer() { return vertex_buffer; }
	vector<Vertex*>* GetNormalBuffer() { return normal_buffer; }
	void AddVertex(float _x, float _y, float _z = 0, float _w = 1) {
		vertex_buffer->push_back(new Vertex(_x, _y, _z, _w));
	}
	void AddNormal(float _x, float _y, float _z = 0, float _w = 1) {
		normal_buffer->push_back(new Vertex(_x, _y, _z, _w));
	}
	void ClearVertexBuffer() {
		for (vector<Vertex*>::iterator it = vertex_buffer->begin(); it != vertex_buffer->end(); ++it)
			delete (*it);
		vertex_buffer->clear();
		for (vector<Vertex*>::iterator it = normal_buffer->begin(); it != normal_buffer->end(); ++it)
			delete (*it);
		normal_buffer->clear();
	}
	void SetVertexBuffer(vector<Vertex*>* new_vertex_buffer) {
		vertex_buffer = new_vertex_buffer;
	}
	void SetNormalBuffer(vector<Vertex*>* new_normal_buffer) {
		normal_buffer = new_normal_buffer;
	}
	bool normalsOK() {
		//cout << this->normal_buffer->size() << " vs. " << this->vertex_buffer->size() << endl;
		if (this->normal_buffer->size() == this->vertex_buffer->size()) return true;
		return false;
	}

private:
	int id;
	unsigned int width, height;
	unsigned int canvas_size;
	float* canvas; //[RGB] x width x height
	float* z_buffer;
	vector<Vertex*>* vertex_buffer;
	vector<Vertex*>* normal_buffer;
};

class TransformationStack {
public:
	static bool scaleActualized;					// flag if scale needs recalculation

	TransformationStack() {
		this->current = NULL;
	}
	~TransformationStack() {
		while (!stack.empty()) {
			delete[] this->current;
			current = this->stack.back();
			this->stack.pop_back();
		}
		delete[] this->current;
	}
	float* GetCurrent() {
		return current;
	}
	void setCurrent(float* matrix) {
		if (this->current != NULL) delete[] this->current;
		this->current = matrix;

		TransformationStack::scaleActualized = false;
	}
	void pop() {
		delete[] this->current;
		if (!stack.empty()) {
			current = this->stack.back();
			this->stack.pop_back();
		}
		else
			current = NULL;
	}
	void push() {
		this->stack.push_back(this->current);
		current = NULL;
	}
	bool checkEmptyStack() { return stack.empty(); }
private:
	float* current;
	vector<float*> stack;
};

class Ray
{
public:
	Ray(float x_start, float y_start, float z_start, float x_end, float y_end, float z_end) {
		origin = new Vertex(x_start, y_start, z_start, 1);
		direction = new Vertex(x_end - x_start, y_end - y_start, z_end - z_start, 1);
		direction->normalizeThis();
	}
	~Ray() {
		delete origin;
		delete direction;
	}

	void adjust(float x_start, float y_start, float z_start, float x_end, float y_end, float z_end) {
		delete origin;
		delete direction;
		origin = new Vertex(x_start, y_start, z_start, 1);
		direction = new Vertex(x_end - x_start, y_end - y_start, z_end - z_start, 1);
		direction->normalizeThis();
	}

	Vertex* origin;
	Vertex* direction;
};

class MaterialBase
{
public:
	MaterialBase() { emissive_material = false; }
	MaterialBase(bool emissive) { emissive_material = emissive; }
	virtual ~MaterialBase() {}
	virtual MaterialBase * DeepCopy() { return new MaterialBase(); };
	bool Emissive() { return emissive_material; }
private:
	bool emissive_material;
};

class Material : public MaterialBase
{
public:
	Material(float _r, float _g, float _b, float _kd, float _ks, float _shine, float _T, float _ior)
		:MaterialBase(false), r(_r), g(_g), b(_b), kd(_kd), ks(_ks), shine(_shine), T(_T), ior(_ior) {}
	virtual ~Material() {}
	Vector3f GetDiffuse() { return Vector3f(r*kd, g*kd, b*kd); }
	Vector3f GetSpecular() { return Vector3f(ks, ks, ks); }
	float getShine() { return this->shine; }
	float getT() { return this->T; }
	float getIor() { return this->ior; }
	Material * DeepCopy() { return new Material(r, g, b, kd, ks, shine, T, ior); }
private:
	float r, g, b;	// color
	float kd;		// diffuse coef.
	float ks;		// specular coef.
	float shine;	// Phong cosine power for highlights
	float T;		// transmittance (fraction of contribution of the transmitting ray)
	float ior;		// index of refraction
};

class MaterialEmissive : public MaterialBase
{
public:
	MaterialEmissive(float _r, float _g, float _b, float _c0, float _c1, float _c2)
		:MaterialBase(true), r(_r), g(_g), b(_b), c0(_c0), c1(_c1), c2(_c2) {}
	virtual ~MaterialEmissive() {}
	Vector3f GetIntensity() { return Vector3f(r, g, b); }
	Vector3f GetAttentuationCons() { return Vector3f(c0, c1, c2); }
	MaterialEmissive * DeepCopy() { return new MaterialEmissive(r, g, b, c0, c1, c2); }
private:
	float r, g, b;		// color
	float c0, c1, c2;	// attenuationm
};

class DrawObject {
public:
	DrawObject() {}
	DrawObject(string _name, MaterialBase* used_material) {
		this->material = used_material->DeepCopy();
		this->name = _name + " " + to_string(number);
		//number++;
	}
	virtual ~DrawObject() { delete material; }
	MaterialBase* GetMaterial() { return material; }
	virtual bool FindIntersection(Ray &ray, float &t) { return false; }
	virtual Vector3f getNormal(Vertex hit) { return Vector3f(0.0f, 1.0f, 0.0f); }
	static int number;
	string GetName() { return name; }
private:
	MaterialBase* material;
	string name;
};

class Sphere : public DrawObject {
public:
	Sphere(float x, float y, float z, float _radius, MaterialBase* _used_material) :DrawObject("Sphere", _used_material), radius(_radius) { center = new Vertex(x, y, z, 1); }
	virtual ~Sphere() { delete center; }
	virtual bool FindIntersection(Ray &ray, float &distance) {
		Vertex dst = (*ray.origin) - (*center);
		float angle = Vertex::DotProduct(*ray.direction, dst.normalize());
		if (angle > 0)
			return false; // behind ray start

		float b = Vertex::DotProduct(dst, (*ray.direction));
		float c = Vertex::DotProduct(dst, dst) - radius*radius;
		float d = b*b - c;

		if (d > 0) {
			distance = -b - sqrtf(d);
			if (distance < 0.01f)
				distance = -b + sqrtf(d);

			return true;
		}
		return false;
	}
	virtual Vector3f getNormal(Vertex hit) {
		return Vector3f(hit.getX() - this->center->getX(), hit.getY() - this->center->getY(), hit.getZ() - this->center->getZ());
	}
private:
	Vertex* center;
	float radius;
};

class Triangle : public DrawObject
{
public:
	Triangle() {
		this->vertices = NULL;
	}
	Triangle(sglEElementType _object_type, vector<Vertex*>* _vertices, MaterialBase* _used_material) :DrawObject("Triangle", _used_material), object_type(_object_type), vertices(_vertices) {
		vertices_linear = new Vertex[3];

		vertices_linear[0] = *_vertices->at(0);
		vertices_linear[1] = *_vertices->at(1);
		vertices_linear[2] = *_vertices->at(2);

		Vertex u = vertices_linear[1] - vertices_linear[0];
		Vertex v = vertices_linear[2] - vertices_linear[0];

		//cout << "B" << endl;

		this->normals = NULL;
		this->normals_linear = NULL;

		cout << "NULL" << endl;

		this->normal = *(new Vector3f(
			(u.getY()*v.getZ() - u.getZ()*v.getY()),
			(u.getZ()*v.getX() - u.getX()*v.getZ()),
			(u.getX()*v.getY() - u.getY()*v.getX())));
		this->normal.Normalize();
	}
	Triangle(sglEElementType _object_type, vector<Vertex*>* _vertices, vector<Vertex*>* _normals, MaterialBase* _used_material) :DrawObject("Triangle", _used_material), object_type(_object_type), vertices(_vertices), normals(_normals) {
		vertices_linear = new Vertex[3];

		vertices_linear[0] = *_vertices->at(0);
		vertices_linear[1] = *_vertices->at(1);
		vertices_linear[2] = *_vertices->at(2);

		Vertex u = vertices_linear[1] - vertices_linear[0];
		Vertex v = vertices_linear[2] - vertices_linear[0];

		/*cout << "Got " << _normals->size() << endl;
		cout << "Ended with " << this->normals->size() << endl;*/
		//cout << "A" << endl;

		//cout << this->normals->size() << endl;

		for (int i = 0; i < normals->size(); i++)
		{
			normals->at(i)->normalizeThis();
		}

		this->normal = *(new Vector3f(
			(u.getY()*v.getZ() - u.getZ()*v.getY()),
			(u.getZ()*v.getX() - u.getX()*v.getZ()),
			(u.getX()*v.getY() - u.getY()*v.getX())));
		this->normal.Normalize();
	}
	virtual ~Triangle() {
		//if (this->vertices == NULL) return;
		//cout << "Destructor called for " << this << endl;
		/*for (vector<Vertex*>::iterator it = vertices->begin(); it != vertices->end(); ++it)
		delete *it;*/

		delete vertices_linear;
		if (normals) delete normals;
		//delete normal;
	}
	void normals_print() {
		cout << this->normals->size() << endl;
	}
	void print() {
		cout << "(" << this->vertices_linear[0].getX() << "," << this->vertices_linear[0].getY() << "," << this->vertices_linear[0].getZ() << ") ";
		cout << "(" << this->vertices_linear[1].getX() << "," << this->vertices_linear[1].getY() << "," << this->vertices_linear[1].getZ() << ") ";
		cout << "(" << this->vertices_linear[2].getX() << "," << this->vertices_linear[2].getY() << "," << this->vertices_linear[2].getZ() << ")" << endl;
	}
	virtual Vector3f getNormal(Vertex hit) {
		//cout << this->normals->size() << endl;
		//return *this->normal; // DEBUG
		if (this->normals == NULL) return this->normal;

		//cout << "Normals available" << endl;

		Vertex v2_1, v2_3, v2_t;

		double bary[3];
		v2_1 = this->vertices_linear[0] - this->vertices_linear[1];
		v2_3 = this->vertices_linear[2] - this->vertices_linear[1];
		v2_t = hit - this->vertices_linear[1];

		float d00 = Vertex::DotProduct(v2_1, v2_1);
		float d01 = Vertex::DotProduct(v2_1, v2_3);
		float d11 = Vertex::DotProduct(v2_3, v2_3);
		float denom = d00 * d11 - d01 * d01;

		float d20 = Vertex::DotProduct(v2_t, v2_1);
		float d21 = Vertex::DotProduct(v2_t, v2_3);
		bary[0] = (d11 * d20 - d01 * d21) / denom;
		bary[1] = (d00 * d21 - d01 * d20) / denom;
		bary[2] = 1.0f - bary[0] - bary[1];

		return (Vector3f(&(*this->normals->at(0) * bary[0] + *this->normals->at(1) * bary[2] + *this->normals->at(2) * bary[1])));
	}
	virtual bool FindIntersection(Ray &ray, float &tHit, bool unCull = false) {

		float SMALL_NUM = 0.000000001f;
		float		r, a, b;              // params to calc ray-plane intersect

										  //cout << ray.direction->getX() << "-" << ray.direction->getY() << "-" << ray.direction->getZ() << ", test:" << this << endl;

		float angle = dotProduct(normal, *ray.direction);
		if (!unCull && angle > 0)
			return false; // behind ray start

						  // get triangle edge vectors and plane normal
		Vertex u = vertices_linear[1] - vertices_linear[0];
		Vertex v = vertices_linear[2] - vertices_linear[0];
		Vertex n = Vertex(this->normal.x, this->normal.y, this->normal.z, 1.0f); // cross product
																				 //if (n == (Vector)0)			// triangle is degenerate
																				 //	return -1;					// do not deal with this case

																				 //dir = R.P1 - R.P0;			// ray direction vector
		Vertex w0 = *ray.origin - vertices_linear[0];
		a = (-1)*(Vertex::DotProduct(n, w0));
		b = Vertex::DotProduct(n, *ray.direction);
		if (fabs(b) < SMALL_NUM) {		// ray is  parallel to triangle plane
			return false;				// ray disjoint from plane
		}

		// get intersect point of ray with triangle plane
		r = a / b;
		if (r < 0.02)                    // ray goes away from triangle
			return false;				// => no intersect
										// for a segment, also test if (r > 1.0) => no intersect

		Vertex I = *ray.origin + (*ray.direction)*r;            // intersect point of ray and plane
																//if (debug) cout << I.getX() << " " << I.getY() << " " << I.getZ() << endl;

																// is I inside T?
		float    uu, uv, vv, wu, wv, D;
		uu = Vertex::DotProduct(u, u);
		uv = Vertex::DotProduct(u, v);
		vv = Vertex::DotProduct(v, v);
		Vertex w = I - vertices_linear[0];
		wu = Vertex::DotProduct(w, u);
		wv = Vertex::DotProduct(w, v);
		D = uv * uv - uu * vv;

		// get and test parametric coords
		float s, t;
		s = (uv * wv - vv * wu) / D;
		if (s < 0.0 || s > 1.0)         // I is outside T
			return 0;
		t = (uv * wu - uu * wv) / D;
		if (t < 0.0 || (s + t) > 1.0)  // I is outside T
			return 0;

		I = I - *ray.origin; // Z I udelam pouze vektor od zacatku k bodu pruniku pro vypocitani vzdalenosti
		tHit = sqrt((I.getX() * I.getX()) + (I.getY()* I.getY()) + (I.getZ() * I.getZ()));

		return 1;                       // I is in T


	}
	Vertex randomSample() {
		// uniform random point in triangle
		float r1 = (double)rand() / (RAND_MAX);
		float r2 = (double)rand() / (RAND_MAX);

		float u, v;
		Vertex e1 = (vertices_linear[1] - vertices_linear[0]);
		Vertex e2 = (vertices_linear[2] - vertices_linear[0]);
		if (r1 + r2 > 1) {
			u = 1 - r1;
			v = 1 - r2;
		}
		else {
			u = r1;
			v = r2;
		}

		return (vertices_linear[0] + e1*u + e2*v);
	}

	float areaSize() {
		float a = Vertex::Distance(vertices_linear[0], vertices_linear[1]);
		float b = Vertex::Distance(vertices_linear[1], vertices_linear[2]);
		float c = Vertex::Distance(vertices_linear[2], vertices_linear[0]);

		// Heron's formula
		float s = (a + b + c) / 2;
		return sqrtf(s*(s - a)*(s - b)*(s - c));
	}

	vector<Vertex*>* getVertices() { return this->vertices; }

private:
	Vector3f normal;
	sglEElementType object_type;
	vector<Vertex*>* vertices;
	Vertex *vertices_linear = NULL;
	vector<Vertex*>* normals = NULL;
	Vertex *normals_linear = NULL;

};

