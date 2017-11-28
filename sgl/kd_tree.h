//---------------------------------------------------------------------------
// kd_tree.h
// Implementation of KD-Tree structure with O(N log N) construction
// Based on "On building fast KD-trees for ray tracing, and on doing that in O(N log N)" by Wald and Havran
// Also based on implementation of slower, O(N log^2 N) algorithm by Miguel Granados and Richard Socher (University of Saarland)
// Date:  2017/04
// Author: Michal Kuèera, CTU Prague
//---------------------------------------------------------------------------

#pragma once


#include "primitives.h"

#include <cassert>
#include <limits>

// ==========================================================================
//  O(N log N) construction of kd-tree. Semester project for DPG
// ==========================================================================

class config{
public:
	static int K_T;// Traversal cost
	static int K_I; // Intersection cost
	static float epsilon;
	static bool debug;
	static float voxel_size_cap; // DEBUG pro VS omezení RAM
};

struct triangle_id {
	int id;
	Triangle* t;
};

/**
* Class used for split plane representation
*/
class splitPlane {
public:
	float position;
	int dimension;
	bool left;

	splitPlane() {
	}

	splitPlane(float position, int dimension, bool left) {
		this->position = position;
		this->dimension = dimension;
		this->left = left;
	}

	~splitPlane() {
		//delete this->position; 
	}

	bool equal( const splitPlane* prev ) {
		if (prev == NULL) return false;

		if ((this->dimension == prev->dimension ) &&
			(this->position == prev->position)
			) return true;

		return false;
	}
};


/**
* Class used for voxel representation
*/
class Voxel {
public:
	Vector3f position; /*!< Position of the corner with smallest coordinates */
	float dX; /*!< Size in X axis */
	float dY; /*!< Size in Y axis */
	float dZ; /*!< Size in Z axis */

	Voxel() {}

	Voxel(Vector3f position, float dX, float dY, float dZ) {
		this->position = position;
		this->dX = dX;
		this->dY = dY;
		this->dZ = dZ;
	}

	float vol() {
		return this->dX * this->dY * this->dZ;
	}

	void setVoxelParameters(Triangle* T, int size) {

		this->position = Vector3f(T[0].getVertices()->at(0));
		this->dX = 0.0;
		this->dY = 0.0;	
		this->dZ = 0.0;

		for (int i = 1; i < size; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				if (T[i].getVertices()->at(j)->getX() < this->position.x) {
					this->dX = this->dX + this->position.x - T[i].getVertices()->at(j)->getX();
					this->position.x = T[i].getVertices()->at(j)->getX();
				}
				if (T[i].getVertices()->at(j)->getX() > (this->position.x + this->dX)) {
					this->dX = T[i].getVertices()->at(j)->getX() - this->position.x;
				}

				if (T[i].getVertices()->at(j)->getY() < this->position.y) {
					this->dY = this->dY + this->position.y - T[i].getVertices()->at(j)->getY();
					this->position.y = T[i].getVertices()->at(j)->getY();
				}
				if (T[i].getVertices()->at(j)->getY() > (this->position.y + this->dY)) {
					this->dY = T[i].getVertices()->at(j)->getY() - this->position.y;
				}

				if (T[i].getVertices()->at(j)->getZ() < this->position.z) {
					this->dZ = this->dZ + this->position.z - T[i].getVertices()->at(j)->getZ();
					this->position.z = T[i].getVertices()->at(j)->getZ();
				}
				if (T[i].getVertices()->at(j)->getZ() > (this->position.z + this->dZ)) {
					this->dZ = T[i].getVertices()->at(j)->getZ() - this->position.z;
				}
			}

		}

	}

	void setPosition(Vector3f position) { this->position = position; };

	/**
	*  Returns two voxels as a result of split operation of THIS voxel by given plane p
	*/
	pair<Voxel*, Voxel*> splitByPlane(splitPlane* p) {
		pair<Voxel*, Voxel*> result;

		Voxel *left, *right;

		left = new Voxel(this->position, this->dX, this->dY, this->dZ);
		right = new Voxel(this->position, this->dX, this->dY, this->dZ);

		switch (p->dimension)
		{
		case 0:
			right->position.x = p->position;
			left->dX = p->position - left->position.x;
			left->dX = (left->dX < 0.0f) ? 0.0f : left->dX;
			right->dX = right->dX - left->dX;
			right->dX = (right->dX < 0.0f) ? 0.0f : right->dX;
			break;
		case 1:
			right->position.y = p->position;
			left->dY = p->position - left->position.y;
			left->dY = (left->dY < 0.0f) ? 0.0f : left->dY;
			right->dY = right->dY - left->dY;
			right->dY = (right->dY < 0.0f) ? 0.0f : right->dY;
			break;
		case 2:
			right->position.z = p->position;
			left->dZ = p->position - left->position.z;
			left->dZ = (left->dZ < 0.0f) ? 0.0f : left->dZ;
			right->dZ = right->dZ - left->dZ;
			right->dZ = (right->dZ < 0.0f) ? 0.0f : right->dZ;
			break;
		default:
			//cout << "Something is wrong in voxel splitting" << endl;
			break;
		}

		result.first = left;
		result.second = right;

		return result;
	}
};



							 /**
							 * Class used for representation of an AABB - Axis Aligned Bounding Box
							 */
class boundingBox {
public:
	float x_min; /*!< Smallest X coordinate */
	float x_max; /*!< Largest X coordinate */
	float y_min; /*!< Smallest Y coordinate */
	float y_max; /*!< Largest Y coordinate */
	float z_min; /*!< Smallest Z cordinate */
	float z_max; /*!< Largest Z coordinate */

	boundingBox() {}

	boundingBox(Triangle *T) {
		this->x_min = this->x_max = T->getVertices()->at(0)->getX();
		this->y_min = this->y_max = T->getVertices()->at(0)->getY();
		this->z_min = this->z_max = T->getVertices()->at(0)->getZ();

		if (T->getVertices()->at(1)->getX() > this->x_max) this->x_max = T->getVertices()->at(1)->getX();
		if (T->getVertices()->at(1)->getX() < this->x_min) this->x_min = T->getVertices()->at(1)->getX();
		if (T->getVertices()->at(1)->getY() > this->y_max) this->y_max = T->getVertices()->at(1)->getY();
		if (T->getVertices()->at(1)->getY() < this->y_min) this->y_min = T->getVertices()->at(1)->getY();
		if (T->getVertices()->at(1)->getZ() > this->z_max) this->z_max = T->getVertices()->at(1)->getZ();
		if (T->getVertices()->at(1)->getZ() < this->z_min) this->z_min = T->getVertices()->at(1)->getZ();

		if (T->getVertices()->at(2)->getX() > this->x_max) this->x_max = T->getVertices()->at(2)->getX();
		if (T->getVertices()->at(2)->getX() < this->x_min) this->x_min = T->getVertices()->at(2)->getX();
		if (T->getVertices()->at(2)->getY() > this->y_max) this->y_max = T->getVertices()->at(2)->getY();
		if (T->getVertices()->at(2)->getY() < this->y_min) this->y_min = T->getVertices()->at(2)->getY();
		if (T->getVertices()->at(2)->getZ() > this->z_max) this->z_max = T->getVertices()->at(2)->getZ();
		if (T->getVertices()->at(2)->getZ() < this->z_min) this->z_min = T->getVertices()->at(2)->getZ();

	}

	/**
	*  Adjusts this AABB's bounds to perfectly bound the given triangle
	*/
	bool boundTriangle(Triangle *T) {
		this->x_min = this->x_max = T->getVertices()->at(0)->getX();
		this->y_min = this->y_max = T->getVertices()->at(0)->getY();
		this->z_min = this->z_max = T->getVertices()->at(0)->getZ();

		if (T->getVertices()->at(1)->getX() > this->x_max) this->x_max = T->getVertices()->at(1)->getX();
		if (T->getVertices()->at(1)->getX() < this->x_min) this->x_min = T->getVertices()->at(1)->getX();
		if (T->getVertices()->at(1)->getY() > this->y_max) this->y_max = T->getVertices()->at(1)->getY();
		if (T->getVertices()->at(1)->getY() < this->y_min) this->y_min = T->getVertices()->at(1)->getY();
		if (T->getVertices()->at(1)->getZ() > this->z_max) this->z_max = T->getVertices()->at(1)->getZ();
		if (T->getVertices()->at(1)->getZ() < this->z_min) this->z_min = T->getVertices()->at(1)->getZ();

		if (T->getVertices()->at(2)->getX() > this->x_max) this->x_max = T->getVertices()->at(2)->getX();
		if (T->getVertices()->at(2)->getX() < this->x_min) this->x_min = T->getVertices()->at(2)->getX();
		if (T->getVertices()->at(2)->getY() > this->y_max) this->y_max = T->getVertices()->at(2)->getY();
		if (T->getVertices()->at(2)->getY() < this->y_min) this->y_min = T->getVertices()->at(2)->getY();
		if (T->getVertices()->at(2)->getZ() > this->z_max) this->z_max = T->getVertices()->at(2)->getZ();
		if (T->getVertices()->at(2)->getZ() < this->z_min) this->z_min = T->getVertices()->at(2)->getZ();

		return true;
	}

	/**
	*  Adjusts this AABB's bound to perfectly bound the given triangle clipped by given voxel
	*/
	bool clipToVoxel(Voxel *V, Triangle *T) {
		vector< Vector3f > points;
		Ray *voxelRay;
		Vertex *a, *b, *line_dir;
		Vector3f dir, intersection;
		float t;
		float tmp1, tmp, tmp2;

		//cout << "Clipping to voxel" << endl;

		// intersections of triangle lines with voxel

		for (int i = 0; i < 3; i++)
		{
			if (T->getVertices()->at(i)->getX() + config::epsilon >= V->position.x &&
				T->getVertices()->at(i)->getX() - config::epsilon <= (V->position.x + V->dX) &&
				T->getVertices()->at(i)->getY() + config::epsilon >= V->position.y &&
				T->getVertices()->at(i)->getY() - config::epsilon <= (V->position.y + V->dY) &&
				T->getVertices()->at(i)->getZ() + config::epsilon >= V->position.z &&
				T->getVertices()->at(i)->getZ() - config::epsilon <= (V->position.z + V->dZ))
				points.push_back(Vector3f(T->getVertices()->at(i)));


			a = T->getVertices()->at(i);
			b = T->getVertices()->at((i + 1) % 3);

			dir = Vector3f(a->getX() - b->getX(), a->getY() - b->getY(), a->getZ() - b->getZ());


			if ((a->getX() < V->position.x && b->getX() > V->position.x) || (a->getX() > V->position.x && b->getX() < V->position.x)) {

				t = (V->position.x - b->getX()) / dir.x;

				if (((b->getY() + dir.y*t) < (V->position.y - config::epsilon)) ||
					((b->getY() + dir.y*t - config::epsilon) > (V->position.y + V->dY)) ||
					((b->getZ() + dir.z*t) < (V->position.z - config::epsilon)) ||
					((b->getZ() + dir.z*t - config::epsilon) > (V->position.z + V->dZ))
					) {
					if (config::debug) {
						cout << ((b->getY() + dir.y*t) < (V->position.y - config::epsilon)) << endl;
						cout << ((b->getY() + dir.y*t - config::epsilon) > (V->position.y + V->dY)) << endl;
						cout << ((b->getZ() + dir.z*t) < (V->position.z - config::epsilon)) << endl;
						cout << ((b->getZ() + dir.z*t - config::epsilon) > (V->position.z + V->dZ)) << endl;
						cout << "a---" << endl;

					}
					points.push_back(Vector3f(b) + dir*t);
				}
				else {
					points.push_back(Vector3f(b) + dir*t);
				}
			}

			if ((a->getX() < (V->position.x + V->dX) && b->getX() >(V->position.x + V->dX)) || (a->getX() > (V->position.x + V->dX) && b->getX() < (V->position.x + V->dX))) {

				t = ((V->position.x + V->dX) - b->getX()) / dir.x;

				if (((b->getY() + dir.y*t) < (V->position.y - config::epsilon)) ||
					((b->getY() + dir.y*t - config::epsilon) > (V->position.y + V->dY)) ||
					((b->getZ() + dir.z*t) < (V->position.z - config::epsilon)) ||
					((b->getZ() + dir.z*t - config::epsilon) > (V->position.z + V->dZ))
					) {
					if (config::debug) {
						cout << ((b->getY() + dir.y*t) < (V->position.y - config::epsilon)) << endl;
						cout << ((b->getY() + dir.y*t - config::epsilon) > (V->position.y + V->dY)) << endl;
						cout << ((b->getZ() + dir.z*t) < (V->position.z - config::epsilon)) << endl;
						cout << ((b->getZ() + dir.z*t - config::epsilon) > (V->position.z + V->dZ)) << endl;

						cout << "b----" << endl;

					}
					points.push_back(Vector3f(b) + dir*t);
				}
				else {
					points.push_back(Vector3f(b) + dir*t);
				}
			}

			if ((a->getY() < V->position.y && b->getY() > V->position.y) || (a->getY() > V->position.y && b->getY() < V->position.y)) {

				t = (V->position.y - b->getY()) / dir.y;

				if (((b->getX() + dir.x*t - config::epsilon) > (V->position.x + V->dX)) ||
					((b->getX() + dir.x*t) < (V->position.x - config::epsilon)) ||
					((b->getZ() + dir.z*t) < (V->position.z - config::epsilon)) ||
					((b->getZ() + dir.z*t - config::epsilon) > (V->position.z + V->dZ))
					) {
					if (config::debug) {
						cout << ((b->getX() + dir.x*t) < (V->position.x - config::epsilon)) << endl;
						cout << ((b->getX() + dir.x*t - config::epsilon) > (V->position.x + V->dX)) << endl;
						cout << ((b->getZ() + dir.z*t) < (V->position.z - config::epsilon)) << endl;
						cout << ((b->getZ() + dir.z*t - config::epsilon) > (V->position.z + V->dZ)) << endl;
						cout << "c---" << endl;

					}
					points.push_back(Vector3f(b) + dir*t);
				}
				else {
					points.push_back(Vector3f(b) + dir*t);
				}
			}

			if ((a->getY() < (V->position.y + V->dY) && b->getY() >(V->position.y + V->dY)) || (a->getY() >(V->position.y + V->dY) && b->getY() < (V->position.y + V->dY))) {

				t = ((V->position.y + V->dY) - b->getY()) / dir.y;

				if (((b->getX() + dir.x*t - config::epsilon) > (V->position.x + V->dX)) ||
					((b->getX() + dir.x*t) < (V->position.x - config::epsilon)) ||
					((b->getZ() + dir.z*t) < (V->position.z - config::epsilon)) ||
					((b->getZ() + dir.z*t - config::epsilon) > (V->position.z + V->dZ))
					) {
					if (config::debug) {
						cout << ((b->getX() + dir.x*t) < (V->position.x - config::epsilon)) << endl;
						cout << ((b->getX() + dir.x*t - config::epsilon) > (V->position.x + V->dX)) << endl;
						cout << ((b->getZ() + dir.z*t) < (V->position.z - config::epsilon)) << endl;
						cout << ((b->getZ() + dir.z*t - config::epsilon) > (V->position.z + V->dZ)) << endl;
						cout << "d---" << endl;
					}
					points.push_back(Vector3f(b) + dir*t);
				}
				else {
					points.push_back(Vector3f(b) + dir*t);
				}
			}

			if ((a->getZ() < V->position.z && b->getZ() > V->position.z) || (a->getZ() > V->position.z && b->getZ() < V->position.z)) {

				t = (V->position.z - b->getZ()) / dir.z;

				if (((b->getX() + dir.x*t - config::epsilon) > (V->position.x + V->dX)) ||
					((b->getX() + dir.x*t) < (V->position.x - config::epsilon)) ||
					((b->getY() + dir.y*t) < (V->position.y - config::epsilon)) ||
					((b->getY() + dir.y*t - config::epsilon) > (V->position.y + V->dY))
					) {
					if (config::debug) {
						cout << ((b->getX() + dir.x*t) < (V->position.x - config::epsilon)) << endl;
						cout << ((b->getX() + dir.x*t - config::epsilon) > (V->position.x + V->dX)) << endl;
						cout << ((b->getY() + dir.y*t) < (V->position.y - config::epsilon)) << endl;
						cout << ((b->getY() + dir.y*t - config::epsilon) > (V->position.y + V->dY)) << endl;
						cout << ((b->getZ() + dir.z*t) < (V->position.z - config::epsilon)) << endl;
						cout << ((b->getZ() + dir.z*t - config::epsilon) > (V->position.z + V->dZ)) << endl;
						cout << "e---" << endl;
					}
					points.push_back(Vector3f(b) + dir*t); // !!!
				}
				else {
					points.push_back(Vector3f(b) + dir*t);
				}
			}

			if ((a->getZ() < (V->position.z + V->dZ) && b->getZ() >(V->position.z + V->dZ)) || (a->getZ() >(V->position.z + V->dZ) && b->getZ() < (V->position.z + V->dZ))) {

				t = ((V->position.z + V->dZ) - b->getZ()) / dir.z;

				if (((b->getX() + dir.x*t - config::epsilon) > (V->position.x + V->dX)) ||
					((b->getX() + dir.x*t) < (V->position.x - config::epsilon)) ||
					((b->getY() + dir.y*t) < (V->position.y - config::epsilon)) ||
					((b->getY() + dir.y*t - config::epsilon) > (V->position.y + V->dY))
					) {
					if (config::debug) {
						cout << ((b->getX() + dir.x*t - config::epsilon) > (V->position.x + V->dX)) << endl;
						cout << ((b->getX() + dir.x*t) < (V->position.x - config::epsilon)) << endl;
						cout << ((b->getY() + dir.y*t) < (V->position.y - config::epsilon)) << endl;
						cout << ((b->getY() + dir.y*t - config::epsilon) > (V->position.y + V->dY)) << endl;
						cout << "f---" << endl;
					}
					points.push_back(Vector3f(b) + dir*t); // !!!
				}
				else {
					points.push_back(Vector3f(b) + dir*t);
				}
			}


		}

		/*
		for (int i = 0; i < 3; i++)
		{
		if (T->getVertices()->at(i)->getX() >= V->position.x &&
		T->getVertices()->at(i)->getX() <= (V->position.x + V->dX) &&
		T->getVertices()->at(i)->getY() >= V->position.y &&
		T->getVertices()->at(i)->getY() <= (V->position.y + V->dY) &&
		T->getVertices()->at(i)->getZ() >= V->position.z &&
		T->getVertices()->at(i)->getZ() <= (V->position.z + V->dZ))
		points.push_back(Vector3f(T->getVertices()->at(i)));

		a = T->getVertices()->at(i);
		b = T->getVertices()->at((i + 1) % 3);
		line_dir = &(*a - *b);



		if (!(fabs(line_dir->getX()) < epsilon)) {
		tmp1 = (V->position.x - b->getX()) / line_dir->getX();
		tmp2 = (V->position.x + V->dX - b->getX()) / line_dir->getX();

		//cout << "Candidate are: " << tmp1 << " " << tmp2 << endl;
		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));
		//cout << "Finals are: " << tmp << " " << t << endl;

		}

		if (!(fabs(line_dir->getY()) < epsilon)) {
		tmp1 = (V->position.y - b->getY()) / line_dir->getY();
		tmp2 = (V->position.y + V->dY - b->getY()) / line_dir->getY();

		//cout << "Candidate are: " << tmp1 << " " << tmp2 << endl;
		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));
		//cout << "Finals are: " << tmp << " " << t << endl;
		}

		if (!(fabs(line_dir->getZ()) < epsilon)) {
		tmp1 = (V->position.z - b->getZ()) / line_dir->getZ();
		tmp2 = (V->position.z + V->dZ - b->getZ()) / line_dir->getZ();

		//cout << "Candidate are: " << tmp1 << " " << tmp2 << endl;
		tmp = max(tmp, min(tmp1, tmp2));
		t = min(t, max(tmp1, tmp2));

		}


		if ( tmp >= 0.0f &&
		tmp <= 1.0f &&
		(((*b) + (*line_dir)*tmp).getX()) >= V->position.x &&
		(((*b) + (*line_dir)*tmp).getX()) <= (V->position.x + V->dX) &&
		(((*b) + (*line_dir)*tmp).getY()) >= V->position.y &&
		(((*b) + (*line_dir)*tmp).getY()) <= (V->position.y + V->dY) &&
		(((*b) + (*line_dir)*tmp).getZ()) >= V->position.z &&
		(((*b) + (*line_dir)*tmp).getZ()) <= (V->position.z + V->dZ)) {
		points.push_back(Vector3f(&((*b) + (*line_dir)*tmp)));
		}

		if (t >= 0.0f &&
		t <= 1.0f &&
		(((*b) + (*line_dir)*t).getX()) >= V->position.x &&
		(((*b) + (*line_dir)*t).getX()) <= (V->position.x + V->dX) &&
		(((*b) + (*line_dir)*t).getY()) >= V->position.y &&
		(((*b) + (*line_dir)*t).getY()) <= (V->position.y + V->dY) &&
		(((*b) + (*line_dir)*t).getZ()) >= V->position.z &&
		(((*b) + (*line_dir)*t).getZ()) <= (V->position.z + V->dZ)) {
		points.push_back(Vector3f(&((*b) + (*line_dir)*t)));
		}

		}
		*/
		//cout << "First part done" << endl;

		// intersections of voxel lines with triangle
		{

			voxelRay = new Ray(V->position.x - 1.0, V->position.y, V->position.z, V->position.x + V->dX, V->position.y, V->position.z);
			if (T->FindIntersection(*voxelRay, t, true)) {
				//cout << "x: " << t << endl;
				intersection = Vector3f(voxelRay->origin);
				intersection = intersection + Vector3f(voxelRay->direction)*t;
				if ((V->position.x + V->dX + config::epsilon >= intersection.x) &&
					(V->position.x - config::epsilon <= intersection.x)
					) {
					//	cout << "ok" << endl;
					points.push_back(intersection);
				}
			}

			voxelRay->adjust(V->position.x - 1.0, V->position.y + V->dY, V->position.z, V->position.x + V->dX, V->position.y + V->dY, V->position.z);
			if (T->FindIntersection(*voxelRay, t, true)) {
				//cout << "x: " << t << endl;
				intersection = Vector3f(voxelRay->origin);
				intersection = intersection + Vector3f(voxelRay->direction)*t;
				if ((V->position.x + V->dX + config::epsilon >= intersection.x) &&
					(V->position.x - config::epsilon <= intersection.x)
					) {
					//cout << "ok" << endl;
					points.push_back(intersection);
				}
			}

			voxelRay->adjust(V->position.x - 1.0, V->position.y + V->dY, V->position.z + V->dZ, V->position.x + V->dX, V->position.y + V->dY, V->position.z + V->dZ);
			if (T->FindIntersection(*voxelRay, t, true)) {
				//cout << "x: " << t << endl;
				intersection = Vector3f(voxelRay->origin);
				intersection = intersection + Vector3f(voxelRay->direction)*t;
				if ((V->position.x + V->dX + config::epsilon >= intersection.x) &&
					(V->position.x - config::epsilon <= intersection.x)
					) {
					//cout << "ok" << endl;
					points.push_back(intersection);
				}
			}

			voxelRay->adjust(V->position.x - 1.0, V->position.y, V->position.z + V->dZ, V->position.x + V->dX, V->position.y, V->position.z + V->dZ);
			if (T->FindIntersection(*voxelRay, t, true)) {
				//cout << "x: " << t << endl;
				intersection = Vector3f(voxelRay->origin);
				intersection = intersection + Vector3f(voxelRay->direction)*t;
				if ((V->position.x + V->dX + config::epsilon >= intersection.x) &&
					(V->position.x - config::epsilon <= intersection.x)
					) {
					//cout << "ok" << endl;
					points.push_back(intersection);
				}
			}
			//delete voxelRay;

			voxelRay->adjust(V->position.x, V->position.y - 1.0, V->position.z, V->position.x, V->position.y + V->dY, V->position.z);
			if (T->FindIntersection(*voxelRay, t, true)) {
				//cout << "y: " << t << endl;
				intersection = Vector3f(voxelRay->origin);
				intersection = intersection + Vector3f(voxelRay->direction)*t;
				if ((V->position.y + V->dY + config::epsilon >= intersection.y) &&
					(V->position.y - config::epsilon <= intersection.y)
					) {
					//cout << "ok" << endl;
					points.push_back(intersection);
				}
			}
			//delete voxelRay;

			voxelRay->adjust(V->position.x + V->dX, V->position.y - 1.0, V->position.z, V->position.x + V->dX, V->position.y + V->dY, V->position.z);
			if (T->FindIntersection(*voxelRay, t, true)) {
				//cout << "y: " << t << endl;
				intersection = Vector3f(voxelRay->origin);
				intersection = intersection + Vector3f(voxelRay->direction)*t;
				if ((V->position.y + V->dY + config::epsilon >= intersection.y) &&
					(V->position.y - config::epsilon <= intersection.y)
					) {
					//cout << "ok" << endl;
					points.push_back(intersection);
				}
			}
			//delete voxelRay;

			voxelRay->adjust(V->position.x, V->position.y - 1.0, V->position.z + V->dZ, V->position.x, V->position.y + V->dY, V->position.z + V->dZ);
			if (T->FindIntersection(*voxelRay, t, true)) {
				//cout << "y: " << t << endl;
				intersection = Vector3f(voxelRay->origin);
				intersection = intersection + Vector3f(voxelRay->direction)*t;
				if ((V->position.y + V->dY + config::epsilon >= intersection.y) &&
					(V->position.y - config::epsilon <= intersection.y)
					) {
					//cout << "ok" << endl;
					points.push_back(intersection);
				}
			}
			//delete voxelRay;

			voxelRay->adjust(V->position.x + V->dX, V->position.y - 1.0, V->position.z + V->dZ, V->position.x + V->dX, V->position.y + V->dY, V->position.z + V->dZ);
			if (T->FindIntersection(*voxelRay, t, true)) {
				//cout << "y: " << t << endl;
				intersection = Vector3f(voxelRay->origin);
				intersection = intersection + Vector3f(voxelRay->direction)*t;
				if ((V->position.y + V->dY + config::epsilon >= intersection.y) &&
					(V->position.y - config::epsilon <= intersection.y)
					) {
					//cout << "ok" << endl;
					points.push_back(intersection);
				}
			}
			//delete voxelRay;

			voxelRay->adjust(V->position.x, V->position.y, V->position.z - 1.0, V->position.x, V->position.y, V->position.z + V->dZ);
			if (T->FindIntersection(*voxelRay, t, true)) {
				//cout << "z1: " << t << endl;
				intersection = Vector3f(voxelRay->origin);
				intersection = intersection + Vector3f(voxelRay->direction)*t;
				if ((V->position.z + V->dZ + config::epsilon >= intersection.z) &&
					(V->position.z - config::epsilon <= intersection.z)
					) {
					//cout << "ok" << endl;
					points.push_back(intersection);
				}
			}
			//delete voxelRay;

			voxelRay->adjust(V->position.x + V->dX, V->position.y, V->position.z - 1.0, V->position.x + V->dX, V->position.y, V->position.z + V->dZ);
			if (T->FindIntersection(*voxelRay, t, true)) {
				//cout << "z2: " << t << endl;
				intersection = Vector3f(voxelRay->origin);
				intersection = intersection + Vector3f(voxelRay->direction)*t;
				if ((V->position.z + V->dZ + config::epsilon >= intersection.z) &&
					(V->position.z - config::epsilon <= intersection.z)
					) {
					//cout << "ok" << endl;
					points.push_back(intersection);
				}
			}
			//delete voxelRay;

			voxelRay->adjust(V->position.x, V->position.y + V->dY, V->position.z - 1.0, V->position.x, V->position.y + V->dY, V->position.z + V->dZ);
			if (T->FindIntersection(*voxelRay, t, true)) {
				//cout << "z3: " << t << endl;
				intersection = Vector3f(voxelRay->origin);
				intersection = intersection + Vector3f(voxelRay->direction)*t;
				if ((V->position.z + V->dZ + config::epsilon >= intersection.z) &&
					(V->position.z - config::epsilon <= intersection.z)
					) {
					//cout << "ok" << endl;
					points.push_back(intersection);
				}
			}
			//delete voxelRay;

			voxelRay->adjust(V->position.x + V->dX, V->position.y + V->dY, V->position.z - 1.0, V->position.x + V->dX, V->position.y + V->dY, V->position.z + V->dZ);
			if (T->FindIntersection(*voxelRay, t, true)) {
				//cout << "z4: " << t << endl;
				intersection = Vector3f(voxelRay->origin);
				intersection = intersection + Vector3f(voxelRay->direction)*t;
				if ((V->position.z + V->dZ + config::epsilon >= intersection.z) &&
					(V->position.z - config::epsilon <= intersection.z)
					) {
					//cout << "ok" << endl;
					points.push_back(intersection);
				}
			}
		}

		delete voxelRay;

		// DEBUG:
		// maybe events are wrong? If the triangle doesnt even go through the split plane, there will be no point in the intersection
		//cout << "Setting for initial point" << endl;
		/*cout << "---------------------------" << endl;
		cout << points.size() << endl;
		cout << V->position.x << ":" << V->position.y << ":" << V->position.z << endl;
		cout << V->dX << ":" << V->dY << ":" << V->dZ << endl;
		cout << "Triangle points:" << endl;
		cout << T->getVertices()->at(0)->getX() << "-" << T->getVertices()->at(0)->getY() << "-" << T->getVertices()->at(0)->getZ() << endl;
		cout << T->getVertices()->at(1)->getX() << "-" << T->getVertices()->at(1)->getY() << "-" << T->getVertices()->at(1)->getZ() << endl;
		cout << T->getVertices()->at(2)->getX() << "-" << T->getVertices()->at(2)->getY() << "-" << T->getVertices()->at(2)->getZ() << endl;
		cout << "---------------------------" << endl;*/

		if (points.size() < 1) return false;

		this->x_min = this->x_max = points[0].x;
		this->y_min = this->y_max = points[0].y;
		this->z_min = this->z_max = points[0].z;

		//cout << "Setting for the rest of " << points.size() << " points" << endl;
		for (int i = 1; i < points.size(); i++)
		{
			if (points[i].x > this->x_max) this->x_max = points[i].x;
			if (points[i].x < this->x_min) this->x_min = points[i].x;
			if (points[i].y > this->y_max) this->y_max = points[i].y;
			if (points[i].y < this->y_min) this->y_min = points[i].y;
			if (points[i].z > this->z_max) this->z_max = points[i].z;
			if (points[i].z < this->z_min) this->z_min = points[i].z;
		}

		//cout << "Second part done" << endl;

		return true;
	}
};

/**
* Class used for representation of a triangle event
*/
class triangleEvent {
public:
	int dimension; /*!< Dimension to which this events add a potential split candidate */
	float coordinate; /*!< position of the event in dimension specified by the above variable dimension */
	int type; /*!<  consistently with the Tau function, type == 0 if the events is an end event, 1 if it is a planar event and 2 if it is a start event  */
	int triangleID; /*!< ID of corresponding triangle in the triangle vector (specific to it's recursion level) */
};

/**
* Event comparator
*/
struct eventComparator
{
	/**
	*  Event comparator as described in source paper ( <_E )
	*/
	inline bool operator() (const triangleEvent* a, const triangleEvent* b)
	{
		if (fabs(a->coordinate - b->coordinate) < config::epsilon) return a->type < b->type; // Chtìlo by to ještì prozkoumat
		return a->coordinate < b->coordinate;
	}
};

/**
* Structure representing one node in the kd-tree
*/
struct kdNode {
	kdNode *left = NULL; /*!< Left child */
	kdNode *right = NULL; /*!< Right child*/
	splitPlane *p = NULL; /*!< Split plane */
	vector<int>* triangles = NULL;
};


struct kdNodeLinear {
	int left; /*!< Left child */
	int right; /*!< Right child*/
	bool splits = false;
	splitPlane p; /*!< Split plane */
	int len;
	int* triangles = NULL;

	kdNodeLinear() {

	}

	kdNodeLinear(int _left, int _right, splitPlane _p, int* _triangles): left(_left), right(_right), p(_p), triangles(_triangles) {}
};


/**
* Lambda function used biasing the cost function in favor of cutting away empty voxels
*/
float lambda(int N_L, int N_R);


/**
* Surface area of a voxel
*/
float SA(const Voxel* V);


/**
* SAH cost function. Uses simplified formula.
*/
float Cost(float P_L, float P_R, int N_L, int N_R);


/**
* SA heuristic function, return a rank for the given plane split
*/
pair<float, bool> SAH(splitPlane *p, Voxel* V, int N_L, int N_R, int N_P);



/**
* Function used for classification of triangles into their respective sub-voxels
*/
vector<int>* ClassifyLeftRightBoth(vector<Triangle*> *T, vector<triangleEvent*>* E, splitPlane *p);

/**
* Structure containing a split plane candidate, its cost and whether the triangles in the plane should be moved to left or right
*/
struct planeSolution {
	splitPlane* plane;
	float cost;
};

/**
* Function for finding split plane with the best cost
*/
planeSolution* FindPlane(int N, Voxel *V, vector<triangleEvent*>* E);

/**
* Function deciding whether the recursion should be terminated. Returns TRUE if cost of splitting the voxel with the best plane is higher than not splitting at all.
*/
bool terminate(int N, float Cost);

/**
* Merge function for events. Creates a new event list.
*/
vector<triangleEvent*>* mergeEvents(vector<triangleEvent*>* E1, vector<triangleEvent*>* E2);

/**
* Recursive tree-building function
*/
kdNode* RecBuild(vector<Triangle*>* T, Voxel *V, vector<triangleEvent*>* E, splitPlane* pp);

/**
* Main function for building of a kd-tree using SAH cost function
*/
kdNode* BuildKdTree(Triangle* T_src, int size, int &count);

struct  traversal_structure
{
	kdNode *w;
	float mint;
	float maxt;

	traversal_structure() {
		this->w = NULL;
	}

	traversal_structure(kdNode *w, float mint, float maxt) {
		this->w = w;
		this->mint = mint;
		this->maxt = maxt;
	}
};

struct  traversal_structure_linear
{
	int node;
	float mint;
	float maxt;

	traversal_structure_linear() {
	}

	traversal_structure_linear(int node, float mint, float maxt) {
		this->node = node;
		this->mint = mint;
		this->maxt = maxt;
	}
};

class KD_Tree {
private:
	vector<kdNode> tree;
	kdNode* root;
	Voxel* V;
	int node_count = 0;

	int linearizeTreeRecursive( kdNodeLinear* result, int &size, kdNode* node ) {
		int index_used = size++;

		if (node->p != NULL) {
			result[index_used].splits = true;
			result[index_used].p = *node->p;
		}
		else {
			result[index_used].splits = false;
		}

		if (node->triangles != NULL) {
			result[index_used].triangles = new int[ node->triangles->size() ];
			result[index_used].len = node->triangles->size();

			for (size_t i = 0; i < node->triangles->size(); i++)
			{
				result[index_used].triangles[i] = node->triangles->at(i);
			}
		}
		else {
			result[index_used].triangles = NULL;
			result[index_used].len = 0;
		}

		result[index_used].left		= node->left != NULL	? linearizeTreeRecursive(result, size, node->left)	: -1;
		result[index_used].right	= node->right != NULL	? linearizeTreeRecursive(result, size, node->right)	: -1;


		return index_used;
	}

public:
	Voxel* getVoxel() {
		return this->V;
	}

	kdNodeLinear* linearizeTree() {
		kdNodeLinear* result = new kdNodeLinear[ this->node_count ];
		cout << "Kd-tree node count: " << this->node_count << endl;
		int size = 0;

		linearizeTreeRecursive( result, size, this->root );

		cout << "Max linear Kd-tree index is " << size << endl;

		return result;
	}

	void doTheBuild(Triangle* T, int size) {
		this->V = new Voxel();
		this->V->setVoxelParameters(T, size);

		root = BuildKdTree(T, size, this->node_count);

		cout << "Tree built" << endl;
	}

	void printLevel(kdNode* node, int level) {
		if (node->p != NULL) {
			for (int i = 0; i < level; ++i) cout << "|  ";
			cout << "Inner node:" << endl;
			for (int i = 0; i < level; ++i) cout << "|  ";
			cout << node->p->dimension << "->" << node->p->position << endl;
			for (int i = 0; i < level; ++i) cout << "|  ";
			cout << " <- Left:" << endl;
			printLevel(node->left, level + 1);
			for (int i = 0; i < level; ++i) cout << "|  ";
			cout << " -> Right:" << endl;
			printLevel(node->right, level + 1);

		}
		else {
			for (int i = 0; i < level; ++i) cout << "|  ";
			cout << "Leaf node:" << endl;
			for (int i = 0; i < level; ++i) cout << "|  ";
			cout << node->triangles->size() << endl;
			for (int i = 0; i < node->triangles->size(); i++)
			{
				for (int i = 0; i < level; ++i) cout << "|  ";
				//node->triangles->at(i)->print();
			}
		}
	}

	void printOutTree() {
		cout << this->V->position.x << "  -  " << this->V->position.y << "  -  " << this->V->position.z << endl;
		cout << this->V->position.x + this->V->dX << "  -  " << this->V->position.y + this->V->dY << "  -  " << this->V->position.z + this->V->dZ << endl;
		this->printLevel(this->root, 0);
	}

	Triangle* FindKDIntersection(Ray &ray, float & dist, Triangle* triangles/*, Triangle* obj = NULL*/) {
		stack<traversal_structure> stacktrav;
		traversal_structure actual;
		Triangle* object = NULL;
		dist = numeric_limits<float>::max();
		float tmp = numeric_limits<float>::min();
		float t = numeric_limits<float>::max();
		int axis;
		float tmp1;
		float tmp2;
		float value;
		float ray_origin_axis = 1.0;
		float ray_dir_axis = 1.0;
		kdNode *nearChild, *farChild;

		//rays++;

		/*std::cout << "Traversal start" << std::endl;
		std::cout << ray.direction->getX() << std::endl;
		std::cout << ray.direction->getY() << std::endl;
		std::cout << ray.direction->getZ() << std::endl;
		cout << "----" << endl;
		std::cout << ray.origin->getX() << std::endl;
		std::cout << ray.origin->getY() << std::endl;
		std::cout << ray.origin->getZ() << std::endl;*/


		if (!(fabs(ray.direction->getX()) < config::epsilon)) {
			//cout << this->V << endl;
			//cout << this->V->position.x << endl;
			//cout << this->V->dX << endl;
			tmp1 = (this->V->position.x - ray.origin->getX()) / ray.direction->getX();
			tmp2 = (this->V->position.x + this->V->dX - ray.origin->getX()) / ray.direction->getX();

			tmp = max(tmp, min(tmp1, tmp2));
			t = min(t, max(tmp1, tmp2));

		}

		if (!(fabs(ray.direction->getY()) < config::epsilon)) {
			tmp1 = (this->V->position.y - ray.origin->getY()) / ray.direction->getY();
			tmp2 = (this->V->position.y + this->V->dY - ray.origin->getY()) / ray.direction->getY();

			tmp = max(tmp, min(tmp1, tmp2));
			t = min(t, max(tmp1, tmp2));

		}

		if (!(fabs(ray.direction->getZ()) < config::epsilon)) {
			tmp1 = (this->V->position.z - ray.origin->getZ()) / ray.direction->getZ();
			tmp2 = (this->V->position.z + this->V->dZ - ray.origin->getZ()) / ray.direction->getZ();

			tmp = max(tmp, min(tmp1, tmp2));
			t = min(t, max(tmp1, tmp2));

		}

		//cout << "Final voxel bounds are are: " << tmp << " " << t << endl;

		stacktrav.push(traversal_structure(this->root, tmp, t));
		//std::cout << "Pushed in " << this->root << " " << tmp << " " << t << std::endl;

		while (!stacktrav.empty()) {
			actual = stacktrav.top();
			//std::cout << "Popped out " << actual.w << " " << actual.mint << " " << actual.maxt << std::endl;
			stacktrav.pop();

			while (actual.w->p != NULL) {
				//trav_steps++;
				axis = actual.w->p->dimension;
				value = actual.w->p->position;
				//std::cout << "Split plane " << value << " (" << axis << ")" << std::endl;

				switch (axis) {
				case 0:
					ray_origin_axis = ray.origin->getX();
					ray_dir_axis = ray.direction->getX();
					break;
				case 1:
					ray_origin_axis = ray.origin->getY();
					ray_dir_axis = ray.direction->getY();
					break;
				case 2:
					ray_origin_axis = ray.origin->getZ();
					ray_dir_axis = ray.direction->getZ();
					break;
				}


				if (value < ray_origin_axis) {
					nearChild = actual.w->right;
					farChild = actual.w->left;
				}
				else {
					nearChild = actual.w->left;
					farChild = actual.w->right;
				}


				if (!(fabs(ray_dir_axis) < config::epsilon)) {
					t = (value - ray_origin_axis) / ray_dir_axis;
				}
				else {
					t = numeric_limits<float>::max();
				}

				//std::cout << t << endl;

				if (t < 0 || t > actual.maxt) { // near
					actual.w = nearChild;
				}
				else {
					if (t < actual.mint) { // far
						actual.w = farChild;
					}
					else { // wherever you are
						stacktrav.push(traversal_structure(farChild, t, actual.maxt));
						actual.w = nearChild;
						actual.maxt = t;
					}
				}

			} // while not leaf

			if (actual.w->triangles != NULL) {
				for (int i = 0; i < actual.w->triangles->size(); i++)
				{
					//inters_steps++; 
					//if (obj != NULL && obj == &triangles[ actual.w->triangles->at(i) ] ) continue;
					if ( triangles[ actual.w->triangles->at(i) ].FindIntersection(ray, tmp) && tmp < dist) { // most definitely not ok
						dist = tmp;
						object = &triangles[ actual.w->triangles->at(i) ];
					}
				}

				if (object != NULL) {
					//cout << "Returning the triangle" << endl; 
					return object;
				}
				else {
					//cout << "No triangles intersected" << endl;
				}

			}

		} //while stack not empty

		return NULL;
	}

};

