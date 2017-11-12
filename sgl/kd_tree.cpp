
#pragma once

/*! \mainpage Fast KD-tree construction
*
* \section intro_sec Introduction
*
* This is an implementation of semester project for B4M39DPG
*
*/



#include "kd_tree.h"

int config::K_T = 1;
int config::K_I = 3;
float config::epsilon = 0.00000001;
bool config::debug = false;
float config::voxel_size_cap = 0.01; // DEBUG pro VS omezení RAM


/**
* Lambda function used biasing the cost function in favor of cutting away empty voxels
*/
float lambda(int N_L, int N_R) {
	//if (N_L == 0 || N_R == 0) return 0.8;
	return 1;
}


/**
* Surface area of a voxel
*/
float SA(const Voxel* V) {
	return 2 * V->dX*V->dY + 2 * V->dX*V->dZ + 2 * V->dY*V->dZ;
}


/**
* SAH cost function. Uses simplified formula.
*/
float Cost(float P_L, float P_R, int N_L, int N_R) {
	//cout << "SAH: " << endl;
	//cout << K_I*P_L*N_L << " - " << N_L << endl;
	//cout << K_I*P_R*N_R << " - " << N_R << endl;
	return lambda(N_L, N_R) * (config::K_T + config::K_I * (P_L*N_L + P_R*N_R));
}


/**
* SA heuristic function, return a rank for the given plane split
*/
pair<float, bool> SAH(splitPlane *p, Voxel* V, int N_L, int N_R, int N_P) {
	float cp_l, cp_r, P_L, P_R;
	Voxel *V_L, *V_R;
	pair<Voxel*, Voxel*> splitted;

	//if (debug) cout << "Split plane cand:" << p->position << " " << p->dimension << endl;
	//if (debug) cout << "SAH:" << N_L << " " << N_P << " " << N_R << endl;

	splitted = V->splitByPlane(p);

	V_L = splitted.first;
	V_R = splitted.second;

	//if (debug) cout << "Volume left: " << V_L->vol() << endl;
	//if (debug) cout << "Volume right: " << V_R->vol() << endl;

	if ((V_R->vol() < config::epsilon && (N_R + N_P == 0)) || (V_L->vol() < config::epsilon && (N_L + N_P == 0)))
		return make_pair(numeric_limits<float>::max(), true);

	P_L = SA(V_L) / SA(V);
	P_R = SA(V_R) / SA(V);

	cp_l = Cost(P_L, P_R, N_L + N_P, N_R);
	if (V_R->vol() < config::epsilon && N_R == 0) cp_l = numeric_limits<float>::max();
	if (V_L->vol() < config::epsilon && N_L + N_P == 0) cp_l = numeric_limits<float>::max();
	cp_r = Cost(P_L, P_R, N_L, N_R + N_P);
	if (V_L->vol() < config::epsilon && N_L == 0) cp_r = numeric_limits<float>::max();
	if (V_R->vol() < config::epsilon && N_R + N_P == 0) cp_r = numeric_limits<float>::max();


	delete V_L;
	delete V_R;

	if (cp_l < cp_r) {
		//if (debug) cout << "Cost " << cp_l << endl;
		return make_pair(cp_l, true);
	}
	//if (debug) cout << "Cost " << cp_r << endl;
	return make_pair(cp_r, false);
}



/**
* Function used for classification of triangles into their respective sub-voxels
*/
vector<int>* ClassifyLeftRightBoth(vector<int> *T, vector<triangleEvent*>* E, splitPlane *p) {

	vector<int> *triangleSides = new vector<int>(T->size(), 2); // 0 = left, 1 = right, 2 = both. We first assume all triangles are on both sides

																//cout << "Classification, plane position: " << p->position << " in dimension " << p->dimension << endl;

	for (int i = 0; i < E->size(); i++)
	{
		if (E->at(i)->dimension != p->dimension) continue;

		/*cout << "Event with:" << endl;
		cout << E->at(i)->coordinate << " " << E->at(i)->dimension << " " << E->at(i)->type << endl;
		cout << "For triangle " << E->at(i)->triangleID << endl;*/

		if ((E->at(i)->type == 0) && (E->at(i)->coordinate <= p->position)) triangleSides->at(E->at(i)->triangleID) = 0;
		if ((E->at(i)->type == 2) && (E->at(i)->coordinate >= p->position)) triangleSides->at(E->at(i)->triangleID) = 1;

		if (E->at(i)->type == 1) {
			if (E->at(i)->coordinate < p->position) triangleSides->at(E->at(i)->triangleID) = 0;
			if (E->at(i)->coordinate > p->position) triangleSides->at(E->at(i)->triangleID) = 1;

			// !! float rovnost
			if (fabs(E->at(i)->coordinate - p->position) < config::epsilon) {
				if (p->left) {
					triangleSides->at(E->at(i)->triangleID) = 0;
				}
				else {
					triangleSides->at(E->at(i)->triangleID) = 1;
				}
			}
		}
	}

	//cout << "Result:" << endl;
	/*for (size_t i = 0; i < triangleSides->size(); i++)
	{
	cout << triangleSides->at(i) << " ";
	}*/
	//cout << endl;

	return triangleSides;
}


/**
* Function for finding split plane with the best cost
*/
planeSolution* FindPlane(int N, Voxel *V, vector<triangleEvent*>* E) {
	vector<int> N_L, N_R, N_P;
	int p_plus, p_minus, p_pipe;
	int i = 0;
	splitPlane *p;
	planeSolution* sol;
	pair<float, bool> candidate;
	bool no_solution_yet = true;

	//cout << "Finding plane for " << N << " triangles" << endl;

	sol = new planeSolution;
	sol->cost = -1.0;
	sol->plane = NULL;

	/*for (int i = 0; i < E->size(); i++)
	{
	cout << "Coord:" << E->at(i)->coordinate << ",Dim:" << E->at(i)->dimension << ",Type:" << E->at(i)->type << ",Tri:" << E->at(i)->triangleID << endl;
	}*/

	for (int k = 0; k < 3; k++)
	{	// start with all triangles on right side
		N_L.push_back(0);
		N_P.push_back(0);
		N_R.push_back(N);
	}

	//cout << E->size() << "::" << endl;


	while (i < E->size()) //for (int i = 0; i < E->size(); i++)
	{
		p = new splitPlane(E->at(i)->coordinate, E->at(i)->dimension, false);
		p_plus = 0;
		p_minus = 0;
		p_pipe = 0;

		while ((i < E->size()) && (E->at(i)->dimension == p->dimension) && (fabs(p->position - E->at(i)->coordinate) < config::epsilon) && (E->at(i)->type == 0))
		{
			p_minus++;
			i++;
		}

		while ((i < E->size()) && (E->at(i)->dimension == p->dimension) && (fabs(p->position - E->at(i)->coordinate) < config::epsilon) && (E->at(i)->type == 1))
		{
			p_pipe++;
			i++;
		}

		while ((i < E->size()) && (E->at(i)->dimension == p->dimension) && (fabs(p->position - E->at(i)->coordinate) < config::epsilon) && (E->at(i)->type == 2))
		{
			p_plus++;
			i++;
		}

		// found all the event types

		//cout << "In dimension: " << p->dimension << ": " << p_minus << " " << p_pipe << " " << p_plus << endl;

		N_P[p->dimension] = p_pipe;
		N_R[p->dimension] -= p_pipe;
		N_R[p->dimension] -= p_minus;

		candidate = SAH(p, V, N_L[p->dimension], N_R[p->dimension], N_P[p->dimension]);

		if (candidate.first < sol->cost || no_solution_yet) {
			no_solution_yet = false;
			if (sol->plane != NULL) delete sol->plane;
			sol->cost = candidate.first;
			sol->plane = p;
			p->left = candidate.second;
		}

		N_L[p->dimension] += p_plus;
		N_L[p->dimension] += p_pipe;
		N_P[p->dimension] = 0;

	}

	if (config::debug) cout << "Best price " << sol->cost << endl;
	return sol;


}

/**
* Function deciding whether the recursion should be terminated. Returns TRUE if cost of splitting the voxel with the best plane is higher than not splitting at all.
*/
bool terminate(int N, float Cost) {
	//cout << "Cost is: " << Cost << ", term is: " << K_I*N << endl;
	if (Cost > config::K_I*N) return true;
	return false;
}


/**
* Merge function for events. Creates a new event list.
*/
vector<triangleEvent*>* mergeEvents(vector<triangleEvent*>* E1, vector<triangleEvent*>* E2) {
	vector<triangleEvent*>* E;
	eventComparator compare;
	int i1, i2;

	i1 = i2 = 0;

	E = new vector<triangleEvent*>((E1->size() + E2->size()), NULL);

	for (int i = 0; i < (E1->size() + E2->size()); i++)
	{
		if (i1 == E1->size()) {
			// E1 inserted completely
			E->at(i) = E2->at(i2);
			i2++;
			continue;
		}
		if (i2 == E2->size()) {
			// E2 inserted completely
			E->at(i) = E1->at(i1);
			i1++;
			continue;
		}

		if (compare(E1->at(i1), E2->at(i2))) {
			// E1 has lower value
			E->at(i) = E1->at(i1);
			i1++;
			continue;
		}
		else {
			// E2 has lower value
			E->at(i) = E2->at(i2);
			i2++;
			continue;
		}

	}

	delete E1;
	delete E2;
	return E;
}

/**
* Recursive tree-building function
*/
kdNode* RecBuild(vector<int>* T, Voxel *V, vector<triangleEvent*>* E, splitPlane* pp, Triangle* triangles) {
	planeSolution *best;
	kdNode* result = new kdNode();
	pair<Voxel*, Voxel*> splitted;
	Voxel *V1, *V2;
	triangleEvent *event;
	boundingBox AABB1, AABB2;
	vector<int> scrambledIndexes_L(T->size(), -1);
	vector<int> scrambledIndexes_R(T->size(), -1);
	vector<int> *classification;
	vector<int> *T_L, *T_R;
	vector<triangleEvent*> *E_L, *E_R, *E_R_new, *E_L_new;

	if (config::debug) cout << "Recursion with " << T->size() << " triangles" << endl;
	if (config::debug) cout << "Voxel bounds are " << V->position.x << ":" << V->position.y << ":" << V->position.z << "  -  " << V->dX << ":" << V->dY << ":" << V->dZ << endl;

	if (T->size() == 0)
	{
		// Create a leaf node from this
		result->triangles = NULL;

		if (config::debug) cout << "Terminating, creating a node (empty list)" << endl;

		delete V;
		delete E;
		delete T;

		return result;
	}



	// Find the best split plane
	best = FindPlane(T->size(), V, E);

	if (config::debug) cout << "Split plane position is: " << best->plane->position << " in dimension " << best->plane->dimension << endl;

	if (terminate(T->size(), best->cost) || (V->vol() < config::voxel_size_cap) || ((pp != NULL) && (best->plane->dimension == pp->dimension) && (fabs(best->plane->position - pp->position) < config::epsilon)))
	{
		// Create a leaf node from this
		result->triangles = T;

		if (config::debug) cout << "Terminating, creating a node (T/S/C)" << endl;

		delete V;
		delete E;

		return result;
	}

	// Split voxels
	splitted = V->splitByPlane(best->plane);
	V1 = splitted.first;
	V2 = splitted.second;

	if (V1->dX < 0 || V1->dY < 0 || V1->dZ < 0) {
		cout << "Parent bounds: " << V->position.x << ":" << V->position.y << ":" << V->position.z << "  -  " << V->dX << ":" << V->dY << ":" << V->dZ << endl;
		cout << "Bounds1: " << V1->position.x << ":" << V1->position.y << ":" << V1->position.z << "  -  " << V1->dX << ":" << V1->dY << ":" << V1->dZ << endl;
		cout << "Parent split plane: " << best->plane->position << " in dimension " << best->plane->dimension << endl;
		cout << "Something failed badly" << endl;
	}

	if (V2->dX < 0 || V2->dY < 0 || V2->dZ < 0) {
		cout << "Parent bounds: " << V->position.x << ":" << V->position.y << ":" << V->position.z << "  -  " << V->dX << ":" << V->dY << ":" << V->dZ << endl;
		cout << "Bounds2: " << V2->position.x << ":" << V2->position.y << ":" << V2->position.z << "  -  " << V2->dX << ":" << V2->dY << ":" << V2->dZ << endl;
		cout << "Parent split plane: " << best->plane->position << " in dimension " << best->plane->dimension << endl;
		cout << "Something failed badly" << endl;
	}

	//cout << "Classifying triangles" << endl;
	// Classify triangles and create new needed events
	classification = ClassifyLeftRightBoth(T, E, best->plane);
	T_L = new vector<int>;
	T_R = new vector<int>;

	/*for (int i = 0; i < E->size(); i++)
	{

	}*/

	for (int i = 0; i < T->size(); i++)
	{
		if (classification->at(i) == 0) {
			T_L->push_back(T->at(i));
			scrambledIndexes_L[i] = (T_L->size() - 1);
		}
		else {
			if (classification->at(i) == 1) {
				T_R->push_back(T->at(i));
				scrambledIndexes_R[i] = (T_R->size() - 1);
			}
			else {
				T_L->push_back(T->at(i));
				T_R->push_back(T->at(i));

				scrambledIndexes_L[i] = (T_L->size() - 1);
				scrambledIndexes_R[i] = (T_R->size() - 1);
			}
		}
	}

	for (int i = 0; i < T->size(); i++)
	{
		//cout << "Triangle " << i << " is " << scrambledIndexes_L[i] << " " << scrambledIndexes_R[i] << endl;
	}

	// Modify the triangle to which event belongs
	//cout << "Modifying events" << endl;
	E_L = new vector<triangleEvent*>;
	E_R = new vector<triangleEvent*>;
	E_L_new = new vector<triangleEvent*>;
	E_R_new = new vector<triangleEvent*>;


	for (int i = 0; i < E->size(); i++)
	{
		//cout << E->at(i)->triangleID << endl;
		if (classification->at(E->at(i)->triangleID) == 0) { E_L->push_back(E->at(i)); E->at(i)->triangleID = scrambledIndexes_L[E->at(i)->triangleID]; continue; }
		if (classification->at(E->at(i)->triangleID) == 1) { E_R->push_back(E->at(i)); E->at(i)->triangleID = scrambledIndexes_R[E->at(i)->triangleID]; continue; }
		if (classification->at(E->at(i)->triangleID) == 2) {
			classification->at(E->at(i)->triangleID) = -1;

			if (config::debug) cout << "Creating new events for triangle " << E->at(i)->triangleID << endl;
			//if (debug) cout << E->at(i)->triangleID << " vs. " << T->size() << endl;
			//if (debug) cout << T->at(E->at(i)->triangleID) << endl;
			// Create some new events (watch out for appropriate triangle ID)
			if (config::debug)
				for (size_t j = 0; j < 3; j++)
				{
					cout << triangles[ T->at(E->at(i)->triangleID) ].getVertices()->at(j)->getX() << " " << triangles[ T->at(E->at(i)->triangleID) ].getVertices()->at(j)->getY() << " " << triangles[ T->at(E->at(i)->triangleID) ].getVertices()->at(j)->getZ() << endl;
				}
			if (!(AABB1.clipToVoxel(V1, &triangles[ T->at(E->at(i)->triangleID) ]  ))) { if (config::debug) cout << "No triangle intersection with plane in V1" << endl; }
			if (!(AABB2.clipToVoxel(V2, &triangles[ T->at(E->at(i)->triangleID) ]  ))) { if (config::debug) cout << "No triangle intersection with plane in V2" << endl; }

			if (config::debug) {
				cout << "New bounds:" << endl;
				cout << AABB1.x_min << ":" << AABB1.x_max << endl;
				cout << AABB1.y_min << ":" << AABB1.y_max << endl;
				cout << AABB1.z_min << ":" << AABB1.z_max << endl;
				cout << "2:" << endl;
				cout << AABB2.x_min << ":" << AABB2.x_max << endl;
				cout << AABB2.y_min << ":" << AABB2.y_max << endl;
				cout << AABB2.z_min << ":" << AABB2.z_max << endl;
			}

			// event generation
			{
				// V1
				event = new triangleEvent();
				// float rovnost
				if (fabs(AABB1.x_min - AABB1.x_max) < config::epsilon) {
					event->type = 1;
					event->dimension = 0;
					event->coordinate = AABB1.x_min;
					event->triangleID = scrambledIndexes_L[E->at(i)->triangleID];
					E_L_new->push_back(event);
				}
				else {
					event->type = 2;
					event->dimension = 0;
					event->coordinate = AABB1.x_min;
					event->triangleID = scrambledIndexes_L[E->at(i)->triangleID];
					E_L_new->push_back(event);

					event = new triangleEvent();
					event->type = 0;
					event->dimension = 0;
					event->coordinate = AABB1.x_max;
					event->triangleID = scrambledIndexes_L[E->at(i)->triangleID];
					E_L_new->push_back(event);
				}

				event = new triangleEvent();
				if (fabs(AABB1.y_min - AABB1.y_max) < config::epsilon) {
					event->type = 1;
					event->dimension = 1;
					event->coordinate = AABB1.y_min;
					event->triangleID = scrambledIndexes_L[E->at(i)->triangleID];
					E_L_new->push_back(event);
				}
				else {
					event->type = 2;
					event->dimension = 1;
					event->coordinate = AABB1.y_min;
					event->triangleID = scrambledIndexes_L[E->at(i)->triangleID];
					E_L_new->push_back(event);

					event = new triangleEvent();
					event->type = 0;
					event->dimension = 1;
					event->coordinate = AABB1.y_max;
					event->triangleID = scrambledIndexes_L[E->at(i)->triangleID];
					E_L_new->push_back(event);
				}

				event = new triangleEvent();
				if (fabs(AABB1.z_min - AABB1.z_max) < config::epsilon) {
					event->type = 1;
					event->dimension = 2;
					event->coordinate = AABB1.z_min;
					event->triangleID = scrambledIndexes_L[E->at(i)->triangleID];
					E_L_new->push_back(event);
				}
				else {
					event->type = 2;
					event->dimension = 2;
					event->coordinate = AABB1.z_min;
					event->triangleID = scrambledIndexes_L[E->at(i)->triangleID];
					E_L_new->push_back(event);

					event = new triangleEvent();
					event->type = 0;
					event->dimension = 2;
					event->coordinate = AABB1.z_max;
					event->triangleID = scrambledIndexes_L[E->at(i)->triangleID];
					E_L_new->push_back(event);
				}

				// V2
				event = new triangleEvent();
				if (fabs(AABB2.x_min - AABB2.x_max) < config::epsilon) {
					event->type = 1;
					event->dimension = 0;
					event->coordinate = AABB2.x_min;
					event->triangleID = scrambledIndexes_R[E->at(i)->triangleID];
					E_R_new->push_back(event);
				}
				else {
					event->type = 2;
					event->dimension = 0;
					event->coordinate = AABB2.x_min;
					event->triangleID = scrambledIndexes_R[E->at(i)->triangleID];
					E_R_new->push_back(event);

					event = new triangleEvent();
					event->type = 0;
					event->dimension = 0;
					event->coordinate = AABB2.x_max;
					event->triangleID = scrambledIndexes_R[E->at(i)->triangleID];
					E_R_new->push_back(event);
				}

				event = new triangleEvent();
				if (fabs(AABB2.y_min - AABB2.y_max) < config::epsilon) {
					event->type = 1;
					event->dimension = 1;
					event->coordinate = AABB2.y_min;
					event->triangleID = scrambledIndexes_R[E->at(i)->triangleID];
					E_R_new->push_back(event);
				}
				else {
					event->type = 2;
					event->dimension = 1;
					event->coordinate = AABB2.y_min;
					event->triangleID = scrambledIndexes_R[E->at(i)->triangleID];
					E_R_new->push_back(event);

					event = new triangleEvent();
					event->type = 0;
					event->dimension = 1;
					event->coordinate = AABB2.y_max;
					event->triangleID = scrambledIndexes_R[E->at(i)->triangleID];
					E_R_new->push_back(event);
				}

				event = new triangleEvent();
				if (fabs(AABB2.z_min - AABB2.z_max) < config::epsilon) {
					event->type = 1;
					event->dimension = 2;
					event->coordinate = AABB2.z_min;
					event->triangleID = scrambledIndexes_R[E->at(i)->triangleID];
					E_R_new->push_back(event);
				}
				else {
					event->type = 2;
					event->dimension = 2;
					event->coordinate = AABB2.z_min;
					event->triangleID = scrambledIndexes_R[E->at(i)->triangleID];
					E_R_new->push_back(event);

					event = new triangleEvent();
					event->type = 0;
					event->dimension = 2;
					event->coordinate = AABB2.z_max;
					event->triangleID = scrambledIndexes_R[E->at(i)->triangleID];
					E_R_new->push_back(event);
				}
			}

			delete E->at(i); // remove all events for triangles in both split voxels

		}
	}




	//cout << "Sorting events" << endl;
	std::sort(E_L_new->begin(), E_L_new->end(), eventComparator());
	std::sort(E_R_new->begin(), E_R_new->end(), eventComparator());

	//cout << "Merging events" << endl;
	// Modify events by merging
	E_L = mergeEvents(E_L, E_L_new);
	E_R = mergeEvents(E_R, E_R_new);



	//delete V;
	delete T;
	delete E;


	// Fill in the kdNode data
	result->p = best->plane;
	if (config::debug) cout << "Creating left child" << endl;
	result->left = RecBuild(T_L, V1, E_L, result->p, triangles);
	if (config::debug) cout << "Creating right child" << endl;
	result->right = RecBuild(T_R, V2, E_R, result->p, triangles);

	return result;
}

/**
* Main function for building of a kd-tree using SAH cost function
*/
kdNode* BuildKdTree(Triangle* T_src, int size) {
	Voxel *V = new Voxel;
	vector<triangleEvent*>* E = new vector<triangleEvent*>;
	triangleEvent *event;
	vector<int >* T = new vector<int>(size, NULL);
	boundingBox AABB;

	cout << "Build started" << endl;

	// make triangles happen
	for (int i = 0; i < size; i++)
	{
		T->at(i) = i;
	}

	cout << "Data structures done" << endl;
	// create events
	for (int i = 0; i < size; i++)
	{
		AABB.boundTriangle(&T_src[i]);

		event = new triangleEvent();
		if (fabs(AABB.x_min - AABB.x_max) < config::epsilon) {
			event->type = 1;
			event->dimension = 0;
			event->coordinate = AABB.x_min;
			event->triangleID = i;
			E->push_back(event);
		}
		else {
			event->type = 2;
			event->dimension = 0;
			event->coordinate = AABB.x_min;
			event->triangleID = i;
			E->push_back(event);

			event = new triangleEvent();
			event->type = 0;
			event->dimension = 0;
			event->coordinate = AABB.x_max;
			event->triangleID = i;
			E->push_back(event);
		}

		event = new triangleEvent();
		if (fabs(AABB.y_min - AABB.y_max) < config::epsilon) {
			event->type = 1;
			event->dimension = 1;
			event->coordinate = AABB.y_min;
			event->triangleID = i;
			E->push_back(event);
		}
		else {
			event->type = 2;
			event->dimension = 1;
			event->coordinate = AABB.y_min;
			event->triangleID = i;
			E->push_back(event);

			event = new triangleEvent();
			event->type = 0;
			event->dimension = 1;
			event->coordinate = AABB.y_max;
			event->triangleID = i;
			E->push_back(event);
		}

		event = new triangleEvent();
		if (fabs(AABB.z_min - AABB.z_max) < config::epsilon) {
			event->type = 1;
			event->dimension = 2;
			event->coordinate = AABB.z_min;
			event->triangleID = i;
			E->push_back(event);
		}
		else {
			event->type = 2;
			event->dimension = 2;
			event->coordinate = AABB.z_min;
			event->triangleID = i;
			E->push_back(event);

			event = new triangleEvent();
			event->type = 0;
			event->dimension = 2;
			event->coordinate = AABB.z_max;
			event->triangleID = i;
			E->push_back(event);
		}

	}

	// sort them
	sort(E->begin(), E->end(), eventComparator());

	// create the Voxel
	V->setVoxelParameters(T_src, size);

	// call the recursive function
	return RecBuild(T, V, E, NULL, T_src);


}

