
#include "primitives.h"

bool TransformationStack::scaleActualized;
int DrawObject::number;

float dotProduct(Vector3f a, Vertex b) {
	return ((a.x * b.getX()) + (a.y * b.getY()) + (a.z * b.getZ()));
}

float dotProduct(Vertex a, Vertex b) {
	return ((a.getX() * b.getX()) + (a.getY() * b.getY()) + (a.getZ() * b.getZ()));
}

bool EdgeSort(Edge e1, Edge e2) {
	return (e1.GetPositionY() > e2.GetPositionY());
}

bool PairSort(pair<float, float> p1, pair<float, float> p2) {
	return (p1.first < p2.first);
} 