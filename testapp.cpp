//---------------------------------------------------------------------------
// testapp.cpp
// Testing file for the Simple Graphics Library (SGL)
// Author: Jaroslav Krivanek, Jiri Bittner, CTU Prague
//---------------------------------------------------------------------------
// version 2.0, 1/10/2012

/// uncomment the tests you wish to run

//#define TEST0
//#define TEST1
//#define TEST2
//#define TEST3
#define TEST4
//#define TEST5

#ifdef TEST0
#define TEST_0A
#define TEST_0B
#endif

#ifdef TEST1
#define TEST_1A
#define TEST_1B
#define TEST_1C
#endif

#ifdef TEST2
#define TEST_2A
#define TEST_2B
#define TEST_2C
#define TEST_2D
#endif

#ifdef TEST3
//#define TEST_3A
//#define TEST_3B
//#define TEST_3C
#endif

#ifdef TEST4
#define TEST_4A
//#define TEST_4B
//#define TEST_4C
#endif

#ifdef TEST5
#define TEST_5A
//#define TEST_5B
//#define TEST_5C
//#define TEST_5D
#endif


#if !defined(USE_GUI) && defined(_MSC_VER)
#define USE_GUI 1
#endif

#ifdef TEST_0A
#define USE_VIEWPORT 0
#endif

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm> ///// ADDED... max' is not a member of 'std' 

#if USE_GUI
#include <GL/glut.h>
#include <GL/gl.h>
#endif

#include "sgl.h"
#include "timer.h"
#include "nffread.h"
#include "nffstore.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

using namespace std;

#define PRINTVAR(a) cerr << #a"\t" << a << endl;


/// Main windows parameters
//#define WIDTH  512
//#define HEIGHT 512


#define WIDTH  800
#define HEIGHT 600
#define TITLE  "SGL Test Application"
#define NUM_CONTEXTS 8

static int _contexts[10];

float tx=0, ty=0, tz=0, tstep=0.2;
float rot=3.14/3, rotstep=0.1;


/// helper class for sgluLookAt
class Vector3 {
public:
	float x,y,z;
	Vector3(float xx,float yy,float zz) : x(xx),y(yy),z(zz) {}
	inline friend float SqrMagnitude(const Vector3& v) { return v.x*v.x+v.y*v.y+v.z*v.z; }
	Vector3& operator/= (float A) {float a = 1.0f/A; x*=a;y*=a;z*=a;return *this;}
	inline friend Vector3 CrossProd (const Vector3& A, const Vector3& B)
	{return Vector3(A.y * B.z - A.z * B.y, A.z * B.x - A.x * B.z, A.x * B.y - A.y * B.x);}
};


typedef unsigned char uint8;
typedef unsigned int uint;

/// Stores framebuffer as uncompressed .tga image
void WriteTGA(
			  const char *aFilename)
{
	float *data = (float *)sglGetColorBufferPointer();

	if (data == NULL)
		return;
	
	int resX = WIDTH;
	int resY = HEIGHT;
	char header[18];
	header[0] = 0;
	header[1] = 0;
	header[2] = 2;
	header[3] = 0;
	header[4] = 0;
	header[5] = 0;
	header[6] = 0;
	header[7] = 0;
	header[8] = 0;
	header[9] = 0;
	header[10] = 0;
	header[11] = 0;
	header[12] = static_cast<uint8>(resX % 0x100);
	header[13] = static_cast<uint8>(resX / 0x100);
	header[14] = static_cast<uint8>(resY % 0x100);
	header[15] = static_cast<uint8>(resY / 0x100);
	header[16] = 24;
	header[17] = 0; // set to 32 to flip the image

	std::ofstream outFile;
	outFile.open(aFilename, std::ios::binary);
	if(!outFile) {
	  //throw std::exception("Could not open required file");
	  cerr<<"Could not open required file "<<aFilename<<endl;
	}
	
	outFile.write(reinterpret_cast<char*>(&header), 18);
	
		for(uint i=0; i<(3*resX*resY);) {
		uint8 r, g, b;
		r = static_cast<uint8>(
							   std::max(0.f, std::min(1.f, data[i++])) * 255.f);
		g = static_cast<uint8>(
							   std::max(0.f, std::min(1.f, data[i++])) * 255.f);
		b = static_cast<uint8>(
							   std::max(0.f, std::min(1.f, data[i++])) * 255.f);
		outFile.write(reinterpret_cast<char*>(&b), sizeof(uint8));
		outFile.write(reinterpret_cast<char*>(&g), sizeof(uint8));
		outFile.write(reinterpret_cast<char*>(&r), sizeof(uint8));
	}
	outFile.flush();
	outFile.close();
}

/// like gluLookAt
void sgluLookAt(float eyex   , float eyey   , float eyez,
                float centerx, float centery, float centerz,
                float upx    , float upy    , float upz)
{
  float    sqrmag;

  /* Make rotation matrix */

  /* Z vector */
  Vector3  z(eyex-centerx,eyey-centery,eyez-centerz);
  sqrmag = SqrMagnitude(z);
  if(sqrmag)
    z /= sqrtf(sqrmag);

  /* Y vector */
  Vector3 y(upx,upy,upz);

  /* X vector = Y cross Z */
  Vector3 x(CrossProd(y,z));

  sqrmag = SqrMagnitude(x);
  if(sqrmag)
    x /= sqrtf(sqrmag);

  /* Recompute Y = Z cross X */
  y = CrossProd(z,x);

  sqrmag = SqrMagnitude(y);
  if(sqrmag)
    y /= sqrtf(sqrmag);

  float m[] = {
    x.x, y.x, z.x, 0, // col 1
    x.y, y.y, z.y, 0, // col 2
    x.z, y.z, z.z, 0, // col 3
    - eyex*x.x - eyey*x.y - eyez*x.z , // col 4
    - eyex*y.x - eyey*y.y - eyez*y.z , // col 4
    - eyex*z.x - eyey*z.y - eyez*z.z , // col 4
    1.0};                              // col 4

	sglMultMatrix(m);
}

/// like gluPerspective
void sgluPerspective( float fovy, float aspect, float zNear, float zFar )
{
  fovy *= (3.1415926535/180);
  float h2 = tan(fovy/2)*zNear;
  float w2 = h2*aspect;
  sglFrustum(-w2,w2,-h2,h2,zNear,zFar);
}



// ==========================================================================
//  Test cases
// ==========================================================================

/// render a wireframe 2D box
void box()
{
  sglBegin(SGL_LINE_LOOP);
  sglVertex2f(-1,-1);
  sglVertex2f(-1,1);
  sglVertex2f(1,1);
  sglVertex2f(1,-1);
  sglEnd();
  sglBegin(SGL_LINES);
  sglVertex2f(0,-1);
  sglVertex2f(0,1);
  sglEnd();
}

/// render a 4-sided polygon
void poly4(int idx[4], float p[][3])
{
  sglBegin(SGL_POLYGON);
  for (int i = 0; i < 4; i++) {
	  sglVertex3f(p[idx[i]][0], p[idx[i]][1], p[idx[i]][2]);
	  //cout << "Pridvan vertex " << p[idx[i]][0] << "," << p[idx[i]][1] << "," << p[idx[i]][2] << endl;
  }
  sglEnd();
}

/// render a cube [-1,1]^3
void cube()
{
  float vertices[][3] =
  { {0,0,0} ,
  { 1, 1, 1},
  { 1, 1,-1},
  { 1,-1, 1},
  { 1,-1,-1},
  {-1, 1, 1},
  {-1, 1,-1},
  {-1,-1, 1},
  {-1,-1,-1}};
  int indices[][4] =
  {
    {2,1,3,4},
    {5,6,8,7},
    {1,2,6,5},
    {4,3,7,8},
    {3,1,5,7},
    {2,4,8,6}
  };

  sglColor3f(0,0,1);
  poly4(indices[0],vertices);
  sglColor3f(0,1,0);
  poly4(indices[1],vertices);
  sglColor3f(0,1,1);
  poly4(indices[2],vertices);
  sglColor3f(1,0,0);
  poly4(indices[3],vertices);
  sglColor3f(1,0,1);
  poly4(indices[4],vertices);
  sglColor3f(1,1,0);
  poly4(indices[5],vertices);
}

/// render cube at (x,y,z) rotate by 'rot' radians
void placeCube(float x,float y,float z)
{
	//cout << "delam kostku" << endl;
  sglPushMatrix();
  sglTranslate(x,y,z);
  sglRotateY(rot);
  cube();
  sglPopMatrix();
}


/// 3D cube drawing test
void DrawTestScene2C(void) 
{
  // set viewport
  sglViewport(0, 0, WIDTH, HEIGHT);

  //cout << "now" << endl;

  // set the projection matrix
  sglMatrixMode(SGL_PROJECTION);
  sglLoadIdentity();

  
  sgluPerspective(45,(float)WIDTH/HEIGHT, 0.1, 10.0);


  //cout << "now2" << endl;

  // set the modelview matrix
  sglMatrixMode(SGL_MODELVIEW);
  sglLoadIdentity();

  sglTranslate(tx,ty,tz-10);

  sglAreaMode(SGL_FILL);

  // upper row with depth test
  sglEnable(SGL_DEPTH_TEST);
  placeCube(-1,2,0);
  placeCube(1,2,0);
  placeCube(0,1.5,0);

  // lower row without depth test
  sglDisable(SGL_DEPTH_TEST);
  placeCube(-1,-2,0);
  placeCube(1,-2,0);
  placeCube(0,-1.5,0);
}

/// render a non-convex polygon
void butterfly()
{
  sglBegin(SGL_POLYGON);
  sglVertex2f(-2,-2);
  sglVertex2f( 0, 1);
  sglVertex2f( 2,-2);
  sglVertex2f( 2, 2);
  sglVertex2f( 0, -1);
  sglVertex2f(-2, 2);

  //cout << "eyyy" << endl;
  sglEnd();
}


/// render a bunch of colorful ellipses
void ellipses(float r, float R, float G, float B)
{
  int numInitSegments = 10;
  for(int i=0; i<numInitSegments; i++) {
    float c = float(i)/numInitSegments;
    sglColor3f(c*R,c*G,c*B);
    sglEllipse(0,0,0,r*float(i+1)/numInitSegments,r*float(numInitSegments-i)/numInitSegments);
  }
}

/// render a bunch of colorful arcs
void arcs(float initr, float R, float G, float B)
{
  for(float r=initr ; r>0.1; r-=0.2) {
    float c = r/initr;;
    sglColor3f(c*R,c*G,c*B);
    for(float seg=0; seg<M_PI*2; seg+=M_PI*2/8)
    sglArc(0,0,0,r,seg,seg+M_PI*2/16);
  }
}

/// Element filling test
void DrawTestScene2B()
{
  sglDisable(SGL_DEPTH_TEST);

  // set viewport
  sglViewport(0, 0, WIDTH, HEIGHT);

  // set the projection matrix
  sglMatrixMode(SGL_PROJECTION);
  sglLoadIdentity();
  sglOrtho(0, 8, -3, 3, -1, 1);

  // set the modelview matrix
  sglMatrixMode(SGL_MODELVIEW);
  sglLoadIdentity();

  sglTranslate(1.5,0,0);

  float r;
  // circles
  for(r=1.2; r>0.1; r-=0.1) {
    sglColor3f(0,r,r);
    sglAreaMode(SGL_FILL);
    sglCircle(0, 0, 0, r);
    sglColor3f(r,r,0);
    sglAreaMode(SGL_LINE);
    sglCircle(0, 0, 0, r);
    sglColor3f(r,r,r);
    sglAreaMode(SGL_POINT);
    sglCircle(0, 0, 0, r);
  }

  // ellipses
  sglTranslate(2.5,0,0);

  sglPushMatrix();
  sglTranslate(0,1.5,0);
  sglAreaMode(SGL_FILL);
  ellipses(1.4, 1,1,0);
  sglAreaMode(SGL_LINE);
  ellipses(1.4, 0,1,1);
  sglTranslate(0,-3,0);
  sglRotate2D(M_PI*0.25,0,0);
  sglAreaMode(SGL_FILL);
  ellipses(1.4, 1,1,0);
  sglAreaMode(SGL_LINE);
  ellipses(1.4, 0,1,1);
  sglPopMatrix();

  // arcs
  sglTranslate(2.5,0,0);
  sglAreaMode(SGL_FILL);
  arcs(1.3, 1,0,1);
  sglAreaMode(SGL_LINE);
  arcs(1.3, 0,1,0);
}


/// Polygon fill test
void DrawTestScene2A(void) 
{
  sglDisable(SGL_DEPTH_TEST);

  // set viewport
  sglViewport(0, 0, WIDTH, HEIGHT);

  // set the projection matrix
  sglMatrixMode(SGL_PROJECTION);
  sglLoadIdentity();
  sglOrtho(-8, 8, -6, 6, -1, 1);

  // set the modelview matrix
  sglMatrixMode(SGL_MODELVIEW);
  sglLoadIdentity();



  sglColor3f(1,0,0);
  sglAreaMode(SGL_FILL);
  //sglAreaMode(SGL_LINE);
  //cout << "eyyy" << endl;
  butterfly();

  //cout << "eyyy2" << endl;
  sglColor3f(1,1,0);
  sglAreaMode(SGL_LINE);
  butterfly();



  sglColor3f(0,1,1);
  sglAreaMode(SGL_FILL);
  //sglAreaMode(SGL_LINE);

  sglTranslate(4,0,0);
  sglRotate2D(1,0,0);
  butterfly();
  sglColor3f(0,1,0);
  sglAreaMode(SGL_LINE);
  butterfly();

  sglLoadIdentity();
  sglTranslate(-4,-1,0);
  sglRotate2D(-0.5,0,0);
  sglAreaMode(SGL_FILL);
  //sglAreaMode(SGL_LINE);
  butterfly();
  sglColor3f(1,1,0);
  sglAreaMode(SGL_LINE);
  butterfly();
}


/// render a branch of a tree
void branch(int depth,float rot)
{
  sglPushMatrix();
  for(int i=0; i<3; i++) {
    sglScale(0.87,0.87,1);
    sglRotate2D(rot*0.15,0,1);
    sglTranslate(0,2,0);
    box();
  }
  if(depth<3) {
    branch(depth+1,1);
    sglColor3f(0,1,0);
    branch(depth+1,-1);
  }
  sglPopMatrix();
}

/// Matrix stack test
void DrawTestScene1C(void) 
{
  sglDisable(SGL_DEPTH_TEST);

  // set viewport
  sglViewport(0, 0, WIDTH, HEIGHT);

  // set the projection matrix
  sglMatrixMode(SGL_PROJECTION);
  sglLoadIdentity();
  sglOrtho(-8, 8, -1, 12, -1, 1);

  // set the modelview matrix
  sglMatrixMode(SGL_MODELVIEW);
  sglLoadIdentity();

  // draw coord axes
  sglColor3f(0.5,0.5,0.5);
  sglBegin(SGL_LINES);
  sglVertex2f(-7,0);
  sglVertex2f( 7,0);
  sglVertex2f( 0,0);
  sglVertex2f(0,10);
  sglEnd();

  sglColor3f(1,1,1);

  branch(0,1);
}


/// Transformation test
void DrawTestScene1B(void) 
{
  sglDisable(SGL_DEPTH_TEST);
  sglAreaMode(SGL_LINE);

  // set viewport
  sglViewport(0, 0, WIDTH, HEIGHT);

  // set the projection matrix
  sglMatrixMode(SGL_PROJECTION);
  sglLoadIdentity();
  sglOrtho(-8, 8, -6, 6, -1, 1);

  // set the modelview matrix
  sglMatrixMode(SGL_MODELVIEW);
  sglLoadIdentity();

  // draw coord axes
  sglColor3f(0.5,0.5,0.5);
  sglBegin(SGL_LINES);
  sglVertex2f(-7,0);
  sglVertex2f(7,0);
  sglVertex2f(0,-5);
  sglVertex2f(0,5);
  sglEnd();

  // BOX
  sglColor3f(1,1,1);
  box();

  sglColor3f(0,1,0);
  sglTranslate(-3,0,0);
  box();

  sglColor3f(1,0,0);
  sglTranslate(-3,0,0);
  sglRotate2D(M_PI*1/8,0,0);
  box();

  sglColor3f(0,0,1);
  sglLoadIdentity();
  sglTranslate(3,0,0);
  sglScale(0.5,2,1);
  box();

  sglColor3f(0,1,1);
  sglLoadIdentity();
  sglTranslate(6,0,0);
  sglRotate2D(M_PI*1/8,0,0);
  sglScale(0.5,2,1);
  box();

  // ARC
  sglLoadIdentity(); sglTranslate(0,4,0);

  sglColor3f(1,1,1);
  sglArc(0,0,0,1,0,4);

  sglColor3f(0,1,0);
  sglTranslate(-3,0,0);
  sglArc(0,0,0,1,0,4);

  sglColor3f(1,0,0);
  sglTranslate(-3,0,0);
  sglRotate2D(M_PI*1/8,0,0);
  sglArc(0,0,0,1,0,4);

  sglColor3f(0,0,1);
  sglLoadIdentity(); sglTranslate(0,4,0);
  sglTranslate(3,0,0);
  sglScale(0.5,2,1);
  sglArc(0,0,0,1,0,4);

  sglColor3f(0,1,1);
  sglLoadIdentity(); sglTranslate(0,4,0);
  sglTranslate(6,0,0);
  sglRotate2D(M_PI*1/8,0,0);
  sglScale(0.5,2,1);
  sglArc(0,0,0,1,0,4);

  // ELLIPSE
  sglLoadIdentity(); sglTranslate(0,-4,0);

  sglColor3f(1,1,1);
  sglEllipse(0,0,0,1.5,0.7);

  sglColor3f(0,1,0);
  sglTranslate(-3,0,0);
  sglEllipse(0,0,0,1.5,0.7);

  sglColor3f(1,0,0);
  sglTranslate(-3,0,0);
  sglRotate2D(M_PI*1/8,0,0);
  sglEllipse(0,0,0,1.5,0.7);

  sglColor3f(0,0,1);
  sglLoadIdentity(); sglTranslate(0,-4,0);
  sglTranslate(3,0,0);
  sglScale(0.5,2,1);
  sglEllipse(0,0,0,1.5,0.7);

  sglColor3f(0,1,1);
  sglLoadIdentity(); sglTranslate(0,-4,0);
  sglTranslate(6,0,0);
  sglRotate2D(M_PI*1/8,0,0);
  sglScale(0.5,2,1);
  sglEllipse(0,0,0,1.5,0.7);
}


/// Drawing simnple primitives, while assuming incomplete
/// transformation chain (no modeling, projection and window transformations)
/// Warning: will not run properly once the viewport transofrmation is implemented!!
void DrawTestScene0A(void) 
{
  sglViewport(-1,-1,2,2);
  sglDisable(SGL_DEPTH_TEST);
  sglMatrixMode(SGL_PROJECTION);
  sglLoadIdentity();
  sglMatrixMode(SGL_MODELVIEW);
  sglLoadIdentity();

  float centerX=0, centerY=0;
  //	float centerX=0, centerY=0;

  int numSegments = 32;
  float angleStep = 0.5*M_PI/(float)numSegments;
  
  int numCircles = 64;
  float radiusStep = (HEIGHT)/numCircles;
  //float radiusStep = .5f/numCircles;
  float r = radiusStep;
  for(int j=0; j < numCircles; j++, r+=radiusStep) {
    // point test
    sglColor3f(j/(float)numCircles,j/(float)numCircles,j/(float)numCircles);
    sglPointSize(1);
    sglBegin(SGL_POINTS);
    {
      for(int i=0; i<numSegments; i++) {
        float angle = angleStep*(float)i;
        sglVertex2f(centerX + r*cosf(angle), centerY + r*sinf(angle));
      }
    }
    sglEnd();
  }
}

void DrawTestScene0B(void) 
{
  sglViewport(-1,-1,2,2);
  sglDisable(SGL_DEPTH_TEST);
  sglMatrixMode(SGL_PROJECTION);
  sglLoadIdentity();
  sglMatrixMode(SGL_MODELVIEW);
  sglLoadIdentity();

  sglDisable(SGL_DEPTH_TEST);

  float centerX=0, centerY=0;
  //	float centerX=0, centerY=0;
  int numSegments = 64;
  float angleStep = 0.25f*M_PI/(float)numSegments;
  float r=HEIGHT;
  //float r=0.5f;
  // line test
  sglPointSize(1);
  for(int i=0; i<numSegments; i++) {
	sglColor3f(i/(float)numSegments,i/(float)numSegments,i/(float)numSegments);
	float angle = angleStep*(float)i;
	sglBegin(SGL_LINES);
	sglVertex2f(centerX, centerY );
	sglVertex2f(centerX + r*cosf(angle), centerY + r*sinf(angle));
	sglEnd();
  }
}

/// Drawing of all elements - orthographic tranformation only (scale+translation in 2d)
void DrawTestScene1A(void) 
{
  sglDisable(SGL_DEPTH_TEST);

  // set viewport
  sglViewport(0, 0, WIDTH, HEIGHT);

  // set the projection matrix
  sglMatrixMode(SGL_PROJECTION);
  sglLoadIdentity();
  sglOrtho(-10*WIDTH/HEIGHT, 10*WIDTH/HEIGHT, -10, 10, -1, 1);

  // set the modelview matrix
  sglMatrixMode(SGL_MODELVIEW);
  sglLoadIdentity();

  float offsetx, offsety = -6;
  int numInitSegments = 8, i;
  float segment = 2.0*M_PI/(float)numInitSegments;

  float r;
  for(r=0.2; r<=2.5; r+=0.2) {

    offsetx = -7.25;

    // point test
    sglColor3f(r/3,r/3,r/3);
    sglPointSize(3);
    sglBegin(SGL_POINTS);
    {
      for(i=0; i<numInitSegments; i++) {
        float angle = segment*(float)i;
        sglVertex2f(offsetx + r*cosf(angle),offsety + r*sinf(angle));
      }
    }
    sglEnd();

    offsetx += 5;

    // line test
    sglColor3f(1,0,0);
    sglBegin(SGL_LINES);
    for(i=0; i<numInitSegments; i++) {
      float angle = segment*(float)i;
      sglVertex2f(offsetx+r*cos(angle),offsety+r*sin(angle));
    }
    sglEnd();

    offsetx += 5;

    // line strip test
    sglColor3f(0,1,0);
    sglBegin(SGL_LINE_STRIP);
    for(i=0; i<numInitSegments; i++) {
      float angle = segment*(float)i;
      sglVertex2f(offsetx+r*cos(angle),offsety+r*sin(angle));
    }
    sglEnd();

    offsetx += 5;


    // line loop test
    sglColor3f(0,0,1);
    sglBegin(SGL_LINE_LOOP);
    for(i=0; i<numInitSegments; i++) {
      float angle = segment*(float)i;
      sglVertex2f(offsetx+r*cos(angle),offsety+r*sin(angle));
    }
    sglEnd();

  } // for r...

  offsety += 2.5*r;

  sglAreaMode(SGL_LINE);

  r=2;

  // ellipse drawing
  sglColor3f(1,0,1);
  for(i=0; i<numInitSegments; i++) {
    sglEllipse(0,0,0,(i+1)*2.5*r/numInitSegments,(numInitSegments-i)*2.5*r/numInitSegments);
  }

  sglColor3f(0,1,1);

  sglEllipse(-7,-1,0,2.9,0.2);
  sglEllipse(7,0,0,0.2,2.9);

  offsetx = -7.25;
  offsety = 6;

  sglColor3f(1,1,0);

  for(float rr=r/5; rr<=r; rr+=r/5) {
    sglCircle(offsetx-1,offsety,0,0.5*rr);
    sglCircle(offsetx+3,offsety,0,1.5*rr);
  }
  
  offsetx = r;

  sglColor3f(1,1,1);

  for(float rr=0.4; rr<=4; rr+=0.4) {
    sglArc(offsetx,offsety,  0,rr,0,M_PI/2);
    sglArc(offsetx+4,offsety,0,rr,M_PI*2/2,M_PI*3/2);
  }
}

 











/// NFF drawing test
void DrawTestScene2D(NFFStore &nffstore)
{
  sglAreaMode(SGL_FILL);
  sglEnable(SGL_DEPTH_TEST);
  // set viewport
  sglViewport(0, 0, WIDTH, HEIGHT);

  // projection transformation
  sglMatrixMode(SGL_PROJECTION);
  sglLoadIdentity();
  sgluPerspective (nffstore.angle, (float)WIDTH/HEIGHT, 1.0, 1800.0);
  
  // modelview transformation
  sglMatrixMode(SGL_MODELVIEW);
  sglLoadIdentity();
  sgluLookAt(
    nffstore.from.x,
    nffstore.from.y,
    nffstore.from.z,
    nffstore.at.x,
    nffstore.at.y,
    nffstore.at.z,
    nffstore.up.x,
    nffstore.up.y,
    nffstore.up.z
    );

  // render the geometry from the NFF file
  NFFStore::TMatGroupList::const_iterator giter = nffstore.matgroups.begin();
  for ( ; giter != nffstore.matgroups.end(); ++giter) {
                const NFFStore::Material &m = giter->material;
		sglColor3f(m.col.r*m.kd, m.col.g*m.kd, m.col.b*m.kd );
    const NFFStore::TriangleList &tlist = giter->geometry;
    NFFStore::TriangleList::const_iterator titer = tlist.begin();
    for ( ; titer != tlist.end(); ++titer ) {
      const NFFStore::Triangle &t = *titer;
      sglBegin(SGL_POLYGON);
      for(int i=0; i<3; i++)
        sglVertex3f(t.vertices[i].x,t.vertices[i].y,t.vertices[i].z);
      sglEnd();
    }
  }
}




/// NFF drawing test
float
RayTraceScene(const char *scenename)
{
  NFFStore nffstore(false);
  Timer timer;

  FILE *f = fopen(scenename,"rt");
  if(!f) {
    cerr << "RTS: Could not open " << scenename << " for reading." << std::endl;
    return 0;
  }
  
  char errstring[4000];
  if( ReadNFF(f,errstring,&nffstore) < 0 ) {
    cerr << "Error in NFF file " << scenename << ":\n" << errstring << std::endl;
    return 0;
  }

  cout << "NFF file " << scenename << " successfully parsed." << endl;

  timer.Restart();

  // projection transformation
  sglMatrixMode(SGL_PROJECTION);
  sglLoadIdentity();
  // modelview transformation
  sglMatrixMode(SGL_MODELVIEW);
  sglLoadIdentity();
  
  /// BEGIN SCENE DEFINITION
  sglBeginScene();
  // iterate over all the geometry from the NFF file
  NFFStore::TMatGroupList::const_iterator giter = nffstore.matgroups.begin();
  for ( ; giter != nffstore.matgroups.end(); ++giter) {
	const NFFStore::Material &m = giter->material;
	sglMaterial(m.col.r,
				m.col.g,
				m.col.b,
				m.kd,
				m.ks,
				m.shine,
				m.T,
				m.ior);
	
	/// store all polygons (converted into triangles)
	const NFFStore::TriangleList &tlist = giter->geometry;
	NFFStore::TriangleList::const_iterator titer = tlist.begin();
	for ( ; titer != tlist.end(); ++titer ) {
	  const NFFStore::Triangle &t = *titer;
	  sglBegin(SGL_POLYGON);
	  for(int i=0; i<3; i++)
		sglVertex3f(t.vertices[i].x,t.vertices[i].y,t.vertices[i].z);
	  sglEnd();
	}
	/// store spheres
	const NFFStore::SphereList &slist = giter->spheres;
	NFFStore::SphereList::const_iterator siter = slist.begin();
	for ( ; siter != slist.end(); ++siter ) {
	  const NFFStore::Sphere &s = *siter;
	  sglSphere(s.center.x,
				s.center.y,
				s.center.z,
				s.radius);
	}

  }
  
  // iterate over all point lights from the NFF file
  std::list<NFFStore::PointLight>::const_iterator liter = nffstore.pointLights.begin();
  for ( ; liter != nffstore.pointLights.end(); ++liter ) {
	const NFFStore::PointLight &l = *liter;
	sglPointLight(l.position.x,
				  l.position.y,
				  l.position.z,
				  l.intensity.r,
				  l.intensity.g,
				  l.intensity.b);
  }


  // iterate over all the geometry from the NFF file
  NFFStore::TLightGroupList::const_iterator aliter = nffstore.lightgroups.begin();
  for ( ; aliter != nffstore.lightgroups.end(); ++aliter) {
	sglEmissiveMaterial(aliter->intensity.r,
						aliter->intensity.g,
						aliter->intensity.b,
						aliter->atten.x,
						aliter->atten.y,
						aliter->atten.z);
						
	
	/// store all polygons (converted into triangles)
	const NFFStore::TriangleList &tlist = aliter->geometry;
	NFFStore::TriangleList::const_iterator titer = tlist.begin();
	for ( ; titer != tlist.end(); ++titer ) {
	  const NFFStore::Triangle &t = *titer;
	  sglBegin(SGL_POLYGON);
	  for(int i=0; i<3; i++)
		sglVertex3f(t.vertices[i].x,t.vertices[i].y,t.vertices[i].z);
	  sglEnd();
	}
  }

  if (nffstore.envMap.cols) {
	sglEnvironmentMap(nffstore.envMap.width,
					  nffstore.envMap.height,
					  nffstore.envMap.cols);
  }
  
  sglEndScene();
  /// END SCENE DEFINITION


  sglAreaMode(SGL_FILL);
  sglEnable(SGL_DEPTH_TEST);
  sglClearColor(nffstore.bg_col.r, nffstore.bg_col.g, nffstore.bg_col.b, 1);
  sglClear(SGL_COLOR_BUFFER_BIT|SGL_DEPTH_BUFFER_BIT);

  // set the viewport transform
  sglViewport(0, 0, WIDTH, HEIGHT);
  
  // setup the camera using appopriate projection transformation
  // note that the resolution stored in the nff file is ignored
  sglMatrixMode(SGL_PROJECTION);
  sglLoadIdentity();
  sgluPerspective (nffstore.angle, (float)WIDTH/HEIGHT, 1.0, 1800.0);
  
  // modelview transformation
  sglMatrixMode(SGL_MODELVIEW);
  sglLoadIdentity();
  sgluLookAt(
			 nffstore.from.x,
			 nffstore.from.y,
			 nffstore.from.z,
			 nffstore.at.x,
			 nffstore.at.y,
			 nffstore.at.z,
			 nffstore.up.x,
			 nffstore.up.y,
			 nffstore.up.z
			 );


  // compute a ray traced image and store it in the color buffer
  sglBuildKdTree();
  sglRayTraceScene();
  //sglRasterizeScene();
  return timer.UserTime();
}


float
RayTraceAssimpScene(const char *scenename)
{	
	Timer timer;
	Assimp::Importer importer;
	string scenename_obj = string(scenename) + ".obj";
	const aiScene *scene = importer.ReadFile(scenename_obj, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_PreTransformVertices );

	if (scene == NULL) {
		cerr << "Assimp: Could not open " << scenename_obj << " for assimp reading." << std::endl;
		return 0.0;
	}


	cout << "OBJ file " << scenename << " successfully parsed." << endl;


	// projection transformation
	sglMatrixMode(SGL_PROJECTION);
	sglLoadIdentity();
	// modelview transformation
	sglMatrixMode(SGL_MODELVIEW);
	sglLoadIdentity();

	/// BEGIN SCENE DEFINITION
	sglBeginScene();
	
	for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
		// MTL
		aiMesh *mesh = scene->mMeshes[i];
		aiMaterial *mat = scene->mMaterials[mesh->mMaterialIndex];
		aiColor3D color;
		mat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
		aiColor3D specular;
		mat->Get(AI_MATKEY_COLOR_SPECULAR, specular);
		float shininess;
		mat->Get(AI_MATKEY_SHININESS, shininess);
		sglMaterial(color.r, color.g, color.b, 1.0f, (specular.r + specular.g + specular.b) / 3.0f, shininess, 0.0f, 0.0f);

		for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
			aiFace *face = &mesh->mFaces[j];

			sglBegin(SGL_POLYGON);

			for (unsigned int k = 0; k < face->mNumIndices; k++) {
				aiVector3D * vertex = &mesh->mVertices[face->mIndices[k]];
				sglVertex3f(vertex->x, vertex->y, vertex->z);
			}

			sglEnd();
		}
	}

	// VIEW
	// default values � overwritten if present in file
	float from_x = 0.0f;
	float from_y = 0.0f;
	float from_z = -5.0f;

	float at_x = 0.0f;
	float at_y = 0.0f;
	float at_z = 1.0f;

	float up_x = 0.0f;
	float up_y = 1.0f;
	float up_z = 0.0f;

	float angle(80.0);
	float close(1.0f);
	float far(500.0f);

	string scenename_view = string(scenename) + ".view";
	std::ifstream file(scenename_view);
	if (file.fail()) {
		std::cerr << "Error: Could not open \"" << scenename_view << "\" for reading." << std::endl;
	}
	else {
		std::string input;
		while (!file.eof()) {
			file >> input;

			if (input.compare("from") == 0) {
				file >> input;
				from_x = stof(input);
				file >> input;
				from_y = stof(input);
				file >> input;
				from_z = stof(input);
			}
			else if (input.compare("at") == 0) {
				file >> input;
				at_x = stof(input);
				file >> input;
				at_y = stof(input);
				file >> input;
				at_z = stof(input);
			}
			else if (input.compare("up") == 0) {
				file >> input;
				up_x = stof(input);
				file >> input;
				up_y = stof(input);
				file >> input;
				up_z = stof(input);
			}
			else if (input.compare("angle") == 0) {
				file >> input;
				angle = stof(input);
			}
			else if (input.compare("hither") == 0) {
				file >> input;
				close = stof(input);
			}
			else if (input.compare("yon") == 0) {
				file >> input;
				far = stof(input);
			}
		}

		file.close();
	}

	sglPointLight(from_x, from_y + 2.0f, from_z - 1.0f, 1.0f, 1.0f, 1.0f);
	sglEndScene();
	/// END SCENE DEFINITION


	sglAreaMode(SGL_FILL);
	sglEnable(SGL_DEPTH_TEST);
	sglClearColor(0.0f, 0.0f, 0.0f, 1);
	sglClear(SGL_COLOR_BUFFER_BIT | SGL_DEPTH_BUFFER_BIT);

	// set the viewport transform
	sglViewport(0, 0, WIDTH, HEIGHT);

	// setup the camera using appopriate projection transformation
	// note that the resolution stored in the nff file is ignored
	sglMatrixMode(SGL_PROJECTION);
	sgluPerspective(angle, (float)WIDTH / HEIGHT, close, far);

	// modelview transformation
	sglMatrixMode(SGL_MODELVIEW);
	sgluLookAt(
		from_x, from_y, from_z, at_x, at_y, at_z, up_x, up_y, up_z );

	timer.Restart();
	sglBuildKdTree();
	cout << "KD-tree built in: " << timer.UserTime() << endl;
	timer.Restart();
	// compute a ray traced image and store it in the color buffer
	sglRayTraceScene();
	cout << "Rendering done in: " << timer.UserTime() << endl;
	//sglRasterizeScene();
	return timer.UserTime();
}


/// Init SGL
void Init(void) 
{
  sglInit();
  for(int i=0; i<10; i++) _contexts[i]=-1;
  for(int i=0; i<NUM_CONTEXTS; i++)
    _contexts[i] = sglCreateContext(WIDTH, HEIGHT);
  sglSetContext(_contexts[0]);
}


/// Clean up SGL
void CleanUp(void) 
{
  /// destroys all created contexts
  sglFinish();
}

#if USE_GUI
////////// GLUT bindings //////////////

/// redraw the main window - copy pixels from the current SGL context
void
myDisplay(void) 
{ 
  float *cb = sglGetColorBufferPointer();
	
  if(cb)
    glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_FLOAT, cb);
	
  // swap buffers (float buffering)
  glutSwapBuffers();
}


/// called upon window size change
void myReshape(int width, int height) 
{
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(-1.0F, 1.0F, -1.0F, 1.0F);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

/// Callback for processing special key input
void mySpecial (int key, int x, int y)
{
  /// rotate and translate cubes in test 2C
  switch(key) {
    case GLUT_KEY_LEFT: 
      tx -= tstep; break;
    case GLUT_KEY_UP:
      ty += tstep; break;
    case GLUT_KEY_RIGHT:
      tx += tstep; break;
    case GLUT_KEY_DOWN:
      ty -= tstep; break;
    case GLUT_KEY_PAGE_UP:
      tz += tstep; break;
    case GLUT_KEY_PAGE_DOWN:
      tz -= tstep; break;
    case 'r':
      rot += rotstep; break;
  }

  sglSetContext(_contexts[5]);
  sglClearColor(0, 0, 0, 1);
  //cout << "now" << endl;
  sglClear(SGL_COLOR_BUFFER_BIT|SGL_DEPTH_BUFFER_BIT);
  //DrawTestScene2C();

  glutPostRedisplay();
}


/// Callback for processing keyboard input
void myKeyboard (unsigned char key, int x, int y)
{
  switch (key) {
    // application finishes upon pressing q
    case 'r':
      mySpecial(key,x,y); break;
    case 'q':
    case 'Q':
    case 27:
      CleanUp();
      exit (0);
      break;
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9': 
      {
        sglSetContext(_contexts[key-'0']);
        sglEErrorCode err = sglGetError();
        if(err != SGL_NO_ERROR)
          cerr << "Failed to switch context: " << sglGetErrorString(err) << endl;
      }
      glutPostRedisplay();
      break;
  }
}

#endif

int main(int argc, char **argv) 
{

#if USE_GUI
	// Initialize GLUT
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowPosition(5, 50);
  glutInitWindowSize(WIDTH, HEIGHT);

  // Create main window and set callbacks
  glutCreateWindow(TITLE);
  glutDisplayFunc(myDisplay);
  glutReshapeFunc(myReshape);
  glutKeyboardFunc(myKeyboard);
  glutSpecialFunc(mySpecial);
#endif
  
  // init SGL
  Init();
  
  // run and benchmark the tests
  Timer timer;
  float time;
  
  double totalTime = 0;

  ofstream resultsInfo("results/desc.neon");
  resultsInfo<<"res:"<<endl;

  
#ifdef TEST_0A
  cout << "test 0a..." << flush;
  timer.Restart();
  sglSetContext(_contexts[0]);
  sglClearColor(0, 0, 0, 1);
  sglClear(SGL_COLOR_BUFFER_BIT);
  DrawTestScene0A();
  cout << "done in " << timer.UserTime() << " sec." << endl;
  totalTime += timer.UserTime();
#endif
 /*
#ifdef TEST_0B
  cout << "test 0b..." << flush;
  timer.Restart();
  sglSetContext(_contexts[1]);
  sglClearColor(0, 0, 0, 1);
  sglClear(SGL_COLOR_BUFFER_BIT);
  DrawTestScene0B();
  cout << "done in " << timer.UserTime() << " sec." << endl;
  totalTime += timer.UserTime();
#endif
*/
#ifdef TEST_1A
  cout << "test 1a..." << flush;
  timer.Restart();

  sglSetContext(_contexts[0]);
  sglClearColor(0, 0, 0, 1);
  sglClear(SGL_COLOR_BUFFER_BIT);
  //for(int i=0; i<1000; i++) {
	DrawTestScene1A();
  //}
  WriteTGA("results/test1a.tga");
  resultsInfo<<"    test1a.png : "<<timer.UserTime()<<endl;
  cout << "done in " << timer.UserTime() << " sec." << endl;
  totalTime += timer.UserTime();
#endif
  
#ifdef TEST_1B
  cout << "test 1b..." << flush;
  timer.Restart();

  sglSetContext(_contexts[1]);
  sglClearColor(0, 0, 0, 1);
  sglClear(SGL_COLOR_BUFFER_BIT);
  //for(int i=0; i<1000; i++)
    DrawTestScene1B();
  WriteTGA("results/test1b.tga");
  resultsInfo<<"    test1b.png : "<<timer.UserTime()<<endl;
  cout << "done in " << timer.UserTime() << " sec." << endl;
  totalTime += timer.UserTime();
#endif
  
  
#ifdef TEST_1C
  cout << "test 1c..." << flush;
  timer.Restart();

  sglSetContext(_contexts[2]);
  sglClearColor(0, 0, 0, 1);
  sglClear(SGL_COLOR_BUFFER_BIT);
  //for(int i=0; i<1000; i++)
    DrawTestScene1C();
  WriteTGA("results/test1c.tga");
  resultsInfo<<"    test1c.png : "<<timer.UserTime()<<endl;

  cout << "done in " << timer.UserTime() << " sec." << endl;
  totalTime += timer.UserTime();
#endif
  

  //  cout << "Total time for test 1: " << test1time << " sec. " << std::endl;
  //  resultsInfo<<"TotalTime : "<<timer.UserTime()<<endl;

#ifdef TEST_2A
  cout << "test 2a..." << flush;
  timer.Restart();

  sglSetContext(_contexts[3]);
  sglClearColor(0, 0, 0, 1);
  sglClear(SGL_COLOR_BUFFER_BIT);
  //for(int i=0; i<150; i++)
    DrawTestScene2A();
  resultsInfo<<"    test2a.png : "<<timer.UserTime()<<endl;
  WriteTGA("results/test2a.tga");

  cout << "done in " << timer.UserTime() << " sec." << endl;
  totalTime += timer.UserTime();
#endif
  
#ifdef TEST_2B
  cout << "test 2b..." << flush;
  timer.Restart();

  sglSetContext(_contexts[4]);
  sglClearColor(0, 0, 0, 1);
  sglClear(SGL_COLOR_BUFFER_BIT);
  //for(int i=0; i<15; i++)
    DrawTestScene2B();
  resultsInfo<<"    test2b.png : "<<timer.UserTime()<<endl;
  WriteTGA("results/test2b.tga");

  cout << "done in " << timer.UserTime() << " sec." << endl;
  totalTime += timer.UserTime();
#endif
  
#ifdef TEST_2C
  cout << "test 2c..." << flush;
  timer.Restart();

  sglSetContext(_contexts[5]);
  sglClearColor(0, 0, 0, 1);
  //cout << "now" << endl;
  sglClear(SGL_COLOR_BUFFER_BIT|SGL_DEPTH_BUFFER_BIT);
  //for(int i=0; i<15; i++)
    DrawTestScene2C();
  resultsInfo<<"    test2c.png : "<<timer.UserTime()<<endl;
  WriteTGA("results/test2c.tga");

  cout << "done in " << timer.UserTime() << " sec." << endl;
  totalTime += timer.UserTime();
#endif
  
  
#ifdef TEST_2D
 {
   NFFStore nffstore;
   
   /// read in the NFF file
   const char *scenename = "cornell-spheres.nff";
   
   FILE *f = fopen(scenename,"rt");
  if(!f) {
    cerr << "2D: Could not open " << scenename << " for reading." << std::endl;
    return 2;
  }
  
  char errstring[4000];
  if( ReadNFF(f,errstring,&nffstore) < 0 ) {
    cerr << "Error in NFF file " << scenename << ":\n" << errstring << std::endl;
    return 2;
  }

  cerr << "NFF file " << scenename << " successfully parsed." << endl;

  cout << "test 2d..." << flush;
  timer.Restart();

  sglSetContext(_contexts[0]);
  sglClearColor(0, 0, 0, 1);
  sglClear(SGL_COLOR_BUFFER_BIT|SGL_DEPTH_BUFFER_BIT);
  //for(int i=0; i<1; i++)
    DrawTestScene2D(nffstore);
  
  resultsInfo<<"    test2d.png : "<<timer.UserTime()<<endl;
  WriteTGA("results/test2d.tga");


  totalTime += timer.UserTime();
 }

  cout << "Total time for test 2: " << totalTime << " sec. " << std::endl;
#endif
  
  
  // general ray tracer code invoken when a parameter is used to execute
  // the file
 /* if (argc > 1) {
	sglSetContext(_contexts[0]);
	float time = RayTraceScene(argv[1]);
	cout<<"Time:"<<time<<endl;
	char savename[256];
	sprintf(savename, "%s.tga", argv[1]);
	WriteTGA(savename);

	// ignore the rest of the code!
#if USE_GUI
	// execute main application loop for event processing
	glutMainLoop();
#else
	CleanUp();
	return 0;
#endif
  }*/
  
#ifdef TEST_3A
 {
   /// read in the NFF file
   const char *sceneFile = "sphere.nff";
   sglSetContext(_contexts[1]);
   float time = RayTraceScene(sceneFile);
   totalTime += time;

   resultsInfo<<"    test3a.png : "<<time<<endl;
   cout<<"    test3a.png : "<<time<<endl;
   WriteTGA("results/test3a.tga");
 }
#endif

#ifdef TEST_3B
 {
   /// read in the NFF file
   const char *sceneFile = "cornell.nff";
   sglSetContext(_contexts[2]);
   float time = RayTraceScene(sceneFile);
   totalTime += time;
   
   resultsInfo<<"    test3b.png : "<<time<<endl;
   cout<<"    test3b.png : "<<time<<endl;
   WriteTGA("results/test3b.tga");
 }
#endif

#ifdef TEST_3C
 {
   /// read in the NFF file
   const char *sceneFile = "floor_sph.nff";
   sglSetContext(_contexts[3]);
   float time = RayTraceScene(sceneFile);
   totalTime += time;
   
   resultsInfo<<"    test3c.png : "<<time<<endl;
   cout<<"    test3c.png : "<<time<<endl;
   WriteTGA("results/test3c.tga");
 }
#endif

#ifdef TEST_4A
 {
	 /// read in the NFF file
	 /*const char *sceneFile = "cornell-blocks.nff";
	 sglSetContext(_contexts[3]);
	 float time = RayTraceScene(sceneFile);
	 //cout << "Starting with " << sceneFile << endl;
	 //float time = RayTraceAssimpScene(sceneFile);
	 totalTime += time;

	 resultsInfo << "    test4a.png : " << time << endl;
	 cout << "    test4a.png : " << time << endl;
	 WriteTGA("results/test4a.tga");*/
	 
	
   /// read in the NFF file
   
   //const char *sceneFile = "Data/A10";
   //const char *sceneFile = "Data/Armadillo";
   //const char *sceneFile = "Data/blob";
   //const char *sceneFile = "Data/asianDragon";			// p�es 2GB RAM
   //const char *sceneFile = "Data/Park";
   //const char *sceneFile = "Data/City";
   //const char *sceneFile = "Data/City2";					// tenhle model se tv��� divn� i v Blenderu
   //const char *sceneFile = "Data/teapots";
   //const char *sceneFile = "Data/sibenik";				// pozor na um�st�n� sv�tla
   //const char *sceneFile = "Data/fforest";
   //const char *sceneFile = "Data/conference";
   const char *sceneFile = "Data/plysak_normalized";		// naprosto �patn� materi�l

   if (argc > 1) {
	   //cout << argv[1] << endl;
	   sceneFile = argv[1];
   }
   string filename = "results/";
   filename.append(sceneFile);
   filename.append(".tga");

   sglSetContext(_contexts[3]);
   //float time = RayTraceScene(sceneFile);
   cout << "Starting with " << sceneFile << endl;
   float time = RayTraceAssimpScene(sceneFile);
   totalTime += time;
   
   resultsInfo<<"    test4a.png : "<<time<<endl;
   //cout<<"    test4a.png : "<<time<<endl;
   //WriteTGA( filename.c_str() );
 }
#endif

#ifdef TEST_4B
 {
   /// read in the NFF file
   const char *sceneFile = "cornell-spheres-raytrace.nff";
   sglSetContext(_contexts[4]);
   float time = RayTraceScene(sceneFile);
   totalTime += time;
   
   resultsInfo<<"    test4b.png : "<<time<<endl;
   cout<<"    test4b.png : "<<time<<endl;
   WriteTGA("results/test4b.tga");
 }
#endif

#ifdef TEST_4C
 {
   /// read in the NFF file
   const char *sceneFile = "cornell-spheres-crazy.nff";
   sglSetContext(_contexts[5]);
   float time = RayTraceScene(sceneFile);
   totalTime += time;
   
   resultsInfo<<"    test4c.png : "<<time<<endl;
   cout<<"    test4c.png : "<<time<<endl;
   WriteTGA("results/test4c.tga");
 }
#endif

#ifdef TEST_5A
 {
   /// read in the NFF file
   const char *sceneFile = "cornell-blocks-arealight.nff";
   sglSetContext(_contexts[0]);
   float time = RayTraceScene(sceneFile);
   totalTime += time;
   
   resultsInfo<<"    test5a.png : "<<time<<endl;
   cout<<"    test5a.png : "<<time<<endl;
   WriteTGA("results/test5a.tga");
 }
#endif

#ifdef TEST_5B
 {
   /// read in the NFF file
   const char *sceneFile = "cornell-blocks-arealight-color.nff";
   sglSetContext(_contexts[1]);
   float time = RayTraceScene(sceneFile);
   totalTime += time;
   
   resultsInfo<<"    test5b.png : "<<time<<endl;
   cout<<"    test5b.png : "<<time<<endl;
   WriteTGA("results/test5b.tga");
 }
#endif

#ifdef TEST_5C
 {
   /// read in the NFF file
   const char *sceneFile = "basilica.nff";
   sglSetContext(_contexts[2]);
   float time = RayTraceScene(sceneFile);
   totalTime += time;
   
   resultsInfo<<"    test5c.png : "<<time<<endl;
   cout<<"    test5c.png : "<<time<<endl;
   WriteTGA("results/test5c.tga");
 }
#endif

#ifdef TEST_5D
 {
   /// read in the NFF file
   const char *sceneFile = "uffizi.nff";
   sglSetContext(_contexts[3]);
   float time = RayTraceScene(sceneFile);
   totalTime += time;
   
   resultsInfo<<"    test5d.png : "<<time<<endl;
   cout<<"    test5d.png : "<<time<<endl;
   WriteTGA("results/test5d.tga");
 }
#endif

  
#if USE_GUI
  // execute main application loop for event processing
  glutMainLoop();
#else
  resultsInfo<<"TotalTime : "<<totalTime<<endl;
  cout<<"TotalTime : "<<totalTime<<endl;
  CleanUp();
#endif
  
  return 0;
}
