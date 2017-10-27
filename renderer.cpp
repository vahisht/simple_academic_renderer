//---------------------------------------------------------------------------
// renderer.cpp
// Main file of this project
// Based on testing application of CTU's APG
// Date:  2017/06
// Author: Michal Kuèera, CTU Prague
//---------------------------------------------------------------------------


#if !defined(USE_GUI) && defined(_MSC_VER)
#define USE_GUI 1
#endif


#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>
#include <fstream>
#include <algorithm> ///// ADDED... max' is not a member of 'std' 

#if USE_GUI
#include <GL/glut.h>
#include <GL/gl.h>
#endif

#include "sgl.h"
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


#define WIDTH  1024
#define HEIGHT 1024
#define TITLE  "Simple Renderer"
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


/// NFF drawing test
float
RayTraceScene(const char *scenename)
{
  NFFStore nffstore(false);
  
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
  return 0.0f;
}


float
RayTraceAssimpScene(const char *scenename)
{	
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
		aiString name;
		mat->Get(AI_MATKEY_NAME, name);
		//cout << name.C_Str() << endl;
		//cout << ": " << color.r << " " << color.g << " " << color.b << endl;
		aiColor3D specular;
		mat->Get(AI_MATKEY_COLOR_SPECULAR, specular);
		//cout << ": " << specular.r << " " << specular.g << " " << specular.b << endl;
		aiColor3D transparent;
		mat->Get(AI_MATKEY_COLOR_TRANSPARENT, transparent);
		//cout << ": " << transparent.r << " " << transparent.g << " " << transparent.b << endl;
		float shininess;
		mat->Get(AI_MATKEY_SHININESS, shininess);
		float opacity;
		mat->Get(AI_MATKEY_OPACITY, opacity);
		//cout << "Op: " << opacity << endl;
		float ior;
		mat->Get(AI_MATKEY_REFRACTI, ior);
		int light_model;
		mat->Get(AI_MATKEY_SHADING_MODEL, light_model);
		//cout << light_model << endl;
		sglMaterial(color.r, color.g, color.b, 1.0f, (specular.r + specular.g + specular.b) / 3.0f, shininess, 1.0f - opacity, ior);


		for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
			aiFace *face = &mesh->mFaces[j];

			//mesh->mN

			sglBegin(SGL_POLYGON);

			for (unsigned int k = 0; k < face->mNumIndices; k++) {
				aiVector3D * vertex = &mesh->mVertices[face->mIndices[k]];
				if (&mesh->mNormals[face->mIndices[k]] != NULL) {
					aiVector3D * normal = &mesh->mNormals[face->mIndices[k]];
					sglVertex3f(vertex->x, vertex->y, vertex->z, normal->x, normal->y, normal->z);
				}
				else {
					sglVertex3f(vertex->x, vertex->y, vertex->z);
				}
			}

			sglEnd();
		}
	}

	// VIEW
	// default values – overwritten if present in file
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

	sglPointLight(from_x, from_y + 0.5f, from_z - 1.0f, 1.0f, 1.0f, 1.0f);
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

	sglBuildKdTree();
	cout << "KD-tree built" << endl;
	// compute a ray traced image and store it in the color buffer
	sglRayTraceScene();
	cout << "Rendering done" << endl;
	//sglRasterizeScene();
	return 0.0f;
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

/// Function splitting given file path into a vector of strings, where each string is a folder/file on the path to the file
/// The last element contains the filename
std::vector<std::string> splitpath(
	const std::string& str
	, const std::set<char> delimiters)
{
	std::vector<std::string> result;

	char const* pch = str.c_str();
	char const* start = pch;
	for (; *pch; ++pch)
	{
		if (delimiters.find(*pch) != delimiters.end())
		{
			if (start != pch)
			{
				std::string str(start, pch);
				result.push_back(str);
			}
			else
			{
				result.push_back("");
			}
			start = pch + 1;
		}
	}
	result.push_back(start);

	return result;
}

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
  
  ofstream resultsInfo("results/desc.neon");
  resultsInfo<<"res:"<<endl;

  


	 /// read in the NFF file
	 /*const char *sceneFile = "cornell-blocks.nff";
	 sglSetContext(_contexts[3]);
	 //cout << "Starting with " << sceneFile << endl;
	 
	 WriteTGA("results/test4a.tga");*/
	 
	
   /// read in the NFF file
   
   //const char *sceneFile = "Data/A10";
   //const char *sceneFile = "Data/Armadillo";
   //const char *sceneFile = "Data/blob";
   //const char *sceneFile = "Data/Park";
   //const char *sceneFile = "Data/City";
   //const char *sceneFile = "Data/City2";					// tenhle model se tváøí divnì i v Blenderu
   //const char *sceneFile = "Data/teapots";
   //const char *sceneFile = "Data/sibenik";				// pozor na umístìní svìtla
   //const char *sceneFile = "Data/fforest";
   //const char *sceneFile = "Data/conference";
   //const char *sceneFile = "Data/plysak_normalized";		
   //const char *sceneFile = "Data/cornellbox-empty-rg";
   const char *sceneFile = "Data/cornellbox-sphere";

   if (argc > 1) {
	   //cout << argv[1] << endl;
	   sceneFile = argv[1];
   }


   
   std::set<char> delims{ '/' };

   std::vector<std::string> path = splitpath(sceneFile, delims);
   string scene_name = path.back();


   string filename = "results/";
   filename.append(scene_name);
   filename.append(".tga");

   sglSetContext(_contexts[3]);
   //float time = RayTraceScene(sceneFile);
   cout << "Starting with " << sceneFile << endl;
   RayTraceAssimpScene(sceneFile);
   
   resultsInfo<<"    test4a.png "<<endl;
   cout<< filename.c_str() <<endl;
   WriteTGA( filename.c_str() );

  
#if USE_GUI
  // execute main application loop for event processing
  glutMainLoop();
#else
  CleanUp();
#endif
  
  return 0;
}
