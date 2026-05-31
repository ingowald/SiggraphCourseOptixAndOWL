// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// include deviceCode.h; this automaticlly pulls in owl.h and owl-common
#include "deviceCode.h"
// a std::vector to store the pixels in when we copy them to the host
#include <vector>
// owlviewer base class (to derive our viewer from). this must
// be included _after_ owl/owl::common are already included.
#include "owlViewer/OWLViewer.h"
#include "stb/stb_image_write.h"
#include <random>

// the "embedded" precompiled PTX-code from our deviceCode.cu. 
extern "C" char deviceCode_ptx[];

struct SampleViewer : public owl::viewer::OWLViewer
{
  SampleViewer();
  ~SampleViewer();
  
  /*! create/initialize all context- and pipeline related stuff: owl
      context, module(s), geometry types, etc */
  void buildPipeline();

  /*! build all the acutal scene geometry that the respective
      app/sample requires, including all geometries, groups, BLAS and
      TLAS, etc */
  void buildSceneGeometry();

  /*! the actual render frame callback - just fill the
      OWLViewer::fbPointer[] with OWLViewer::fbSize rgba8 pixel
      values; the viewer base class will do the rest */
  void render() override;

  /*! this gets called when the user presses a key on the keyboard ... */
  void key(char key, const vec2i &/*where*/) override;

  /*! window notifies us that we got resized */     
  void resize(const vec2i &newSize) override;
  
  /*! this function gets called whenever any camera manipulator
    updates the camera. gets called AFTER all values have been updated */
  void cameraChanged() override;
  
  /*! @{ all constext/pipeline related stuff; will be set in
      initPipeline() */
  OWLContext      owl = 0;
  OWLModule       mod;
  OWLRayGen       rg = 0;
  OWLLaunchParams lp = 0;
  /* the OWL geometry type for the geometries we're using (right now
     only a single one */
  OWLGeomType     spheresGT = 0;
  /*! @} */

  /*! the top-level accel struct (our samples will only have one);
      will be set in buildSceneGeometry */
  OWLGroup        tlas = 0;
  OWLBuffer       accumBuffer = 0;

  /*! track some linear sequence number for each frame, so each frame
      can use a new set of random numbers. this also serves as
      acounting how many frames have been accumulated already, so will
      get reset to 0 every time the camera or other render settings
      change. */
  int accumID = 0;
};

SampleViewer::SampleViewer()
{
  buildPipeline();
  buildSceneGeometry();
}

void SampleViewer::resize(const vec2i &newSize)
{
  OWLViewer::resize(newSize);
    
  if (!accumBuffer)
    accumBuffer = owlDeviceBufferCreate
      (owl,OWL_FLOAT3,newSize.x*newSize.y,nullptr);
  else 
    owlBufferResize(accumBuffer,newSize.x*newSize.y);
}

/*! this function gets called whenever any camera manipulator
  updates the camera. gets called AFTER all values have been updated */
void SampleViewer::cameraChanged()
{
  accumID = 0;
}

/*! this gets called when the user presses a key on the keyboard ... */
void SampleViewer::key(char key, const vec2i &where) 
{
  switch (key) {
  case ' ': {
    vec3f from = vec3f(13, 2, 3);
    vec3f at = vec3f(0, 0, 0);
    vec3f up = vec3f(0.f,1.f,0.f);
    float fovy = 20.f;
    setCameraOrientation(from,at,up,fovy);
  } break;
  default:
    owl::viewer::OWLViewer::key(key,where);
  }
}

void SampleViewer::buildPipeline()
{
  // ==================================================================
  std::cout << OWL_TERMINAL_LIGHT_BLUE
            << "\nBuilding basic RTX pipeline ingredients:\n"
            << OWL_TERMINAL_DEFAULT;
  // ------------------------------------------------------------------
  std::cout << "- creating OWL context...\n";
  owl = owlContextCreate(nullptr,1);
  // ##################################################################
  // for this sample we go back to single ray type; the Pete Shirley
  // RTOW scene doesn't have lights...
  // ##################################################################
  owlContextSetRayTypeCount(owl,1);
  assert(owl && "could not create context");

  // ------------------------------------------------------------------
  std::cout << "- creating module from precompiled device code...\n";
  mod = owlModuleCreate(owl,deviceCode_ptx);
  assert(mod && "could not create module");
  
  // ------------------------------------------------------------------
  std::cout << "- creating raygen program (data now moved to globals)...\n";
  OWLVarDecl rgVars[] = {
    // empty for now; we moved all data to optixGlobals
    { nullptr },
  };
  rg  = owlRayGenCreate
    (owl,mod,"renderFrame",sizeof(RayGenData),rgVars,-1);
  assert(rg && "could not create raygen");

  // ------------------------------------------------------------------
  std::cout << "- creating LaunchParams object to set optixGlobals...\n";
  OWLVarDecl lpVars[] = {
    // ------------------------------------------------------------------
    { "fb.size",
      OWL_INT2,
      OWL_OFFSETOF(OptixGlobals,frameBuffer.size) },
    // ------------------------------------------------------------------
    { "fb.color",
      OWL_RAW_POINTER,
      OWL_OFFSETOF(OptixGlobals,frameBuffer.color) },
    // ------------------------------------------------------------------
    { "fb.accum",
      OWL_BUFPTR,
      OWL_OFFSETOF(OptixGlobals,frameBuffer.accum) },
    // ------------------------------------------------------------------
    { "camera.position",
      OWL_FLOAT3,
      OWL_OFFSETOF(OptixGlobals,camera.position) },
    { "camera.dir_00",
      OWL_FLOAT3,
      OWL_OFFSETOF(OptixGlobals,camera.dir_00) },
    { "camera.dir_du",
      OWL_FLOAT3,
      OWL_OFFSETOF(OptixGlobals,camera.dir_du) },
    { "camera.dir_dv",
      OWL_FLOAT3,
      OWL_OFFSETOF(OptixGlobals,camera.dir_dv) },
    // ------------------------------------------------------------------
    { "world",
      OWL_GROUP,
      OWL_OFFSETOF(OptixGlobals,world) },
    // ------------------------------------------------------------------
    // delta: add some frame ID
    { "accumID",
      OWL_INT,
      OWL_OFFSETOF(OptixGlobals,frameBuffer.accumID) },
    // ------------------------------------------------------------------
    { nullptr },
  };

  lp  = owlParamsCreate(owl,sizeof(OptixGlobals),lpVars,-1);
  assert(lp && "could not create launch params");

  // ------------------------------------------------------------------
  OWLVarDecl gtVars[] = {
    { "spheres",
      OWL_BUFPTR,
      OWL_OFFSETOF(SpheresGeomData,spheres) },
    { "materialsData",
      OWL_BUFPTR,
      OWL_OFFSETOF(SpheresGeomData,materialsData) },
    { "materialType",
      OWL_INT,
      OWL_OFFSETOF(SpheresGeomData,materialType) },
    { nullptr },
  };
  spheresGT = owlGeomTypeCreate(owl,OWL_GEOM_USER,
                         sizeof(SpheresGeomData),
                         gtVars,-1);
  assert(spheresGT && "could not create geometry type");
                                     
  // ------------------------------------------------------------------
  std::cout << "- setting CH program on geom type...\n";
  // ##################################################################
  // Delta from last sample: set CH and AH program(s) for different
  // ray types
  // ##################################################################
  
  // in this sample we only trace one type of ray, and need only CH
  // program
  owlGeomTypeSetClosestHit(spheresGT,
                           /* ray type */0,
                           /* module where to look for code*/mod,
                           /* entry point */"Spheres_CH");

  // Delta from prev sample: set intersection and bounds program(s) */
  owlGeomTypeSetBoundsProg(spheresGT,
                           /* module where to look for code*/mod,
                           /* entry point */"Spheres_Bounds");
  // Delta from prev sample: set intersection and bounds program(s) */
  owlGeomTypeSetIntersectProg(spheresGT,
                              /* ray type */0,
                              /* module where to look for code*/mod,
                              /* entry point */"Spheres_Intersect");
  
  /* ****************************************************************** */
  // important: this is where we build the actual programs and
  // pipeline; after all the types and programs have been defined, but
  // before we build actual geometries or accels with them
  /* ****************************************************************** */
  // ------------------------------------------------------------------
  std::cout << "- building actual device programs...\n";
  owlBuildPrograms(owl);
  
  // ------------------------------------------------------------------
  std::cout << "- building RTX pipeline...\n";
  owlBuildPipeline(owl);
}

inline float rnd()
{
  static std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with rd()
  static std::uniform_real_distribution<float> dis(0.f, 1.f);
  return dis(gen);
}

inline vec3f rnd3f() { return vec3f(rnd(),rnd(),rnd()); }

void SampleViewer::buildSceneGeometry()
{
  // ==================================================================
  // set up some host-side containers to store the data we generate
  // ==================================================================
  std::vector<Metal>       metalMaterials;
  std::vector<Sphere>      metalSpheres;
  std::vector<Lambertian>  lambertianMaterials;
  std::vector<Sphere>      lambertianSpheres;
  std::vector<Dielectric>  dielectricMaterials;
  std::vector<Sphere>      dielectricSpheres;

  // ==================================================================
  // let's generate the acutal scene - this is largely stolen from
  // Pete Shirley's RTOW scene
  // ==================================================================
  lambertianSpheres.push_back(Sphere{vec3f(0.f, -1000.0f, -1.f), 1000.f});
  lambertianMaterials.push_back(Lambertian{vec3f(0.5f, 0.5f, 0.5f)});

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = rnd();
      vec3f center(a + rnd(), 0.2f, b + rnd());
      if (choose_mat < .6//0.8f
          ) {
        lambertianSpheres.push_back(Sphere{center, 0.2f});
        lambertianMaterials.push_back(Lambertian{rnd3f()*rnd3f()});
      } else if (choose_mat < .85//0.95f
                 ) {
        metalSpheres.push_back(Sphere{center, 0.2f});
        metalMaterials.push_back(Metal{0.5f*(1.f+rnd3f()),0.5f*rnd()});
      } else {
        dielectricSpheres.push_back(Sphere{center, 0.2f});
        dielectricMaterials.push_back(Dielectric{1.5f});
      }
    }
  }
  dielectricSpheres.push_back(Sphere{vec3f(0.f, 1.f, 0.f), 1.f});
  dielectricMaterials.push_back(Dielectric{1.5f});
  lambertianSpheres.push_back(Sphere{vec3f(-4.f,1.f, 0.f), 1.f});
  lambertianMaterials.push_back(Lambertian{vec3f(0.4f, 0.2f, 0.1f)});
  metalSpheres.push_back(Sphere{vec3f(4.f, 1.f, 0.f), 1.f});
  metalMaterials.push_back(Metal{vec3f(0.7f, 0.6f, 0.5f), 0.0f});

  // ==================================================================
  // now declare all that to OWL
  // ==================================================================

  // ----------- metal -----------
  OWLBuffer metalMaterialsBuffer
    = owlDeviceBufferCreate(owl,OWL_USER_TYPE(Metal),
                            metalMaterials.size(),
                            metalMaterials.data());
  OWLBuffer metalSpheresBuffer
    = owlDeviceBufferCreate(owl,OWL_USER_TYPE(Sphere),
                            metalSpheres.size(),
                            metalSpheres.data());
  OWLGeom metalSpheresGeom
    = owlGeomCreate(owl,spheresGT);
  owlGeomSetPrimCount(metalSpheresGeom,(int)metalSpheres.size());
  owlGeomSet1i(metalSpheresGeom,"materialType",(int)METAL);
  owlGeomSetBuffer(metalSpheresGeom,"materialsData",metalMaterialsBuffer);
  owlGeomSetBuffer(metalSpheresGeom,"spheres",metalSpheresBuffer);
 
  // ----------- lambertian -----------
  OWLBuffer lambertianMaterialsBuffer
    = owlDeviceBufferCreate(owl,OWL_USER_TYPE(Lambertian),
                            lambertianMaterials.size(),
                            lambertianMaterials.data());
  OWLBuffer lambertianSpheresBuffer
    = owlDeviceBufferCreate(owl,OWL_USER_TYPE(Sphere),
                            lambertianSpheres.size(),
                            lambertianSpheres.data());
  OWLGeom lambertianSpheresGeom
    = owlGeomCreate(owl,spheresGT);
  owlGeomSetPrimCount(lambertianSpheresGeom,(int)lambertianSpheres.size());
  owlGeomSet1i(lambertianSpheresGeom,"materialType",(int)LAMBERTIAN);
  owlGeomSetBuffer(lambertianSpheresGeom,"materialsData",lambertianMaterialsBuffer);
  owlGeomSetBuffer(lambertianSpheresGeom,"spheres",lambertianSpheresBuffer);
 
  // ----------- dielectric -----------
  OWLBuffer dielectricMaterialsBuffer
    = owlDeviceBufferCreate(owl,OWL_USER_TYPE(Dielectric),
                            dielectricMaterials.size(),
                            dielectricMaterials.data());
  OWLBuffer dielectricSpheresBuffer
    = owlDeviceBufferCreate(owl,OWL_USER_TYPE(Sphere),
                            dielectricSpheres.size(),
                            dielectricSpheres.data());
  OWLGeom dielectricSpheresGeom
    = owlGeomCreate(owl,spheresGT);
  owlGeomSetPrimCount(dielectricSpheresGeom,(int)dielectricSpheres.size());
  owlGeomSet1i(dielectricSpheresGeom,"materialType",(int)DIELECTRIC);
  owlGeomSetBuffer(dielectricSpheresGeom,"materialsData",dielectricMaterialsBuffer);
  owlGeomSetBuffer(dielectricSpheresGeom,"spheres",dielectricSpheresBuffer);
 
  std::vector<OWLGeom> geoms = {
    lambertianSpheresGeom,
    dielectricSpheresGeom,
    metalSpheresGeom
  };
  // ------------------------------------------------------------------
  OWLGroup blas = owlUserGeomGroupCreate
    (owl,geoms.size(),geoms.data());
  assert(blas && "could not create bottom level accel struct group");
  // ------------------------------------------------------------------
  owlGroupBuildAccel(blas);
  
  tlas = owlInstanceGroupCreate(owl,1,&blas,
                                // no instance IDs
                                nullptr,
                                // no instance transforms
                                nullptr);
  assert(tlas && "could not create instance/top level accel struct group");
  
  // ------------------------------------------------------------------
  std::cout << "- build the instance group's accel (the tlas)...\n";
  owlGroupBuildAccel(tlas);

  // ------------------------------------------------------------------
  std::cout << "- building SBT (all program data set, all groups built)...\n";
  owlBuildSBT(owl);
}

void SampleViewer::render()
{
  if (!accumBuffer)
    // we haven't been properly resized yet...
    return;
  int thisFrameID = accumID++;

  owlParamsSet2i(lp,"fb.size",fbSize.x,fbSize.y);
  owlParamsSetPointer(lp,"fb.color",fbPointer);
  owlParamsSetBuffer(lp,"fb.accum",accumBuffer);
  owlParamsSet1i(lp,"accumID",thisFrameID);
  
  owlParamsSetGroup(lp,"world",tlas);

  auto camera = this->getSimplifiedCamera();
  owlParamsSet3f(lp,"camera.position",
                 camera.lens.center.x,
                 camera.lens.center.y,
                 camera.lens.center.z);
  owlParamsSet3f(lp,"camera.dir_00",
                 camera.focalPlane.lower_left.x,
                 camera.focalPlane.lower_left.y,
                 camera.focalPlane.lower_left.z);
  owlParamsSet3f(lp,"camera.dir_du",
                 camera.focalPlane.horizontal.x,
                 camera.focalPlane.horizontal.y,
                 camera.focalPlane.horizontal.z);
  owlParamsSet3f(lp,"camera.dir_dv",
                 camera.focalPlane.vertical.x,
                 camera.focalPlane.vertical.y,
                 camera.focalPlane.vertical.z);
  // ------------------------------------------------------------------
  owlLaunch2D(rg,fbSize.x,fbSize.y,lp);
}


SampleViewer::~SampleViewer()
{
  // ------------------------------------------------------------------
  std::cout << "- releasing launchparams...\n";
  owlParamsRelease(lp);
  std::cout << "- releasing raygen...\n";
  owlRayGenRelease(rg);
  std::cout << "- releasing module...\n";
  owlModuleRelease(mod);
  std::cout << "- destroying OWL context...\n";
  owlContextDestroy(owl); 
}

int main(int ac, char **av)
{
  std::cout
    << OWL_TERMINAL_LIGHT_BLUE
    << "**********************************************************************\n"
    << CHAPTER_NAME << "\n"
    << "**********************************************************************\n"
    << OWL_TERMINAL_DEFAULT;
  
  std::cout << R"(
This sample introduces the concept of optix
'per-ray-data' (PRD) to communicate between different
pipeline programs, and how owl offers some convenient
helpers to make it easier to use this.
)";

  vec3f from = vec3f(0.f);
  vec3f at   = vec3f(0.f);
  vec3f up   = vec3f(0.f);
  float fovy = 60.f;
  std::string sceneFileName = "";
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg == "--camera") {
      from.x = std::stof(av[i+1]);
      from.y = std::stof(av[i+2]);
      from.z = std::stof(av[i+3]);
      at.x = std::stof(av[i+4]);
      at.y = std::stof(av[i+5]);
      at.z = std::stof(av[i+6]);
      up.x = std::stof(av[i+7]);
      up.y = std::stof(av[i+8]);
      up.z = std::stof(av[i+9]);
      i += 9;
    } else if (arg == "-fovy") {
      fovy = std::stof(av[i+1]);
      i += 1;
    } else if (arg[0] != '-') {
      sceneFileName = arg;
    } else if (arg == "-v0") {
      from = vec3f(-1388.46, 1937.5, 4309.81);
      at = vec3f(-3865.96, -1793.97, 6061.06);
      up = vec3f(0, 1, 0);
      fovy = 60.f;
    } else if (arg == "-v1") {
      from = vec3f(-678.018, 618.124, 3672.22);
      at   = vec3f(-1733.15, -3330.65, 1137.9);
      up   = vec3f(0, 1, 0);
      fovy = 60.f;
    } else
      throw std::runtime_error("un-recognized cmdline arg '"+arg+"'");
  }

  // ==================================================================
  std::cout << "- creating viewer (and letting it initialize itself)...\n";
  SampleViewer *sampleViewer = new SampleViewer;
  sampleViewer->setTitle(CHAPTER_NAME);
  // enable viewer's 'fly' and 'inspect' camera modes (last one
  // becomes default)
  sampleViewer->enableFlyMode();
  sampleViewer->enableInspectMode();

  from = vec3f(13, 2, 3);
  at = vec3f(0, 0, 0);
  up = vec3f(0.f,1.f,0.f);
  fovy = 20.f;
  
  sampleViewer->setCameraOrientation
    (from,at,up,fovy);
  sampleViewer->setWorldScale
    (1.f);
  
  std::cout << "- starting the viewer...\n";
  sampleViewer->showAndRun();
  // ==================================================================
  std::cout << OWL_TERMINAL_LIGHT_BLUE
            << "\nImage saved; sample done; cleaning up:\n"
            << OWL_TERMINAL_DEFAULT;
  delete sampleViewer;

  std::cout << OWL_TERMINAL_LIGHT_GREEN
            << "\nAll good! - Sample concluded successfully...\n\n"
            << OWL_TERMINAL_DEFAULT;
  return 0;
}
