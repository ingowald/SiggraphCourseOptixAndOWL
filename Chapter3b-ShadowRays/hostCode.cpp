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
// miniScene, just so we can load some better scene content
#include "miniScene/Scene.h"

// the "embedded" precompiled PTX-code from our deviceCode.cu. 
extern "C" char deviceCode_ptx[];


struct SampleViewer : public owl::viewer::OWLViewer
{
  SampleViewer(mini::Scene::SP scene);
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

  mini::Scene::SP scene;
  
  /*! @{ all constext/pipeline related stuff; will be set in
      initPipeline() */
  OWLContext      owl = 0;
  OWLModule       mod;
  OWLRayGen       rg = 0;
  OWLLaunchParams lp = 0;
  /* the OWL geometry type for the geometries we're using (right now
     only a single one */
  OWLGeomType     gt = 0;
  /*! @} */

  /*! the top-level accel struct (our samples will only have one);
      will be set in buildSceneGeometry */
  OWLGroup        tlas = 0;
};

SampleViewer::SampleViewer(mini::Scene::SP scene)
  : scene(scene)
{
  buildPipeline();
  buildSceneGeometry();
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
    { "fb.data",
      OWL_RAW_POINTER,
      OWL_OFFSETOF(OptixGlobals,frameBuffer.data) },
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
    { nullptr },
  };

  lp  = owlParamsCreate(owl,sizeof(OptixGlobals),lpVars,-1);
  assert(lp && "could not create launch params");

  // ------------------------------------------------------------------
  std::cout << "- creating triangles geom type (no data for now)...\n";
  OWLVarDecl gtVars[] = {
    { "vertices",
      OWL_BUFPTR,
      OWL_OFFSETOF(TrianglesGeomData,vertices) },
    { "indices",
      OWL_BUFPTR,
      OWL_OFFSETOF(TrianglesGeomData,indices) },
    { "baseColor",
      OWL_FLOAT3,
      OWL_OFFSETOF(TrianglesGeomData,baseColor) },
    { nullptr },
  };
  gt = owlGeomTypeCreate(owl,OWL_GEOM_TRIANGLES,
                                     sizeof(TrianglesGeomData),
                                     gtVars,-1);
  assert(gt && "could not create geometry type");
                                     
  // ------------------------------------------------------------------
  std::cout << "- setting CH program on geom type...\n";
  owlGeomTypeSetClosestHit(gt,
                           /* ray type */0,
                           /* module where to look for code*/mod,
                           /* entry point */"TrianglesCH");

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


void SampleViewer::buildSceneGeometry()
{
  /* ****************************************************************** */
  // DELTA: in this sample we no longer ignore instancing, but create
  // the proper top-level accel (TLAS) with instances.
  /* ****************************************************************** */
  
  // ------------------------------------------------------------------
  std::cout << "- first, generating a list of all the geometric objects\n"
            << "  in the input scene - we want to build a BLAS for each\n"
            << "  one, but only ONE for each\n";
  std::set<mini::Object::SP> uniqueObjectsInScene;
  for (auto inst : scene->instances)
    uniqueObjectsInScene.insert(inst->object);
  std::cout << "  -> found " << prettyNumber(uniqueObjectsInScene.size())
            << " unique input objects (w/ possibly multiple meshes each)\n";

  std::cout << "- second, build a blas for each scuch object (once!)\n";
  std::map<mini::Object::SP,OWLGroup> blasForObject;
  for (auto miniObject : uniqueObjectsInScene) {
    // note this is the exact same code as for the previous example,
    // except fore having removed all the prints:
    std::vector<OWLGeom> generatedGeoms;
    for (auto inputMesh : miniObject->meshes) {
      int geomID = (int)generatedGeoms.size();
      
      OWLGeom geom = owlGeomCreate(owl,gt);
      assert(geom && "could not create geometry from geom type");
      
      // ------------------------------------------------------------------
      auto &vertices = inputMesh->vertices;
      OWLBuffer verticesBuffer
        = owlDeviceBufferCreate(owl,OWL_FLOAT3,vertices.size(),vertices.data());
      assert(verticesBuffer && "could not create vertices buffer");
      owlTrianglesSetVertices(geom,
                              /*buffer*/verticesBuffer,
                              /*count*/vertices.size(),
                              /*stride*/sizeof(vec3f),
                              /*offset*/0);
      owlGeomSetBuffer(geom,"vertices",verticesBuffer);
      // release this buffer here, the geom now has a reference anyway
      owlBufferRelease(verticesBuffer);
      
      // ------------------------------------------------------------------
      auto &indices = inputMesh->indices;
      OWLBuffer indicesBuffer
        = owlDeviceBufferCreate(owl,OWL_INT3,indices.size(),indices.data());
      assert(indicesBuffer && "could not create indices buffer");
      owlTrianglesSetIndices(geom,
                             /*buffer*/indicesBuffer,
                             /*count*/indices.size(),
                             /*stride*/sizeof(vec3i),
                             /*offset*/0);
      owlGeomSetBuffer(geom,"indices",indicesBuffer);
      // release this buffer here, the geom now has a reference anyway
      owlBufferRelease(indicesBuffer);
      
      vec3f baseColor = .8f;
      owlGeomSet3f(geom,"baseColor",(const owl3f&)baseColor);
      
      // and add this now fully configured geom to the list of generated
      // geometries.
      generatedGeoms.push_back(geom);
    }
    // ------------------------------------------------------------------
    OWLGroup blas = owlTrianglesGeomGroupCreate
      (owl,generatedGeoms.size(),generatedGeoms.data());
    assert(blas && "could not create bottom level accel struct group");
    // ------------------------------------------------------------------
    owlGroupBuildAccel(blas);
    
    // ... and save that blas for later
    blasForObject[miniObject] = blas;
  }

  // ------------------------------------------------------------------
  std::cout << "- now, create list of instances\n";
  std::cout << "  (each inst has one transform and one blas)\n";

  std::vector<OWLGroup> instancedBlases;
  std::vector<owl4x3f> instanceTransforms;
  for (auto inst : scene->instances) {
    // 'just coincidentally' miniscene has the same matrix format....
    instanceTransforms.push_back((const owl4x3f&)inst->xfm);
    instancedBlases.push_back(blasForObject[inst->object]);
  }
  tlas = owlInstanceGroupCreate(owl,instanceTransforms.size(),
                                instancedBlases.data(),
                                // no instnace IDs
                                nullptr,
                                (float*)instanceTransforms.data());
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
  static int frameID = 0;
  const bool firstFrameOnly = (++frameID == 1);
  // ------------------------------------------------------------------
  if (firstFrameOnly)
    std::cout << "- setting frame buffer data in launch params...\n";
  owlParamsSet2i(lp,"fb.size",fbSize.x,fbSize.y);
  owlParamsSetPointer(lp,"fb.data",fbPointer);
  
  // ------------------------------------------------------------------
  if (firstFrameOnly)
    std::cout << "- setting world in launch params...\n";
  owlParamsSetGroup(lp,"world",tlas);

  // ------------------------------------------------------------------
  if (firstFrameOnly)
    std::cout << "- setting camera in launch params...\n";
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
  if (firstFrameOnly)
    std::cout << "- launching raygen...\n\n";  
  owlLaunch2D(rg,fbSize.x,fbSize.y,lp);
  if (firstFrameOnly) {
    std::cout << OWL_TERMINAL_LIGHT_GREEN
              << "\n(First) frame done rendering... (will only print this once)\n\n"
              << OWL_TERMINAL_DEFAULT;
    
    const std::string outFileName = std::string(CHAPTER_NAME)+".jpg";
    std::cout << "- saving image (via STB) in " << outFileName << "...\n";
    // OWLViewer::fbPointer is in managed memory, so readable on host
    stbi_flip_vertically_on_write(true);
    stbi_write_jpg(outFileName.c_str(),fbSize.x,fbSize.y,4,
                   fbPointer,fbSize.x*sizeof(uint32_t));
    std::cout << "- done saving; now back to interactive rendering...\n";
  }
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
    } else
      throw std::runtime_error("un-recognized cmdline arg '"+arg+"'");
  }

  // ==================================================================
  if (sceneFileName == "")
    sceneFileName = "../data/owls.mini";
  
  std::cout << OWL_TERMINAL_LIGHT_BLUE
            << "\nLoading some input scene (from " << sceneFileName << "):\n"
            << OWL_TERMINAL_DEFAULT;
  mini::Scene::SP scene = mini::Scene::load(sceneFileName);
  std::cout << "done loading scene, found: " << std::endl;
  std::cout << "- num instances: " << scene->instances.size() << std::endl;
  box3f worldBounds;
  (mini::box3f &)worldBounds = scene->getBounds();
  std::cout << "- bounds       : " << worldBounds << std::endl;

  
  std::cout << "- creating viewer (and letting it initialize itself)...\n";
  SampleViewer *sampleViewer = new SampleViewer(scene);
  sampleViewer->setTitle(CHAPTER_NAME);
  // enable viewer's 'fly' and 'inspect' camera modes (last one
  // becomes default)
  sampleViewer->enableFlyMode();
  sampleViewer->enableInspectMode();


  // use camera if one was set on cmdline, otherwise generate an
  // auto-camera similar to previous example
  if (up == vec3f(0.f)) {
    std::cout << "no camera set, auto-generating one\n";
    std::cout << "if this camera doesn't make sense, pass one via\n";
    std::cout << "--camera from.x from.y from.z at.x at.y at.z up.x up.y up.z\n";

    if (scene->instances.size() == 30035) {
      // hack: get soem useful start-up camera for the landscape model
#if 0
      // this a close to the PBRT default view:
      
      // --camera -2976.901123 309.4327087 -3735.08252 1474.776611 1074.60376 4633.866699 0 1 0 -fovy 70
      from = { -2976.901123f, 309.4327087f, -3735.08252f };
      at = { 1474.776611f, 1074.60376f, 4633.866699f };
#else
      // this give a better overview
      // --camera -5980.32 6014.52 -1581.8 896.848 -1414.59 3290.6 0 1 0 -fovy 60
      from = { -5980.32f, 6014.52f, -1581.8f };
      at = { 896.848f, -1414.59f, 3290.6f };
#endif
      up = { 0.f, 1.f, 0.f };
      fovy = 70.f;
    } else {
      up = {0.f,1.f,0.f};
      at = worldBounds.center();
      vec3f diag = worldBounds.size();
      from = at + 1.5f*vec3f(-.15f, .2f, +1.f) * diag.y;
    }
  }
  
  sampleViewer->setCameraOrientation
    (from,at,up,60.f);
  sampleViewer->setWorldScale
    (length(worldBounds.size()));
  
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
