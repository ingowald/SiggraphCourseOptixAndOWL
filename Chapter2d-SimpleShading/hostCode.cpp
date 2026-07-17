// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
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
    (owl,mod,"renderTestFrame",sizeof(RayGenData),rgVars,-1);
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
  std::cout << "- creating miss program...\n";
  OWLMissProg miss = owlMissProgCreate(owl,
                                       // module and entry point name
                                       mod,"missProg",
                                       /*size*/0,
                                       /*vars*/nullptr,-1);
  assert(miss && "could not create miss program");
  
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
  // DELTA: in this sample we take our input geometry from the
  // miniSCene we have have loaded above. Since we can't do real
  // instancing yet we'll only consider the first instance, and just
  // build one triangles geom for each input miniScene::TriangleMesh
  // we find in that first instance.
  /* ****************************************************************** */
  
  // ------------------------------------------------------------------
  std::cout << "- taking first object in the input scene...\n";
  mini::Object::SP miniObject = scene->instances[0]->object;
  std::cout << "- ... and building one triangles geom for each mesh therein\n";
  std::vector<OWLGeom> generatedGeoms;
  for (auto inputMesh : miniObject->meshes) {
    int geomID = (int)generatedGeoms.size();
    
    std::cout << "  - mesh #" << generatedGeoms.size() << " :\n";
    OWLGeom geom = owlGeomCreate(owl,gt);
    assert(geom && "could not create geometry from geom type");
    
    // ------------------------------------------------------------------
    std::cout << "    - set triangle mesh's vertex array...\n";
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
    std::cout << "    - set triangle mesh's index array...\n";
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
  std::cout << "- create an actual geometry group over these "
            << generatedGeoms.size() << " mesh(es)...\n";
  OWLGroup blas = owlTrianglesGeomGroupCreate
    (owl,generatedGeoms.size(),generatedGeoms.data());
  assert(blas && "could not create bottom level accel struct group");
  
  // ------------------------------------------------------------------
  std::cout << "- build the geometry group's accel (the blas)...\n";
  owlGroupBuildAccel(blas);
  
  // ------------------------------------------------------------------
  std::cout << "- create a top-level accel over this blas group...\n";
  tlas = owlInstanceGroupCreate(owl,1,&blas);
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
This sample extends the previous samples to load some real
scene geometry (from a miniScene .mini file), and build
geometries, blas, and tlas for that geometry.
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
#if WIN32
      sceneFileName = "../../data/owls.mini";
#else
      sceneFileName = "../data/owls.mini";
#endif
  
  std::cout << OWL_TERMINAL_LIGHT_BLUE
            << "\nLoading some input scene (from " << sceneFileName << "):\n"
            << OWL_TERMINAL_DEFAULT;
  mini::Scene::SP scene = mini::Scene::load(sceneFileName);
  std::cout << "done loading scene, found: " << std::endl;
  std::cout << "- num instances: " << scene->instances.size() << std::endl;
  if (scene->instances.size() != 1)
    throw std::runtime_error("This sample cannot yet do instances; "
                             "we'll add this in the next sample");
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
    up = {0.f,1.f,0.f};
    at = worldBounds.center();
    vec3f diag = worldBounds.size();
    from = at + 1.5f*vec3f(-.15f, .2f, +1.f) * diag.y;
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
