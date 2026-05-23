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

SampleViewer::SampleViewer()
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
    // empty for now; we moved all data to optixGlobals
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
  // ==================================================================
  std::cout << OWL_TERMINAL_LIGHT_BLUE
            << "\nall scene data built; setting up first frame\n"
            << OWL_TERMINAL_DEFAULT;
  
  // ==================================================================
  std::cout << OWL_TERMINAL_LIGHT_BLUE
            << "\nbuilding scene data (triangle mesh, tlas, and blas)...\n"
            << OWL_TERMINAL_DEFAULT;
  
  // ------------------------------------------------------------------
  std::cout << "- creating an actual geometry with this type...\n";
  OWLGeom geom = owlGeomCreate(owl,gt);
  assert(geom && "could not create geometry from geom type");
  
  // ------------------------------------------------------------------
  std::cout << "- set triangle mesh's vertex array...\n";
  std::vector<vec3f> vertices = {
    { -.8f, -.8f, -.8f },
    { +.8f, -.8f, -.8f },
    { -.8f, -.8f, +.8f },
    { +.8f, -.8f, +.8f },
    { -.8f, +.8f, -.8f },
    { +.8f, +.8f, -.8f },
    { -.8f, +.8f, +.8f },
    { +.8f, +.8f, +.8f }
  };
  OWLBuffer verticesBuffer
    = owlDeviceBufferCreate(owl,OWL_FLOAT3,vertices.size(),vertices.data());
  assert(verticesBuffer && "could not create vertices buffer");
  owlTrianglesSetVertices(geom,
                          /*buffer*/verticesBuffer,
                          /*count*/vertices.size(),
                          /*stride*/sizeof(vec3f),
                          /*offset*/0);

  // ------------------------------------------------------------------
  std::cout << "- set triangle mesh's index array...\n";
  std::vector<vec3i> indices = {
    { 0,1,3 }, { 2,0,3 },
    { 5,7,6 }, { 5,6,4 },
    { 0,4,5 }, { 0,5,1 },
    { 2,3,7 }, { 2,7,6 },
    { 1,5,7 }, { 1,7,3 },
    { 4,0,2 }, { 4,2,6 }
  };
  OWLBuffer indicesBuffer
    = owlDeviceBufferCreate(owl,OWL_INT3,indices.size(),indices.data());
  assert(indicesBuffer && "could not create indices buffer");
  owlTrianglesSetIndices(geom,
                         /*buffer*/indicesBuffer,
                         /*count*/indices.size(),
                         /*stride*/sizeof(vec3i),
                         /*offset*/0);

  // ------------------------------------------------------------------
  std::cout << "- create an actual geometry group over this mesh...\n";
  OWLGroup blas = owlTrianglesGeomGroupCreate(owl,1,&geom);
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
This sample extends the previous samples to use an interactive
3D Viewer (based on the glfwViewer external submodule). This
now only shows how to apply per-frame changes (like camera
or frame buffer) outside the SBT (using LaunchParams), but also
introduces a better camera model (and in particular, automatic
camera pose initialization) that we will need when loading
more interesting content in the coming samples.
)";

  std::cout << "- creating viewer (and letting it initialize itself)...\n";
  SampleViewer *sampleViewer = new SampleViewer;
  sampleViewer->setTitle(CHAPTER_NAME);
  // enable viewer's 'fly' and 'inspect' camera modes (last one
  // becomes default)
  sampleViewer->enableFlyMode();
  sampleViewer->enableInspectMode();

  // define some world bounding box (hardcoded, for this sample) to
  // allow for setting some automate camera
  box3f worldBounds = {vec3f(-.8f),vec3f(+.8f)};
  /* initializes a useful the motion scale (ie, how far the camera
     should move in world space for a given amount of screen-space
     mouse motion; should be roughly about the same as the worls-scale
     size of the model */
  sampleViewer->setWorldScale
    (length(worldBounds.size()));
  sampleViewer->setCameraOrientation
    (// 'from': default: from front (z axis)a, d from slightly left/above
     worldBounds.center()
     + vec3f(-.6f,.8f,+1.5f)*
     worldBounds.size(),
     // 'to': default: point to world center:
     worldBounds.center(),
     // 'up': default y-up
     vec3f(0.f,1.f,0.f),
     // fovy, in degrees
     60.f);
                                     
  
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
