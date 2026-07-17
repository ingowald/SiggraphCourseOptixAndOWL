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

  /*! this gets called when the user presses a key on the keyboard ... */
  void key(char key, const vec2i &/*where*/) override;

  /*! window notifies us that we got resized */     
  void resize(const vec2i &newSize) override;
  
  /*! this function gets called whenever any camera manipulator
    updates the camera. gets called AFTER all values have been updated */
  void cameraChanged() override;
  
  /*! helper function that loads a miniscene texture into an
      OWLTexture (including use of textureLibrary to avoid replicated
      loading */
  OWLTexture loadTexture(mini::Texture::SP miniTexture);
  
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
  OWLBuffer       accumBuffer = 0;

  /*! track some linear sequence number for each frame, so each frame
      can use a new set of random numbers. this also serves as
      acounting how many frames have been accumulated already, so will
      get reset to 0 every time the camera or other render settings
      change. */
  int accumID = 0;
  int disablePathTracing = false;
  
  // use a texture library/texture cache to avoid re-creating the same
  // texture for every geometry that's using it.
  std::map<mini::Texture::SP,OWLTexture> textureLibrary;
};

SampleViewer::SampleViewer(mini::Scene::SP scene)
  : scene(scene)
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
    disablePathTracing = !disablePathTracing;
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
  // Delta from last sample: set number of ray types in context to two
  // - we'll use ray type 0 for regular and type 1 for shadow rays
  // ##################################################################
  owlContextSetRayTypeCount(owl,2);
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
#if FOR_ILLUSTRATION_ONLY
    { "disablePathTracing",
      OWL_INT,
      OWL_OFFSETOF(OptixGlobals,disablePathTracing) },
#endif
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
    { "ior",
      OWL_FLOAT,
      OWL_OFFSETOF(TrianglesGeomData,ior) },
    { "colorTexture",
      OWL_TEXTURE,
      OWL_OFFSETOF(TrianglesGeomData,colorTexture) },
    { "texcoords",
      OWL_BUFPTR,
      OWL_OFFSETOF(TrianglesGeomData,texcoords) },
    { "normals",
      OWL_BUFPTR,
      OWL_OFFSETOF(TrianglesGeomData,normals) },
    { nullptr },
  };
  gt = owlGeomTypeCreate(owl,OWL_GEOM_TRIANGLES,
                                     sizeof(TrianglesGeomData),
                                     gtVars,-1);
  assert(gt && "could not create geometry type");
                                     
  // ------------------------------------------------------------------
  std::cout << "- setting CH program on geom type...\n";
  // ##################################################################
  // Delta from last sample: set CH and AH program(s) for different
  // ray types
  // ##################################################################
  // for regular rays we set both AH and CH
  owlGeomTypeSetClosestHit(gt,
                           /* ray type */0,
                           /* module where to look for code*/mod,
                           /* entry point */"Triangles_CH_RegularRays");
  owlGeomTypeSetAnyHit(gt,
                       /* ray type */0,
                       /* module where to look for code*/mod,
                       /* entry point */"Triangles_AH_RegularRays");
  // now same for shadow rays
  owlGeomTypeSetClosestHit(gt,
                           /* ray type */1,
                           /* module where to look for code*/mod,
                           /* entry point */"Triangles_CH_ShadowRays");
  owlGeomTypeSetAnyHit(gt,
                       /* ray type */1,
                       /* module where to look for code*/mod,
                       /* entry point */"Triangles_AH_ShadowRays");

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

OWLTexture SampleViewer::loadTexture(mini::Texture::SP miniTexture)
{
  if (!miniTexture)
    // no input texture -> use invalid texture handle. in owl that's a
    // perfectly valid texture handle, it'll set a cudaTextureObject
    // that's '0'.
    return OWLTexture{};
  
  // there _is_ a texture on the geom. check if we already
  // loaded this texture:
  if (textureLibrary.find(miniTexture)
      != textureLibrary.end()) 
    // and if so, use it....
    return textureLibrary[miniTexture];
  
  // and if not, create it:
  
  OWLTexelFormat texelFormat;
  switch (miniTexture->format) {
  case mini::Texture::FLOAT4:
    texelFormat = OWL_TEXEL_FORMAT_RGBA32F;
    break;
  case mini::Texture::FLOAT1:
    texelFormat = OWL_TEXEL_FORMAT_R32F;
    break;
  case mini::Texture::RGBA_UINT8:
    texelFormat = OWL_TEXEL_FORMAT_RGBA8;
    break;
  default:
    std::cout << "warning: unsupported mini::Texture format #"
              << (int)miniTexture->format << std::endl;
    return OWLTexture{};
  }
  auto size = miniTexture->size;
  // hardcode bilinear filtering for this sample
  auto filterMode = OWL_TEXTURE_LINEAR;
  // hardcode mirror mode for this sample (ls model needs this)
  auto addressMode = OWL_TEXTURE_MIRROR;
  OWLTexture owlTexture
    = owlTexture2DCreate(owl,texelFormat,size.x,size.y,
                         (const void *)miniTexture->data.data(),
                         filterMode,addressMode,addressMode);
  textureLibrary[miniTexture] = owlTexture;
  return owlTexture;
}

void SampleViewer::buildSceneGeometry()
{
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

      // ------------------------------------------------------------------
      auto &texcoords = inputMesh->texcoords;
      if (!texcoords.empty()) {
        OWLBuffer texcoordsBuffer
          = owlDeviceBufferCreate(owl,OWL_FLOAT2,
                                  texcoords.size(),texcoords.data());
        assert(texcoordsBuffer && "could not create texcoords buffer");
        owlGeomSetBuffer(geom,"texcoords",texcoordsBuffer);
        // release this buffer here, the geom now has a reference anyway
        owlBufferRelease(texcoordsBuffer);
      }
      
      // ------------------------------------------------------------------
      auto &normals = inputMesh->normals;
      if (!normals.empty()) {
        OWLBuffer normalsBuffer
          = owlDeviceBufferCreate(owl,OWL_FLOAT3,
                                  normals.size(),normals.data());
        assert(normalsBuffer && "could not create normals buffer");
        owlGeomSetBuffer(geom,"normals",normalsBuffer);
        // release this buffer here, the geom now has a reference anyway
        owlBufferRelease(normalsBuffer);
      }
      
      // ------------------------------------------------------------------
      mini::DisneyMaterial::SP disney
        = inputMesh->material->as<mini::DisneyMaterial>();
      mini::Dielectric::SP dielectric
        = inputMesh->material->as<mini::Dielectric>();
      if (dielectric) {
        owlGeomSet1f(geom,"ior",dielectric->etaInside);
      } else if (disney) {
        // use basecolor from material
        owlGeomSet3f(geom,"baseColor",(const owl3f&)disney->baseColor);
        if (disney->transmission > 0.f) 
          owlGeomSet1f(geom,"ior",disney->ior);
        else
          owlGeomSet1f(geom,"ior",1.f);
        // and load texture (this'll return a null handle if there is
        // no texture - that's fine.
        OWLTexture owlTexture = loadTexture(disney->colorTexture);
        owlGeomSetTexture(geom,"colorTexture",owlTexture);
      } else {
        // dynamic-cast 'as()' didn't work out, so the model we're
        // currently looking at seems to use a different
        // mini::Material type. let's just use a gray basecolor
        vec3f baseColor = .8f;
        owlGeomSet3f(geom,"baseColor",(const owl3f&)baseColor);
        owlGeomSet1f(geom,"ior",1.f);
      }
      
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
#if FOR_ILLUSTRATION_ONLY
  owlParamsSet1i(lp,"disablePathTracing",disablePathTracing);
#endif  
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
  if (sceneFileName == "")
#if WIN32
      sceneFileName = "../../data/ls.mini";
      // sceneFileName = "../../data/peteAndI.mini";
      // sceneFileName = "../../data/ls.mini";
      // sceneFileName = "../../data/owls.mini";
#else
     sceneFileName = "../data/ls.mini";
     // sceneFileName = "../data/peteAndI.mini";
     // sceneFileName = "../data/ls.mini";
     //  sceneFileName = "../data/owls.mini";
#endif
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
      from = { -5980.32f, 6014.52f, -1581.8f };
      at = { 896.848f, -1414.59f, 3290.6f };
      up = {0.f,1.f,0.f};
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
