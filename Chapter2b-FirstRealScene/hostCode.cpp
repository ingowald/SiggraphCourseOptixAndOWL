// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// include deviceCode.h; this automaticlly pulls in owl.h and owl-common
#include "deviceCode.h"
// a std::vector to store the pixels in when we copy them to the host
#include <vector>
// stb, to write the image
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
// miniScene, just so we can load some better scene content
#include "miniScene/Scene.h"

// the "embedded" precompiled PTX-code from our deviceCode.cu. 
extern "C" char deviceCode_ptx[];

vec3f toOWL(mini::common::vec3f v)
{ return { v.x,v.y,v.z }; }
         
int main(int ac, char **av)
{
  std::cout << OWL_TERMINAL_LIGHT_BLUE << R"(
**********************************************************************
Chapter2b-FirstRealSceneGeometry:
**********************************************************************
            )" << OWL_TERMINAL_DEFAULT;
  
  std::cout << R"(
In this sample we'll create our first device buffer to hold a frame
buffer in, we'll change out raygen value to actually render a simple
test frame into this frame buffer, and show how to get these pixels
back on the host (where we'll save it to disk).
)";

  vec3f from = vec3f(0.f);
  vec3f at   = vec3f(0.f);
  vec3f up   = vec3f(0.f);
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
      float fovy = std::stof(av[i+1]);
      // parse, but ignore for now
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
  mini::box3f worldBounds = scene->getBounds();
  std::cout << "- bounds       : " << worldBounds << std::endl;

  if (up == vec3f(0.f)) {
    std::cout << "no camera set, auto-generating one\n";
    std::cout << "if this camera doesn't make sense, pass one via\n";
    std::cout << "--camera from.x from.y from.z at.x at.y at.z up.x up.y up.z\n";
    up = {0.f,1.f,0.f};
    at = toOWL(worldBounds.center());
    vec3f diag = toOWL(worldBounds.size());
    from = at + vec3f(-.1f, .2f, +2.f) * diag;
  }
  
  if (scene->instances.size() != 1)
    throw std::runtime_error("This sample cannot yet do instances; "
                             "we'll add this in the next sample");
  
  // ==================================================================
  std::cout << OWL_TERMINAL_LIGHT_BLUE
            << "\nBuilding basic RTX pipeline ingredients:\n"
            << OWL_TERMINAL_DEFAULT;
  // ------------------------------------------------------------------
  std::cout << "- creating OWL context...\n";
  OWLContext owl = owlContextCreate(nullptr,1);
  assert(owl && "could not create context");

  // ------------------------------------------------------------------
  std::cout << "- creating module from precompiled device code...\n";
  OWLModule mod = owlModuleCreate(owl,deviceCode_ptx);
  assert(mod && "could not create module");
  
  // ------------------------------------------------------------------
  std::cout << "- creating raygen program (data now moved to globals)...\n";
  OWLVarDecl rgVars[] = {
    // empty for now; we moved all data to optixGlobals
    { nullptr },
  };
  OWLRayGen rg  = owlRayGenCreate
    (owl,mod,"renderTestFrame",sizeof(RayGenData),rgVars,-1);
  assert(rg && "could not create raygen");

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
  OWLGeomType gt = owlGeomTypeCreate(owl,OWL_GEOM_TRIANGLES,
                                     sizeof(TrianglesGeomData),
                                     gtVars,-1);
  assert(gt && "could not create geometry type");
                                     
  // ------------------------------------------------------------------
  std::cout << "- setting CH program on geom type...\n";
  owlGeomTypeSetClosestHit(gt,
                           /* ray type */0,
                           /* module where to look for code*/mod,
                           /* entry point */"TrianglesCH");

  // ------------------------------------------------------------------
  std::cout << "- building actual device programs...\n";
  owlBuildPrograms(owl);
  
  // ------------------------------------------------------------------
  std::cout << "- building RTX pipeline...\n";
  owlBuildPipeline(owl);
  
  // ==================================================================
  std::cout << OWL_TERMINAL_LIGHT_BLUE
            << "\nbuilding scene data (triangle mesh, tlas, and blas)...\n"
            << OWL_TERMINAL_DEFAULT;

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
  OWLGroup tlas = owlInstanceGroupCreate(owl,1,&blas);
  assert(tlas && "could not create instance/top level accel struct group");
  
  // ------------------------------------------------------------------
  std::cout << "- build the instance group's accel (the tlas)...\n";
  owlGroupBuildAccel(tlas);

  // ------------------------------------------------------------------
  std::cout << "- building SBT (all program data set, all groups built)...\n";
  owlBuildSBT(owl);
  
  // ==================================================================
  std::cout << OWL_TERMINAL_LIGHT_BLUE
            << "\nall scene data built; setting up first frame\n"
            << OWL_TERMINAL_DEFAULT;

  // ------------------------------------------------------------------
  std::cout << "- creating LaunchParams object to set optixGlobals...\n";
  OWLVarDecl lpVars[] = {
    // ------------------------------------------------------------------
    { "fb.size",
      OWL_INT2,
      OWL_OFFSETOF(OptixGlobals,frameBuffer.size) },
    // ------------------------------------------------------------------
    { "fb.data",
      OWL_BUFPTR,
      OWL_OFFSETOF(OptixGlobals,frameBuffer.data) },
    // ------------------------------------------------------------------
    { "world",
      OWL_GROUP,
      OWL_OFFSETOF(OptixGlobals,world) },
    // ------------------------------------------------------------------
    { "camera",
      OWL_USER_TYPE(Camera),
      OWL_OFFSETOF(OptixGlobals,camera) },
    // ------------------------------------------------------------------
    { nullptr },
  };
  OWLLaunchParams lp  = owlParamsCreate(owl,sizeof(OptixGlobals),lpVars,-1);
  assert(lp && "could not create launch params");

  // ------------------------------------------------------------------
  std::cout << "- creating a buffer to store the frame in...\n";
  // let's do a 1200x800 image...
  vec2i fbSize = vec2i(1200,800);
  OWLBuffer fb = owlDeviceBufferCreate(owl,OWL_INT,fbSize.x*fbSize.y,
                                       /* no host data to upload */
                                       nullptr);
  
  // ------------------------------------------------------------------
  std::cout << "- setting frame buffer data in launch params...\n";
  Camera camera(from,at,up,fbSize);
  PRINT(camera.dir_00);
  owlParamsSetRaw(lp,"camera",&camera);
  owlParamsSet2i(lp,"fb.size",fbSize.x,fbSize.y);
  owlParamsSetBuffer(lp,"fb.data",fb);
  std::cout << "- setting world in launch params...\n";
  owlParamsSetGroup(lp,"world",tlas);

  // ------------------------------------------------------------------
  std::cout << "- launching raygen...\n\n";
  owlLaunch2D(rg,fbSize.x,fbSize.y,lp);
  
  // ==================================================================
  std::cout << OWL_TERMINAL_LIGHT_BLUE
            << "Launch completed; downloading rendered image:\n"
            << OWL_TERMINAL_DEFAULT;

  std::cout << "- downloading pixels (via cudaMemcpy)...\n";
  std::vector<uint32_t> hostPixels(fbSize.x*fbSize.y);
  cudaMemcpy(// pointer to host data to store pixels in
             hostPixels.data(),
             // *device* pointer of the frame buffer we used
             owlBufferGetPointer(fb,0),
             fbSize.x*fbSize.y*sizeof(uint32_t),
             cudaMemcpyDefault);
  cudaDeviceSynchronize();

  const char *outFileName = "c2b-FirstRealScene.jpg";
  std::cout << "- saving image (via STB) in " << outFileName << "...\n";
  
  stbi_flip_vertically_on_write(true);
  stbi_write_jpg(outFileName,fbSize.x,fbSize.y,4,
                 hostPixels.data(),fbSize.x*sizeof(uint32_t));
  
  // ==================================================================
  std::cout << OWL_TERMINAL_LIGHT_BLUE
            << "\nImage saved; sample done; cleaning up:\n"
            << OWL_TERMINAL_DEFAULT;

  // ------------------------------------------------------------------
  std::cout << "- releasing launchparams...\n";
  owlParamsRelease(lp);
  std::cout << "- releasing buffer...\n";
  owlBufferRelease(fb);
  std::cout << "- releasing raygen...\n";
  owlRayGenRelease(rg);
  std::cout << "- releasing module...\n";
  owlModuleRelease(mod);
  std::cout << "- destroying OWL context...\n";
  owlContextDestroy(owl);
  
  std::cout << OWL_TERMINAL_LIGHT_GREEN
            << "\nAll good! - Sample concluded successfully...\n\n"
            << OWL_TERMINAL_DEFAULT;
  return 0;
}
