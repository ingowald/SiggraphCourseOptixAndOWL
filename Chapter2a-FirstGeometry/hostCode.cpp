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

// the "embedded" precompiled PTX-code from our deviceCode.cu. 
extern "C" char deviceCode_ptx[];

int main(int ac, char **av)
{
  std::cout
    << OWL_TERMINAL_LIGHT_BLUE
    << "**********************************************************************\n"
    << CHAPTER_NAME << "\n"
    << "**********************************************************************\n"
    << OWL_TERMINAL_DEFAULT;
  
  std::cout << R"(
In this sample we'll create our first device buffer to hold a frame
buffer in, we'll change out raygen value to actually render a simple
test frame into this frame buffer, and show how to get these pixels
back on the host (where we'll save it to disk).
)";

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
    { nullptr },
  };


  /* ****************************************************************** */
  // DELTA: in this sample we create soem actual geometry type (for a
  // triangle mesh, with the CH program we have written), and a
  // geometry that uses that. we also need a miss program to write
  // pixels that didn't hit anything
  /* ****************************************************************** */

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


  /* ****************************************************************** */
  // now, we can build som actual geometies, a geometry group (fro the
  // bottom level accel) and a instance group (for the top level
  // accel)
  /* ****************************************************************** */
  
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
  
  OWLLaunchParams lp  = owlParamsCreate(owl,sizeof(OptixGlobals),lpVars,-1);
  assert(lp && "could not create launch params");

  // ------------------------------------------------------------------
  std::cout << "- creating a buffer to store the frame in...\n";
  // let's do a 1024^2 image. Note this is intentionally square
  // because the trival hardcoded 'camera' that this samples' raygen
  // program uses can't do aspect ration, yet
  vec2i fbSize = vec2i(1600,1200);
  OWLBuffer fb = owlDeviceBufferCreate(owl,OWL_INT,fbSize.x*fbSize.y,
                                       /* no host data to upload */
                                       nullptr);
  
  // ------------------------------------------------------------------
  std::cout << "- setting frame buffer data in launch params...\n";
  owlParamsSet2i(lp,"fb.size",fbSize.x,fbSize.y);
  owlParamsSetBuffer(lp,"fb.data",fb);
  std::cout << "- setting world in launch params...\n";
  owlParamsSetGroup(lp,"world",tlas);

  // ------------------------------------------------------------------
  std::cout << "- launching raygen...\n\n";
  owlLaunch2D(rg,fbSize.x,fbSize.y,lp);
  
  // ==================================================================
  std::cout << OWL_TERMINAL_LIGHT_BLUE
            << "\nLaunch completed; downloading rendered image:\n"
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

  const std::string outFileName = std::string(CHAPTER_NAME)+".jpg";
  std::cout << "- saving image (via STB) in " << outFileName << "...\n";
  stbi_flip_vertically_on_write(true);
  stbi_write_jpg(outFileName.c_str(),fbSize.x,fbSize.y,4,
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
