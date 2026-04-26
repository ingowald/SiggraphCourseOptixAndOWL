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
  std::cout << OWL_TERMINAL_LIGHT_BLUE << R"(
**********************************************************************
Chapter1d-FirstBufferAndTestFrame:
**********************************************************************
            )" << OWL_TERMINAL_DEFAULT;
  
  std::cout << R"(
In this sample we'll create our first device buffer to hold a frame
buffer in, we'll change out raygen value to actually render a simple
test frame into this frame buffer, and show how to get these pixels
back on the host (where we'll save it to disk).
)";

  // ==================================================================
  std::cout << OWL_TERMINAL_LIGHT_BLUE
            << "\nGetting ready for our first launch:\n"
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
  std::cout << "- creating raygen program object (now with program data)...\n";
     /* ****************************************************************** */
     /* difference from previous sample: we actually set useful values :-) */
     /* ****************************************************************** */
  OWLVarDecl rgVars[] = {
    // ------------------------------------------------------------------
    { "fb.size",
      OWL_INT2,
      OWL_OFFSETOF(RayGenData,frameBuffer.size) },
    // ------------------------------------------------------------------
    { "fb.data",
      OWL_BUFPTR,
      OWL_OFFSETOF(RayGenData,frameBuffer.data) },
    // ------------------------------------------------------------------
    /* last entry is a nullptr for name; this serves as 'sentinel' to
       mark end of list */
    { nullptr },
  };
  OWLRayGen rg  = owlRayGenCreate
    (owl,mod,"renderTestFrame",
     /* ****************************************************************** */
     /* difference from previous sample: passing size of ray gen data,
        and the varDecls that describe its members. */
     /* ****************************************************************** */
     sizeof(RayGenData),
     // array of vardecls that describe the members of that struct
     rgVars,
     // '-1' = rely on nullptr sentinal to mark end of list
     -1);
  assert(rg && "could not create raygen");

  // ------------------------------------------------------------------
  std::cout << "- building actual device programs...\n";
  owlBuildPrograms(owl);
  
  // ------------------------------------------------------------------
  std::cout << "- building RTX pipeline...\n";
  owlBuildPipeline(owl);

  // ------------------------------------------------------------------
  std::cout << "- creating a buffer to store the frame in...\n";
  // let's do a 1200x800 image...
  vec2i fbSize = vec2i(1200,800);
  OWLBuffer fb = owlDeviceBufferCreate(owl,OWL_INT,fbSize.x*fbSize.y,
                                       /* no host data to upload */
                                       nullptr);
  
  // ------------------------------------------------------------------
  std::cout << "- setting the raygen data values (BEFORE building sbt!!!)...\n";
  owlRayGenSet2i(rg,"fb.size",fbSize.x,fbSize.y);
  owlRayGenSetBuffer(rg,"fb.data",fb);

  // ------------------------------------------------------------------
  std::cout << "- building SBT (no geoms, but raygen is part of SBT, too!)...\n";
  owlBuildSBT(owl);
  
  // ------------------------------------------------------------------
  std::cout << "- launching raygen...\n\n";
  owlRayGenLaunch2D(rg,fbSize.x,fbSize.y);
  
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

  const char *outFileName = "c1d-FirstBufferAndTestFrame.jpg";
  std::cout << "- saving image (via STB) in " << outFileName << "...\n";
  stbi_write_jpg(outFileName,fbSize.x,fbSize.y,4,
                 hostPixels.data(),fbSize.x*sizeof(uint32_t));
  
  // ==================================================================
  std::cout << OWL_TERMINAL_LIGHT_BLUE
            << "\nImage saved; sample done; cleaning up:\n"
            << OWL_TERMINAL_DEFAULT;
  
  // ------------------------------------------------------------------
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
