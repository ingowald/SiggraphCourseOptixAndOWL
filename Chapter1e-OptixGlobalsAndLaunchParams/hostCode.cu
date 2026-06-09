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

__global__ void dummy(int *ptr)
{
if (threadIdx.x == 0)
printf("cuda kernel sees managedmem %p[0] = %i\n",
ptr,ptr[0]);
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
In this sample we show how to use __global__ optixLauchParams instead
of SBT data to pass data to device program(s). Otherwise this computes
the same image as sample 1d.
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
    { "frameID",
      OWL_INT,
      OWL_OFFSETOF(OptixGlobals,frameID) },
    { "managedMem",
      OWL_RAW_POINTER,
      OWL_OFFSETOF(OptixGlobals,managedMem) },
    // ------------------------------------------------------------------
    /* last entry is a nullptr for name; this serves as 'sentinel' to
       mark end of list */
    { nullptr },
  };

  // ******************************************************************
  // DELTA: in this sample we create some laumch params object; the
  // data of which is what will go into the optixGlobals. We create
  // this through the same parameters mechanism as for a raygen;
  // except we have no module and no entrypoint becuse it's not a
  // program, it just stores data.
  // ******************************************************************
  OWLLaunchParams lp  = owlParamsCreate(owl,sizeof(OptixGlobals),lpVars,-1);
  assert(lp && "could not create launch params");

  // ------------------------------------------------------------------
  std::cout << "- building actual device programs...\n";
  owlBuildPrograms(owl);
  
  // ------------------------------------------------------------------
  std::cout << "- building RTX pipeline...\n";
  owlBuildPipeline(owl);

  // ------------------------------------------------------------------
  std::cout << "- building SBT (can do that before setting launch params)...\n";
  owlBuildSBT(owl);
  
  
  // ------------------------------------------------------------------
  std::cout << "- creating a buffer to store the frame in...\n";
  // let's do a 1200x800 image...
  vec2i fbSize = vec2i(1200,800);
  OWLBuffer fb = owlDeviceBufferCreate(owl,OWL_INT,fbSize.x*fbSize.y,
                                       /* no host data to upload */
                                       nullptr);
  
  // ------------------------------------------------------------------
  std::cout << "- setting the launch params data values...\n";
  owlParamsSet2i(lp,"fb.size",fbSize.x,fbSize.y);
  owlParamsSetBuffer(lp,"fb.data",fb);

  // ------------------------------------------------------------------
  std::cout << "- launching raygen programs with launch params...\n\n";

int *managedMem;
cudaMallocManaged((void **)&managedMem,1024);
owlParamsSetPointer(lp,"managedMem",managedMem);

std::cout << "==== withOUT some dummy cuda kernel call =====" << std::endl;
for (int i=0;i<10;i++) {
owlParamsSet1i(lp,"frameID",i);
  owlLaunch2D(rg,fbSize.x,fbSize.y,lp);
  cudaDeviceSynchronize();
  printf("on host: managedMem %p[0] = %i\n",managedMem,managedMem[0]);
  }

std::cout << "==== WITH some dummy cuda kernel call =====" << std::endl;
for (int i=0;i<10;i++) {
owlParamsSet1i(lp,"frameID",i);
  owlLaunch2D(rg,fbSize.x,fbSize.y,lp);
  cudaDeviceSynchronize();
  dummy<<<1,32>>>(managedMem);
  cudaDeviceSynchronize();
  printf("on host: managedMem %p[0] = %i\n",managedMem,managedMem[0]);
  }


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
