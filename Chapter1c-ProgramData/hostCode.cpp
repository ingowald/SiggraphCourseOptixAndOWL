// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// include deviceCode.h; this automaticlly pulls in owl.h and owl-common
#include "deviceCode.h"

// the "embedded" precompiled PTX-code from our deviceCode.cu. This
// will get linked in under a symbol with the same node as the input
// cuda file that contains the device code:
//
// i.e., embed_ptx(deviceCode.cu) -> char deviceCode_ptx[]
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
This sample shows hwo to declare and set 'program data' for device programs.
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
     /* difference from previous sample: passing size of ray gen data,
        and the varDecls that describe its members. */
     /* ****************************************************************** */
  OWLVarDecl rgVars[] = {
    // ------------------------------------------------------------------
    { "someInteger",
      OWL_INT,
      OWL_OFFSETOF(RayGenData,someInteger) },
    // ------------------------------------------------------------------
    { "somePointOrVector",
      OWL_FLOAT3,
      OWL_OFFSETOF(RayGenData,somePointOrVector) },
    // ------------------------------------------------------------------
    { "aRawUnformattedRegion",
      OWL_USER_TYPE(SomeDummyRawData),
      OWL_OFFSETOF(RayGenData,aRawUnformattedRegion) },
    // ------------------------------------------------------------------
    { "someRawPointer",
      /* OWL_RAW_POINTER: a raw pointer type, without any translation;
         the device value will be exactly the same 64-bit pointer that
         gets set on the host */
      OWL_RAW_POINTER,
      OWL_OFFSETOF(RayGenData,someRawPointer) },
    // ------------------------------------------------------------------
    { "someBuffer",
      /* OWL_BUFPTR: a parameter that needs "translating" from a
         host-side `OWLBuffer` (we'll introduce these later) to a
         device-side pointer. On the host, this gets set to a
         `OWLBuffer`; owl will then "translate" this to that buffer's
         device address for the given GPU, and write that device
         pointer into the program data (ie, the type in the program
         data struct is a C/C++ _pointer_, not a `OWLBuffer` */
      OWL_BUFPTR,
      OWL_OFFSETOF(RayGenData,somePointerToBufferData) },
    // ------------------------------------------------------------------
    { "someTexture",
      /*! a OWL_TEXTURE: parameter that gets set to a `OWLTexture` on
        the host, and which OWL translates to a `cudaTextureObject_t`
        on the device side */
      OWL_TEXTURE,
      OWL_OFFSETOF(RayGenData,someTextureHandle) },
    // ------------------------------------------------------------------
    { "someGroup",
      /*! OWL_GROUP: a paramter that requires translating from a
          host-side `OWLGroup` to a device-side `OptixTraversableHandle`
          type. */
      OWL_GROUP,
      OWL_OFFSETOF(RayGenData,someOptixAccelHandle) },
    // ------------------------------------------------------------------
    /* last entry is a nullptr for name; this serves as 'sentinel' to
       mark end of list */
    { nullptr },
  };
  OWLRayGen rg  = owlRayGenCreate
    (owl,mod,"helloOWL",
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
  std::cout << "- now setting some raygen data values (BEFORE building sbt!!!)...\n";
  // sets one int
  owlRayGenSet1i(rg,"someInteger",1234567);
  // set three floats
  owlRayGenSet3f(rg,"somePointOrVector",1.2f,3.4f,5.67f);
  // sets a 64-bit pointer value
  owlRayGenSetPointer(rg,"someRawPointer",(void*)0x1234567);
    
  char someString[sizeof(SomeDummyRawData)] = "s1234567s";
  // this just copies 'sizeof()' bytes, without caring what it is.
  owlRayGenSetRaw(rg,"aRawUnformattedRegion",someString);
  // set all of the advanted types to NULL handles; we'll introduce
  // those in later samples, for now just show how they _could_ be set
  // if we had any.
  owlRayGenSetGroup(rg,"someGroup",(OWLGroup)0);
  owlRayGenSetTexture(rg,"someTexture",(OWLTexture)0);
  owlRayGenSetBuffer(rg,"someBuffer",(OWLBuffer)0);
  // ------------------------------------------------------------------
  std::cout << "- building SBT (no geoms, but raygen is part of SBT, too!)...\n";
  owlBuildSBT(owl);
  
  // ------------------------------------------------------------------
  std::cout << "- launching raygen...\n\n";
  owlRayGenLaunch2D(rg,/* launch with launch dims of 100x100: */100,100);
  
  // ==================================================================
  std::cout << OWL_TERMINAL_LIGHT_BLUE
            << "\nLaunch completed. Sample done; cleaning up:\n"
            << OWL_TERMINAL_DEFAULT;
  
  // ------------------------------------------------------------------
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
