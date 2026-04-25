// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// the actual OWL API
#include "owl/owl.h"
// some owl helper functions we will use (eg pretty printing)
#include "owl/common/owl-common.h"
#include <cassert>

// the "embedded" precompiled PTX-code from our deviceCode.cu. This
// will get linked in under a symbol with the same node as the input
// cuda file that contains the device code:
//
// i.e., embed_ptx(deviceCode.cu) -> char deviceCode_ptx[]
extern "C" char deviceCode_ptx[];

int main(int ac, char **av)
{
  std::cout << OWL_TERMINAL_LIGHT_BLUE << R"(
**********************************************************************
Chapter1b-HelloRayGen:
**********************************************************************
            )" << OWL_TERMINAL_DEFAULT;
  
  std::cout << R"(
This sample will create a first OptiX pipeline with (only) a raygen
program, then launch this raygen program, and have it print a 'hello
world' from the device.

This sample will not do anything useful yet (no geometries being
created, no acceleration structures built, no rays being traced,...),
but already walks through the process of how to write device-side
programs, how to embed them in the generated binary, how to create a
optix raygen program from this embedded code, how to create a proper
pipeline with this, and how to launch this raygen program (which is
quite a lot, actually... and which would be a *real* lot without OWL!)
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
  std::cout << "- creating raygen program object...\n";
  OWLRayGen rg  = owlRayGenCreate(owl,mod,"helloOWL",
                                  0,
                                  nullptr,0);
  assert(mod && "could not create raygen");

  // ------------------------------------------------------------------
  std::cout << "- building actual device programs...\n";
  owlBuildPrograms(owl);
  
  // ------------------------------------------------------------------
  std::cout << "- building RTX pipeline...\n";
  owlBuildPipeline(owl);

  // ------------------------------------------------------------------
  std::cout << "- building SBT (no geoms, but raygen is part of SBT, too!)...\n";
  owlBuildSBT(owl);
  
  // ------------------------------------------------------------------
  std::cout << "- launching raygen...\n\n";
  owlRayGenLaunch2D(rg,/* launch with launch dims of 100x100: */100,100);
  assert(mod && "could not create raygen");

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
