// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// the actual OWL API
#include "owl/owl.h"
// some owl helper functions we will use (eg pretty printing)
#include "owl/common/owl-common.h"

int main(int ac, char **av)
{
  std::cout << OWL_TERMINAL_LIGHT_BLUE << R"(
**********************************************************************
Chapter1a-HelloOWL:
**********************************************************************
            )" << OWL_TERMINAL_DEFAULT;
  
  std::cout << R"(
This sample won't do anything useful (nor produce any outputs beyond
these messages); it primarily exists to allow for checking whether the
host system contains everything to build and run an OptiX/OWL program

So: Here we go - trying to create an OWL context. If this fails,
please make sure to check that you actually have a OptiX-capable
driver running.
)";
  
  // create an OWL context, on GPU 0
  int gpuID = 0;
  OWLContext owl = owlContextCreate(&gpuID,1);
  
  std::cout << OWL_TERMINAL_LIGHT_BLUE << R"(
Hello OWL!!!

OWL Context is now active, and we could now use this to create
geometries, build BVHes, etc...  We'll do this in a later sample, for
now this will just shut down again...
)" << OWL_TERMINAL_DEFAULT;


  owlContextDestroy(owl);
  
  std::cout << OWL_TERMINAL_LIGHT_GREEN << R"(
All good! - if you see this message we could properly create and take
down a OWL context... your system is ready for use with OptiX and OWL!
)" << OWL_TERMINAL_DEFAULT << std::endl;
  return 0;
}
