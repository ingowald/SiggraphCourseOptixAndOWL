// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// the actual OWL API
#include "owl/owl.h"
// some owl helper functions we will use (eg pretty printing)
#include "owl/common/owl-common.h"

int main(int ac, char **av)
{
  std::cout << OWL_TERMINAL_LIGHT_BLUE
            << "*********************************************************\n"
            << "Chapter1a-HelloOWL:\n"
            << "*********************************************************\n"
            << OWL_TERMINAL_DEFAULT;
  std::cout << "\n";
  std::cout << "This sample won't do anything useful (nor produce\n";
  std::cout << "any outputs beyond these messages); it primarily exists\n";
  std::cout << "to allow for checking whether the host system contains\n";
  std::cout << "everything to build and run an OptiX/OWL program.\n";
  std::cout << "\n";

  // create an OWL context, on GPU 0
  int gpuID = 0;
  std::cout << OWL_TERMINAL_LIGHT_BLUE
            << "Here we go - trying to create an OWL context.\n"
            << "(if this fails, please make sure to check that you\n"
            << "actually have a OptiX-capable driver running).\n\n"
            << OWL_TERMINAL_DEFAULT;
  OWLContext owl = owlContextCreate(&gpuID,1);
  std::cout << OWL_TERMINAL_GREEN
            << "Hello OWL!!!\n\n"
            << OWL_TERMINAL_DEFAULT;
  std::cout << "OWL Context is now active, and we could now use\n"
            << "this to create geometries, build BVHes, etc...\n"
            << "We'll do this in a later sample, for now this will\n" 
            << "just shut down again...\n\n";
  owlContextDestroy(owl);
  std::cout << OWL_TERMINAL_GREEN
            << "All good! - if you see this message we could properly\n"
            << "create and take down a OWL context... your system is\n"
            << "ready for use with OptiX and OWL!\n"
            << OWL_TERMINAL_DEFAULT << std::endl;
  return 0;
}
