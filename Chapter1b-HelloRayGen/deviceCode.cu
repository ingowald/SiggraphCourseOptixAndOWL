// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "owl/owl_device.h"
#include "owl/common/owl-common.h"

OPTIX_RAYGEN_PROGRAM(helloOWL)()
{
  if (optixGetLaunchIndex().x == 0 &&
      optixGetLaunchIndex().y == 0)
    {
      printf(OWL_TERMINAL_GREEN);
      printf("********************************************\n");
      printf("Hello OWL (from device!)\n");
      printf("********************************************\n");
      printf(OWL_TERMINAL_DEFAULT);
    }
}

