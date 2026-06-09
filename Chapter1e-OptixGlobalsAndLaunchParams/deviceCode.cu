// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "deviceCode.h"

__constant__ OptixGlobals optixLaunchParams;

OPTIX_RAYGEN_PROGRAM(renderTestFrame)()
{
  auto fb = optixLaunchParams.frameBuffer;
  
  vec2i pixel = owl::getLaunchIndex();
  if (pixel.x >= fb.size.x) return;
  if (pixel.y >= fb.size.y) return;

  // compute rgba values for a simple test pattern
  uint32_t r = pixel.x & 0xff;
  uint32_t g = pixel.y & 0xff;
  uint32_t b = (pixel.x+pixel.y) & 0xff;
  uint32_t a = 0xff;
  
  uint32_t rgba8
    = (r << 0)
    | (g << 8)
    | (b << 16)
    | (a << 24);
  
  fb.data[pixel.x+pixel.y*fb.size.x] = rgba8;

if (pixel == vec2i(0,0)) {
for (int i=0;i<10;i++)
   optixLaunchParams.managedMem[i] = i+optixLaunchParams.frameID;
   printf("on device: wrote %p[0] = %i\n",optixLaunchParams.managedMem,
   optixLaunchParams.managedMem[0]);
}
}

