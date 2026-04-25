// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "deviceCode.h"

OPTIX_RAYGEN_PROGRAM(renderTestFrame)()
{
  RayGenData self = owl::getProgramData<RayGenData>();

  vec2i pixelIdx = owl::getLaunchIndex();
  if (pixelIdx.x >= self.frameBuffer.size.x) return;
  if (pixelIdx.y >= self.frameBuffer.size.y) return;

  // compute rgba values for a simple test pattern
  uint32_t r = pixelIdx.x & 0xff;
  uint32_t g = pixelIdx.y & 0xff;
  uint32_t b = (pixelIdx.x+pixelIdx.y) & 0xff;
  uint32_t a = 0xff;
  
  uint32_t pixelValue_rgba8
    = (r << 0)
    | (g << 8)
    | (b << 16)
    | (a << 24);
  
  int fbOfs
    = pixelIdx.x
    + pixelIdx.y * self.frameBuffer.size.x;
  self.frameBuffer.data[fbOfs] = pixelValue_rgba8;
}

