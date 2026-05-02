// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "deviceCode.h"

__constant__ OptixGlobals optixLaunchParams;

OPTIX_MISS_PROGRAM(missProg)()
{
  auto fb = optixLaunchParams.frameBuffer;
  // slightly off-white background:
  uint32_t bgColor = 0xffdddddd;
  
  vec2i pixel = owl::getLaunchIndex();
  fb.data[pixel.x+pixel.y*fb.size.x] = bgColor;
}

OPTIX_CLOSEST_HIT_PROGRAM(TrianglesCH)()
{
  auto fb = optixLaunchParams.frameBuffer;
#if 0
  // set pixel to some pseudo-random color based on prim
  // ID. owl-device has some helper functions for doing this, as well
  // as for converting from float3 color to rgba8.
  int primID = optixGetPrimitiveIndex();
  vec3f shadeColor = owl::randomColor(primID);
#else
  // set pixel color to barycentric coordinates of hit
  vec2f bary = optixGetTriangleBarycentrics(); 
  vec3f shadeColor = { bary.x, bary.y, 1.f-bary.x-bary.y };
#endif
  
  // poor-man's "gamma correction"
  shadeColor = sqrt(shadeColor);
  
  vec2i pixel = owl::getLaunchIndex();
  fb.data[pixel.x+pixel.y*fb.size.x] = owl::make_rgba(shadeColor);
}

OPTIX_RAYGEN_PROGRAM(renderTestFrame)()
{
  auto fb = optixLaunchParams.frameBuffer;
  
  vec2i pixel = owl::getLaunchIndex();
  if (pixel.x >= fb.size.x) return;
  if (pixel.y >= fb.size.y) return;

  // relative position on the image, in [0,1]^2
  float u = (pixel.x+.5f) / fb.size.x;
  float v = (pixel.y+.5f) / fb.size.y;

  // hard-coded camera for a camera at y=-2, and a 2x2 sized image
  // plane around the origin (ie, from x=-1,z=-1 to x=+1,z=+1)
  auto camera = optixLaunchParams.camera;
  vec3f rayOrigin
    = camera.position;
  vec3f rayDirection
    = camera.dir_00
    + u * camera.dir_du
    + v * camera.dir_dv;
  
  // generate the OWL ray, and trace it into the scene
  owl::Ray ray(rayOrigin,rayDirection,0.f,CUDART_INF);
  
  // we'll introduce what per ray data is in the next example, but
  // already need 'something' to pass to traceRays(), so let's just
  // use some dummy integer value.
  int someDummyPerRayData = 0;
  owl::traceRay(optixLaunchParams.world,ray,someDummyPerRayData);

  // in this example, there is nothing more to do here - the CH and
  // miss programs write into the frame buffer directly, so our job
  // here is done
}

