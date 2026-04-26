// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "deviceCode.h"

__constant__ OptixGlobals optixLaunchParams;

OPTIX_MISS_PROGRAM(missProg)()
{
  auto fb = optixLaunchParams.frameBuffer;
  // full white background:
  uint32_t bgColor = 0xffffffff;
  
  vec2i pixel = owl::getLaunchIndex();
  fb.data[pixel.x+pixel.y*fb.size.x] = bgColor;
}

OPTIX_CLOSEST_HIT_PROGRAM(TrianglesCH)()
{
  auto fb = optixLaunchParams.frameBuffer;
  auto self = owl::getProgramData<TrianglesGeomData>();

  int primID = optixGetPrimitiveIndex();
  vec3i triangle = self.indices[primID];
  vec3f a = self.vertices[triangle.x];
  vec3f b = self.vertices[triangle.y];
  vec3f c = self.vertices[triangle.z];
  vec3f n = normalize(cross(b-a,c-a));

  // set pixel to some pseudo-random color based on prim
  // ID. owl-device has some helper functions for doing this, as well
  // as for converting from float3 color to rgba8.
  
  vec3f dir = optixGetWorldRayDirection();
  dir = normalize(dir);
  vec3f baseColor
    = 0.2f*owl::randomColor(primID)
    + 0.8f*vec3f(1.f);
  vec3f shadeColor = (.2f+.6f*fabsf(dot(n,dir)))*baseColor;

  vec2i pixel = owl::getLaunchIndex();
  fb.data[pixel.x+pixel.y*fb.size.x] = owl::make_rgba(shadeColor);
}

OPTIX_RAYGEN_PROGRAM(renderTestFrame)()
{
  auto &lp = optixLaunchParams;
  auto fb = lp.frameBuffer;
  
  vec2i pixel = owl::getLaunchIndex();
  if (pixel.x >= fb.size.x) return;
  if (pixel.y >= fb.size.y) return;

  // generate the OWL ray, and trace it into the scene
  vec2f pixelSample = { pixel.x+.5f, pixel.y+.5f };
  vec3f origin
    = lp.camera.position;
  vec3f direction
    = lp.camera.dir_00
    + (pixel.x+.5f) * lp.camera.dir_du
    + (pixel.y+.5f) * lp.camera.dir_dv;
  if (pixel == vec2i(0,0)) {
    printf("origin %f %f %f\n",
           origin.x,
           origin.y,
           origin.z);
    printf("direction %f %f %f\n",
           direction.x,
           direction.y,
           direction.z);
  }
  owl::Ray ray(origin,direction,0.f,CUDART_INF);
  
  // we'll introduce what per ray data is in the next example, but
  // already need 'something' to pass to traceRays(), so let's just
  // use some dummy integer value.
  int someDummyPerRayData = 0;
  owl::traceRay(optixLaunchParams.world,ray,someDummyPerRayData);

  // in this example, there is nothing more to do here - the CH and
  // miss programs write into the frame buffer directly, so our job
  // here is done
}

