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
  auto self = owl::getProgramData<TrianglesGeomData>();

  int primID = optixGetPrimitiveIndex();
  vec3i triangle = self.indices[primID];
  vec3f a = self.vertices[triangle.x];
  vec3f b = self.vertices[triangle.y];
  vec3f c = self.vertices[triangle.z];
  vec3f n = normalize(cross(b-a,c-a));

  vec3f dir = optixGetWorldRayDirection();
  dir = normalize(dir);
  vec3f baseColor
    = 0.2f*owl::randomColor(primID)
    + 0.8f*self.baseColor;
  vec3f shadeColor = (.2f+.6f*fabsf(dot(n,dir)))*baseColor;


  auto fb = optixLaunchParams.frameBuffer;
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

