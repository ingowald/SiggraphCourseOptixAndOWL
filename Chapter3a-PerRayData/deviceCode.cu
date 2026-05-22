// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "deviceCode.h"

__constant__ OptixGlobals optixLaunchParams;

struct PerRayData {
  bool hadHit = false;
  struct {
    vec3f normal;
    vec3f baseColor;
  } hit;
};

OPTIX_CLOSEST_HIT_PROGRAM(TrianglesCH)()
{
  auto self = owl::getProgramData<TrianglesGeomData>();
  // ask owl for a (writeable reference to) the per-ray data we passed
  // in raygen. note the template type here _has_ to match what you
  // actually provided; getPRD will simply do a typecast with whatever
  // type you specify here.
  auto &prd = owl::getPRD<PerRayData>();

  int primID = optixGetPrimitiveIndex();
  int instID = optixGetInstanceIndex();
  vec3i triangle = self.indices[primID];
  vec3f a = self.vertices[triangle.x];
  vec3f b = self.vertices[triangle.y];
  vec3f c = self.vertices[triangle.z];
  vec3f n = cross(b-a,c-a);

  n = optixTransformNormalFromObjectToWorldSpace(n);

#if 1
  // "geom ID" shading - color primarily from object, no matter which
  // instnace of it it we hit.
  int geomID = optixGetSbtGASIndex();
  vec3f baseColor
    = 0.2f*owl::randomColor(primID)
    + 0.8f*owl::randomColor(geomID);
#else
  // "instance ID" shading - shade color primarily from instance index
  vec3f baseColor
    = 0.2f*owl::randomColor(primID)
    + 0.8f*owl::randomColor(instID);
#endif

  // since prd is a *reference* to raygen's local data these changes
  // will be visible to raygen:
  prd.hadHit        = true;
  prd.hit.baseColor = baseColor;
  prd.hit.normal    = n;
}

OPTIX_RAYGEN_PROGRAM(renderFrame)()
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

  /* *************************************************************** */
  // DELTA TO PREV SAMPLE: we use per-ray data in this example.  The
  // way this works is that the raygen can define and locally
  // pre-initialize a local struct of whatevr members it desires, then
  // pass that to owl::traceRay(); traceray will then encode a
  // pointer/reference to this struct that the ray carries along using
  // optix PRD registers, and the respective AH/CH program(s) can then
  // get a refernce to this struct using owl::getPRD<>() (see usage of
  // that in the CH program above)
  /* *************************************************************** */

  // create PRD, and pre-initialize:
  PerRayData prd;
  prd.hadHit = false;
  /* pass (reference to) PRD as third param; traceRay() is templated
     over the type of that third param (optixTrace only needs a
     pointer, anyway), so you can pass whatever struct type want */
  owl::traceRay(optixLaunchParams.world,ray,/* --->>> */prd);

  /* *************************************************************** */
  // DELTA TO PREV SAMPLE: now we do all the actual shading and frame
  // buffer stuff in a single location, at the end of raygen.
  /* *************************************************************** */
  uint32_t pixelValue = 0;
  if (!prd.hadHit) {
    const uint32_t bgColor = 0xffdddddd;
    pixelValue = bgColor;
  } else {
    float eyelightTerm
      = fabsf(dot(normalize(ray.direction),
                  normalize(prd.hit.normal)));
    vec3f shadeColor
      = (.2f+.6f*eyelightTerm)*prd.hit.baseColor;
    // poor-man's gamma correction
    shadeColor = sqrt(shadeColor);
    pixelValue = owl::make_rgba(shadeColor);
  }
  fb.data[pixel.x+pixel.y*fb.size.x] = pixelValue;
}

