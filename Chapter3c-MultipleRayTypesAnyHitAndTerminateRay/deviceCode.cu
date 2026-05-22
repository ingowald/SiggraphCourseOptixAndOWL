// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "deviceCode.h"

__constant__ OptixGlobals optixLaunchParams;

// use owl's templating to define the two ray types. note (and this is important):
//
// a) this use of Ray<x,2> ONLY makes sense if the app also calls
// owlContextSetRayTypeCount(2), and properly sets the different ray
// types' different AH/CH programs (see hostCode.cu)
//
// b) since our hostcode does set rayTypeCount to 2 we _have_ to use
// these two ray types that also use numRayTypes=2 in the device
// code. If we used the same owl::Ray (which defaults to 1 ray type)
// we'd get wrong results because a one-ray-tyep Ray<> would then
// index into a two-ray-type SBT.

typedef owl::RayT</* ray type 0 of 2:*/0,2> RegularRay;
typedef owl::RayT</* ray type 1 of 2:*/1,2> ShadowRay;


// in this sample, we use two differnet ray types; just to show that
// this possible we also use different PRD types for each type of
// ray. this one is for shadow rays - we'll eventually extend that to
// fractional occlusion (through partially transparent surfaces), but
// for now we'll juse use a binary is/isn't occluded, just to show how
// it works.
struct ShadowRayPRD {
  bool isOccluded = false;
};

// the same per-ray data we had in the previous sample (without the
// 'isShadowRay' tag - this will only ever get used on non-shadow
// rays, anyway, using the ray type mechanism
struct RegularRayPRD {
  bool hadHit = false;
  struct {
    /*! WORLD space position */
    vec3f position;
    /*! WORLD space normal */
    vec3f normal;
    vec3f baseColor;
  } hit;
};

OPTIX_CLOSEST_HIT_PROGRAM(Triangles_CH_RegularRays)()
{
  auto &prd = owl::getPRD<RegularRayPRD>();

  auto self = owl::getProgramData<TrianglesGeomData>();

  int primID = optixGetPrimitiveIndex();
  int instID = optixGetInstanceIndex();
  vec3i triangle = self.indices[primID];
  // get vertex positions (in object space, if we use instancing)
  vec3f a = self.vertices[triangle.x];
  vec3f b = self.vertices[triangle.y];
  vec3f c = self.vertices[triangle.z];
  // transform to world space
  a = optixTransformPointFromObjectToWorldSpace(a);
  b = optixTransformPointFromObjectToWorldSpace(b);
  c = optixTransformPointFromObjectToWorldSpace(c);
  vec3f n = cross(b-a,c-a);

#if 1
  // compute hit point through trilerp of (world space)
  // vertices. could also compute through origin+tHit*direction, but
  // this way is more numerically stable (see ray tracing gem by
  // carsten waechter)
  vec2f uv = optixGetTriangleBarycentrics();
  vec3f p = (1.f-uv.x-uv.y)*a + uv.x*b + uv.y*c;
#else
  vec3f p
    = optixGetWorldRayOrigin()
    + optixGetRayTmax()
    * vec3f(optixGetWorldRayDirection());
#endif
  int geomID = optixGetSbtGASIndex();
  vec3f baseColor
    = 0.2f*owl::randomColor(primID)
    + 0.8f*owl::randomColor(geomID);

  // since prd is a *reference* to raygen's local data these changes
  // will be visible to raygen:
  prd.hadHit        = true;
  prd.hit.baseColor = baseColor;
  prd.hit.normal    = n;
  prd.hit.position  = p;
}

OPTIX_ANY_HIT_PROGRAM(Triangles_AH_RegularRays)()
{
  /* not doing anything, yet; we'll fill this out in a future
     sample */
}

OPTIX_CLOSEST_HIT_PROGRAM(Triangles_CH_ShadowRays)()
{
  /* for shadow rays we won't even need a CH program, since we'll do
     all the work in the AH program; but for didactic purposes we
     leave this in here so user can see how to do it _if_ it is ever
     needed */
}

// AH program we use for (only) shadow rays
OPTIX_ANY_HIT_PROGRAM(Triangles_AH_ShadowRays)()
{
  auto &prd = owl::getPRD<ShadowRayPRD>();
  /* for now, we don't yet handle semi-transparent hits - we'll add
     this in a later sample - so let's simply set this to occluded */
  prd.isOccluded = true;

  /* optimization: once we tagged the ray as occluded there's no need
     to trace it any further (it won't get unoccluded from here on
     out), so let's tell optix to stop searching. */
  optixTerminateRay();
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
  
  // generate the OWL ray, and trace it into the scene. Note we _have_
  // to use the ray types that use numRayTypes=2 in this sample
  // because the host does build an SBT for two ray types. If we used
  // a regular owl::Ray (which defaults to 1 ray type) we'd get wrong
  // results because a one-ray-tyep Ray<> would then index into a
  // two-ray-type SBT.
  RegularRay ray(rayOrigin,rayDirection,0.f,CUDART_INF);

  // ------------------------------------------------------------------
  // first trace: this is the primary ray from the camera to find the
  // surface we're going to shade
  // ------------------------------------------------------------------
  RegularRayPRD prd{};
  owl::traceRay(optixLaunchParams.world,ray,/* --->>> */prd);

  uint32_t pixelValue = 0;
  if (!prd.hadHit) {
    // ------------------------------------------------------------------
    // primary ray didn't hit anything; shade with background color,
    // and no need to shoot any shadow rays
    // ------------------------------------------------------------------
    const uint32_t bgColor = 0xffdddddd;
    pixelValue = bgColor;
  } else {
    // ------------------------------------------------------------------
    // primary ray *did* hit something; let's shoot a shadow ray from
    // where we hit. for now we'll do a hardcoded directional light
    // ------------------------------------------------------------------

    // get normal from the PRD, normalize, and face-forward it
    vec3f Ng = normalize(prd.hit.normal);
    if (dot(Ng,rayDirection) > 0.f)
      Ng = -Ng;
    
    // second, offset the hit point a tiny bit in normal direction to
    // make sure we're not self-shadowing on the surface we're
    // shading.  use a normal that's relative to world-space position
    // so as to always offset relative to scale of the world
    float offset
      = 1e-6f*reduce_max(abs(prd.hit.position));
    vec3f shadowRayOrigin
      = prd.hit.position + offset * Ng;
    // hardcoded light for now:
    vec3f lightDir = normalize(vec3f(.1f,1.f,.1f));
    float lightRadiance = 1.f;

    // ------------------------------------------------------------------
    // now: construct and trace a shadow ray. we'll just re-use the
    // same PRD we used for the primary rays
    // ------------------------------------------------------------------
    ShadowRay shadowRay(shadowRayOrigin,lightDir,0.f,CUDART_INF);
    ShadowRayPRD shadowPRD{};
    owl::traceRay(optixLaunchParams.world,shadowRay,shadowPRD);
    bool isIlluminated = !shadowPRD.isOccluded;

    // compute one term for ambient illumiation, using the same
    // 'eyelight' term as before, just toned down by 10x.
    rayDirection = normalize(rayDirection);
    float ambientEyelightTerm 
      = .05 + .1f * -dot(Ng,ray.direction);
    float lightTerm
      = isIlluminated
      ? lightRadiance * max(0.f,dot(normalize(lightDir),Ng))
      : 0.f;
    vec3f shadeColor
      = (ambientEyelightTerm+lightTerm)*prd.hit.baseColor;
    // poor-man's gamma correction
    shadeColor = sqrt(shadeColor);
    // ...clamp...
    shadeColor = min(shadeColor,vec3f(1.f));
    // ... and convert to rgba8
    pixelValue = owl::make_rgba(shadeColor);
  }
  fb.data[pixel.x+pixel.y*fb.size.x] = pixelValue;
}

