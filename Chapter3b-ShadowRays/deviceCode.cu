// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "deviceCode.h"

__constant__ OptixGlobals optixLaunchParams;

struct PerRayData {
  bool hadHit = false;
  /*! we could also use two different ray types (OWL can do that),
      then each could have had its own PRD, and its own set of H/AH
      programs ...but this is the much easier solution... */
  bool isShadowRay;
  struct {
    /*! WORLD space position */
    vec3f position;
    /*! WORLD space normal */
    vec3f normal;
    vec3f baseColor;
  } hit;
};

OPTIX_CLOSEST_HIT_PROGRAM(TrianglesCH)()
{
  auto &prd = owl::getPRD<PerRayData>();

  // for shadow rays we don't need position etc, so let's just
  // early-out here. if we had used different ray types we'd have had
  // the shadow ray's CH prog do only that, too
  if (prd.isShadowRay) { prd.hadHit = true; return; }
  
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

  // ------------------------------------------------------------------
  // first trace: this is the primary ray from the camera to find the
  // surface we're going to shade
  // ------------------------------------------------------------------
  PerRayData prd;
  prd.hadHit = false;
  prd.isShadowRay = false;
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
    owl::Ray shadowRay(shadowRayOrigin,lightDir,0.f,CUDART_INF);
    prd.hadHit = false;
    prd.isShadowRay = true;
    owl::traceRay(optixLaunchParams.world,
                  shadowRay,
                  prd);
    bool isIlluminated = !prd.hadHit;

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

