// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "deviceCode.h"
// include own helper for random numbers
#include <owl/common/math/random.h>

__constant__ OptixGlobals optixLaunchParams;

// use owl's templating to define the two ray types. See earlier
// examples for explanation
typedef owl::RayT</* ray type 0 of 2:*/0,2> RegularRay;
typedef owl::RayT</* ray type 1 of 2:*/1,2> ShadowRay;

struct ShadowRayPRD {
  bool isOccluded = false;
};

/*! how many diffuse bounces we want to do */
#define MAX_DIFFUSE_BOUNCES 2

struct RegularRayPRD {
  bool hadHit = false;
  struct {
    vec3f position;
    vec3f normal;
  } hit;
  // this - obviously - isn't a real BRDF, yet, but this is where this
  // would be going in a more serious renderer
  struct {
    vec3f diffuse;
  } brdf;
};

using RNG = owl::common::LCG<4>;

inline __device__
vec3f uniformRandomDirection(RNG &rng)
{
  // NOTE: this is the 'pete shirley' style hacky way of generating a
  // random direction. it's not exactly the best method for doing
  // this, but easy to understand so for didacting reaons let's do it
  // this way. If you're going to use that as a baseline for your own
  // renderer, please read up on sampling spheres, hemispheres, etc,
  // for example by reading Phil Dutre's (still!) excellent "Global
  // Illumination Compendium".
  while (true) {
    vec3f U(rng(),rng(),rng());
    vec3f D = 2.f*U - 1.f;
    if (dot(D,D) > 1.f) /* too long, try again */continue;
    return normalize(D);
  }
}

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

  // compute hit point through trilerp of (world space)
  // vertices. could also compute through origin+tHit*direction, but
  // this way is more numerically stable (see ray tracing gem by
  // carsten waechter)
  vec2f uv = optixGetTriangleBarycentrics();
  vec3f p = (1.f-uv.x-uv.y)*a + uv.x*b + uv.y*c;
  int geomID = optixGetSbtGASIndex();

  vec3f baseColor = self.baseColor;
  if (self.colorTexture) {
    vec2f tc = uv;
    if (self.texcoords) {
      tc
        = (1.f-uv.x-uv.y) * self.texcoords[triangle.x]
        +      uv.x       * self.texcoords[triangle.y]
        +           uv.y  * self.texcoords[triangle.z];
    }
    // use plain cuda tex3D<>():
    float4 textureColor = tex2D<float4>(self.colorTexture,tc.x,tc.y);
    // for now, ignore alpha component - we'll deal with this in next
    // sample. 
    baseColor = (const vec3f&)textureColor;
  }

  // since prd is a *reference* to raygen's local data these changes
  // will be visible to raygen:
  prd.hadHit        = true;
  prd.brdf.diffuse  = baseColor;
  prd.hit.normal    = n;
  prd.hit.position  = p;
}

OPTIX_ANY_HIT_PROGRAM(Triangles_AH_RegularRays)()
{
  auto self = owl::getProgramData<TrianglesGeomData>();
  if (!self.colorTexture)
    // no color texture, nothing to skip - this does nothing, which
    // means the hit gets accepted, and will get reported to CH
    // (unless anything closer still gets fround after this .
    return;

  vec2f uv = optixGetTriangleBarycentrics();
  vec2f tc = uv;
  if (self.texcoords) {
    int primID = optixGetPrimitiveIndex();
    vec3i triangle = self.indices[primID];
    tc
      = (1.f-uv.x-uv.y) * self.texcoords[triangle.x]
      +      uv.x       * self.texcoords[triangle.y]
      +           uv.y  * self.texcoords[triangle.z];
  }
  float4 texSample = tex2D<float4>(self.colorTexture,tc.x,tc.y);
  
  // now, check alpha channel - we'll use a simple hard-coded cutoff
  // at 50% transparenty to decide whether or not surface is
  // opaque. Would usually use some random number generator and decide
  // this via monte carlo, but for this simple sample this will do.
  if (texSample.w < .5f) {
    // surface is more than 50% transparent here - let's tell optix to
    // ignore this hit. This will also terminate the AH program.
    optixIgnoreIntersection();
  } else {
    // surface is mostly opaque; let optix accept it as potential
    // closest hit. Note this means we might do the same texture
    // lookup *again* in the CH program (to get RGB); that's not ideal
    // but ... shrug
  }
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
  ShadowRayPRD &prd = owl::getPRD<ShadowRayPRD>();
  auto self = owl::getProgramData<TrianglesGeomData>();
  if (!self.colorTexture) {
    // doesn't even have a texture; can't be transparent: mark as
    // occluded and end traversal.
    prd.isOccluded = true;
    optixTerminateRay();
    // not strictly necessary - terminatRay() should end the program
    // anyway, but this is easier on the eye.
    return;
  }

  vec2f uv = optixGetTriangleBarycentrics();
  vec2f tc = uv;
  if (self.texcoords) {
    int primID = optixGetPrimitiveIndex();
    vec3i triangle = self.indices[primID];
    tc
      = (1.f-uv.x-uv.y) * self.texcoords[triangle.x]
      +      uv.x       * self.texcoords[triangle.y]
      +           uv.y  * self.texcoords[triangle.z];
  }
  float4 texSample = tex2D<float4>(self.colorTexture,tc.x,tc.y);
  if (texSample.w > 0.5f) {
    // texture is more than 50% opaque - let's accept as occluded and
    // kill. for true semi-transparent shadows we'd want to accumulate
    // rgb transmittance here (and also consider sources of
    // transparency other than texture.w), but our test models don't
    // have that.
    prd.isOccluded = true;
    optixTerminateRay();
    // not strictly necessary - terminatRay() should end the program
    // anyway, but this is easier on the eye.
    return;
  } else {
    // texture is mostly transparent here - let's NOT mark as
    // occluded, and tell optix to ignore this hit (so it
    optixIgnoreIntersection();
  }
}


OPTIX_RAYGEN_PROGRAM(renderFrame)()
{
  auto &lp = optixLaunchParams;
  auto  fb = lp.frameBuffer;
  
  vec2i pixel = owl::getLaunchIndex();
  if (pixel.x >= fb.size.x) return;
  if (pixel.y >= fb.size.y) return;

  // delta: initialize a random number generator based on pixel coord
  RNG rng(pixel.x+lp.frameID*fb.size.x,
          pixel.y+lp.frameID*fb.size.y);
  
  // relative position on the image, in [0,1]^2.
#if 0
  // from previous sample: trace through center of each pixel
  float u = (pixel.x+.5f) / fb.size.x;
  float v = (pixel.y+.5f) / fb.size.y;
#else
  /* delta: instead of tracing throuwe now use a 'random' offset in
     the pixel we use a rng-chosen offset */
  float u = (pixel.x+rng()) / fb.size.x;
  float v = (pixel.y+rng()) / fb.size.y;
#endif

  // hard-coded camera for a camera at y=-2, and a 2x2 sized image
  // plane around the origin (ie, from x=-1,z=-1 to x=+1,z=+1)
  auto camera = lp.camera;
  vec3f rayOrigin
    = camera.position;
  vec3f rayDirection
    = camera.dir_00
    + u * camera.dir_du
    + v * camera.dir_dv;

  RegularRay ray(rayOrigin,rayDirection,0.f,CUDART_INF);

  // ------------------------------------------------------------------
  // first trace: this is the primary ray from the camera to find the
  // surface we're going to shade
  // ------------------------------------------------------------------
  RegularRayPRD prd{};
  owl::traceRay(lp.world,ray,/* --->>> */prd);

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

    int numBounces = 0;
    vec3f pathThroughput = 1.f;
    vec3f radiance = 0.f;
    while (true) {
      // ==================================================================
      // FIRST: 'localize' the hit point, normal, brdf, etc
      // ==================================================================
      
      // get normal from the PRD, normalize, and face-forward it
      vec3f Ng = normalize(prd.hit.normal);
      if (dot(Ng,rayDirection) > 0.f)
        Ng = -Ng;

      /* world-space shade point */
      vec3f P = prd.hit.position;
      /*! how much we offset shade point from surface */
      float offset = 1e-6f*reduce_max(abs(prd.hit.position));
      P += offset * Ng;
      
      // ==================================================================
      // SECOND (for this sample): do explicit shadow(s) to any light
      // sources we want to sample directly (in this example, we only
      // do a single dirlight, but you might also have other non-area
      // lights, or might even want to sample some HDRI light,
      // etc). Remember this is not a course in advanced global
      // illumination, this is just a sample path tracer to show some
      // example of how one could be written in optix/owl.
      // ==================================================================

      // hardcoded dirlight for now:
      vec3f lightDir = normalize(vec3f(.1f,1.f,.1f));
      float lightRadiance = .5f;

      vec3f lightTerm
        = pathThroughput * prd.brdf.diffuse * dot(lightDir,Ng) * lightRadiance;
      if (reduce_max(lightTerm) > 1e-3f) {
        // trace shadow ray only if there is 'some' contribution...
        ShadowRay shadowRay(P,lightDir,0.f,CUDART_INF);
        ShadowRayPRD shadowPRD{};
        owl::traceRay(lp.world,shadowRay,shadowPRD);
        if (!shadowPRD.isOccluded)
          radiance += lightTerm;
      }

#if FOR_ILLUSTRATION_ONLY
      // allows for showing how image looks _wihtout_ taking secondary
      // illumination into account; in this case we only consider the
      // direct ligthing, and exit.
      if (lp.disablePathTracing)
        break;
#endif
      
      // ==================================================================
      // THIRD (for this sample): sample the 'brdf' (only diffuse for
      // now) to get a new outgoing ray direction, and adjust path
      // trhoughput accordingly
      // ==================================================================
      
      // for a diffuse bounce, just generate uniform random direction ...
      vec3f bounceDirection = uniformRandomDirection(rng);
      // ... in same hemisphere as face-forwarded surface normal
      if (dot(bounceDirection,Ng) < 0.f)
        bounceDirection = -bounceDirection;

      vec3f brdfTerm = prd.brdf.diffuse * dot(bounceDirection,Ng);
      pathThroughput *= brdfTerm;

      // quick check if path throughput even matters any more:
      if (reduce_max(pathThroughput) < 1e-3f) break;

      // genearte new outgoing ray:
      ray = RegularRay(P,bounceDirection,0.f,CUDART_INF);
      prd.hadHit = false;

      // trace outgoing ray:
      owl::traceRay(lp.world,ray,/* --->>> */prd);
      if (!prd.hadHit) {
        // bounce ray did NOT hit anything - 'shade' it with the
        // environment illumination. "eventually" we'll want to get
        // that from the HDRI map....
        vec3f radianceFromEnv = 1.f;
        radiance += pathThroughput * radianceFromEnv;
        // and done, because there's no surface to bounce.
        break;
      }

      if (++numBounces > MAX_DIFFUSE_BOUNCES)
        // too many bounces - assume this path got lost somewhere in
        // the geometry and won't ever make it out.
        break;
      
      // bounce was valid AND found a valid hit, let's process it in
      // the next iteration.
    }
    
    // poor-man's gamma correction
    radiance = sqrt(radiance);
    // ...clamp...
    radiance = min(radiance,vec3f(1.f));
    // ... and convert to rgba8
    pixelValue = owl::make_rgba(radiance);
  }
  fb.data[pixel.x+pixel.y*fb.size.x] = pixelValue;
}

