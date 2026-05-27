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
#define MAX_SPECULAR_BOUNCES 16

inline __device__ bool dbg()
{ return vec2i(optixGetLaunchIndex()) == vec2i(optixGetLaunchDimensions())/2; }

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
    // index of refraction from the disney brdf. if equal to 1.f we'll
    // treat this as a diffuse surface in these samples; otherwise we
    // ignore basecolor and treat this as glass/water with this IOR
    float ior;
  } brdf;
};

using RNG = owl::common::LCG<4>;

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
  vec2f tc = uv;
  if (self.texcoords) {
    tc
      = (1.f-uv.x-uv.y) * self.texcoords[triangle.x]
      +      uv.x       * self.texcoords[triangle.y]
      +           uv.y  * self.texcoords[triangle.z];
  }
  if (self.colorTexture) {
    // use plain cuda tex3D<>():
    float4 textureColor = tex2D<float4>(self.colorTexture,tc.x,tc.y);
    // for now, ignore alpha component - we'll deal with this in next
    // sample. 
    baseColor = (const vec3f&)textureColor;
  }
  if (self.normals) {
    n 
      = (1.f-uv.x-uv.y) * self.normals[triangle.x]
      +      uv.x       * self.normals[triangle.y]
      +           uv.y  * self.normals[triangle.z];
    n = optixTransformNormalFromObjectToWorldSpace(n);
  }
  
  // since prd is a *reference* to raygen's local data these changes
  // will be visible to raygen:
  prd.hadHit        = true;
  prd.brdf.diffuse  = baseColor;
  prd.brdf.ior      = self.ior;
  prd.hit.normal    = normalize(n);
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

  if (self.ior != 1.f) {
    // for this simple set of sample(s) we treat all surfaces with IOR
    // other than 1.f as if they were glass/water. for these, we'll
    // simply have shadow rays pass straight rhough
    optixIgnoreIntersection();
    return;
  }
  
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

/*! largely stolen from pete shirley's RTOW samples */
struct ScatterEvent {
  struct {
    vec3f P;
    vec3f N;
    vec3f dir;
  } incoming;
  struct{
    vec3f dir;
    vec3f atten;
  } outgoing;
};

/*! stolen from pete shirley's RTOW samples - we'll use this on
  diffuse surfaces */
struct Lambertian {
  inline __device__ bool scatter(ScatterEvent &event, RNG &rng);
  vec3f albedo;
};

/*! stolen from pete shirley's RTOW samples - we'll use this on
  water/glass */
struct Dielectric {
  inline __device__ bool scatter(ScatterEvent &event, RNG &rng);
  float ior;
};

inline __device__
float schlick(float cosine,
              float ref_idx)
{
  float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
  r0 = r0 * r0;
  return r0 + (1.0f - r0)*powf((1.0f - cosine), 5.0f);
}

inline __device__
bool refract(const vec3f &v,
             const vec3f &n,
             float ni_over_nt,
             vec3f &refracted)
{
  vec3f uv = normalize(v);
  float dt = dot(uv, n);
  float discriminant = 1.f - ni_over_nt * ni_over_nt*(1.f - dt * dt);
  if (discriminant > 0.f) {
    refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
    return true;
  }
  else
    return false;
}

inline __device__
vec3f reflect(const vec3f &v,
              const vec3f &n)
{
  return v - 2.f*dot(v, n)*n;
}

inline __device__
bool Lambertian::scatter(ScatterEvent &event, RNG &rng)
{
  vec3f N         = event.incoming.N;
  const vec3f dir = event.incoming.dir;

  if (dot(N,dir) > 0.f)
    N = -N;
  N = normalize(N);

  // consine-weighed probability:
  vec3f D = (N + uniformRandomDirection(rng));

  // return scattering event
  event.outgoing.dir    = normalize(D);
  event.outgoing.atten  = this->albedo;
  return true;
}

inline __device__
bool Dielectric::scatter(ScatterEvent &event, RNG &rng)
{
  vec3f N = normalize(event.incoming.N);
  vec3f dir = normalize(event.incoming.dir);
  vec3f reflected = reflect(dir,N);
  event.outgoing.atten = vec3f(1.f, 1.f, 1.f);

  vec3f outward_normal;
  float ni_over_nt;
  vec3f refracted;
  float reflect_prob;
  
  float cosine = dot(dir,N);
  if (cosine > 0.f) {
    // normal and incoming dir point in same direction - we're
    // *leaving* the medium
    outward_normal = -N;
    ni_over_nt = ior;
    cosine = sqrtf(1.f - ior*ior*(1.f-cosine*cosine));
  }
  else {
    // we're entering
    outward_normal = N;
    ni_over_nt = 1.f/ior;
    cosine = -cosine;
  }
  
  if (refract(dir, outward_normal, ni_over_nt, refracted)) {
    reflect_prob = schlick(cosine, ior);
  } else 
    reflect_prob = 1.f;

  if (rng() < reflect_prob) {
    event.outgoing.dir = reflected;
  } else 
    event.outgoing.dir = refracted;

  event.outgoing.dir = normalize(event.outgoing.dir);
  return true;
}

OPTIX_RAYGEN_PROGRAM(renderFrame)()
{
  auto &lp = optixLaunchParams;
  auto  fb = lp.frameBuffer;
  
  vec2i pixel = owl::getLaunchIndex();
  if (pixel.x >= fb.size.x) return;
  if (pixel.y >= fb.size.y) return;

  // delta: initialize a random number generator based on pixel coord
  RNG rng(pixel.x+fb.accumID*fb.size.x,
          pixel.y+fb.accumID*fb.size.y);
  
  // relative position on the image, in [0,1]^2.
  float u = (pixel.x+rng()) / fb.size.x;
  float v = (pixel.y+rng()) / fb.size.y;

  // hard-coded camera for a camera at y=-2, and a 2x2 sized image
  // plane around the origin (ie, from x=-1,z=-1 to x=+1,z=+1)
  auto camera = lp.camera;
  vec3f rayOrigin
    = camera.position;
  vec3f rayDirection
    = camera.dir_00
    + u * camera.dir_du
    + v * camera.dir_dv;
  rayDirection = normalize(rayDirection);
  
  RegularRay ray(rayOrigin,rayDirection,0.f,CUDART_INF);

  // ------------------------------------------------------------------
  // first trace: this is the primary ray from the camera to find the
  // surface we're going to shade
  // ------------------------------------------------------------------
  RegularRayPRD prd{};
  owl::traceRay(lp.world,ray,/* --->>> */prd);

  // the radiance that reaches the camera through this path
  vec3f radiance = 0.f;
  if (!prd.hadHit) {
    // ------------------------------------------------------------------
    // primary ray didn't hit anything; shade with background color,
    // and no need to shoot any shadow rays
    // ------------------------------------------------------------------

    // use pete shirley's hacky background gradient from his RTOW samples
    const float t = v;
    radiance = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
  } else {
    // ------------------------------------------------------------------
    // primary ray *did* hit something; let's shoot a shadow ray from
    // where we hit. for now we'll do a hardcoded directional light
    // ------------------------------------------------------------------

    int numSpecularBounces = 0;
    int numDiffuseBounces  = 0;
    vec3f pathThroughput = 1.f;
    while (true) {
      // ==================================================================
      // FIRST: 'localize' the hit point, normal, brdf, etc
      // ==================================================================

      ScatterEvent event;
      event.incoming.N   = normalize(prd.hit.normal);
      event.incoming.P   = prd.hit.position;// NOT offset from surface yet
      event.incoming.dir = normalize(ray.direction);

      bool wasSpecular;
      if (prd.brdf.ior == 1.f) {
        // DIFFUSE surface
        Lambertian lambertian = { prd.brdf.diffuse };
        lambertian.scatter(event,rng);
        wasSpecular = false;
      } else {
        Dielectric dielectric = { prd.brdf.ior };
        dielectric.scatter(event,rng);
        wasSpecular = true;
      }
      
      // get normal from the PRD, normalize, and face-forward it
      vec3f Ngff = normalize(event.incoming.N);
      if (dot(Ngff,rayDirection) > 0.f)
        Ngff = -Ngff;
      float epsilon = 1e-6f*reduce_max(abs(event.incoming.P));

      if (!wasSpecular) {
        // hardcoded dirlight for now:
        vec3f lightDir = normalize(vec3f(.1f,1.f,.1f));
        float lightRadiance = .5f;
        
        vec3f lightTerm
          = pathThroughput * prd.brdf.diffuse * dot(lightDir,Ngff) * lightRadiance;
        if (reduce_max(lightTerm) > 1e-3f) {
          // trace shadow ray only if there is 'some' contribution...
          ShadowRay shadowRay(event.incoming.P+epsilon*Ngff,
                              lightDir,0.f,CUDART_INF);
          ShadowRayPRD shadowPRD{};
          owl::traceRay(lp.world,shadowRay,shadowPRD);
          if (!shadowPRD.isOccluded)
            radiance += lightTerm;
        }
      }

      // ==================================================================
      // THIRD (for this sample): sample the 'brdf' (only diffuse for
      // now) to get a new outgoing ray direction, and adjust path
      // trhoughput accordingly
      // ==================================================================

      // check if we go to other side of surface, and if so, adjust
      // the offset direction.
      float nn = dot(Ngff,event.outgoing.dir);
      if (nn < 0.f)
        epsilon = - epsilon;

      pathThroughput *= event.outgoing.atten;

      // quick check if path throughput even matters any more:
      if (reduce_max(pathThroughput) < 1e-3f) break;

      // genearte new outgoing ray:
      ray = RegularRay(event.incoming.P+epsilon*Ngff,
                       event.outgoing.dir,0.f,CUDART_INF);
      
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

      if (wasSpecular) {
        if (++numSpecularBounces > MAX_SPECULAR_BOUNCES)
          break;
      } else {
        if (++numDiffuseBounces > MAX_DIFFUSE_BOUNCES)
          // too many bounces - assume this path got lost somewhere in
          // the geometry and won't ever make it out.
          break;
      }
      
      // bounce was valid AND found a valid hit, let's process it in
      // the next iteration.
    }
  }

  int fbIdx = pixel.x+pixel.y*fb.size.x;
  // k - we know how much *this* path carries.... add this to accum buffer
  if (fb.accumID == 0) {
    // this is the first path ever being accumulated - just write
    fb.accum[fbIdx] = radiance;
  } else {
    // let's add this to what we already had
    radiance += fb.accum[fbIdx];
    // store the added value
    fb.accum[fbIdx] = radiance;
    // and normalize by how many frames we've already accumulated
    radiance *= 1.f/(/*how many we had*/fb.accumID+/*the one we just added*/1);
  }

  // now all the usual gamma, clamping, and 8-bit conversion on final
  // accumulated value:
  radiance  = min(sqrt(radiance),vec3f(1.f));
  fb.color[fbIdx] = owl::make_rgba(radiance);
}

