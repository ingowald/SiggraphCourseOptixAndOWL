// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "deviceCode.h"
// include own helper for random numbers
#include <owl/common/math/random.h>

__constant__ OptixGlobals optixLaunchParams;

/*! how many bounces we want to do */
#define MAX_BOUNCES 32

struct PRD {
  bool hadHit = false;
  struct {
    vec3f position;
    vec3f normal;
  } hit;
  struct {
    uint32_t type:4;
    uint32_t index:28;
    // raygen can use 'type' to typecase to proper type.
    void    *buffer;
  } material;
};


OPTIX_BOUNDS_PROGRAM(Spheres_Bounds)(/*! pointer to the SBT data we created */
                                     const void *geomData,
                                     /*! return value: the bounding
                                         box fo the given primitivie
                                         for which this program is
                                         being run */
                                     box3f &primBounds,
                                     /*! the prim index for which we
                                         are to compute the BBox */
                                     int primID)
{
  const SpheresGeomData &self = *(const SpheresGeomData*)geomData;
  const Sphere sphere = self.spheres[primID];
  primBounds = box3f()
    .extend(sphere.center - sphere.radius)
    .extend(sphere.center + sphere.radius);
}

OPTIX_INTERSECT_PROGRAM(Spheres_Intersect)()
{
  const int primID = optixGetPrimitiveIndex();
  // unlike in other examples in this course, here we use 'self' to
  // refer to a single sphere rather than the entire group of spehres.
  const auto &self
    = owl::getProgramData<SpheresGeomData>().spheres[primID];

  const vec3f org  = optixGetObjectRayOrigin();
  const vec3f dir  = optixGetObjectRayDirection();
  float hit_t      = optixGetRayTmax();
  const float tmin = optixGetRayTmin();

  const vec3f oc = org - self.center;
  const float a = dot(dir,dir);
  const float b = dot(oc, dir);
  const float c = dot(oc, oc) - self.radius * self.radius;
  const float discriminant = b * b - a * c;
  
  if (discriminant < 0.f)
    // return without 'optixReportIntersection()' -> no hit
    return;

  {
    float temp = (-b - sqrtf(discriminant)) / a;
    if (temp < hit_t && temp > tmin) 
      hit_t = temp;
  }
      
  {
    float temp = (-b + sqrtf(discriminant)) / a;
    if (temp < hit_t && temp > tmin) 
      hit_t = temp;
  }
  if (hit_t < optixGetRayTmax()) {
    // 'k - store this hit!

    /* let's store the normal in object space, just because that's
       numerically more stable (note we _can_ also access/modify PRD
       here. have to be careful here to make sure that these
       modificaitons remain consistent with whatever CH will do, but
       in this case that's the case: if this _will_ eventually turn
       out to be the closest hit then we're the last one to write this
       value, and CH will write otehr matching data; if not whatever
       other closer intersection found will also overwrite the data we
       write here!). Also note that in isec program all these values
       are *object* space; we'll need to still transform them in CH */
    auto &prd = owl::getPRD<PRD>();
    prd.hit.position = org + hit_t * dir;
    prd.hit.normal = prd.hit.position - self.center;
#if 1
    // iw - this (^^^^) is the obvious way of computing the hit
    // position (as org+tHit*dir), but this is rather instable on the
    // numerical front (meaning the computed position may be actually
    // quite a bit under or below the intended surface of the
    // sphere). Let's 'move' it back to the surface by shiting along
    // the normal.
#else
    prd.hit.position = self.center + self.radius * normalize(prd.hit.normal);
#endif      
    // aaaand, let's make it official:
    optixReportIntersection(hit_t, 0);
  }
}

  
OPTIX_CLOSEST_HIT_PROGRAM(Spheres_CH)()
{
  auto &prd = owl::getPRD<PRD>();

  auto self = owl::getProgramData<SpheresGeomData>();

  int primID = optixGetPrimitiveIndex();
  int instID = optixGetInstanceIndex();

  prd.hadHit        = true;

  // isec program stored these in object space (which is numerically
  // more stable) - have to transform them here because our RG expects
  // world space.
  prd.hit.position
    = optixTransformPointFromObjectToWorldSpace(prd.hit.position);
  prd.hit.normal
    = optixTransformNormalFromObjectToWorldSpace(prd.hit.normal);

  prd.material.type   = self.materialType;
  prd.material.buffer = self.materialsData;
  prd.material.index  = primID;
}












inline __device__
vec3f randomPointInUnitSphere(RNG &rng)
{
  // NOTE: this is the 'pete shirley' style hacky way of generating a
  // random direction. it's not exactly the best method for doing
  // this, but easy to understand so for didacting reaons let's do it
  // this way. If you're going to use that as a baseline for your own
  // renderer, please read up on sampling spheres, hemispheres, etc,
  // for example by reading Phil Dutre's (still!) excellent "Global
  // Illumination Compendium".
  vec3f p;
  do {
    p = 2.f*vec3f(rng(),rng(),rng()) - vec3f(1.f);
  } while (dot(p,p) > 1.f);
  return p;
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
  return normalize(randomPointInUnitSphere(rng));
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
bool Metal::scatter(ScatterEvent &event, RNG &rng)
{
  vec3f N         = event.incoming.N;
  const vec3f dir = event.incoming.dir;

  if (dot(N,dir)  > 0.f)
    N = -N;
  N = normalize(N);
  
  vec3f reflected = reflect(normalize(dir),N);
  event.outgoing.dir
    = (reflected+this->fuzz*randomPointInUnitSphere(rng));
  event.outgoing.dir = normalize(event.outgoing.dir);
  event.outgoing.atten
    = this->albedo;
  return (dot(event.outgoing.dir, N) > 0.f);
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
  
  owl::Ray ray(rayOrigin,normalize(rayDirection),0.f,CUDART_INF);

  // ------------------------------------------------------------------
  // first trace: this is the primary ray from the camera to find the
  // surface we're going to shade
  // ------------------------------------------------------------------
  PRD prd{};
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

    int numBounces = 0;
    vec3f pathThroughput = 1.f;
    while (true) {
      // ==================================================================
      // FIRST: 'localize' the hit point, normal, brdf, etc
      // ==================================================================

      ScatterEvent event;
      event.incoming.N   = normalize(prd.hit.normal);
      event.incoming.P   = prd.hit.position;// NOT offset from surface yet
      event.incoming.dir = ray.direction;

      bool valid;
      switch(prd.material.type) {
      case METAL: {
        auto material = ((Metal *)prd.material.buffer)[prd.material.index];
        valid = material.scatter(event,rng);
      } break;
      case DIELECTRIC: {
        auto material = ((Dielectric *)prd.material.buffer)[prd.material.index];
        valid = material.scatter(event,rng);
      } break;
      case LAMBERTIAN: {
        auto material = ((Lambertian *)prd.material.buffer)[prd.material.index];
        valid = material.scatter(event,rng);
      } break;
      default:
        break;
      }
      if (!valid) break;

      event.outgoing.dir = normalize(event.outgoing.dir);
      
      // get normal from the PRD, normalize, and face-forward it
      vec3f Ngff = normalize(event.incoming.N);
      if (dot(Ngff,rayDirection) > 0.f)
        Ngff = -Ngff;
      float epsilon = 1e-5f*reduce_max(abs(event.incoming.P));


      // check if we go to other side of surface, and if so, adjust
      // the offset direction.
      float nn = dot(Ngff,event.outgoing.dir);
      if (nn < 0.f)
        epsilon = - epsilon;

      pathThroughput *= event.outgoing.atten;

      // quick check if path throughput even matters any more:
      if (reduce_max(pathThroughput) < 1e-3f) break;

      // generate new outgoing ray:
      ray = owl::Ray(event.incoming.P+epsilon*Ngff,
                     event.outgoing.dir,1e-3f,CUDART_INF);
      
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
      
      if (++numBounces > MAX_BOUNCES)
        break;
      
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

