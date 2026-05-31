// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// we'll include this in both host and device, so include the main
// 'owl.h' that'll automatically switch to the right 'content' based
// on whether it's included in host code or device code.
#include "owl/owl.h"
#include "owl/common/owl-common.h"
// include helper vector classes
#include "owl/common/math/vec.h"
#include "owl/common/math/random.h"

using namespace owl::common;

typedef enum { METAL=0, DIELECTRIC, LAMBERTIAN } MaterialType;


/*! forward definitions so we can use those types in method
    declarations */
struct ScatterEvent;
using RNG = owl::common::LCG<4>;

/*! largely stolen from pete shirley's RTOW samples */
struct Lambertian {
#ifdef __CUDA_ARCH__
  /* this function should only be visible in cuda device code */
  inline __device__ bool scatter(ScatterEvent &event, RNG &rng);
#endif
  vec3f albedo;
};

/*! largely stolen from pete shirley's RTOW samples */
struct Dielectric {
#ifdef __CUDA_ARCH__
  /* this function should only be visible in cuda device code */
  inline __device__ bool scatter(ScatterEvent &event, RNG &rng);
#endif
  float ior;
};

/*! largely stolen from pete shirley's RTOW samples */
struct Metal {
#ifdef __CUDA_ARCH__
  /* this function should only be visible in cuda device code */
  inline __device__ bool scatter(ScatterEvent &event, RNG &rng);
#endif
  vec3f albedo;
  float fuzz;
};


/*! the OptixGlobals that we'll store in 'global' constant memory;
    this means we could change this data every frame (or even across
    multiple parallel launches) without having to rebuild the SBT
    (raygen is a program, so its data is stroed in the SBT; globals
    are not). */
struct OptixGlobals {
  struct {
    vec2i     size;
    uint32_t *color;
    vec3f    *accum;
    int       accumID;
  } frameBuffer;
  struct {
    vec3f position;
    vec3f dir_00;
    vec3f dir_du;
    vec3f dir_dv;
  } camera;
  OptixTraversableHandle world;
};

/*! the pure geometric data - position and radius - required for
    running bounding box and/or intersection program(s). Note that in
    this sample this will _not_ be its own SBT entry, but rather, one
    array element in an entire array of such spheres that our SBT
    entry will represent. We could also realize this sample with one
    SBT entry per sphere, but this would be less efficient both in
    terms of memory (an SBT entry is much more expensive than an array
    element) as well as in build/update time (running N bounds
    programs in a single kernel vs running N different one-element
    bounds kernels; or building an SBT with one etnry per material
    type vs building one with one entry per sphere) */
struct Sphere {
  vec3f center;
  float radius;
};

/*! we use one OWL geom for all spheres that have the same type; i.e.,
    there'll be one such geom for all metal spheres, one for all
    lambertian spheres, etc. This isn't the only way this could be
    done, feel free to experiment with other variants. Note though
    that even though different spheres end up in differnt geoms
    they'll still all end up in the same BLAS becasue we'll put all
    these into the same OWLUserGeomGroup (and in OWL, the accel is
    handled on the Group level, not on the Geom level. */
struct SpheresGeomData {
  Sphere *spheres;
  /* this is a pointer into a device buffer that holds 'numSpheres'
     materials of the given 'materialType'. This being CUDA we could
     of course have used templates to used properly typed pointers,
     but chose to do the untyped way here just to show that this is
     perfectly valid, too - it'll be the raygen program to type-cast
     this to the proper material type (indicated in mateiraltype), and
     unlike some more type-strict shading languages in CUDA/OptiX it
     is perfectly valid to do that! Note this points to an entire
     buffer with exactly as many entires as there are spheres in this
     geometry. */
  void    *materialsData;
  uint32_t materialType;
};

/*! the raygen program data values. for now we moved everything into
    the laumch params, so the raygen actually contains nothing at all
    right now */
struct RayGenData {
};
  

