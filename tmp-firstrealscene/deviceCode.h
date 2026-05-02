// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Camera.h"

using namespace owl::common;

/* DELTA: as compared to the previous sample we have moved the frame
  buffer data from RayGen *program* data to the
  "not-stored-in-the-SBT" OptixGlobals. Exactly the same data, and we
  use it the same way, we just store it outside of the SBT to show how
  optixGlobals and LaunchParams are working.
*/

/*! the OptixGlobals that we'll store in 'global' constant memory;
    this means we could change this data every frame (or even across
    multiple parallel launches) without having to rebuild the SBT
    (raygen is a program, so its data is stroed in the SBT; globals
    are not). */
struct OptixGlobals {
  struct {
    vec2i     size;
    uint32_t *data;
  } frameBuffer;
  OptixTraversableHandle world;
  Camera camera;
};

/*! data for the triangles geometry we're going to use */
struct TrianglesGeomData {
  vec3f *vertices;
  vec3i *indices;
  vec3f baseColor;
};

/*! the raygen program data values. for now we moved everything into
    the laumch params, so the raygen actually contains nothing at all
    right now */
struct RayGenData {
};
  

