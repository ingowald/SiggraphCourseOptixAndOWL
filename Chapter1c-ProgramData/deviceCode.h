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

using namespace owl::common;

/*! the data we want to store with the ray gen program. For this
    introductory sample we just use some random choice of different
    data to show what's possible; we're not even setting the buffer,
    accel, or texture to any useful values yet (we haven't even
    introduced those, yet), but this sample can already show how to
    declare and set them. */
struct RayGenData {
  int                 someInteger;
  vec3f               somePointOrVector;
  uint8_t             aRawUnformattedRegion[31];
  
  // we're not actually setting this in this sample, yet; this just
  // shows how it can eb declared.
  vec4f              *somePointerToBufferData;
  
  // a raw pointer value, *without* translation from a buffer to
  // device address. It's the user's job to make sure that this
  // poitner is a valid device-accessible pointer.
  void               *someRawPointer;
  
  // we're not actually setting this in this sample, yet; this just
  // shows how it can eb declared.
  cudaTextureObject_t someTextureHandle;
  
  // we're not actually setting this in this sample, yet; this just
  // shows how it can be declared.
  OptixTraversableHandle someOptixAccelHandle;
};
  

