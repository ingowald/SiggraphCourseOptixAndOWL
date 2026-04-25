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

/*! as compared to the previous sample we have remove all the dummy
    values we had in this sample, and now use some 'real' payload: a
    frame buffer to store our ray gen's computed pixel values in.
*/
struct RayGenData {
  struct {
    vec2i     size;
    uint32_t *data;
  } frameBuffer;
};
  

