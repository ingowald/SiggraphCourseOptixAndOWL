// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// we'll include this in both host and device, so include the main
// 'owl.h' that'll automatically switch to the right 'content' based
// on whether it's included in host code or device code.
#include "owl/owl.h"
#include "owl/common/owl-common.h"
// include helper vector classes
#include "owl/common/math/vec.h"

using namespace owl::common;

struct Camera {
  Camera() = default;
  __host__ Camera(vec3f from,
                  vec3f to,
                  vec3f up,
                  vec2i fbSize);

  vec3f position;
  vec3f dir_00;
  vec3f dir_du;
  vec3f dir_dv;
};

__host__ Camera::Camera(vec3f from,
                        vec3f at,
                        vec3f up,
                        vec2i fbSize)
{
  vec3f direction = normalize(at-from);
  const float cosFovy = 0.5f;
  const float aspect = fbSize.x / float(fbSize.y);
  // horizontal and vertical edges of image plane at distance 1
  const vec3f horizontal
    = 2.f * cosFovy * aspect * normalize(cross(direction,up));
  const vec3f vertical
    = 2.f * cosFovy * normalize(cross(horizontal,direction));

  position = from;
  dir_00
    = direction
    - .5f*horizontal
    - .5f*vertical;
  dir_du = horizontal * (1.f/fbSize.x);
  dir_dv = vertical   * (1.f/fbSize.y);
}

