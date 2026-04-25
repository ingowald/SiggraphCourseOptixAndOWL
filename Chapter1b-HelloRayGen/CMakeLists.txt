# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

set(CMAKE_BUILD_TYPE_INIT "Release")

project(Chapter1b-HelloRayGen LANGUAGES CXX CUDA)

if (NOT(TARGET owl::owl))
  add_subdirectory(../submodules/owl builddir-owl EXCLUDE_FROM_ALL)
endif()

# ##################################################################
# the main difference to the previous sample: we have some device-side
# code that we have to (pre-)compile to PTX, the result of which we
# have to embed (as a C string) into the generated binary. We can use
# the OWL 'embed_ptx()' helper macro for that:
# ##################################################################
embed_ptx(
  OUTPUT_TARGET
  # this is what the main app will link against:
  embedded_deviceCode
  
  PTX_LINK_LIBRARIES
  owl::owl
  
  SOURCES
  deviceCode.cu
  )


# define what this chapter builds
add_executable(c1b-HelloRayGen
  hostCode.cpp
)

# (cmake-)link this to the owl::OWL target; this will automatically
# set the include paths, add the dependency to the OWL library, etc.
target_link_libraries(c1b-HelloRayGen
  owl::owl
  embedded_deviceCode
)
