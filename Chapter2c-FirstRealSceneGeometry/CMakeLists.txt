# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

set(CMAKE_BUILD_TYPE_INIT "Release")

project(Chapter2c-FirstRealSceneGeometry LANGUAGES CXX CUDA)

if (NOT(TARGET owl::owl))
  add_subdirectory(../submodules/owl builddir-owl EXCLUDE_FROM_ALL)
endif()
if(NOT(TARGET owlGLFWViewer))
  # interactive viewer base
  # Bundled GLFW declares compatibility with CMake 3.0, which CMake 4 rejects.
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
  add_subdirectory(../submodules/glfwViewer builddir-glfwViewer EXCLUDE_FROM_ALL)
endif()
if (NOT(TARGET miniScene))
  add_subdirectory(../submodules/miniScene builddir-miniScene EXCLUDE_FROM_ALL)
endif()

# precompile and embed the device code
embed_ptx(
  OUTPUT_TARGET
  # this is what the main app will link against:
  c2c_embedded_deviceCode
  
  PTX_LINK_LIBRARIES
  owl::owl_static
  
  SOURCES
  deviceCode.cu
  )

# define what this chapter builds
add_executable(c2c-FirstRealSceneGeometry
  hostCode.cpp
)

# (cmake-)link this to the owl::OWL target; this will automatically
# set the include paths, add the dependency to the OWL library, etc.
target_link_libraries(c2c-FirstRealSceneGeometry
  owl::owl_static
  c2c_embedded_deviceCode
  owlViewer-config
  owlGLFWViewer
  miniScene
)
