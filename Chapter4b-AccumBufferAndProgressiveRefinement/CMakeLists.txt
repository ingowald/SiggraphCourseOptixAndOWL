# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

set(CMAKE_BUILD_TYPE_INIT "Release")

set(chapter_number "4b")
set(chapter_topic  "AccumBufferAndProgressiveRefinement")
set(chapter_name "Chapter${chapter_number}_${chapter_topic}")

project(${chapter_name} LANGUAGES CXX CUDA)

if (NOT(TARGET owl::owl))
  add_subdirectory(../submodules/owl builddir-owl EXCLUDE_FROM_ALL)
endif()
if(NOT(TARGET owlGLFWViewer))
  # interactive viewer base
  add_subdirectory(../submodules/glfwViewer builddir-glfwViewer EXCLUDE_FROM_ALL)
endif()
if (NOT(TARGET miniScene))
  add_subdirectory(../submodules/miniScene builddir-miniScene EXCLUDE_FROM_ALL)
endif()

# precompile and embed the device code
embed_ptx(
  OUTPUT_TARGET
  # this is what the main app will link against:
  c${chapter_number}_embedded_deviceCode
  
  PTX_LINK_LIBRARIES
  owl::owl_static
  
  SOURCES
  deviceCode.cu
  )

# define what this chapter builds
add_executable(c${chapter_number}-${chapter_topic}
  hostCode.cpp
)

# (cmake-)link this to the owl::OWL target; this will automatically
# set the include paths, add the dependency to the OWL library, etc.
target_link_libraries(c${chapter_number}-${chapter_topic}
  owl::owl_static
  c${chapter_number}_embedded_deviceCode
  owlViewer-config
  owlGLFWViewer
  miniScene
)
target_compile_definitions(c${chapter_number}-${chapter_topic}
  PRIVATE
  -DCHAPTER_NAME=\"${chapter_name}\")
  
