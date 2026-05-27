# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

set(CMAKE_BUILD_TYPE_INIT "Release")

set(chapter_number "1a")
set(chapter_topic  "HelloOWL")
set(chapter_name "Chapter${chapter_number}_${chapter_topic}")

project(${chapter_name})

# iw - only include OWL if we build this sample standalone (if we build
# it as part of the parent directory this will already be defined)
if (NOT(TARGET owl::owl))
  add_subdirectory(../submodules/owl builddir-owl EXCLUDE_FROM_ALL)
endif()

# define what this chapter builds
add_executable(c${chapter_number}-${chapter_topic}
  hostCode.cpp
)

# (cmake-)link this to the owl::OWL target; this will automatically
# set the include paths, add the dependency to the OWL library, etc.
target_link_libraries(c${chapter_number}-${chapter_topic}
  owl::owl_static
)
target_compile_definitions(c${chapter_number}-${chapter_topic}
  PRIVATE
  -DCHAPTER_NAME=\"${chapter_name}\")
  
