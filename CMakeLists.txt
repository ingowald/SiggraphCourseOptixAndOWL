# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

set(CMAKE_BUILD_TYPE_INIT "Release")
set(CMAKE_CXX_STANDARD 20)

project(SiggraphCourseOptixAndOWL)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

add_subdirectory(Chapter1a-HelloOWL)
add_subdirectory(Chapter1b-HelloRayGen)
add_subdirectory(Chapter1c-ProgramData)
add_subdirectory(Chapter1d-FirstBufferAndTestFrame)
add_subdirectory(Chapter1e-OptixGlobalsAndLaunchParams)

add_subdirectory(Chapter2a-FirstGeometry)
add_subdirectory(Chapter2b-InteractiveViewer)
add_subdirectory(Chapter2c-FirstRealSceneGeometry)
add_subdirectory(Chapter2d-SimpleShading)

