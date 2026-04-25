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
