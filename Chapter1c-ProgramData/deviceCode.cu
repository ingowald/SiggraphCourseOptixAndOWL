// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "deviceCode.h"

OPTIX_RAYGEN_PROGRAM(helloOWL)()
{
  // first, we need to ask optix/owl for the program data that we have
  // set this type of raygen program up with (that data is stored in
  // the SBT and optix allows us to get a pointer to that SBT
  // data). There's two different -- and totally equivalent --- ways
  // of accessing this:
#if 0
  // option 1: the most low-level variant, using the respective optix
  // call directly. This gives us a untyped device pointer into the
  // SBT that we can then typecast and use. This is what a regular
  // optix program would (have to) use, but it will also work in an
  // owl program.
  RayGenData rgData = *(const RayGenData *)optixGetSbtDataPointer();
#else
  // option 2: the owl variant, with some syntactic sugar that makes
  // this more readbale; but under the hood this is doing *exactly*
  // the same as option 1.
  RayGenData rgData = owl::getProgramData<RayGenData>();
#endif

  if (optixGetLaunchIndex().x != 0 ||
      optixGetLaunchIndex().y != 0)
    // exit for all but launch index (0,0), so we're not flooding the
    // screen
    return;
  
  printf(OWL_TERMINAL_GREEN);
  printf("********************************************\n");
  printf("RayGen Program Data Test (printed on device)\n");
  printf("********************************************\n");
  printf("Data passed to us:\n");
  printf(" - someInteger %i\n",
         rgData.someInteger);
  printf(" - somePointOrVector %f %f %f\n",
         rgData.somePointOrVector.x,
         rgData.somePointOrVector.y,
         rgData.somePointOrVector.z);
  printf(" - aRawUnformattedRegion: %s\n",
         (const char *)&rgData.aRawUnformattedRegion[0]);
  printf("   (note _we_ knew that that data passed by the app\n");
  printf("    was actually a string, even through OWL didn't)\n");
  printf(" - someDummyPointerToABuffer: %p\n",
         rgData.somePointerToBufferData);
  printf("   (this is a null pointer because we haven't even introduced\n");
  printf("    buffers yet; this is OK, we'll do this later on)\n");
  printf(" - someRawPointer: %p\n",
         rgData.someRawPointer);
  printf("   (this should be 0x1234567, since that's what the host is\n");
  printf("    supposed to set in this example. As a 'RAW_POINTER' type\n");
  printf("    OWL will simply fill in whatever the app asked it to use,\n");
  printf("    without any translation or checking whatsoever)\n");
  printf(OWL_TERMINAL_DEFAULT);
}

