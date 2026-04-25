# SiggraphCourseOptixAndOWL

This repo is work-in-progress; it is supposed to host the different
tutorial 'chapters' for the upcoming Siggraph 2026 "OptiX and OWL"
course.

progress so far:

# Chapter 1

## Step 1a: "Hello OWL" ([Chapter1a-HelloOWL](http://Chapter1a-HelloOWL))

Tests basic set-up of OptiX/OWL. Creates an OWL context, prints a
"hello world", then simply exist.
  
Won't do anything useful in itself, but serves as a useful minimum
testing sample: if this one doesn't compile or run then clearly
something is missing in the host set-up: missing or wrong cmake,
compiler, CUDA, etc; wrong version of CUDA, no optix-capable GPU or
driver, etc. 

Shows:

- how to configure a CMake script to include and use OWL

- how to create a OWLContext on a specific GPU

- hot to properly clean up and exit

## Step 1b: Hello RayGen ([Chapter1b-HelloRayGen](http://Chapter1b-HelloRayGen))

Extends the previous step to create an actual OptiX RayGen program,
lauches this, and prints the "Hello World" from within this RayGen
program running on the device.
  
Shows:

- how to write device programs; this sample only generates a single
  (RayGen) device program, but other device programs later on will work
  exactly the same way.
  
- how to use OWLs `embed_ptx()` cmake macro to precompile devicecode
  into PTX, and how to embed that into the executable
  
- how to create a `OWLModule` device code module from embedded PTX code

- how to create a `RayGen` program object for the corresponding device code

- how to launch a raygen program

- some basic functionality available within a raygen program, like querying
  launch index, or doing `printf`s.
  

## Step 1c: Program Data ([Chapter1c-ProgramData](http://Chapter1c-ProgramData))

Adds various kinds of program data to our raygen, and shows how to declare that, and how
  
Shows:

- how to add various kinds of program data to a program (in this case,
  the raygen program, but the same will work for other programs too)
  
- how to use arrays of `OWLVarDecl` to describe this program data to
  OWL (so the app can then set these variables)
  
- how to use `owl<ProgramType>Set()` functions to set program data
  from the host.

- how to use `owlBuildSBT()` to (re-)build a shader binding table
  (SBT) with the proper program data that was set.


## Step 1d: First Buffer, and Rendering of First (Test-)Frame  ([Chapter1d-FirstBufferAndTestFrame](http://Chapter1d-FirstBufferAndTestFrame))

Our first sample that actually renders a (test-)frame! 

This sample uses an `OWLBuffer` to hold a frame buffer that then gets
filled in---with a simple test pattern for now---by an appropriately
extended raygen program. This buffer then gets downloaded onto the
host, and saved in a jpeg file on disk.

Shows:

- how to create a (device) buffer of a given size and data type

- how to assign a `OWLBuffer` to a program (in this case, the raygen
  program that needs to write into that)
  
- how OWL can translate from abstract host types like `OWLBuffer` to 
  more low-level device representations (in this case, the device pointer
  that this buffer represents)
  
- how to use `owlBufferGetPointer()` to query a device buffer's device
  address on the host, and then using CUDA interop (`cudaMemcpy()`) to
  copy that buffer's content to host memory.
  
## Step 1e: OptiX "Globals" and OWL "Launch Parameters"  ([Chapter1e-OptixGlobalsAndLaunchParams](http://Chapter1e-OptixGlobalsAndLaunchParams))

This sample introduces the concept of OptiX per-launch "globals" (data
stored in GPU constant memory rather than in the SBT), and how to use
the OWL "launch params" (`OWLLaunchParams`) to describe and set this
data.

This sample pretty muche exactly reproduces the previous sample's
functionality, but stores frame buffer size and pointer in optix
globals rather than in the raygen's program data. This does not
actually buy much in this simple example (we only build the SBT once,
anyway), but storing 'per frame'/'per launch' data outside of the SBT
is an important concept to understand and master going forward (and
way too often sadly neglected when talking about OptiX), so introduced
here early on in a more prominent manner.

Shows:

- how to use a `__constant__ SomeUserType optixLaunchParams;` to make
  per-launch constant data available to all programs. This data is
  "constant" within a given launch, but (unlike the SBT) can actually
  have different values across multiple concurrently running launches,
  and can be changed between different launches without rebuilding the
  SBT.
  
- how to use the `OWLLaunchParams` type to describe per-launch optix
  globals, and how to set them.
  
- how to issue a optix launch with a given set of launch param values.


  
