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
  
  


