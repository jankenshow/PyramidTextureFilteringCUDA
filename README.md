# Pyramid Texture Filtering in CUDA/Python

This is a CUDA/Python implementation of [pyramid texture filtering](https://rewindl.github.io/pyramid_texture_filtering/) for edge-preserving image smoothing.

Currently, CUDA code does not work well.

## Prerequisites

### CUDA

- CUDA Toolkit (11.0 or later)
- OpenCV (4.0 or later)
- CMake (3.16 or later)
- C++17 compatible compiler

### Python

- opencv, numpy

## Building

### CUDA

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Running

### CUDA

```bash
cd build
./demo
```

### Python

```bash
python python_implementation/demo.py
```
