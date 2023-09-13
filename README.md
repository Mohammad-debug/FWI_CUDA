# Seismic Full Waveform Inversion with GPU Acceleration

## Overview

This is Seismic Full Waveform Inversion (FWI) is a powerful technique used to estimate subsurface parameters by analyzing seismic measurements obtained at the surface. However, due to the large volume of data, complex model sizes, and non-linear iterative procedures involved, numerical computations for FWI are often computationally intensive and time-consuming. This project addresses these challenges by implementing parallel computation techniques with Graphical Processing Units (GPUs) via CUDA to significantly accelerate the FWI process.

**Note:** This project is an implementation of the research paper described in [this paper](https://www.mdpi.com/2076-3417/12/17/8844).


## Implementation

- Host code is written in C++ to manage the overall project structure and coordinate computations.
- Parallel computation codes are written in CUDA C, a language optimized for GPU processing.

## Performance Comparison

The project includes a comprehensive performance evaluation:

- **Comparing CUDA C and OpenMP:** The computational time and performance achieved through CUDA C and OpenMP parallel computation are compared to a serial code implementation.

- **Scaling with Model Dimensions:** The project demonstrates that as model dimensions increase, the performance improvement is enhanced. It remains nearly constant after reaching a certain threshold.

- **Impressive GPU Performance Gain:** In our experiments, we achieved a GPU performance boost of up to 90 times compared to the serial code, underscoring the substantial benefits of GPU acceleration.

## Prerequisites

To run this project, you will need the following:

- NVIDIA GPU with CUDA support.
- CUDA Toolkit installed ([Download CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)).
- C/C++ Compiler (e.g., `nvcc` for CUDA code and `g++` for CPU code).
- Git (optional, for cloning the repository).
