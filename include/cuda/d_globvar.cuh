//d_globvar.h

/* 
* Created by: Min Basnet
* 2020.April.13
* Kathmandu, Nepal
*/

// This is temporary file, later this file will contain the inputs from host to device

#ifndef DEVICE_GLOBVAR_H				
#define DEVICE_GLOBVAR_H

#include "cuda_runtime.h"//
#include "device_launch_parameters.h"//
#include<time.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <thrust/copy.h>


#define cudaCheckError(code)                                             \
  {                                                                      \
    if ((code) != cudaSuccess) {                                         \
      fprintf(stderr, "Cuda failure %s:%d: '%s' \n", __FILE__, __LINE__, \
              cudaGetErrorString(code)); exit(0);                                \
    }                                                                    \
  }

using real = double;
#define PI 3.14159265

#endif