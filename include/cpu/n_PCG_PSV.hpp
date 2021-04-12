//d_PCG_PSV.cpp

/* 
* Created by: Min Basnet
* 2021.January.15
* Kathmandu, Nepal
*/

// Preconditioned Gradient Method for PSV gradients

#ifndef PCG_PSV_H		
#define PCG_PSV_H	

#include "n_globvar.hpp"

void PCG_PSV(real **&PCG_dir, real **&PCG_old, real **&grad_mat, int nz, int nx);

#endif