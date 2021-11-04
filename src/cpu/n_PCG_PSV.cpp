//d_PCG_PSV.cpp

/* 
* Created by: Min Basnet
* 2021.January.15
* Kathmandu, Nepal
*/

// Preconditioned Gradient Method for PSV gradients
#include "n_PCG_PSV.hpp"
#include <iostream>

void PCG_PSV(real **&PCG_dir, real **&PCG_old, real **&grad_mat, int nz, int nx){
    // applies Preconditined Conjugate Gradient Method for PSV wave
    //std::cout << "step 0. ";
    real beta_PCG = 0, beta_i= 0, beta_j= 0;
    // getting the descent (Negative directions for now) done below for now
    //std::cout << "step 1. ";
   
    // Applying PSG method
    for (int iz=0;iz<nz;iz++){
        for (int ix=0;ix<nx;ix++){

            // Fletcher-Reeves [Fletcher and Reeves, 1964]:
            beta_i += grad_mat[iz][ix] * grad_mat[iz][ix];
            beta_j += PCG_old[iz][ix] * PCG_old[iz][ix];
            PCG_old[iz][ix] = -grad_mat[iz][ix]; 
           
        }
    }
    beta_PCG = beta_i/beta_j;
    beta_PCG = (beta_PCG >0) ? beta_PCG : 0.0;

    //std::cout << "step 3. ";
  
    for (int iz=0;iz<nz;iz++){
        for (int ix=0;ix<nx;ix++){
            PCG_dir[iz][ix] = -grad_mat[iz][ix] + beta_PCG * PCG_dir[iz][ix]; // Getting PCG direction
            grad_mat[iz][ix] = PCG_dir[iz][ix]; // Getting PCG_dir to gradient vectors
           
        }
    }
   
}

