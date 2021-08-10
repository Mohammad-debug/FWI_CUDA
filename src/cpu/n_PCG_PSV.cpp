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
    std::cout << "step 0. ";
    real beta_PCG = 0, beta_i= 0, beta_j= 0;
    // getting the descent (Negative directions for now) done below for now
    std::cout << "step 1. ";
   
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

    std::cout << "step 3. ";
  
    for (int iz=0;iz<nz;iz++){
        for (int ix=0;ix<nx;ix++){
            PCG_dir[iz][ix] = -grad_mat[iz][ix] + beta_PCG * PCG_dir[iz][ix]; // Getting PCG direction
            grad_mat[iz][ix] = PCG_dir[iz][ix]; // Getting PCG_dir to gradient vectors
           
        }
    }
   
}

/*

void PCG_PSV(real *&PCG_dir, real *&PCG_new, real *&PCG_old, real **&grad_mat, int nz, int nx, int matindex){
    // applies Preconditined Conjugate Gradient Method for PSV wave
    std::cout << "step 0. ";
    real beta_PCG = 0, beta_i= 0, beta_j= 0;
    int ih;
    // getting the descent (Negative directions for now) done below for now
    std::cout << "step 1. ";
    // get the descend and store to the PCG new
    ih = matindex*nz*nx; // matindex = 0 for lam, 1 for mu and 2 for rho
    for (int iz=0;iz<nz;iz++){
        for (int ix=0;ix<nx;ix++){
            PCG_new[ih] = -grad_mat[iz][ix]; // Storing new gradient as old one ((Negative value is the descend))

            ih++;
        }
    }
    std::cout << "step 2. ";
    // Applying PSG method
    ih = matindex*nz*nx;
    for (int iz=0;iz<nz;iz++){
        for (int ix=0;ix<nx;ix++){

            // Fletcher-Reeves [Fletcher and Reeves, 1964]:
            beta_i += PCG_new[ih] * PCG_new[ih];
            beta_j += PCG_old[ih] * PCG_old[ih];
            ih++;
        }
    }
    beta_PCG = beta_i/beta_j;
    std::cout << "step 3. ";
    ih = matindex*nz*nx; // matindex = 0 for lam, 1 for mu and 2 for rho
    for (int iz=0;iz<nz;iz++){
        for (int ix=0;ix<nx;ix++){
            PCG_dir[ih] = PCG_new[ih] + beta_PCG * PCG_dir[ih]; // Getting PCG direction
            ih++;
        }
    }
    std::cout << "step 4. ";
    // Extract the PSG_dir to gradients
    ih = matindex*nz*nx; // matindex = 0 for lam, 1 for mu and 2 for rho
    for (int iz=0;iz<nz;iz++){
        for (int ix=0;ix<nx;ix++){
            grad_mat[iz][ix] = PCG_dir[ih]; // Getting PCG_dir to gradient vectors
            ih++;
        }
    }
    std::cout << "step 5. ";
    // Store the PCG to old one for later calculation
    ih = matindex*nz*nx; // matindex = 0 for lam, 1 for mu and 2 for rho
    for (int iz=0;iz<nz;iz++){
        for (int ix=0;ix<nx;ix++){
            PCG_old[ih] = PCG_new[ih]; 
            ih++;
        }
    }  
    std::cout << "step 6. ";  
}
*/