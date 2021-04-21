

#ifndef SEISMIC_KERNEL_LIB2_H		
#define SEISMIC_KERNEL_LIB2_H	


#include "d_globvar.cuh"
void mat_av2_GPU(
    // Medium arguments
    real *&lam, real *&mu, real *&rho,
    real *&mu_zx, real *&rho_zp, real *&rho_xp, // inverse of densityint dimz, int dimx
    real &C_lam, real &C_mu, real &C_rho, // scalar averages
    int nz, int nx);


#endif