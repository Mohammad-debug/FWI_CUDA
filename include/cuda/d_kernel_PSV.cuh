
#ifndef KERNEL_PSV_H		
#define KERNEL_PSV_H

#include "d_globvar.cuh"
#include "d_kernel_lib_PSV.cuh"


void kernel_PSV_GPU(int ishot, // shot index
    int nt, int nz, int nx, // Time and space grid arguments
    real dt, real dx, real dz, 
    // surface incides (0.top, 1.bottom, 2.left, 3.right)
    bool surf, int *&isurf,
    // computationsl arguments
    real *&hc, int fdorder, 
    // Wave arguments (velocity, displacement and stress tensors)
    real *&vz, real *&vx,  real *&uz, real *&ux, 
    real *&szz, real *&szx, real *&sxx,  real *&We,
    // Spatial derivatives (for internal computations)
    real *&dz_z, real *&dx_z, real *&dz_x, real *&dx_x, 
    // Medium arguments
    real *&lam, real *&mu, real *&mu_zx, 
    real *&rho_zp, real *&rho_xp, // inverse of density
    // Gradients of the medium
    bool grad, real *&grad_lam, real *&grad_mu, real *&grad_rho,
    //PML arguments
    bool pml_z, real *&a_z, real *&b_z, real *&K_z, 
    real *&a_half_z, real *&b_half_z, real *&K_half_z,
    bool pml_x, real *&a_x, real *&b_x, real *&K_x, 
    real *&a_half_x, real *&b_half_x, real *&K_half_x, 
    // PML memory arrays
    real *&mem_vz_z, real *&mem_vx_z, real *&mem_szz_z, real *&mem_szx_z, 
    real *&mem_vz_x, real *&mem_vx_x, real *&mem_szx_x, real *&mem_sxx_x,
    // Seismic sources
    int nsrc, int stf_type, real *&stf_z, real *&stf_x, 
    int *&z_src, int *&x_src, int *&src_shot_to_fire,
    // Reciever seismograms
    int nrec, int rtf_type, real *&rtf_uz, real *&rtf_ux, int *&z_rec, int *&x_rec,
    // Accumulate the snap of forward wavefield parameters
    bool accu, real *&accu_vz, real *&accu_vx, //accumulated velocity memory over time
    real *&accu_szz, real *&accu_szx, real *&accu_sxx, //accumulated velocity memory over time
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, // grid boundaries for fwi
    int snap_dt, int snap_dz, int snap_dx // time n space grid intervals to save storage
    );

    

#endif