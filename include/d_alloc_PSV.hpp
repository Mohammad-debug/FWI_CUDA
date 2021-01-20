//seismic_allocation2.hpp

/* 
* Created by: Min Basnet
* 2020.December.03
* Kathmandu, Nepal
*/

#ifndef SEISMIC_ALLOCATION2_H		
#define SEISMIC_ALLOCATION2_H	

#include "d_globvar.hpp"
#include "d_contiguous_arrays.hpp"

// The arrays allocated to get external inputs
void alloc_varin_PSV( real *&hc, int *&isurf, // holberg coefficients and surface indices
    // Medium arguments
    real **&lam, real **&mu, real **&rho,
    //PML arguments (z and x direction)
    real *&a_z, real *&b_z, real *&K_z, 
    real *&a_half_z, real *&b_half_z, real *&K_half_z,
    real *&a_x, real *&b_x, real *&K_x, 
    real *&a_half_x, real *&b_half_x, real *&K_half_x, 
    // Seismic sources
    int *&z_src, int *&x_src, // source locations
    int *&src_shot_to_fire, // which source to fire on which shot index
    real **&stf_z, real **&stf_x, // source time functions
    // Reciever seismograms
    int *&z_rec, int *&x_rec,
    real **&rtf_z_true, real **&rtf_x_true, // Field measurements for receivers
    // Scalar variables for allocation
    int fdorder, bool surf,
    bool pml_z, bool pml_x, int nsrc, int nrec, bool rtf_true,
    int nt, int nz, int nx);

// Allocate the main variables (excluding external inputs)
void alloc_varmain_PSV(
    // Wave arguments 
    real **&vz, real **&vx,  // velocity
    real **&uz, real **&ux, // Displacement
    real **&We, real **&We_adj, // Energy fwd and adj
    real **&szz, real **&szx, real **&sxx,
    // Spatial derivatives (for internal computations)
    real **&dz_z, real **&dx_z, real **&dz_x, real **&dx_x, 
    // PML memory arrays for spatial derivatives
    real **&mem_vz_z, real **&mem_vx_z, 
    real **&mem_szz_z,  real **&mem_szx_z, 
    real **&mem_vz_x, real **&mem_vx_x,  
    real **&mem_szx_x, real **&mem_sxx_x,
    // Material average arrays
    real **&mu_zx, real **&rho_zp, real **&rho_xp,
    // Copy old material while updating
    real **&lam_copy, real **&mu_copy, real **&rho_copy,
    // Gradients of the medium
    real **&grad_lam, real **&grad_mu, real **&grad_rho,
    // Gradients for each shot 
    real **&grad_lam_shot, real **&grad_mu_shot, real **&grad_rho_shot,
    // reciever time functions
    real **&rtf_uz, real **&rtf_ux,
    // Accumulate the snap of forward wavefield parameters
    real ***&accu_vz, real ***&accu_vx, //accumulated velocity memory over time
    real ***&accu_szz, real ***&accu_szx, real ***&accu_sxx, //accumulated velocity memory over time
    bool pml_z, bool pml_x, int nrec, bool accu, bool grad, 
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, // grid boundaries for fwi
    int snap_dt, int snap_dz, int snap_dx, // time n space grid intervals to save storage
    int nt, int nz, int nx);

#endif