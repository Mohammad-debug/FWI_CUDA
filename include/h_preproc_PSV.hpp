//h_preproc_PSV.hpp

/* 
* Created by: Min Basnet
* 2020.December.09
* Kathmandu, Nepal
*/

#ifndef H_PREPROC_PSV_H		
#define H_PREPROC_PSV_H		

#include "h_globvar.hpp"
#include "n_contiguous_arrays.hpp"

// Allocate the input variables in the host
void alloc_varpre_PSV( real *&hc, int *&isurf, int *&npml, // holberg coefficients, surface indices and number pml in each side
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
    real ***&rtf_z_true, real ***&rtf_x_true, // Field measurements for receivers
    // Scalar variables for allocation
    int fdorder, 
    bool pml_z, bool pml_x, int nshot, int nsrc, int nrec,
    int nt, int nz, int nx);

// Staggering of scalar material over the grid
void mat_stag_PSV(real **&lam, real **&mu, real **&rho, real scalar_lam, real scalar_mu, real scalar_rho, 
                int nz, int nx);

// Create CPML coefficient array
void cpml_PSV(real *&a, real *&b, real *&K, 
        real *&a_half, real *&b_half, real *&K_half,
        real npower, real damp_v_PML, real rcoef, real k_max_PML,
        real freq, int npml_h1, int npml_h2, int fpad, int nh, real dt, real dh);


// Create a single source wavelet in one direction only
void wavelet(real *&signal, int nt, real dt, real amp, real fc, real ts, int shape);

void write_dat(real ***&vz, real ***&vx, real ***&szz, real ***&szx, real ***&sxx, 
                int nt, int nz, int nx, int snap_z1, int snap_z2, int snap_x1, 
                int snap_x2, int snap_dt, int snap_dz, int snap_dx);

                
#endif