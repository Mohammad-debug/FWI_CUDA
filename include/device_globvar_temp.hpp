//device_globvar_temp.hpp

/* 
* Created by: Min Basnet
* 2020.April.13
* Kathmandu, Nepal
*/

#ifndef DEVICE_GLOBVAR_TEMP_H				
#define DEVICE_GLOBVAR_TEMP_H

#include "device_globvar.hpp"

// ---------------------------------------
// 1. VARIABLES TO BE COPIED FROM CPU (HOST)
// --------------------------------------

// Geometric parameters
real time = 3.0;
real dt = 0.9e-3, dz = 5.0, dx = 5.0;
int nt = time/dt, nx = 801, nz = 401;

// FWI grids
int snap_z1 = 1, snap_z2 = nz-1, snap_x1=1, snap_x2=nx-1, snap_dt=4, snap_dz=2, snap_dx=2;

// Geometric boundaries definition
int surf [4] = {1, 0, 0, 0}; //(top surface on grid index 1)
int pml [4] = {0, 20, 20, 20}; // 20 PML grids in each side except top

// Order of finite difference and holberg coefficients
int fdorder = 2; // currently second order only
real hc[2] = {0.0, 1.0}; // Assuming Holberg's coefficient

// full waveform inversion parameters
bool fwinv = false; // true= fwi, false = no fwi computation

// Medium arguments (currently elastic)
real ** lam, ** mu, ** rho; 

// source, receivers and shots
int n_shots = 1;
int nsrc = 3; // number of sources
int stf_type = 1; // type of source (0 = Displacement, 1 = Velocity)
real **stf_z, **stf_x; // source time functions 
int *z_src, *x_src; // source locations

int nrec = 6; // Number of sources
int *z_rec, *x_rec; // grid index for receiver locations
real ** rtf_uz_true, ** rtf_ux_true; // Seismograms if available (in case of FWI)

/*
// temporary array inputs (source locations)
z_src[3] = {5, 5, 5}; x_src[3] = {200, 400, 600};

// Temporary compute STF
stf_z[nsrc][nt];
stf_x[nsrc][nt];

// temporary array inputs (reciever locations)
z_src[6] = {350, 350, 350, 350, 350, 350}; 
x_src[6] = {200, 300, 400, 500, 600, 700};

// Temporary compute true RTF
rtf_uz_true[nrec][nt];
rtf_ux_true[nrec][nt];
*/



// ------------------------------------------------------------------
// 2. VARIABLES TO BE INITIALIZED IN GPU (computed in PREPROCESSING)
// ------------------------------------------------------------------

// Medium arguments (currently elastic)
real ** mu_zx, ** rho_zp, ** rho_xp; //  average mu and inverse of density

//PML arguments
real ** a, ** b, ** K; 
real ** a_half, ** b_half, ** K_half;

// -------------------------------------------------------------------
// 3. VARIABLES TO BE INITIALIZED IN GPU (computed in SEISMIC KERNELS)
// -------------------------------------------------------------------

// Wave arguments (velocity, displacement and stress tensors)
real ** vx, ** vz,  ** ux,  ** uz;
real ** sxx, ** szx,  ** szz,  ** We;

// Spatial derivatives (for internal computations)
real **vz_z,  **vx_z,  **vz_x,  **vx_x;
real **szz_z,  **szx_z,  **szx_x, **sxx_x;

// Gradients of the medium (FWI parameters)
real **grad_lam,  ** grad_mu,  ** grad_rho;
real L2_norm; // L2 norm for residual computation

// PML memory arrays (only if PML exists in the system)
real ** mem_vx_x, ** mem_vx_z, ** mem_vz_x, ** mem_vz_z;
real ** mem_sxx_x, ** mem_szx_x, ** mem_szz_z, ** mem_szx_z;

// FWI parameters (FWI storage arrays)
real ***accu_vz, ***accu_vx; // accumulated memory storage over time
real ***accu_szz, ***accu_szx, ***accu_sxx; // accumulated memory storage over time

// Forward or Adjoint Kernel
bool adj_kernel = false; // false = forward kernel, true = adjoint kernel

// Reciever seismograms
int rtf_type = 0; //RTF type (0 = Displacement)
real **rtf_uz, **rtf_ux;

// Adjoint sources
int a_stf_type = 0; // adjoint source type
real ** a_stf_uz, ** a_stf_ux; // Adjoint source vectors
               

#endif