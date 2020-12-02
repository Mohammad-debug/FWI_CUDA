

#ifndef SEIS_KERNEL2_H		
#define SEIS_KERNEL2_H	

#include "device_globvar.hpp"

void reset_sv2(
    // wave arguments (velocity) & Energy weights
    real **vz, real **vx, real ** sxx, real ** szx, real ** szz, real **We, 
    // time & space grids (size of the arrays)
    real nz, real nx);


void vdiff2(
    // spatial velocity derivatives
    real **vz_z, real **vx_z, real **vz_x, real **vx_x,
    // wave arguments (velocity)
    real **vz, real **vx,
    // holberg coefficient
    real *hc,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, int dz, int dx);


void pml_vdiff2(
    // spatial velocity derivatives
    real **vz_z, real **vx_z, real **vz_x, real **vx_x,
    // PML memory arrays
    real** mem_vz_z, real** mem_vx_z, real** mem_vz_x, real ** mem_vx_x, 
    //PML arguments
    real ** a, real ** b, real ** K, 
    real ** a_half, real ** b_half, real ** K_half,
    // time space grids
    int nz1, int nz2, int nx1, int nx2);


void update_s2(
    // Wave arguments (stress)
    real ** sxx, real ** szx, real ** szz,
    // spatial velocity derivatives
    real **vz_z, real **vx_z, real **vz_x, real **vx_x,
    // Medium arguments
    real ** lam, real ** mu, real ** mu_zx,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, int dt);


void sdiff2(
    // spatial stress derivatives
    real **szz_z, real **szx_z, real **szx_x, real **sxx_x,
    // Wave arguments (stress)
    real ** sxx, real ** szx, real ** szz,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, int dz, int dx,
    // holberg coefficient
    real *hc);


void pml_sdiff2();


void update_v2(
    // wave arguments (velocity) & Energy weights
    real **vz, real **vx, 
    // displacement and energy arrays 
    real **uz, real **ux, real **We,
    // spatial stress derivatives
    real **szz_z, real **szx_z, real **szx_x, real **sxx_x,
    // Medium arguments
    real ** rho_zp, real ** rho_xp,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, int dt);


void surf_mirror(
    // Wave arguments (stress & velocity derivatives)
    real ** sxx, real ** szx, real ** szz, real **vz_z, real **vx_x,
    // Medium arguments
    real ** lam, real ** mu,
    // surface indices for four directions(0.top, 1.bottom, 2.left, 3.right)
    int *surf,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, int dt);


void gard_fwd_storage2(
    // forward storage for full waveform inversion 
    real ***accu_vz, real ***accu_vx, real ***accu_szz, real ***accu_szx, real ***accu_sxx,
    // velocity and stress tensors
    real **vz, real **vx, real **szz, real **szx, real **sxx,
    // time and space parameters
    int dt, int fw_it, int snap_z1, int snap_z2, int snap_x1, int snap_x2, int snap_dz, int snap_dx);


void fwi_grad2(
    // Gradient of the materials
    real **grad_lam, real ** grad_mu, real ** grad_rho,
    // forward storage for full waveform inversion 
    real ***accu_vz, real ***accu_vx, real ***accu_szz, real ***accu_szx, real ***accu_sxx,
    // displacement and stress tensors
    real **uz, real **ux, real **szz, real **szx, real **sxx,
    // Medium arguments
    real ** lam, real ** mu,
    // time and space parameters
    int dt, int tf, int snap_dt, int snap_z1, int snap_z2, int snap_x1, int snap_x2, int snap_dz, int snap_dx);

void vsrc2(
    // Velocity tensor arrays
    real **vz, real **vx, 
    // inverse of density arrays
    real **rho_zp, real **rho_xp,
    // source parameters
    int nsrc, int stf_type, real **stf_z, real **stf_x, 
    int *z_src, int *x_src, int it,
    int dt, int dz, int dx);

void urec2(int rtf_type,
    // reciever time functions
    real **rtf_uz, real **rtf_ux, 
    // velocity tensors
    real **vz, real **vx,
    // reciever time and space grids
    int nrec, int *rz, int *rx, int it, real dt, real dz, real dx);

real adjsrc2(int a_stf_type, real ** a_stf_uz, real ** a_stf_ux, 
            int rtf_type, real ** rtf_uz_true, real ** rtf_ux_true, 
            real ** rtf_uz_mod, real ** rtf_ux_mod,             
            real dt, int nseis, int nt);


#endif