

#ifndef SEISMIC_KERNEL_LIB2_H		
#define SEISMIC_KERNEL_LIB2_H	


#include "d_globvar.hpp"

void reset_sv2(
    // wave arguments (velocity) & Energy weights
    real **&vz, real **&vx, real **&uz, real **&ux, 
    real **&szz, real **&szx, real **&sxx, real **&We, 
    // time & space grids (size of the arrays)
    real nz, real nx);


void reset_PML_memory2(
    // PML memory arrays
    real **&mem_vz_z, real **&mem_vx_z, real **&mem_vz_x, real **&mem_vx_x, 
    // time & space grids (size of the arrays)
    real nz, real nx);

void reset_grad_shot2(real **&grad_lam, real **&grad_mu, real **&grad_rho,
					int snap_z1, int snap_z2, int snap_x1, int snap_x2,
					int snap_dz, int snap_dx);

void vdiff2(
    // spatial velocity derivatives
    real **&vz_z, real **&vx_z, real **&vz_x, real **&vx_x,
    // wave arguments (velocity)
    real **&vz, real **&vx,
    // holberg coefficient
    real *&hc,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dz, real dx);

void pml_diff2(bool pml_z, bool pml_x,
    // spatial derivatives
    real **&dz_z, real **&dx_z, real **&dz_x, real **&dx_x,
    //PML arguments (z and x direction)
    real *&a_z, real *&b_z, real *&K_z, 
    real *&a_half_z, real *&b_half_z, real *&K_half_z,
    real *&a_x, real *&b_x, real *&K_x, 
    real *&a_half_x, real *&b_half_x, real *&K_half_x, 
    // PML memory arrays for spatial derivatives
    real **&mem_z_z, real **&mem_x_z, 
    real **&mem_z_x, real **&mem_x_x,  
    // time space grids
    int nz1, int nz2, int nx1, int nx2);

void update_s2(
    // Wave arguments (stress)
    real **&szz, real **&szx, real **&sxx, 
    // spatial velocity derivatives
    real **&vz_z, real **&vx_z, real **&vz_x, real **&vx_x,
    // Medium arguments
    real **&lam, real **&mu, real **&mu_zx,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dt);

void sdiff2(
    // spatial stress derivatives
    real **&szz_z, real **&szx_z, real **&szx_x, real **&sxx_x,
    // Wave arguments (stress)
    real **&szz, real **&szx, real **&sxx, 
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dz, real dx,
    // holberg coefficient
    real *&hc);

void update_v2(
    // wave arguments (velocity) & Energy weights
    real **&vz, real **&vx, 
    // displacement and energy arrays 
    real **&uz, real **&ux, real **&We,
    // spatial stress derivatives
    real **&szz_z, real **&szx_z, real **&szx_x, real **&sxx_x,
    // Medium arguments
    real **&rho_zp, real **&rho_xp,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dt);


void surf_mirror(
    // Wave arguments (stress & velocity derivatives)
    real **&szz, real **&szx, real **&sxx, real **&vz_z, real **&vx_x,
    // Medium arguments
    real **&lam, real **&mu,
    // surface indices for four directions(0.top, 1.bottom, 2.left, 3.right)
    int *&surf,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dt);

void gard_fwd_storage2(
    // forward storage for full waveform inversion 
    real ***&accu_vz, real ***&accu_vx, 
    real ***&accu_szz, real ***&accu_szx, real ***&accu_sxx,
    // velocity and stress tensors
    real **&vz, real **&vx, real **&szz, real **&szx, real **&sxx,
    // time and space parameters
    real dt, int itf, int snap_z1, int snap_z2, 
    int snap_x1, int snap_x2, int snap_dz, int snap_dx);

void fwi_grad2(
    // Gradient of the materials
    real **&grad_lam, real **&grad_mu, real **&grad_rho,
    // forward storage for full waveform inversion 
    real ***&accu_vz, real ***&accu_vx, 
    real ***&accu_szz, real ***&accu_szx, real ***&accu_sxx,
    // displacement and stress tensors
    real **&uz, real **&ux, real **&szz, real **&szx, real **&sxx,
    // Medium arguments
    real **&lam, real **&mu,
    // time and space parameters
    real dt, int tf, int snap_dt, int snap_z1, int snap_z2, 
    int snap_x1, int snap_x2, int snap_dz, int snap_dx);

void vsrc2(
    // Velocity tensor arrays
    real **&vz, real **&vx, 
    // inverse of density arrays
    real **&rho_zp, real **&rho_xp,
    // source parameters
    int nsrc, int stf_type, real **&stf_z, real **&stf_x, 
    int *&z_src, int *&x_src, int *&src_shot_to_fire,
    int ishot, int it, real dt, real dz, real dx);

void urec2(int rtf_type,
    // reciever time functions
    real **&rtf_uz, real **&rtf_ux, 
    // velocity tensors
    real **&vz, real **&vx,
    // reciever 
    int nrec, int *&rz, int *&rx, 
    // time and space grids
    int it, real dt, real dz, real dx);


real adjsrc2(int *&a_stf_type, real **&a_stf_uz, real **&a_stf_ux, 
            int rtf_type, real **&rtf_uz_true, real **&rtf_ux_true, 
            real **&rtf_uz_mod, real **&rtf_ux_mod,             
            real dt, int nseis, int nt);

void interpol_grad2(
    // Global and shot gradient
    real **&grad, real **&grad_shot, 
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, 
    int snap_dz, int snap_dx);

void energy_weights2(
    // Energy Weights (forward and reverse)
    real **&We, real **&We_adj, 
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2);

void scale_grad_E2(
    // Gradients, material average and energy weights
    real **&grad, real **&grad_shot, real mat_av, real **&We,
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2);

void update_mat2(real **&mat, real **&grad_mat, 
            real step_length, int nz, int nx);

void mat_av2(
    // Medium arguments
    real **&lam, real **&mu, real **&rho,
    real **&mu_zx, real **&rho_zp, real **&rho_xp, // inverse of densityint dimz, int dimx
    real &C_lam, real &C_mu, real &C_rho, // scalar averages
    int nz, int nx);

void mat_grid2(real **&lam, real **&mu, real **&rho, 
    real lam_sc, real mu_sc, real rho_sc, int nz, int nx);



#endif