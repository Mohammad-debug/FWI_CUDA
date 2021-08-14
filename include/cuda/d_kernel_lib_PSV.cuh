

#ifndef SEISMIC_KERNEL_LIB2_H
#define SEISMIC_KERNEL_LIB2_H

#include "d_globvar.cuh"
void mat_grid2_GPU(
    // Gradients, material average and energy weights
    real* lam, real* mu, real * rho,
    real lam_sc, real mu_sc, real rho_sc,
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, int nz, int nx);

void copy_mat_GPU( real *lam_copy, real *mu_copy,  real *rho_copy,
        real *lam, real *mu,  real *rho, int nz, int nx);


void scale_grad_E2_GPU(
    // Gradients, material average and energy weights
    real *grad, real *grad_shot,
    real mat_av, real *We,
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, int nz, int nx);




real adjsrc2_GPU(int ishot, int *&a_stf_type, real *&a_stf_uz, real *&a_stf_ux, 
            int rtf_type, real *&rtf_uz_true, real *&rtf_ux_true, 
            real *&rtf_uz_mod, real *&rtf_ux_mod,             
            real dt, int nseis, int nt);


void energy_weights2_GPU(
    // Energy Weights (forward and reverse)
    real *&We, real *&We_adj, 
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, int nx);

void interpol_grad2_GPU( // Global and shot gradient
    real *grad, real *grad_shot, 
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, 
    int snap_dz, int snap_dx, int nx);

void update_mat2_GPU(real *&mat, real *&mat_old, real *&grad_mat,
                     real mat_max, real mat_min, real step_length, int nz, int nx);


void taper2_GPU(
    // Gradients, material average and energy weights
    real* A, int nz, int nx,
    int snap_z1, int snap_z2, int snap_x1, int snap_x2,
    int& taper_t1, int& taper_t2, int& taper_b1, int& taper_b2,
    int& taper_l1, int& taper_l2, int& taper_r1, int& taper_r2);
    
void mat_av2_GPU(
    // Medium arguments
    real *&lam, real *&mu, real *&rho,
    real *&mu_zx, real *&rho_zp, real *&rho_xp, // inverse of densityint dimz, int dimx
    real &C_lam, real &C_mu, real &C_rho,       // scalar averages
    int nz, int nx);

void reset_sv2_GPU(
    // wave arguments _GPU(velocity) & Energy weights
    real *&vz, real *&vx, real *&uz, real *&ux,
    real *&szz, real *&szx, real *&sxx, real *&We,
    // time & space grids _GPU(size of the arrays)
    real nz, real nx);

void reset_PML_memory2_GPU(
    // PML memory arrays
    real *&mem_vz_z, real *&mem_vx_z, real *&mem_vz_x, real *&mem_vx_x,
    // time & space grids _GPU(size of the arrays)
    real nz, real nx);

void reset_grad_shot2_GPU(real *&grad_lam, real *&grad_mu, real *&grad_rho,
                          int snap_z1, int snap_z2, int snap_x1, int snap_x2,
                          int snap_dz, int snap_dx, int nx);

void vdiff2_GPU(
    // spatial velocity derivatives
    real *&vz_z, real *&vx_z, real *&vz_x, real *&vx_x,
    // wave arguments _GPU(velocity)
    real *&vz, real *&vx,
    // holberg coefficient
    real *&hc,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dz, real dx, int nx);

void pml_diff2_GPU(bool pml_z, bool pml_x,
                   // spatial derivatives
                   real *&dz_z, real *&dx_z, real *&dz_x, real *&dx_x,
                   //PML arguments _GPU(z and x direction)
                   real *&a_z, real *&b_z, real *&K_z,
                   real *&a_half_z, real *&b_half_z, real *&K_half_z,
                   real *&a_x, real *&b_x, real *&K_x,
                   real *&a_half_x, real *&b_half_x, real *&K_half_x,
                   // PML memory arrays for spatial derivatives
                   real *&mem_z_z, real *&mem_x_z,
                   real *&mem_z_x, real *&mem_x_x,
                   // time space grids
                   int nz1, int nz2, int nx1, int nx2, int nx);

void update_s2_GPU(
    // Wave arguments _GPU(stress)
    real *&szz, real *&szx, real *&sxx,
    // spatial velocity derivatives
    real *&vz_z, real *&vx_z, real *&vz_x, real *&vx_x,
    // Medium arguments
    real *&lam, real *&mu, real *&mu_zx,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dt, int nx);

void sdiff2_GPU(
    // spatial stress derivatives
    real *&szz_z, real *&szx_z, real *&szx_x, real *&sxx_x,
    // Wave arguments _GPU(stress)
    real *&szz, real *&szx, real *&sxx,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dz, real dx,
    // holberg coefficient
    real *&hc, int nx);

void update_v2_GPU(
    // wave arguments _GPU(velocity) & Energy weights
    real *&vz, real *&vx,
    // displacement and energy arrays
    real *&uz, real *&ux, real *&We,
    // spatial stress derivatives
    real *&szz_z, real *&szx_z, real *&szx_x, real *&sxx_x,
    // Medium arguments
    real *&rho_zp, real *&rho_xp,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dt, int nx);

void surf_mirror_GPU(
    // Wave arguments _GPU(stress & velocity derivatives)
    real *&szz, real *&szx, real *&sxx, real *&vz_z, real *&vx_x,
    // Medium arguments
    real *&lam, real *&mu,
    // surface indices for four directions_GPU(0.top, 1.bottom, 2.left, 3.right)
    int *&surf,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dt, int nx);

void gard_fwd_storage2_GPU(
    // forward storage for full waveform inversion
    real *&accu_vz, real *&accu_vx,
    real *&accu_szz, real *&accu_szx, real *&accu_sxx,
    // velocity and stress tensors
    real *&vz, real *&vx, real *&szz, real *&szx, real *&sxx,
    // time and space parameters
    real dt, int itf, int snap_z1, int snap_z2,
    int snap_x1, int snap_x2, int snap_dz, int snap_dx, int nx);

void fwi_grad2_GPU(
    // Gradient of the materials
    real *&grad_lam, real *&grad_mu, real *&grad_rho,
    // forward storage for full waveform inversion
    real *&accu_vz, real *&accu_vx,
    real *&accu_szz, real *&accu_szx, real *&accu_sxx,
    // displacement and stress tensors
    real *&uz, real *&ux, real *&szz, real *&szx, real *&sxx,
    // Medium arguments
    real *&lam, real *&mu,
    // time and space parameters
    real dt, int tf, int snap_dt, int snap_z1, int snap_z2,
    int snap_x1, int snap_x2, int snap_dz, int snap_dx, int nx);

void urec2_GPU(int rtf_type,
               // reciever time functions
               real *&rtf_uz, real *&rtf_ux,
               // velocity tensors
               real *&vz, real *&vx,
               // reciever
               int nrec, int *&rz, int *&rx,
               // time and space grids
               int it, real dt, real dz, real dx, int nt, int nx);

void vsrc2_GPU(
    // Velocity tensor arrays
    real *&vz, real *&vx,
    // inverse of density arrays
    real *&rho_zp, real *&rho_xp,
    // source parameters
    int nsrc, int stf_type, real *&stf_z, real *&stf_x,
    int *&z_src, int *&x_src, int *&src_shot_to_fire,
    int ishot, int it, real dt, real dz, real dx, int nx);

#endif