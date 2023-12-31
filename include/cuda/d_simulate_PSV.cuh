//fwi_simulate.cpp

/* 
* Created by: Min Basnet
* 2020.November.26
* Kathmandu, Nepal
*/

// full waveform simulation of 2D plane (P/SV) seismic wave problems
#ifndef DEVICE_SIMULATE2_H
#define DEVICE_SIMULATE2_H

#include "d_globvar.cuh"
#include "d_alloc_PSV.cuh"
#include "d_kernel_lib_PSV.cuh"
#include "d_kernel_PSV.cuh"
#include "d_step_length_PSV.cuh"
#include "../h_io_PSV.hpp"
/*
#include "d_kernel_PSV.cuh"
#include "d_alloc_PSV.cuh"
#include "d_kernel_lib_PSV.cuh"
#include "d_step_length_PSV.cuh"
include "h_io_PSV.hpp"
//#include "d_PCG_PSV.cuh"
*/
void g_simulate_PSV(int *&npml, int nt, int nz, int nx, real dt, real dz, real dx,
                    int snap_z1, int snap_z2, int snap_x1, int snap_x2, int snap_dt, int snap_dz, int snap_dx,
                    bool surf, bool pml_z, bool pml_x, int nsrc, int nrec, int nshot, int stf_type, int rtf_type,
                    int fdorder, real scalar_lam, real scalar_mu, real scalar_rho,
                    real *&hc, int *&isurf, real **&lam, real **&mu, real **&rho,
                    real *&a_z, real *&b_z, real *&K_z, real *&a_half_z, real *&b_half_z, real *&K_half_z,
                    real *&a_x, real *&b_x, real *&K_x, real *&a_half_x, real *&b_half_x, real *&K_half_x,
                    int *&z_src, int *&x_src, int *&z_rec, int *&x_rec,
                    int *&src_shot_to_fire, real **&stf_z, real **&stf_x,
                    bool accu_save, bool seismo_save,
                    real ***&rtf_z_true, real ***&rtf_x_true,
                    int mat_save_interval, int &taper_t1, int &taper_t2, int &taper_b1, int &taper_b2,
                    int &taper_l1, int &taper_l2, int &taper_r1, int &taper_r2, bool fwinv);
                    
void simulate_fwi_PSV_GPU(int nt, int nz, int nx, real dt, real dz, real dx, 
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, int snap_dt, int snap_dz, int snap_dx, 
    bool surf, bool pml_z, bool pml_x, int nsrc, int nrec, int nshot, int stf_type, int rtf_type, 
    int fdorder, real scalar_lam, real scalar_mu, real scalar_rho,
    real *&hc, int *&isurf, real *&lam, real *&mu, real *&rho, 
    real *&a_z, real *&b_z, real *&K_z, real *&a_half_z, real *&b_half_z, real *&K_half_z,
    real *&a_x, real *&b_x, real *&K_x, real *&a_half_x, real *&b_half_x, real *&K_half_x,
    int *&z_src, int *&x_src, int *&z_rec, int *&x_rec,
    int *&src_shot_to_fire, real *&stf_z, real *&stf_x, real *&rtf_z_true, real *&rtf_x_true,
    int mat_save_interval, int &taper_t1, int &taper_t2, int &taper_b1, int &taper_b2, 
    int &taper_l1, int &taper_l2, int &taper_r1, int &taper_r2);

void simulate_fwd_PSV_GPU(int nt, int nz, int nx, real dt, real dz, real dx,
                          int snap_z1, int snap_z2, int snap_x1, int snap_x2, int snap_dt, int snap_dz, int snap_dx,
                          bool surf, bool pml_z, bool pml_x, int nsrc, int nrec, int nshot, int stf_type, int rtf_type,
                          int fdorder, real scalar_lam, real scalar_mu, real scalar_rho,

                          real *&hc, int *&isurf,

                          real *&lam, real *&mu, real *&rho,

                          real *&a_z, real *&b_z, real *&K_z, real *&a_half_z, real *&b_half_z, real *&K_half_z,
                          real *&a_x, real *&b_x, real *&K_x, real *&a_half_x, real *&b_half_x, real *&K_half_x,
                          int *&z_src, int *&x_src, int *&z_rec, int *&x_rec,

                          int *&src_shot_to_fire, real *&stf_z, real *&stf_x,
                          bool accu_save, bool seismo_save);

#endif