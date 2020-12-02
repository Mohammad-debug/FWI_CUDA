
#ifndef FORWARD_KERNEL2_H		
#define FORWARD_KERNEL2_H	

#include "device_globvar.hpp"
#include "seismic_kernel_lib2.hpp"

void seismic_kernel2(
                // Time and space grid arguments
                int nt, int nz, int nx, 
                real dt, real dx, real dz, 
                // surface incides (0.top, 1.bottom, 2.left, 3.right)
                int *surf,
                // computationsl arguments
                real *hc, int fdorder, 
                // Wave arguments (velocity, displacement and stress tensors)
                real ** vx, real ** vz, real ** ux, real ** uz, 
                real ** sxx, real ** szx, real ** szz, real ** We,
                // Spatial derivatives (for internal computations)
                real **vz_z, real **vx_z, real **vz_x, real **vx_x, 
                real **szz_z, real **szx_z, real **szx_x, real **sxx_x, 
                // Medium arguments
                real ** lam, real ** mu, real ** mu_zx, 
                real ** rho_zp, real ** rho_xp, // inverse of density
                // Gradients of the medium
                real **grad_lam, real ** grad_mu, real ** grad_rho,
                //PML arguments
                bool pml, real ** a, real ** b, real ** K, 
                real ** a_half, real ** b_half, real ** K_half, 
                // PML memory arrays
                real ** mem_vx_x, real ** mem_vx_z, real ** mem_vz_x, real ** mem_vz_z,
                real ** mem_sxx_x, real ** mem_szx_x, real ** mem_szz_z, real ** mem_szx_z,
                // Seismic sources
                int nsrc, int stf_type, real **stf_z, real **stf_x, int *z_src, int *x_src,
                // Reciever seismograms
                int nrec, int rtf_type, real **rtf_uz, real **rtf_ux, int *z_rec, int *x_rec,
                // FWI parameters
                bool fwinv, real ***accu_vz, real ***accu_vx, 
                real ***accu_szz, real ***accu_szx, real ***accu_sxx,
                int snap_z1, int snap_z2, int snap_x1, int snap_x2, // grid boundaries for fwi
                int snap_dt, int snap_dz, int snap_dx, // time n space grid intervals to save storage
                // Output and print parameters
                bool adj_kernel);

#endif