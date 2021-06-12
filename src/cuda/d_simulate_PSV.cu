//simulate_PSV.cu

/* 
* Created by: Min Basnet
* 2020.November.26
* Kathmandu, Nepal
*/

/* 
* Translated to CUDA by: 
* 2021.April.12
* 
*/

// Interface between host and device code
// full waveform simulation of 2D plane (P/SV) seismic wave problems
// The device instances are created and memory are copied to the device from host
// The computations take place in CUDA
// The main file for calling gpu code

#include "d_simulate_PSV.cuh"
#include "d_preproc.cuh"
#include <iostream>
#include <math.h>

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
                    int &taper_l1, int &taper_l2, int &taper_r1, int &taper_r2, bool fwinv)
{

    // ---------------------------------------------
    // HOST DEVICE INTERFACE
    // ---------------------------------------------

    // ---------------------------------------------
    // MEMORY ESPECIALLY INPUT, PREPROCESS FOR SIMULATION
    // ---------------------------------------------

    real *d_hc;
    int *d_isurf;
    int *d_npml; // holberg coefficients, surface indices and number pml in each side
    // Medium arguments
    real *d_lam;
    real *d_mu;
    real *d_rho;
    //PML arguments (z and x direction)
    real *d_a_z;
    real *d_b_z;
    real *d_K_z;
    real *d_a_half_z;
    real *d_b_half_z;
    real *d_K_half_z;
    real *d_a_x;
    real *d_b_x;
    real *d_K_x;
    real *d_a_half_x;
    real *d_b_half_x;
    real *d_K_half_x;

    // Seismic sources
    int *d_z_src;
    int *d_x_src;            // source locations
    int *d_src_shot_to_fire; // which source to fire on which shot index
    real *d_stf_z;
    real *d_stf_x; // source time functions
    // Reciever seismograms

    // Reciever seismograms
    int *d_z_rec;
    int *d_x_rec;
    real *d_rtf_z_true;
    real *d_rtf_x_true; // Field measurements for receivers

    alloc_varpre_PSV_GPU(d_hc, d_isurf, d_npml, // holberg coefficients, surface indices and number pml in each side
                                                // Medium arguments
                         d_lam, d_mu, d_rho,
                         //PML arguments (z and x direction)
                         d_a_z, d_b_z, d_K_z,
                         d_a_half_z, d_b_half_z, d_K_half_z,
                         d_a_x, d_b_x, d_K_x,
                         d_a_half_x, d_b_half_x, d_K_half_x,
                         // Seismic sources
                         d_z_src, d_x_src,   // source locations
                         d_src_shot_to_fire, // which source to fire on which shot index
                         d_stf_z, d_stf_x,   // source time functions
                                             // Reciever seismograms
                         d_z_rec, d_x_rec,
                         d_rtf_z_true, d_rtf_x_true, // Field measurements for receivers
                                                     // Scalar variables for allocation
                         fdorder, pml_z, pml_x, nsrc, nrec, nt, nz, nx);

    // ------------------------------------------------------------------------
    //  MEMORY (INPUT) COPY TO THE DEVICE
    // -------------------------------------------------------------------------

    copy_varpre_PSV_CPU_TO_GPU(
        hc, isurf, npml, // holberg coefficients, surface indices and number pml in each side
                         // Medium arguments
        lam, mu, rho,
        //PML arguments (z and x direction)
        a_z, b_z, K_z,
        a_half_z, b_half_z, K_half_z,
        a_x, b_x, K_x,
        a_half_x, b_half_x, K_half_x,
        // Seismic sources
        z_src, x_src,     // source locations
        src_shot_to_fire, // which source to fire on which shot index
        stf_z, stf_x,     // source time functions
                          // Reciever seismograms
        z_rec, x_rec,
        //  rtf_z_true,  rtf_x_true, // Field measurements for receivers
        // Scalar variables for allocation

        /////////////////////////////////  GPU variables /////////////////////////////////////////////////////

        d_hc, d_isurf, d_npml, // holberg coefficients, surface indices and number pml in each side
                               // Medium arguments
        d_lam, d_mu, d_rho,
        //PML arguments (z and x direction)
        d_a_z, d_b_z, d_K_z,
        d_a_half_z, d_b_half_z, d_K_half_z,
        d_a_x, d_b_x, d_K_x,
        d_a_half_x, d_b_half_x, d_K_half_x,
        // Seismic sources
        d_z_src, d_x_src,   // source locations
        d_src_shot_to_fire, // which source to fire on which shot index
        d_stf_z, d_stf_x,   // source time functions
                            // Reciever seismograms
        d_z_rec, d_x_rec,
        d_rtf_z_true, d_rtf_x_true, // Field measurements for receivers
        // Scalar variables for allocation
        fdorder,
        pml_z, pml_x, nsrc, nrec,
        nt, nz, nx);

    // Calling global device functions (forward and fwi kernels)
    if (fwinv)
    {
        std::cout << "Here you go: FWI" << std::endl;
        //simulate_fwi_PSV() // call the global device code with device variables
    }
    else
    {
        std::cout << "Here you go: FWD" << std::endl;

        simulate_fwd_PSV_GPU(nt, nz, nx, dt, dz, dx,
                             snap_z1, snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx,
                             surf, pml_z, pml_x, nsrc, nrec, nshot, stf_type, rtf_type,
                             fdorder, scalar_lam, scalar_mu, scalar_rho,
                             d_hc, d_isurf,
                             d_lam, d_mu, d_rho,
                             d_a_z, d_b_z, d_K_z, d_a_half_z, d_b_half_z, d_K_half_z,
                             d_a_x, d_b_x, d_K_x, d_a_half_x, d_b_half_x, d_K_half_x,
                             d_z_src, d_x_src, d_z_rec, d_x_rec,
                             d_src_shot_to_fire, d_stf_z, d_stf_x,
                             accu_save, seismo_save);

        //simulate_fwd_PSV() // call the global device code with device variables
    }
}

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
                          bool accu_save, bool seismo_save)
{ // forward accumulated storage arrays){
    // Forward modelling in PSV

    // -------------------------------------------------------------------------------------------------------
    // Internally computational arrays
    // --------------------------------------------------------------------------------------------------------
    real *vz, *vx, *uz, *ux;                           // Tensors: velocity, displacement
    real *We, *We_adj, *szz, *szx, *sxx;               // Tensors: Energy fwd and adj, stress (We_adj used as temp grad)
    real *dz_z, *dx_z, *dz_x, *dx_x;                   // spatial derivatives
    real *mem_vz_z, *mem_vx_z, *mem_szz_z, *mem_szx_z; // PML spatial derivative memories: z-direction
    real *mem_vz_x, *mem_vx_x, *mem_szx_x, *mem_sxx_x; // PML spatial derivative memories: x-direction
    real *mu_zx, *rho_zp, *rho_xp;                     // Material averages
    real *lam_copy, *mu_copy, *rho_copy;               // Old material storage while updating (only in fwi)
    real *grad_lam, *grad_mu, *grad_rho;               // Gradients of material (full grid)
    //real *grad_lam_old, *grad_mu_old, *grad_rho_old; // Storing old material gradients for optimization
    real *grad_lam_shot, *grad_mu_shot, *grad_rho_shot;       // Gradient of materials in each shot (snapped)
    real *rtf_uz, *rtf_ux;                                    // receiver time functions (displacements)
    real *accu_vz, *accu_vx, *accu_szz, *accu_szx, *accu_sxx; // forward accumulated storage arrays
    // -----------------------------------------------------------------------------------------------------

    // Internal variables
    bool accu = true, grad = false;

    // int nt, nz, nx; // grid sizes
    // bool surf, pml_z, pml_x;
    // int fdorder; // order of the finite difference
    // int nsrc, nrec

    // allocating main computational arrays
    alloc_varmain_PSV_GPU(vz, vx, uz, ux, We, We_adj, szz, szx, sxx, dz_z, dx_z, dz_x, dx_x,
                          mem_vz_z, mem_vx_z, mem_szz_z, mem_szx_z, mem_vz_x, mem_vx_x, mem_szx_x, mem_sxx_x,
                          mu_zx, rho_zp, rho_xp, lam_copy, mu_copy, rho_copy, grad_lam, grad_mu, grad_rho,
                          grad_lam_shot, grad_mu_shot, grad_rho_shot, rtf_uz, rtf_ux, accu_vz, accu_vx,
                          accu_szz, accu_szx, accu_sxx, pml_z, pml_x, nrec, accu, grad, snap_z1,
                          snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx, nt, nz, nx);

    mat_av2_GPU(lam, mu, rho, mu_zx, rho_zp, rho_xp,
                scalar_lam, scalar_mu, scalar_rho, nz, nx);

    //Seismic forward kernel
    for (int ishot = 0; ishot < nshot; ishot++)
    {
        accu = true;  // Accumulated storage for output
        grad = false; // no gradient computation in forward kernel
        kernel_PSV_GPU(ishot, nt, nz, nx, dt, dx, dz, surf, isurf, hc, fdorder,
                       vz, vx, uz, ux, szz, szx, sxx, We, dz_z, dx_z, dz_x, dx_x,
                       lam, mu, mu_zx, rho_zp, rho_xp, grad, grad_lam, grad_mu, grad_rho,
                       pml_z, a_z, b_z, K_z, a_half_z, b_half_z, K_half_z,
                       pml_x, a_x, b_x, K_x, a_half_x, b_half_x, K_half_x,
                       mem_vz_z, mem_vx_z, mem_szz_z, mem_szx_z,
                       mem_vz_x, mem_vx_x, mem_szx_x, mem_sxx_x,
                       nsrc, stf_type, stf_z, stf_x, z_src, x_src, src_shot_to_fire,
                       nrec, rtf_type, rtf_uz, rtf_ux, z_rec, x_rec,
                       accu, accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx,
                       snap_z1, snap_z2, snap_x1, snap_x2,
                       snap_dt, snap_dz, snap_dx);

        //Saving the Accumulative storage file to a binary file for every shots
        // if (accu_save){
        //     // Writing the accumulation array
        //     std::cout << "Writing accu to binary file for SHOT " << ishot ;
        //     write_accu_GPU(accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx, nt, snap_z1, snap_z2, snap_x1,
        //     snap_x2, snap_dt, snap_dz, snap_dx, ishot);
        //     std::cout <<" <DONE>"<< std::endl;
        // }

        // // Saving the Accumulative storage file to a binary file for every shots
        // if (seismo_save){
        //     // Writing the accumulation array
        //     std::cout << "Writing accu to binary file for SHOT " << ishot  ;
        //     write_seismo_GPU(rtf_uz, rtf_ux, nrec, nt, ishot);
        //     std::cout <<" <DONE>"<< std::endl;
        // }
    }
}
