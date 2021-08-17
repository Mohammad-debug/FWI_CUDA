// kernel_PSV.cpp

/* 
* Created by: Min Basnet
* 2020.November.21
* Kathmandu, Nepal
*/

// Contains the forward kernel for computation of
// Seismic wave propagation in time domain
// Currently elastic case only

#include "d_kernel_PSV.cuh"
#include <iostream>

void kernel_PSV_GPU(int ishot,              // shot index
                    int nt, int nz, int nx, // Time and space grid arguments
                    real dt, real dx, real dz,
                    // surface incides (0.top, 1.bottom, 2.left, 3.right)
                    bool surf, int *&isurf,
                    // computationsl arguments
                    real *&hc, int fdorder,
                    // Wave arguments (velocity, displacement and stress tensors)
                    real *&vz, real *&vx, real *&uz, real *&ux,
                    real *&szz, real *&szx, real *&sxx, real *&We,
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
                    bool accu, real *&accu_vz, real *&accu_vx,          //accumulated velocity memory over time
                    real *&accu_szz, real *&accu_szx, real *&accu_sxx,  //accumulated velocity memory over time
                    int snap_z1, int snap_z2, int snap_x1, int snap_x2, // grid boundaries for fwi
                    int snap_dt, int snap_dz, int snap_dx               // time n space grid intervals to save storage
)
{

    // Computes the forward wave propagation problem in time domain

    // internal variables
    int nz1, nz2, nx1, nx2; // start and end grids for the computation
    int tf;                 // time step for forward storage
    int it;                 // time index parameter

    // start and end indices of the computational FD grids
    int fpad = fdorder / 2;
    nz1 = fpad;
    nz2 = nz - fpad;
    nx1 = fpad;
    nx2 = nx - fpad; // index variables

    // Reset kernels
    reset_sv2_GPU(vz, vx, uz, ux, sxx, szx, szz, We, nz, nx);

    if (pml_z)
    { // Reset PML memory arrays if they exist
        reset_PML_memory2_GPU(mem_vz_z, mem_vx_z, mem_szz_z, mem_szx_z, nz, nx);
    }
    if (pml_x)
    { // Reset PML memory arrays if they exist
        reset_PML_memory2_GPU(mem_vz_x, mem_vx_x, mem_sxx_x, mem_szx_x, nz, nx);
    }

    if (grad)
    { // Reset gradient for each shots
        reset_grad_shot2_GPU(grad_lam, grad_mu, grad_rho,
                             snap_z1, snap_z2, snap_x1, snap_x2,
                             snap_dz, snap_dx, nx);
    }

    for (int jt = 0; jt < nt; jt++)
    {
        if (grad)
        {                     // Adjoint kernel
            it = nt - jt - 1; // Starts from back to front in adjoint modelling
        }
        else
        {
            it = jt; // Forward modelling
        }

        if (!(it % 1000) || (it == nt - 1))
        {
            std::cout << "Time step: " << it << " of " << nt - 1 << std::endl;
        }

        // -----------------------------------------------------------------------11
        // STEP 1: UPDATING STRESS TENSOR
        // -----------------------------------------------------------------------
        //timing start
        clock_t time1,time2;
        double net=0.0;
       // time1=clock();
        // 1.1: Spatial velicity derivatives

         //std::cout << "Time step: " << it  << std::endl;


        vdiff2_GPU(dz_z, dx_z, dz_x, dx_x, vz, vx, hc, nz1, nz2, nx1, nx2, dz, dx, nx);

        // 1.2: PML memory update for velocity gradients (if any)
        if (pml_z || pml_x)
        {
            pml_diff2_GPU(pml_z, pml_x, dz_z, dx_z, dz_x, dx_x,
                          a_z, b_z, K_z, a_half_z, b_half_z, K_half_z,
                          a_x, b_x, K_x, a_half_x, b_half_x, K_half_x,
                          mem_vz_z, mem_vx_z, mem_vz_x, mem_vx_x,
                          nz1, nz2, nx1, nx2, nx);
        }

        // 1.3: Update stress tensor
        update_s2_GPU(szz, szx, sxx, dz_z, dx_z, dz_x, dx_x,
                      lam, mu, mu_zx, nz1, nz2, nx1, nx2, dt, nx);

        // 1.4: Apply mirroring techniques for surfaces conditions (if any)
        if (surf)
        {
            surf_mirror_GPU(szz, szx, sxx, dz_z, dx_x,
                            lam, mu, isurf, nz1, nz2, nx1, nx2, dt, nx);
        }

        // -----------------------------------------------------------------------11

        // -----------------------------------------------------------------------
        // STEP 2: UPDATING VELOCITY TENSOR
        // -----------------------------------------------------------------------

        // 2.1: Spatial stress derivatives
        sdiff2_GPU(dz_z, dx_z, dz_x, dx_x, szz, szx, sxx, nz1, nz2, nx1, nx2, dz, dx, hc, nx);

        // 2.2: PML memory update for stress gradients (if any)
        if (pml_z || pml_x)
        {
            pml_diff2_GPU(pml_z, pml_x, dz_z, dx_z, dz_x, dx_x,
                          a_z, b_z, K_z, a_half_z, b_half_z, K_half_z,
                          a_x, b_x, K_x, a_half_x, b_half_x, K_half_x,
                          mem_szz_z, mem_szx_z, mem_szx_x, mem_sxx_x,
                          nz1, nz2, nx1, nx2, nx);
        }

        // 2.3: Update velocity tensor
        update_v2_GPU(vz, vx, uz, ux, We, dz_z, dx_z, dz_x, dx_x, rho_zp, rho_xp, nz1, nz2, nx1, nx2, dt, nx);
        
        // time2=clock();
        // net =(time2-time1)/ (double)CLOCKS_PER_SEC;
        // printf("\n timing GPU  = %f seconds end\n", net);
        // exit(0);

        // -----------------------------------------------------------------------

        // -----------------------------------------------------------------------
        // STEP 3: SOURCES AND RECEIVERS  (VELOCITY SOURCES)
        // -----------------------------------------------------------------------

        // 3.1: Firing the source terms
        if (nsrc)
        { // source seismograms exist
            // Adding source term corresponding to velocity
            //std::cout <<"The source applied here: "<<std::endl;
             vsrc2_GPU(vz, vx, rho_zp, rho_xp, nsrc, stf_type, stf_z, stf_x,
                       z_src, x_src, src_shot_to_fire, ishot, it, dt, dz, dx, nx);
        }

        // 3.2: Recording the displacements to the recievers
        if (nrec && !grad)
        { // reciever seismograms exist
            // Recording to the receivers
            urec2_GPU(rtf_type, rtf_uz, rtf_ux, vz, vx, nrec, z_rec, x_rec, it, dt, dz, dx, nt, nx);
        }
    //temporary
        // -----------------------------------------------------------------------

        // -----------------------------------------------------------------------
        // STEP 4: OUTPUTS, SCREENPRINTS INFO IF ANY
        // -----------------------------------------------------------------------

        // -----------------------------------------------------------------------

        // -----------------------------------------------------------------------
        // STEP 5: GRADIENT COMPUTATION (STORE TENSORS: FORWARD MODELLING)
        // -----------------------------------------------------------------------

        // Time index in forward accumulated storage array
        if (!(it % snap_dt))
        {
            tf = it / snap_dt;

            if (accu)
            { // Forward kernel (store tensor arrays)

                gard_fwd_storage2_GPU(accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx,
                                      vz, vx, szz, szx, sxx, dt, tf, snap_z1, snap_z2, snap_x1, snap_x2, snap_dz, snap_dx, nx);
            }

            if (grad)
            { // Adjoint kernel (calculate gradients)
                fwi_grad2_GPU(grad_lam, grad_mu, grad_rho, accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx,
                              uz, ux, szz, szx, sxx, lam, mu, dt, tf, snap_dt, snap_z1, snap_z2, snap_x1, snap_x2, snap_dz, snap_dx, nx);
            }
        }
    }
}
