//fwi_simulate.cpp

/* 
* Created by: Min Basnet
* 2020.November.26
* Kathmandu, Nepal
*/

// full waveform simulation of 2D plane (P/SV) seismic wave problems

#include "device_globvar_temp.hpp"
#include "seismic_kernel2.hpp"

// Preprocessing
// calculation of PML arrays

void fwi_simulate(){
    // full waveform inversion modelling
    // fwinv = true for this case
    fwinv = true;

    //-----------------------------------------------
    // 1.0. OUTER PREPROCESSING (IN EVERY FWI LOOPS)
    // ----------------------------------------------
    // Outside every shot loop


    // calculate PML
    int iter; iter = 0;
    while (iter=10){ // currently 10 just for test
        //-----------------------------------------------
        // 1.0. INNER PREPROCESSING (IN EVERY FWI LOOPS)
        // ----------------------------------------------


        // Update kernels (except first step, if not computed in the end of previous loop)
        // lam, mu, rho // Medium update and recalculation together
        // grad_lam, grad_mu, grad_rho, // gradients reset at the same time (CHECK)

        // calculate material averages
        // mu_zx, rho_zp, rho_xp
    
        for (int i_shot = 0; i_shot < n_shots; ++i_shot){
        
        
            // -----------------------------------
            // 2.0. FORWARD MODELLING
            // ------------------------------------
    
            adj_kernel = false;

            // Reset kernels
            // vx, vz, ux, uz, sxx, szx, szz, We, // wave parameters
            // vz_z, vx_z, vz_x, vx_x, szz_z, szx_z, szx_x, sxx_x, //spatial derivatives
            // mem_vx_x, mem_vx_z, mem_vz_x, mem_vz_z, // PML memory
            // mem_sxx_x, mem_szx_x, mem_szz_z, mem_szx_z, // PML memory
            // accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx, //fwi memory
            // rtf_uz, rtf_ux, z_rec, x_rec, // recievers (CHECK IF NECESSARY)

            // Seismic forward kernel
            seismic_kernel2(nt, nz, nx, dt, dx, dz, surf, hc, fdorder, //grids
                        vx, vz, ux, uz, sxx, szx, szz, We, // wave parameters
                        vz_z, vx_z, vz_x, vx_x, szz_z, szx_z, szx_x, sxx_x, //spatial derivatives
                        lam, mu, mu_zx, rho_zp, rho_xp, // Medium
                        grad_lam, grad_mu, grad_rho, // gradients
                        pml, a, b, K, a_half, b_half, K_half, // PML var
                        mem_vx_x, mem_vx_z, mem_vz_x, mem_vz_z, // PML memory
                        mem_sxx_x, mem_szx_x, mem_szz_z, mem_szx_z, // PML memory
                        nsrc, stf_type, stf_z, stf_x, z_src, x_src, // source
                        nrec, rtf_type, rtf_uz, rtf_ux, z_rec, x_rec, // recievers
                        fwinv, accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx, //fwi memory
                        snap_z1, snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx, // grid
                        adj_kernel);

            // -----------------------------------------------
            // 2.0. RESIDUALS AND ADJOINT SOURCE COMPUTATION
            // ------------------------------------------------

            // calculating L2 norm and adjoint sources
            L2_norm = adjsrc2(a_stf_type, a_stf_uz, a_stf_ux, rtf_type, rtf_uz_true, rtf_ux_true,
                            rtf_uz, rtf_ux, dt, nrec, nt);

            // -----------------------------------
            // 4.0. ADJOING MODELLING
            // ------------------------------------
    
            adj_kernel = true;

            // Reset necessary seismic kernels;
            // vx, vz, ux, uz, sxx, szx, szz, We, // wave parameters
            // vz_z, vx_z, vz_x, vx_x, szz_z, szx_z, szx_x, sxx_x, //spatial derivatives
            // mem_vx_x, mem_vx_z, mem_vz_x, mem_vz_z, // PML memory
            // mem_sxx_x, mem_szx_x, mem_szz_z, mem_szx_z, // PML memory

            // Seismic adjoint kernel
            seismic_kernel2(nt, nz, nx, dt, dx, dz, surf, hc, fdorder, //grids
                        vx, vz, ux, uz, sxx, szx, szz, We, // wave parameters
                        vz_z, vx_z, vz_x, vx_x, szz_z, szx_z, szx_x, sxx_x, //spatial derivatives
                        lam, mu, mu_zx, rho_zp, rho_xp, // Medium
                        grad_lam, grad_mu, grad_rho, // gradients
                        pml, a, b, K, a_half, b_half, K_half, // PML var
                        mem_vx_x, mem_vx_z, mem_vz_x, mem_vz_z, // PML memory
                        mem_sxx_x, mem_szx_x, mem_szz_z, mem_szx_z, // PML memory
                        nrec, a_stf_type, a_stf_uz, a_stf_ux, z_rec, x_rec, // adjoint source
                        nrec, rtf_type, rtf_uz, rtf_ux, z_rec, x_rec, // recievers
                        fwinv, accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx, //accumulated tensor memory
                        snap_z1, snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx, // grid
                        adj_kernel);


        }

    
        // -------------------
        // 5.0. OPTIMIZATION
        // -------------------


        
    }



}