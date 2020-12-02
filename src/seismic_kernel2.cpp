//seismic_kernel2.cpp

/* 
* Created by: Min Basnet
* 2020.November.21
* Kathmandu, Nepal
*/

// Contains the forward kernel for computation of 
// Seismic wave propagation in time domain
// Currently elastic case only

#include "seismic_kernel2.hpp"

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
                bool fwinv, real ***accu_vz, real ***accu_vx, //accumulated velocity memory over time
                real ***accu_szz, real ***accu_szx, real ***accu_sxx, //accumulated velocity memory over time
                int snap_z1, int snap_z2, int snap_x1, int snap_x2, // grid boundaries for fwi
                int snap_dt, int snap_dz, int snap_dx, // time n space grid intervals to save storage
                // Output and print parameters
                bool adj_kernel){

    // Computes the forward wave propagation problem in time domain

    // internal variables
    int nz1, nz2, nx1, nx2; // start and end grids for the computation
    int tf; // time step for forward storage
    int it; // time index parameter


    // start and end indices of the computational FD grids
    int fpad = fdorder/2;
    nz1 = fpad; nz2 = nz-fpad; nx1 = fpad; nx2 = nx-fpad; // index variables

    // reset the wave parameters (stress & velocities)
    reset_sv2(vz, vx, sxx, szx, szz, We, nz, nx);

    for(int jt=0; jt<nt; jt++){
        if(adj_kernel) it=nt-jt-1; // Starts from back to front in adjoint modelling 

        // -----------------------------------------------------------------------
        // STEP 1: UPDATING STRESS TENSOR
        // -----------------------------------------------------------------------

        // 1.1: Spatial velicity derivatives
        vdiff2(vz_z, vx_z, vz_x, vx_x, vz, vx, hc, nz1, nz2, nx1, nx2, dz, dx);

        // 1.2: PML memory update for velocity gradients (if any)
        if(pml){
            pml_vdiff2(vz_z, vx_z, vz_x, vx_x, mem_vz_z, mem_vx_z, mem_vz_x, mem_vx_x,
                    a, b, K, a_half, b_half, K_half, nz1, nz2, nx1, nx2);
        }

        // 1.3: Update stress tensor
        update_s2(sxx, szx, szz, vz_z, vx_z, vz_x, vx_x, 
                    lam, mu, mu_zx, nz1, nz2, nx1, nx2, dt);

        // 1.4: Apply mirroring techniques for surfaces conditions (if any)
        surf_mirror(sxx, szx, szz, vz_z, vx_x, 
                    lam, mu, surf, nz1, nz2, nx1, nx2, dt);
        // -----------------------------------------------------------------------


        // -----------------------------------------------------------------------
        // STEP 2: UPDATING VELOCITY TENSOR
        // -----------------------------------------------------------------------

        // 2.1: Spatial stress derivatives
        sdiff2(szz_z, szx_z, szx_x, sxx_x, sxx, szx, szz, nz1, nz2, nx1, nx2, dz, dx, hc);

        // 2.2: PML memory update for stress gradients (if any)
        if(pml){
            pml_vdiff2(szz_z, szx_z, szx_x, sxx_x, mem_szz_z, mem_szx_z, mem_szx_x, mem_sxx_x,
                    a, b, K, a_half, b_half, K_half, nz1, nz2, nx1, nx2);
        }

        // 2.3: Update velocity tensor
        update_v2(vz, vx, uz, ux, We, szz_z, szx_z, szx_x, sxx_x, rho_zp, rho_xp, nz1, nz2, nx1, nx2, dt);

        // -----------------------------------------------------------------------

        // -----------------------------------------------------------------------
        // STEP 3: SOURCES AND RECEIVERS  (VELOCITY SOURCES)  
        // -----------------------------------------------------------------------

        // 3.1: Firing the source terms
        if(nsrc){ // source seismograms exist
            // Adding source term corresponding to velocity
            vsrc2(vz, vx, rho_zp, rho_xp, nsrc, stf_type, stf_z, stf_x, 
                z_src, x_src, it, dt, dz, dx);
        }

        // 3.2: Recording the velocity to the recievers
        if(nrec && !adj_kernel){ // reciever seismograms exist
            // Recording to the receivers
            urec2(rtf_type, rtf_uz, rtf_ux, vz, vx, nrec, z_rec, x_rec, it, dt, dz, dx);
        }

        // -----------------------------------------------------------------------

        // -----------------------------------------------------------------------
        // STEP 4: OUTPUTS, SCREENPRINTS INFO IF ANY
        // -----------------------------------------------------------------------

        // -----------------------------------------------------------------------

        // -----------------------------------------------------------------------
        // STEP 5: GRADIENT COMPUTATION (STORE TENSORS: FORWARD MODELLING)   
        // -----------------------------------------------------------------------

        // Time index in forward storage array
        if (fwinv && !(it%snap_dt)){
            tf = it/snap_dt; 
            if (!adj_kernel){ // Forward kernel (store tensor arrays)

                gard_fwd_storage2(accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx,
                        vz, vx, szz, szx, sxx, dt, tf, snap_z1, snap_z2, snap_x1, snap_x2, snap_dz, snap_dx);

            }

            if (adj_kernel){ // Adjoint kernel (calculate gradients)
                fwi_grad2(grad_lam, grad_mu, grad_rho, accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx,
                        uz, ux, szz, szx, sxx, lam, mu, dt, tf, snap_dt, snap_z1, snap_z2, snap_x1, snap_x2, snap_dz, snap_dx);


            }

        }
        
    }

}
