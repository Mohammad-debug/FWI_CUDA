//alloc_PSV.cpp

/* 
* Created by: Min Basnet
* 2020.December.03
* Kathmandu, Nepal
*/

// Allocation and deallocation of memory for PSV problems 

#include "n_alloc_PSV.hpp"


void alloc_varin_PSV( real *&hc, int *&isurf, int *&npml, // holberg coefficients, surface indices and number pml in each side
    // Medium arguments
    real **&lam, real **&mu, real **&rho,
    //PML arguments (z and x direction)
    real *&a_z, real *&b_z, real *&K_z, 
    real *&a_half_z, real *&b_half_z, real *&K_half_z,
    real *&a_x, real *&b_x, real *&K_x, 
    real *&a_half_x, real *&b_half_x, real *&K_half_x, 
    // Seismic sources
    int *&z_src, int *&x_src, // source locations
    int *&src_shot_to_fire, // which source to fire on which shot index
    real **&stf_z, real **&stf_x, // source time functions
    // Reciever seismograms
    int *&z_rec, int *&x_rec,
    real ***&rtf_z_true, real ***&rtf_x_true, // Field measurements for receivers
    // Scalar variables for allocation
    int fdorder, 
    bool pml_z, bool pml_x, int nshot, int nsrc, int nrec, bool rtf_true,
    int nt, int nz, int nx){
    // Allocates the variables that are to be...
    // Transferred from device to the host
    // These Arrays need to take array readings from (device to the host)
    // Every shots should have same number of time steps

    // Allocating holberg coefficient
    hc = new real[fdorder];
    // Allocating surface indices & npml
    isurf = new int[4]; // four sized in the surface
    npml = new int[4]; // number of PMLs in each side
    // Allocating medium arrays
    allocate_array(lam, nz, nx);
    allocate_array(mu, nz, nx);
    allocate_array(rho, nz, nx);

    // Allocating PML coeffieients
    if (pml_z){
        a_z = new real[nz];
        b_z = new real[nz];
        K_z = new real[nz];

        a_half_z = new real[nz];
        b_half_z = new real[nz];
        K_half_z = new real[nz];
    }

    if (pml_x){
        a_x = new real[nx];
        b_x = new real[nx];
        K_x = new real[nx];

        a_half_x = new real[nx];
        b_half_x = new real[nx];
        K_half_x = new real[nx];
    }

    // Allocating Source locations and time functions
    if (nsrc){
        // locations (grid index)
        z_src = new int[nsrc];
        x_src = new int[nsrc];
        src_shot_to_fire = new int[nsrc];

        // stf
        allocate_array(stf_z, nsrc, nt);
        allocate_array(stf_x, nsrc, nt);
    }

    if (nrec){
        // locations (grid index)
        z_rec = new int[nrec];
        x_rec = new int[nrec];

        if (rtf_true){
            // rtf field measurements
            allocate_array(rtf_z_true, nshot, nrec, nt);
            allocate_array(rtf_x_true, nshot, nrec, nt);
        }

    }

}

void alloc_varmain_PSV(
    // Wave arguments 
    real **&vz, real **&vx,  // velocity
    real **&uz, real **&ux, // Displacement
    real **&We, real **&We_adj, // Energy fwd and adj
    real **&szz, real **&szx, real **&sxx,
    // Spatial derivatives (for internal computations)
    real **&dz_z, real **&dx_z, real **&dz_x, real **&dx_x, 
    // PML memory arrays for spatial derivatives
    real **&mem_vz_z, real **&mem_vx_z, 
    real **&mem_szz_z,  real **&mem_szx_z, 
    real **&mem_vz_x, real **&mem_vx_x,  
    real **&mem_szx_x, real **&mem_sxx_x,
    // Material average arrays
    real **&mu_zx, real **&rho_zp, real **&rho_xp,
    // Copy old material while updating
    real **&lam_copy, real **&mu_copy, real **&rho_copy,
    // Gradients of the medium
    real **&grad_lam, real **&grad_mu, real **&grad_rho, 
    // Gradients for each shot
    real **&grad_lam_shot, real **&grad_mu_shot, real **&grad_rho_shot,
    // reciever time functions
    real **&rtf_uz, real **&rtf_ux,
    // Accumulate the snap of forward wavefield parameters
    real ***&accu_vz, real ***&accu_vx, //accumulated velocity memory over time
    real ***&accu_szz, real ***&accu_szx, real ***&accu_sxx, //accumulated velocity memory over time
    bool pml_z, bool pml_x, int nrec, bool accu, bool grad, 
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, // grid boundaries for fwi
    int snap_dt, int snap_dz, int snap_dx, // time n space grid intervals to save storage
    int nt, int nz, int nx){


    // allocate velocities
    allocate_array(vz, nz, nx); 
    allocate_array(vx, nz, nx); 

    // allocate displacements
    allocate_array(uz, nz, nx); 
    allocate_array(ux, nz, nx); 

    // allocate stress tensors
    allocate_array(szz, nz, nx); 
    allocate_array(szx, nz, nx); 
    allocate_array(sxx, nz, nx); 

    // Allocate Energy for forward modelling
    allocate_array(We, nz, nx); 

    // spatial derivatives 
    // (same arrays can be used for ...
    // ...velocity and stress computations)
    allocate_array(dz_z, nz, nx);
    allocate_array(dx_z, nz, nx);
    allocate_array(dz_x, nz, nx);
    allocate_array(dx_x, nz, nx);

    // Allocating PML memory array for spatial derivatives
    if (pml_z){ // PMLs in z-direction
        allocate_array(mem_vz_z, nz, nx);
        allocate_array(mem_vx_z, nz, nx);
        allocate_array(mem_szz_z, nz, nx);
        allocate_array(mem_szx_z, nz, nx);
    }
    
    if (pml_x){ // PMLs in x-direction
        allocate_array(mem_vz_x, nz, nx);
        allocate_array(mem_vx_x, nz, nx);
        allocate_array(mem_szx_x, nz, nx);
        allocate_array(mem_sxx_x, nz, nx);
    }

    // Allocates material average arrays
    allocate_array(mu_zx, nz, nx);
    allocate_array(rho_zp, nz, nx);
    allocate_array(rho_xp, nz, nx);


    // rtf (used as adjoint sources in adjoint modelling)
    allocate_array(rtf_uz, nrec, nt);
    allocate_array(rtf_ux, nrec, nt);

    // calculation for snaps and accumulation
    if (accu||grad){
        int snap_nt = 1 + (nt-1)/snap_dt;
        int snap_nz = 1 + (snap_z2 - snap_z1)/snap_dz;
        int snap_nx = 1 + (snap_x2 - snap_x1)/snap_dx;
    

        // Allocate accumulation arrays
        if (accu){
            // allocation of the accumulation storage arrays
            allocate_array(accu_vz, snap_nt, snap_nz, snap_nx);
            allocate_array(accu_vx, snap_nt, snap_nz, snap_nx);
            allocate_array(accu_szz, snap_nt, snap_nz, snap_nx);
            allocate_array(accu_szx, snap_nt, snap_nz, snap_nx);
            allocate_array(accu_sxx, snap_nt, snap_nz, snap_nx);
        }
        if (grad){
            // Allocating medium gradient
            allocate_array(grad_lam, nz, nx);
            allocate_array(grad_mu, nz, nx);
            allocate_array(grad_rho, nz, nx);

            // Allocating variables for old material storages
            allocate_array(lam_copy, nz, nx);
            allocate_array(mu_copy, nz, nx);
            allocate_array(rho_copy, nz, nx);

            // Allocate Energy for reverse modelling
            allocate_array(We_adj, nz, nx); // used later as temporary gradient

            // Allocating medium gradient for each shot
            allocate_array(grad_lam_shot, snap_nz, snap_nx);
            allocate_array(grad_mu_shot, snap_nz, snap_nx);
            allocate_array(grad_rho_shot, snap_nz, snap_nx); 

        }
        
    }

}
