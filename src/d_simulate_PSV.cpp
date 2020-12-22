//simulate_PSV.cpp

/* 
* Created by: Min Basnet
* 2020.November.26
* Kathmandu, Nepal
*/

// full waveform simulation of 2D plane (P/SV) seismic wave problems


#include "d_simulate_PSV.hpp"
#include <iostream>


void simulate_fwd_PSV(int nt, int nz, int nx, real dt, real dz, real dx, 
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, int snap_dt, int snap_dz, int snap_dx, 
    bool surf, bool pml_z, bool pml_x, int nsrc, int nrec, int nshot, int stf_type, int rtf_type, 
    bool rtf_true, int fdorder, real scalar_lam, real scalar_mu, real scalar_rho,
    real *&hc, int *&isurf, real **&lam, real **&mu, real **&rho, 
    real *&a_z, real *&b_z, real *&K_z, real *&a_half_z, real *&b_half_z, real *&K_half_z,
    real *&a_x, real *&b_x, real *&K_x, real *&a_half_x, real *&b_half_x, real *&K_half_x,
    int *&z_src, int *&x_src, int *&z_rec, int *&x_rec,
    int *&src_shot_to_fire, real **&stf_z, real **&stf_x, real **&rtf_z_true, real **&rtf_x_true,
    bool accu_save){ // forward accumulated storage arrays){
    // Forward modelling in PSV
    

    // ---------------------------------------------------------------------------------------
    // External input arrays
    // ---------------------------------------------------------------------------------------
    /*
    int nt, nz, nx; // grid sizes
    int dt, dz, dx; // grid intervals
    int snap_z1, snap_z2, snap_x1, snap_x2; // grid snap indices for output and forward accu storage
    int snap_dt, snap_dz, snap_dx; // snap intervals
    bool surf, pml_z, pml_x; // if Surface and PML layers exist
    int nsrc, nrec, nshot; // number of sources, receivers and shots
    int stf_type, rtf_type; // Type of source and receiver time functions
    bool rtf_true; // RTF field measurements exists?
    int fdorder; // The Finite Difference Order
    real scalar_lam, scalar_mu, scalar_rho; // Scalar materials later used as averages
    bool accu, grad; // if to accumulate or to calculate material gradientss
    */
    // ---------------------------------------------------------------------------------------

    // ---------------------------------------------------------------------------------------
    // External input arrays
    // ---------------------------------------------------------------------------------------
    /*
    real *hc; // Holberg Coefficients
    int *isurf; // surface indices ([0] Top, [1] Bottom, [2] Left, [3] Bottom)
    real **lam, **mu, **rho; // Elastic material parameters
    real *a_z, *b_z, *K_z, *a_half_z, *b_half_z, *K_half_z; // PML coefficients: z-direction
    real *a_x, *b_x, *K_x, *a_half_x, *b_half_x, *K_half_x; // PML coefficients: x-direction
    int *z_src, *x_src, *z_rec, *x_rec; // source and receiver location indices
    int *src_shot_to_fire; // Which source to fire on which shot index
    real **stf_z, **stf_x, **rtf_z_true, **rtf_x_true; // source and receiver time functions
    */
    // ---------------------------------------------------------------------------------------

    // -------------------------------------------------------------------------------------------------------
    // Internally computational arrays
    // --------------------------------------------------------------------------------------------------------
    real **vz, **vx, **uz, **ux; // Tensors: velocity, displacement
    real **We, **We_adj, **szz, **szx, **sxx; // Tensors: Energy fwd and adj, stress (We_adj used as temp grad)
    real **dz_z, **dx_z, **dz_x, **dx_x; // spatial derivatives
    real **mem_vz_z, **mem_vx_z, **mem_szz_z, **mem_szx_z; // PML spatial derivative memories: z-direction
    real **mem_vz_x, **mem_vx_x, **mem_szx_x, **mem_sxx_x; // PML spatial derivative memories: x-direction
    real **mu_zx, **rho_zp, **rho_xp; // Material averages
    real **grad_lam, **grad_mu, **grad_rho; // Gradients of material (full grid)
    real **grad_lam_shot, **grad_mu_shot, **grad_rho_shot; // Gradient of materials in each shot (snapped)
    real **rtf_uz, **rtf_ux; // receiver time functions (displacements)
    real ***accu_vz, ***accu_vx, ***accu_szz, ***accu_szx, ***accu_sxx; // forward accumulated storage arrays
    // -----------------------------------------------------------------------------------------------------

    // Internal variables
     bool accu = true, grad = false; 


    // int nt, nz, nx; // grid sizes
    // bool surf, pml_z, pml_x;
    // int fdorder; // order of the finite difference
    // int nsrc, nrec

    // Allocating arrays for array inputs
    /*
    alloc_varin_PSV(hc, isurf, lam, mu, rho, a_z, b_z, K_z, a_half_z, b_half_z, K_half_z,
        a_x, b_x, K_x, a_half_x, b_half_x, K_half_x, z_src, x_src, src_shot_to_fire, 
        stf_z, stf_x, z_rec, x_rec, rtf_z_true, rtf_x_true, fdorder, surf, pml_z, pml_x, 
        nsrc, nrec, rtf_true, nt, nz, nx);
    */
    // get array inputs
    // ..............TO BE ADDED LATER ....

    
    // allocating main computational arrays
    alloc_varmain_PSV(vz, vx, uz, ux, We, We_adj, szz, szx, sxx, dz_z, dx_z, dz_x, dx_x, 
    mem_vz_z, mem_vx_z, mem_szz_z, mem_szx_z, mem_vz_x, mem_vx_x, mem_szx_x, mem_sxx_x,
    mu_zx, rho_zp, rho_xp, grad_lam, grad_mu, grad_rho, grad_lam_shot, grad_mu_shot, grad_rho_shot,
    rtf_uz, rtf_ux, accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx, 
    pml_z, pml_x, nrec, accu, grad, snap_z1, snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx, 
    nt, nz, nx);

    // calculate material average
    mat_av2(lam, mu, rho, mu_zx, rho_zp, rho_xp, 
        scalar_lam, scalar_mu, scalar_rho, nz, nx);
  
    // Seismic forward kernel
    for (int ishot = 0; ishot < nshot; ishot++){
        accu = true; // Accumulated storage for output
        grad = false; // no gradient computation in forward kernel
        kernel_PSV(ishot, nt, nz, nx, dt, dx, dz, surf, isurf, hc, fdorder, 
            vz, vx,  uz, ux, szz, szx, sxx, We, dz_z, dx_z, dz_x, dx_x, 
            lam, mu, mu_zx, rho_zp, rho_xp, grad, grad_lam, grad_mu, grad_rho,
            pml_z, a_z, b_z, K_z, a_half_z, b_half_z, K_half_z,
            pml_x, a_x, b_x, K_x, a_half_x, b_half_x, K_half_x, 
            mem_vz_z, mem_vx_z, mem_szz_z, mem_szx_z,
            mem_vz_x, mem_vx_x, mem_szx_x, mem_sxx_x,
            nsrc, stf_type, stf_z, stf_x, z_src, x_src, src_shot_to_fire,
            nrec, rtf_type, rtf_uz, rtf_ux, z_rec, x_rec,
            accu, accu_vz,accu_vx, accu_szz, accu_szx, accu_sxx, 
            snap_z1, snap_z2, snap_x1, snap_x2, 
            snap_dt, snap_dz, snap_dx);

        // Saving the Accumulative storage file to a binary file for every shots
        if (accu_save){
            // Writing the accumulation array
            std::cout << "Writing accu to binary file for SHOT " << ishot ;
            write_accu(accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx, nt, nz, nx, snap_z1, snap_z2, snap_x1, 
            snap_x2, snap_dt, snap_dz, snap_dx, ishot);
            std::cout <<" <DONE>"<< std::endl;
        }

    }
    
}

/*

void fwi_simulate(){
    // full waveform inversion modelling
    // fwinv = true for this case
    //-----------------------------------------------
    // 0.0. OUTER PREPROCESSING (IN EVERY FWI LOOPS)
    // ----------------------------------------------
    // Outside every shot loop
	
	// Allocate the variables common for every shots and iterations
	
	// Prepare initial medium parameters
	
	// Calculate PML factors if necessary


    // Start of FWI iteration loop
    int iter; iter = 0;
    while (iter<10){ // currently 10 just for test (check the conditions later)
	
        //-----------------------------------------------
        // 1.0. INNER PREPROCESSING (IN EVERY FWI LOOPS)
        // ----------------------------------------------


        // calculate material averages
        // mu_zx, rho_zp, rho_xp, C_lam, C_mu, C_rho
		// Reset gradient matrices: grad_lam, grad_mu, grad_rho; 
    
        for (int i_shot = 0; i_shot < n_shots; ++i_shot){
        
        
            // -----------------------------------
            // 2.0. FORWARD MODELLING
            // ------------------------------------
            
            adj_kernel = false;
            accu = true;
            // Allocate the accumulated storage kernels (deallocated in each shot)
            // Different shots may require different accumulation arrays
            // accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx, //fwi memory

            // Seismic forward kernel
            seismic_kernel2(nt, nz, nx, dt, dx, dz, surf, hc, fdorder, //grids
                    vx, vz, ux, uz, sxx, szx, szz, We, // wave parameters
                    vz_z, vx_z, vz_x, vx_x, szz_z, szx_z, szx_x, sxx_x, //spatial derivatives
                    lam, mu, mu_zx, rho_zp, rho_xp, // Medium
                    grad_lam_shot, grad_mu_shot, grad_rho_shot, // gradients
                    pml, a, b, K, a_half, b_half, K_half, // PML var
                    mem_vx_x, mem_vx_z, mem_vz_x, mem_vz_z, // PML memory
                    mem_sxx_x, mem_szx_x, mem_szz_z, mem_szx_z, // PML memory
                    nsrc, stf_type, stf_z, stf_x, z_src, x_src, // source
                    nrec, rtf_type, rtf_uz, rtf_ux, z_rec, x_rec, // recievers
                    accu, accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx, //fwi memory
                    snap_z1, snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx, // grid
                    adj_kernel);

            // -----------------------------------------------
            // 3.0. RESIDUALS AND ADJOINT SOURCE COMPUTATION
            // ------------------------------------------------

            // calculating L2 norm and adjoint sources
            L2_norm = adjsrc2(a_stf_type, a_stf_uz, a_stf_ux, rtf_type, rtf_uz_true, rtf_ux_true,
                            rtf_uz, rtf_ux, dt, nrec, nt);

            // -----------------------------------
            // 4.0. ADJOING MODELLING
            // ------------------------------------
    
            adj_kernel = true;
            
            // Seismic adjoint kernel
            seismic_kernel2(nt, nz, nx, dt, dx, dz, surf, hc, fdorder, //grids
                    vx, vz, ux, uz, sxx, szx, szz, We_adj, // wave parameters
                    vz_z, vx_z, vz_x, vx_x, szz_z, szx_z, szx_x, sxx_x, //spatial derivatives
                    lam, mu, mu_zx, rho_zp, rho_xp, // Medium
                    grad_lam_shot, grad_mu_shot, grad_rho_shot, // gradients
                    pml, a, b, K, a_half, b_half, K_half, // PML var
                    mem_vx_x, mem_vx_z, mem_vz_x, mem_vz_z, // PML memory
                    mem_sxx_x, mem_szx_x, mem_szz_z, mem_szx_z, // PML memory
                    nrec, *a_stf_type, a_stf_uz, a_stf_ux, z_rec, x_rec, // adjoint source
                    nrec, rtf_type, rtf_uz, rtf_ux, z_rec, x_rec, // recievers
                    accu, accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx, //accumulated tensor memory
                    snap_z1, snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx, // grid
                    adj_kernel);

			
			// Deallocate accu_storage matrices
			
            // Smooth gradients

            // Calculate Energy Weights
            energy_weights2(We, We_adj, snap_z1, snap_z2, snap_x1, snap_x2);

            // GRAD_LAM
            // Interpolate gradients to temporary array
            interpol_grad2(grad_temp, grad_lam_shot, snap_z1, snap_z2, 
                        snap_x1, snap_x2, snap_dz, snap_dx);
            // Scale to energy weight and add to global array 
            scale_grad_E2(grad_lam, grad_temp, lam_av, We,
                snap_z1, snap_z2, snap_x1, snap_x2);

            // GRAD_MU
            // Interpolate gradients to temporary array
            interpol_grad2(grad_temp, grad_mu_shot, snap_z1, snap_z2, 
                        snap_x1, snap_x2, snap_dz, snap_dx);
            // Scale to energy weight and add to global array 
            scale_grad_E2(grad_mu, grad_temp, lam_av, We,
                snap_z1, snap_z2, snap_x1, snap_x2);

            // GRAD_RHO
            // Interpolate gradients to temporary array
            interpol_grad2(grad_temp, grad_rho_shot, snap_z1, snap_z2, 
                        snap_x1, snap_x2, snap_dz, snap_dx);
            // Scale to energy weight and add to global array 
            scale_grad_E2(grad_rho, grad_temp, lam_av, We,
                snap_z1, snap_z2, snap_x1, snap_x2);

        }

		// Smooth the global gradient with taper functions
		
		// Preconditioning of Gradients
    
        // -------------------
        // 5.0. OPTIMIZATION
        // -------------------
		

        // ----------------------
        // 6.0. MATERIAL UPDATE
        // ---------------------
		
		// Step length estimation
		real step_length = 0.005;
        
        // Update material parameters to the gradients !!
		update_mat2(lam, grad_lam, step_length, nz, nx);
        update_mat2(mu, grad_mu, step_length, nz, nx);
        update_mat2(rho, grad_rho, 0.5*step_length, nz, nx);

		// smooth model
        
    }



}
*/
