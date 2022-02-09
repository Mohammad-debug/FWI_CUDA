//simulate_PSV.cpp

/* 
* Created by: Min Basnet
* 2020.November.26
* Kathmandu, Nepal
*/

// full waveform simulation of 2D plane (P/SV) seismic wave problems


#include "n_simulate_PSV.hpp"
#include <iostream>
#include <math.h>
#include <omp.h>


void simulate_fwd_PSV(int nt, int nz, int nx, real dt, real dz, real dx, 
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, int snap_dt, int snap_dz, int snap_dx, 
    bool surf, bool pml_z, bool pml_x, int nsrc, int nrec, int nshot, int stf_type, int rtf_type, 
    int fdorder, real scalar_lam, real scalar_mu, real scalar_rho,
    real *&hc, int *&isurf, real **&lam, real **&mu, real **&rho, 
    real *&a_z, real *&b_z, real *&K_z, real *&a_half_z, real *&b_half_z, real *&K_half_z,
    real *&a_x, real *&b_x, real *&K_x, real *&a_half_x, real *&b_half_x, real *&K_half_x,
    int *&z_src, int *&x_src, int *&z_rec, int *&x_rec,
    int *&src_shot_to_fire, real **&stf_z, real **&stf_x, 
    bool accu_save, bool seismo_save){ // forward accumulated storage arrays){
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
    real **lam_copy, **mu_copy, **rho_copy; // Old material storage while updating (only in fwi)
    real **grad_lam, **grad_mu, **grad_rho; // Gradients of material (full grid)
    //real **grad_lam_old, **grad_mu_old, **grad_rho_old; // Storing old material gradients for optimization
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

    // allocating main computational arrays
    alloc_varmain_PSV(vz, vx, uz, ux, We, We_adj, szz, szx, sxx, dz_z, dx_z, dz_x, dx_x, 
    mem_vz_z, mem_vx_z, mem_szz_z, mem_szx_z, mem_vz_x, mem_vx_x, mem_szx_x, mem_sxx_x,
    mu_zx, rho_zp, rho_xp, lam_copy, mu_copy, rho_copy, grad_lam, grad_mu, grad_rho, 
    grad_lam_shot, grad_mu_shot, grad_rho_shot, rtf_uz, rtf_ux, accu_vz, accu_vx, 
    accu_szz, accu_szx, accu_sxx, pml_z, pml_x, nrec, accu, grad, snap_z1, 
    snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx, nt, nz, nx);

       double dif=0;
       double start = omp_get_wtime();

    // calculate material average
    mat_av2(lam, mu, rho, mu_zx, rho_zp, rho_xp, 
        scalar_lam, scalar_mu, scalar_rho, nz, nx);
  
    // Seismic forward kernel
    for (int ishot = 0; ishot < nshot; ishot++){
        std::cout << "FORWARD KERNEL: SHOT " << ishot << " of " << nshot <<"." << std::endl;
        accu = true; // Accumulated storage for output
        grad = false; // no gradient computation in forward kernel
        start = omp_get_wtime();

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

            double end = omp_get_wtime(); // end the timer
            dif += end - start;            // stores the difference in dif
            
            

        // Saving the Accumulative storage file to a binary file for every shots
        if (accu_save){
            // Writing the accumulation array
            std::cout << "Writing accu to binary file for SHOT " << ishot ;
            write_accu(accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx, nt, snap_z1, snap_z2, snap_x1, 
            snap_x2, snap_dt, snap_dz, snap_dx, ishot);
            std::cout <<" <DONE>"<< std::endl;
        }

        // Saving the Accumulative storage file to a binary file for every shots
        if (seismo_save){
            // Writing the accumulation array
            std::cout << "Writing accu to binary file for SHOT " << ishot ;
            write_seismo(rtf_uz, rtf_ux, nrec, nt, ishot);
            std::cout <<" <DONE>"<< std::endl;
        }

      std::cout << "\n--------\nthe time of CPU Kernel without i/0  = " << dif  << " s\n---------\n";
    }
    
}



void simulate_fwi_PSV(int nt, int nz, int nx, real dt, real dz, real dx, 
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, int snap_dt, int snap_dz, int snap_dx, 
    bool surf, bool pml_z, bool pml_x, int nsrc, int nrec, int nshot, int stf_type, int rtf_type, 
    int fdorder, real scalar_lam, real scalar_mu, real scalar_rho,
    real *&hc, int *&isurf, real **&lam, real **&mu, real **&rho, 
    real *&a_z, real *&b_z, real *&K_z, real *&a_half_z, real *&b_half_z, real *&K_half_z,
    real *&a_x, real *&b_x, real *&K_x, real *&a_half_x, real *&b_half_x, real *&K_half_x,
    int *&z_src, int *&x_src, int *&z_rec, int *&x_rec,
    int *&src_shot_to_fire, real **&stf_z, real **&stf_x, real ***&rtf_z_true, real ***&rtf_x_true,
    int mat_save_interval, int &taper_t1, int &taper_t2, int &taper_b1, int &taper_b2, 
    int &taper_l1, int &taper_l2, int &taper_r1, int &taper_r2){
    // full waveform inversion modelling
    // fwinv = true for this case
    // Internal variables
    bool accu = true, grad = true; 
    int *a_stf_type; // adjoint source type
    int *rec_shot_to_fire;
    rec_shot_to_fire = new int [nrec];
   
    // -------------------------------------------------------------------------------------------------------
    // Internally computational arrays
    // --------------------------------------------------------------------------------------------------------
    real **vz, **vx, **uz, **ux; // Tensors: velocity, displacement
    real **We, **We_adj, **szz, **szx, **sxx; // Tensors: Energy fwd and adj, stress (We_adj used as temp grad)
    real **dz_z, **dx_z, **dz_x, **dx_x; // spatial derivatives
    real **mem_vz_z, **mem_vx_z, **mem_szz_z, **mem_szx_z; // PML spatial derivative memories: z-direction
    real **mem_vz_x, **mem_vx_x, **mem_szx_x, **mem_sxx_x; // PML spatial derivative memories: x-direction
    real **mu_zx, **rho_zp, **rho_xp; // Material averages
    real **lam_copy, **mu_copy, **rho_copy; // Old material storage while updating
    real **grad_lam, **grad_mu, **grad_rho; // Gradients of material (full grid)
    //real **grad_lam_old, **grad_mu_old, **grad_rho_old; // Storing old material gradients for optimization
    real **PCG_lam, **PCG_dir_lam; // Old conjugate gradient storages
    real **PCG_mu, **PCG_dir_mu;
    real **PCG_rho, **PCG_dir_rho;
    real **grad_lam_shot, **grad_mu_shot, **grad_rho_shot; // Gradient of materials in each shot (snapped)
    real **rtf_uz, **rtf_ux; // receiver time functions (displacements)
    real ***accu_vz, ***accu_vx, ***accu_szz, ***accu_szx, ***accu_sxx; // forward accumulated storage arrays
    // -----------------------------------------------------------------------------------------------------
    real beta_PCG, beta_i, beta_j;

    // allocating main computational arrays
    accu = true; grad = true;
    alloc_varmain_PSV(vz, vx, uz, ux, We, We_adj, szz, szx, sxx, dz_z, dx_z, dz_x, dx_x, 
    mem_vz_z, mem_vx_z, mem_szz_z, mem_szx_z, mem_vz_x, mem_vx_x, mem_szx_x, mem_sxx_x,
    mu_zx, rho_zp, rho_xp, lam_copy, mu_copy, rho_copy, grad_lam, grad_mu, grad_rho, 
    grad_lam_shot, grad_mu_shot, grad_rho_shot, rtf_uz, rtf_ux, accu_vz, accu_vx, 
    accu_szz, accu_szx, accu_sxx, pml_z, pml_x, nrec, accu, grad, snap_z1, 
    snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx, nt, nz, nx);


    // Allocating PCG variables
    //PCG_new = new real[nz*nx*3];
    //PCG_old = new real[nz*nx*3];
    //PCG_dir = new real[nz*nx*3];
    //allocate_array(PCG_new, nz, nx);
    // allocate_array(PCG_lam, nz, nx);
    // allocate_array(PCG_mu, nz, nx);
    // allocate_array(PCG_rho, nz, nx);

    // allocate_array(PCG_dir_lam, nz, nx);
    // allocate_array(PCG_dir_mu, nz, nx);
    // allocate_array(PCG_dir_rho, nz, nx);
    
    // for (int iz=0;iz<nz;iz++){
    //     for (int ix=0;ix<nx;ix++){
    //         PCG_dir_lam[iz][ix] = 0.0;
    //         PCG_dir_mu[iz][ix] = 0.0;
    //         PCG_dir_rho[iz][ix] = 0.0;
    //     }
    // }


    //-----------------------------------------------
    // 0.0. OUTER PREPROCESSING (IN EVERY FWI LOOPS)
    // ----------------------------------------------
    // Outside every shot loop
	
	// Allocate the variables common for every shots and iterations
	
	// Prepare initial medium parameters
	
	// Calculate PML factors if necessary


    // Start of FWI iteration loop
    bool iter = true;
    int iterstep = 0;
    int maxIter = 1000; 
    real L2_norm[1000]; // size is maxIter
    for (int ll=0;ll<1000;ll++){ L2_norm[ll] = 0.0;}
    real step_length = 0.01; // step length set to initial
    real step_length_rho = 0.01; // step length set to initial
    
    double dif=0;
   

    while (iter){ // currently 10 just for test (check the conditions later)
        //
         double start = omp_get_wtime();

        std::cout << std::endl << std::endl;
        std::cout << "==================================" << std::endl;
        std::cout << "FWI: Iteration "<< iterstep << std::endl;
        std::cout << "==================================" << std::endl;
        //-----------------------------------------------
        // 1.0. INNER PREPROCESSING (IN EVERY FWI LOOPS)
        // ----------------------------------------------
		// Reset gradient matrices: grad_lam, grad_mu, grad_rho;??

        // Copy updated material for old material storage
        copy_mat(lam_copy, mu_copy, rho_copy, lam, mu,  rho, nz, nx);
        
        // calculate material average
        mat_av2(lam, mu, rho, mu_zx, rho_zp, rho_xp, 
            scalar_lam, scalar_mu, scalar_rho, nz, nx);
    
    
        for (int ishot = 0; ishot < nshot; ishot++){
            std::cout << "FWI KERNEL: SHOT " << ishot << " of " << nshot <<"." << std::endl;
            // -----------------------------------
            // 2.0. FORWARD MODELLING
            // ------------------------------------
            
            // Seismic forward kernel
            accu = true; // Accumulated storage for output
            grad = false; // no gradient computation in forward kernel
            kernel_PSV(ishot, nt, nz, nx, dt, dx, dz, surf, isurf, hc, fdorder, 
                vz, vx,  uz, ux, szz, szx, sxx, We, dz_z, dx_z, dz_x, dx_x, 
                lam, mu, mu_zx, rho_zp, rho_xp, grad, grad_lam_shot, grad_mu_shot, grad_rho_shot,
                pml_z, a_z, b_z, K_z, a_half_z, b_half_z, K_half_z,
                pml_x, a_x, b_x, K_x, a_half_x, b_half_x, K_half_x, 
                mem_vz_z, mem_vx_z, mem_szz_z, mem_szx_z,
                mem_vz_x, mem_vx_x, mem_szx_x, mem_sxx_x,
                nsrc, stf_type, stf_z, stf_x, z_src, x_src, src_shot_to_fire,
                nrec, rtf_type, rtf_uz, rtf_ux, z_rec, x_rec,
                accu, accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx, 
                snap_z1, snap_z2, snap_x1, snap_x2, 
                snap_dt, snap_dz, snap_dx);



// //TEST
// double l=0,m=0,r=0;

//    int snap_nz = 1 + (snap_z2 - snap_z1) / snap_dz;
//    int snap_nx = 1 + (snap_x2 - snap_x1) / snap_dx;

//     for (int iz=0; iz<snap_nz; iz++){
//         for (int ix=0; ix<snap_nx; ix++){
//             l+=grad_lam_shot[iz][ix];
//             m+=grad_mu_shot[iz][ix];
//             r+=grad_rho_shot[iz][ix];
//         }
//     }

//         std::cout << "This is test CPU>FORWARD \nLAM_SHOT=" << l << " \nMU_SHOT=" << m << " \nRHO_SHOT=" << r << " \n\n";





            // -----------------------------------------------
            // 3.0. RESIDUALS AND ADJOINT SOURCE COMPUTATION
            // ------------------------------------------------
            
            
            // calculating L2 norm and adjoint sources
            L2_norm[iterstep] += adjsrc2(ishot, a_stf_type, rtf_uz, rtf_ux, rtf_type, rtf_z_true, rtf_x_true,
                            rtf_uz, rtf_ux, dt, nrec, nt);

            std::cout<<"L2 NORM: " << L2_norm[iterstep]/L2_norm[0] << ", " << L2_norm[iterstep]<< std::endl;
            if (iterstep > 2){
                std::cout<<"L2 Diff: " << abs(L2_norm[iterstep]-L2_norm[iterstep-2])/L2_norm[iterstep-2] << std::endl;
            }
            
        
            
            // -----------------------------------
            // 4.0. ADJOING MODELLING
            // ------------------------------------
            
            // Preparing adjoint shot to fire in the shot
            // Fire all adjoint sources in this shot
            for(int ir=0; ir<nrec; ir++){
                rec_shot_to_fire[ir] = ishot;
            }
            
            // Seismic adjoint kernel
            accu = false; // Accumulated storage for output
            grad = true; // no gradient computation in forward kernel
            
            kernel_PSV(ishot, nt, nz, nx, dt, dx, dz, surf, isurf, hc, fdorder, 
                vz, vx,  uz, ux, szz, szx, sxx, We_adj, dz_z, dx_z, dz_x, dx_x, 
                lam, mu, mu_zx, rho_zp, rho_xp, 
                grad, grad_lam_shot, grad_mu_shot, grad_rho_shot,
                pml_z, a_z, b_z, K_z, a_half_z, b_half_z, K_half_z,
                pml_x, a_x, b_x, K_x, a_half_x, b_half_x, K_half_x, 
                mem_vz_z, mem_vx_z, mem_szz_z, mem_szx_z,
                mem_vz_x, mem_vx_x, mem_szx_x, mem_sxx_x,
                nrec, rtf_type, rtf_uz, rtf_ux, z_rec, x_rec, rec_shot_to_fire, //*a_stf_type = rtf_type
                nrec, rtf_type, rtf_uz, rtf_ux, z_rec, x_rec,
                accu, accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx, 
                snap_z1, snap_z2, snap_x1, snap_x2, 
                snap_dt, snap_dz, snap_dx);

            

            //TEST


// l=0;m=0;r=0;
  
    // for (int iz=0; iz<snap_nz; iz++){
    //     for (int ix=0; ix<snap_nx; ix++){
    //         l+=grad_lam_shot[iz][ix];
    //         m+=grad_mu_shot[iz][ix];
    //         r+=grad_rho_shot[iz][ix];
    //     }
    // }

    //     std::cout << "This is test CPU>ADJOINT \nLAM_SHOT=" << l << " \nMU_SHOT=" << m << " \nRHO_SHOT=" << r << " \n\n";


			
            // Smooth gradients
        
            // Calculate Energy Weights
           energy_weights2(We, We_adj, snap_z1, snap_z2, snap_x1, snap_x2);
            

           //  exit(0);
            // [We_adj used as temporary gradient here after]
            
            // GRAD_LAM
            // ----------------------------------------
            // Interpolate gradients to temporary array
            interpol_grad2(We_adj, grad_lam_shot, snap_z1, snap_z2, 
                       snap_x1, snap_x2, snap_dz, snap_dx);
            
            // Scale to energy weight and add to global array 
            scale_grad_E2(grad_lam, We_adj, scalar_lam, We,
                    snap_z1, snap_z2, snap_x1, snap_x2);
            
            // GRAD_MU
            // ----------------------------------------
            // Interpolate gradients to temporary array
            interpol_grad2(We_adj, grad_mu_shot, snap_z1, snap_z2, 
                       snap_x1, snap_x2, snap_dz, snap_dx);
            // Scale to energy weight and add to global array 
            scale_grad_E2(grad_mu, We_adj, scalar_mu, We,
                    snap_z1, snap_z2, snap_x1, snap_x2);

            // GRAD_RHO
            // ----------------------------------------
            // Interpolate gradients to temporary array
           interpol_grad2(We_adj, grad_rho_shot, snap_z1, snap_z2, 
                       snap_x1, snap_x2, snap_dz, snap_dx);
            // Scale to energy weight and add to global array 
            scale_grad_E2(grad_rho, We_adj, scalar_rho, We,
                    snap_z1, snap_z2, snap_x1, snap_x2);

        }
        
		// Smooth the global gradient with taper functions
		
		// Preconditioning of Gradients
    
        // -------------------
        // 5.0. OPTIMIZATION (Directly computed here)
        // ----------------------------------

        // Congugate Gradient Method
        // -------------------------------
        std::cout << "Applying Preconditioning" << std::endl;
        // Applying Conjugate Gradient Method
        //PCG_PSV(PCG_dir_lam, PCG_lam, grad_lam, nz, nx);
        //PCG_PSV(PCG_dir_mu, PCG_mu, grad_mu, nz, nx);
        //PCG_PSV(PCG_dir_rho, PCG_rho, grad_rho, nz, nx);

        
        //write_mat(grad_lam, grad_mu, grad_rho, nz, nx, 1000*(iterstep+1));


        // Applying taper function 
        // Currently only Tukey Taper function available
        taper2(grad_lam, nz, nx, snap_z1, snap_z2, snap_x1, snap_x2,
                taper_t1, taper_t2, taper_b1, taper_b2, taper_l1, taper_l2, taper_r1, taper_r2);

        taper2(grad_mu, nz, nx, snap_z1, snap_z2, snap_x1, snap_x2,
                taper_t1, taper_t2, taper_b1, taper_b2, taper_l1, taper_l2, taper_r1, taper_r2);

        taper2(grad_rho, nz, nx, snap_z1, snap_z2, snap_x1, snap_x2,
                taper_t1, taper_t2, taper_b1, taper_b2, taper_l1, taper_l2, taper_r1, taper_r2);
        

        /*
        //write_mat(grad_lam, grad_mu, grad_rho, nz, nx, 1000*(iterstep+1)+1);
        // Applying PSG method
        beta_i = 0.0; beta_j = 0.0;
        for (int iz=0;iz<nz;iz++){
            for (int ix=0;ix<nx;ix++){

                // Fletcher-Reeves [Fletcher and Reeves, 1964]:
                beta_i += grad_lam[iz][ix] * (grad_lam[iz][ix] - PCG_lam[iz][ix]);
                beta_i += grad_mu[iz][ix] * (grad_mu[iz][ix] - PCG_mu[iz][ix]);
                beta_i += grad_rho[iz][ix] * (grad_rho[iz][ix] - PCG_rho[iz][ix]);

                beta_j += PCG_lam[iz][ix] * PCG_lam[iz][ix];
                beta_j += PCG_mu[iz][ix] * PCG_mu[iz][ix];
                beta_j += PCG_rho[iz][ix] * PCG_rho[iz][ix];

                PCG_lam[iz][ix] = -grad_lam[iz][ix]; 
                PCG_mu[iz][ix] = -grad_mu[iz][ix]; 
                PCG_rho[iz][ix] = -grad_rho[iz][ix]; 
           
            }
        }
        beta_PCG = (iterstep) ? (beta_i/beta_j) : 0.0;
        std::cout << "beta = "<< beta_PCG ;
        beta_PCG = (beta_PCG >0) ? beta_PCG : 0.0;
        //beta_PCG = (beta_PCG <1.0e6) ? beta_PCG : 1.0e6;
        std::cout << " || adopted: " << beta_PCG<< std::endl;

        
        for (int iz=0;iz<nz;iz++){
            for (int ix=0;ix<nx;ix++){
                PCG_dir_lam[iz][ix] = PCG_lam[iz][ix] + beta_PCG * PCG_dir_lam[iz][ix]; // Getting PCG direction
                PCG_dir_mu[iz][ix] = PCG_mu[iz][ix] + beta_PCG * PCG_dir_mu[iz][ix]; // Getting PCG direction
                PCG_dir_rho[iz][ix] = PCG_rho[iz][ix] + beta_PCG * PCG_dir_rho[iz][ix]; // Getting PCG direction

                grad_lam[iz][ix] = -PCG_dir_lam[iz][ix]; // Getting PCG_dir to gradient vectors
                grad_mu[iz][ix] = -PCG_dir_mu[iz][ix]; // Getting PCG_dir to gradient vectors
                grad_rho[iz][ix] = -PCG_dir_rho[iz][ix]; // Getting PCG_dir to gradient vectors
           
            }
        }
        */

        //write_mat(grad_lam, grad_mu, grad_rho, nz, nx, 1000*(iterstep+1)+2);
        // ----------------------
        // 6.0. MATERIAL UPDATE
        // ---------------------
		
		// Step length estimation for wave parameters

        //uncomment here.....
        
        step_length = step_length_PSV(step_length, L2_norm[iterstep], nshot, nt, nz, nx, dt, dx, dz, surf, isurf, hc, fdorder, 
            vz, vx,  uz, ux, szz, szx, sxx,  We, dz_z, dx_z, dz_x, dx_x, 
            lam, mu, rho, lam_copy, mu_copy, rho_copy, 
            mu_zx, rho_zp, rho_xp, grad, grad_lam, grad_mu, grad_rho,
            pml_z, a_z, b_z, K_z, a_half_z, b_half_z, K_half_z,
            pml_x, a_x, b_x, K_x, a_half_x, b_half_x, K_half_x, 
            mem_vz_z, mem_vx_z, mem_szz_z, mem_szx_z, 
            mem_vz_x, mem_vx_x, mem_szx_x, mem_sxx_x,
            nsrc, stf_type, stf_z, stf_x, z_src, x_src, src_shot_to_fire,
            nrec, rtf_type, rtf_uz, rtf_ux, z_rec, x_rec,
            rtf_z_true, rtf_x_true, accu, accu_vz, accu_vx,  accu_szz, accu_szx, accu_sxx, 
            snap_z1, snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx, 0);
        std::cout<<"\n\n *****STEP LENGTH CPU ******"<<step_length<<"\n";

        // Separate Step length for density update
        /*
        step_length_rho = step_length_PSV(step_length_rho, L2_norm[iterstep], nshot, nt, nz, nx, dt, dx, dz, surf, isurf, hc, fdorder, 
            vz, vx,  uz, ux, szz, szx, sxx,  We, dz_z, dx_z, dz_x, dx_x, 
            lam, mu, rho, lam_copy, mu_copy, rho_copy, 
            mu_zx, rho_zp, rho_xp, grad, grad_lam, grad_mu, grad_rho,
            pml_z, a_z, b_z, K_z, a_half_z, b_half_z, K_half_z,
            pml_x, a_x, b_x, K_x, a_half_x, b_half_x, K_half_x, 
            mem_vz_z, mem_vx_z, mem_szz_z, mem_szx_z, 
            mem_vz_x, mem_vx_x, mem_szx_x, mem_sxx_x,
            nsrc, stf_type, stf_z, stf_x, z_src, x_src, src_shot_to_fire,
            nrec, rtf_type, rtf_uz, rtf_ux, z_rec, x_rec,
            rtf_z_true, rtf_x_true, accu, accu_vz, accu_vx,  accu_szz, accu_szx, accu_sxx, 
            snap_z1, snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx, 2);
        */
     
        // Update material parameters to the gradients !!
		update_mat2(lam, lam_copy, grad_lam, 4.8e+10, 0.0, step_length, nz, nx);
        update_mat2(mu, mu_copy, grad_mu, 2.7e+10, 0.0, step_length, nz, nx);

        step_length_rho = 0.5 * step_length;
        update_mat2(rho, rho_copy, grad_rho, 3000.0, 1.25, step_length_rho, nz, nx);

        double end = omp_get_wtime(); // end the timer
        dif = end - start;            // stores the difference in dif
        std::cout << "==================================" << std::endl;
        std::cout << "the time of single FWI iteration = " << dif << "s\n";
        std::cout << "==================================" << std::endl;

        //return;

        //
        // Saving the Accumulative storage file to a binary file for every shots
        std::cout<<"Iteration step: " <<iterstep<<", "<<mat_save_interval<<", "<< iterstep%mat_save_interval<<std::endl;
        if (mat_save_interval>0 && !(iterstep%mat_save_interval)){
            // Writing the accumulation array
            std::cout << "Writing updated material to binary file for ITERATION " << iterstep ;
           write_mat(lam, mu, rho, nz, nx, iterstep);
            std::cout <<" <DONE>"<< std::endl;
        }
        
 
       //
       iterstep++ ;
       iter = (iterstep < maxIter) ? true : false; // Temporary condition
       if (iterstep > 25){
           iter = (abs((L2_norm[iterstep] - L2_norm[iterstep-2])/L2_norm[iterstep-2])<0.001) ? false : true; // Temporary condition
           std::cout << "The change is less than minimal after " << iterstep << " iteration steps." << std::endl;
       }
    }

    // Saving the Accumulative storage file to a binary file for every shots
    if (mat_save_interval<1){
        // Writing the accumulation array
        std::cout << "Writing updated material to binary file <FINAL> ITERATION " << iterstep ;
        write_mat(lam, mu, rho, nz, nx, iterstep);
        std::cout <<" <DONE>"<< std::endl;
    }
}

