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
#include <chrono>
using namespace std::chrono;
  
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
auto start = high_resolution_clock::now();

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
                         fdorder, pml_z, pml_x, nsrc, nrec, nt, nshot, nz, nx);

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
        rtf_z_true, rtf_x_true, // Field measurements for receivers
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
        nt, nshot, nz, nx);

    // Calling global device functions (forward and fwi kernels)
    if (fwinv)
    {

// Use auto keyword to avoid typing long
// type definitions to get the timepoint
// at this instant use function now()


        std::cout << "Here you go: FWI" << std::endl;
        simulate_fwi_PSV_GPU(nt, nz, nx, dt, dz, dx,
                             snap_z1, snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx,
                             surf, pml_z, pml_x, nsrc, nrec, nshot, stf_type, rtf_type,
                             fdorder, scalar_lam, scalar_mu, scalar_rho,

                             d_hc, d_isurf, d_lam, d_mu, d_rho,
                             d_a_z, d_b_z, d_K_z, d_a_half_z, d_b_half_z, d_K_half_z,
                             d_a_x, d_b_x, d_K_x, d_a_half_x, d_b_half_x, d_K_half_x,
                             d_z_src, d_x_src, d_z_rec, d_x_rec,
                             d_src_shot_to_fire, d_stf_z, d_stf_x, d_rtf_z_true, d_rtf_x_true,
                             mat_save_interval, taper_t1, taper_t2, taper_b1, taper_b2,
                             taper_l1, taper_l2, taper_r1, taper_r2);
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

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);
  
// To get the value of duration use the count()
// member function on the duration object
std::cout << "Time taken by GPU: "
         << duration.count() << " microseconds" << "\n";
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

__global__ void forLoop(int *rec_shot_to_fire, int ishot, int nrec)
{
    int ir = blockDim.x * blockIdx.x + threadIdx.x;
   
    if (ir < nrec && ir >= 0){
        rec_shot_to_fire[ir] = ishot;
        // printf("    %d\n",ir);

    }

    return;
}

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
                          int &taper_l1, int &taper_l2, int &taper_r1, int &taper_r2)
{
    // full waveform inversion modelling
    // fwinv = true for this case
    // Internal variables
    bool accu = true, grad = true;
    int *a_stf_type; // adjoint source type
    int *rec_shot_to_fire;

    cudaCheckError(cudaMalloc(&rec_shot_to_fire, nrec * sizeof(int))); //device allocation

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
    //real **grad_lam_old, **grad_mu_old, **grad_rho_old; // Storing old material gradients for optimization
    // real **PCG_lam, **PCG_dir_lam; // Old conjugate gradient storages
    // real **PCG_mu, **PCG_dir_mu;
    // real **PCG_rho, **PCG_dir_rho;

    // -----------------------------------------------------------------------------------------------------
    // real beta_PCG, beta_i, beta_j;

///HOST ARRAYS TO SAVE THE RESULT
        real **h_lam;
        real **h_mu;
        real **h_rho;
    allocate_array_cpu(h_lam, nz, nx);
    allocate_array_cpu(h_mu, nz, nx);
    allocate_array_cpu(h_rho, nz, nx);


    // allocating main computational arrays
    accu = true;
    grad = true;
    alloc_varmain_PSV_GPU(vz, vx, uz, ux, We, We_adj, szz, szx, sxx, dz_z, dx_z, dz_x, dx_x,
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
    int iterstep = 0; //  0
    int maxIter = 10; // 1000
    real L2_norm[1000]; // size is maxIter
    for (int ll = 0; ll < 1000; ll++)
    {
        L2_norm[ll] = 0.0;
    }
    real step_length = 0.01;     // step length set to initial
    real step_length_rho = 0.01; // step length set to initial

   

    while (iter)
    { // currently 10 just for test (check the conditions later)
        std::cout << std::endl
                  << std::endl;
        std::cout << "==================================" << std::endl;
        std::cout << "FWI: Iteration " << iterstep << std::endl;
        std::cout << "==================================" << std::endl;
        //-----------------------------------------------
        // 1.0. INNER PREPROCESSING (IN EVERY FWI LOOPS)
        // ----------------------------------------------
        // Reset gradient matrices: grad_lam, grad_mu, grad_rho;??

        // Copy updated material for old material storage
        copy_mat_GPU(lam_copy, mu_copy, rho_copy, lam, mu, rho, nz, nx);

        // calculate material average
        mat_av2_GPU(lam, mu, rho, mu_zx, rho_zp, rho_xp,
                    scalar_lam, scalar_mu, scalar_rho, nz, nx);//this gives C_lam

        

        for (int ishot = 0; ishot < nshot; ishot++)
        {
            std::cout << "FWI KERNEL: SHOT " << ishot << " of " << nshot << "." << std::endl;
            // -----------------------------------
            // 2.0. FORWARD MODELLING
            // ------------------------------------
            
            // Seismic forward kernel
            accu = true;  // Accumulated storage for output
            grad = false; // no gradient computation in forward kernel
            kernel_PSV_GPU(ishot, nt, nz, nx, dt, dx, dz, surf, isurf, hc, fdorder,
                           vz, vx, uz, ux, szz, szx, sxx, We, dz_z, dx_z, dz_x, dx_x,
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

            // -----------------------------------------------
            // 3.0. RESIDUALS AND ADJOINT SOURCE COMPUTATION
            // ------------------------------------------------

            // calculating L2 norm and adjoint sources
            L2_norm[iterstep] += adjsrc2_GPU(ishot, a_stf_type, rtf_uz, rtf_ux, rtf_type, rtf_z_true, rtf_x_true,
                                             rtf_uz, rtf_ux, dt, nrec, nt);

            std::cout << "L2 NORM: " << L2_norm[iterstep] / L2_norm[0] << ", " << L2_norm[iterstep] << std::endl;
            if (iterstep > 2)
            {
                std::cout << "L2 Diff: " << abs(L2_norm[iterstep] - L2_norm[iterstep - 2]) / L2_norm[iterstep - 2] << std::endl;
            }



            


   double l=0,m=0,r=0;
   int snap_nz = 1 + (snap_z2 - snap_z1) / snap_dz;
   int snap_nx = 1 + (snap_x2 - snap_x1) / snap_dx;

    thrust::device_ptr<real> dev_ptr11 = thrust::device_pointer_cast(grad_lam_shot);
    thrust::device_ptr<real> dev_ptr22 = thrust::device_pointer_cast(grad_mu_shot);
    thrust::device_ptr<real> dev_ptr33 = thrust::device_pointer_cast(grad_rho_shot);

    // l += thrust::reduce(dev_ptr11 , dev_ptr11 + snap_nz*snap_nx, 0.0, thrust::plus<real>());
    // m += thrust::reduce(dev_ptr22 , dev_ptr22 + snap_nz*snap_nx, 0.0, thrust::plus<real>());
    // r += thrust::reduce(dev_ptr33 , dev_ptr33 + snap_nz*snap_nx, 0.0, thrust::plus<real>());

  //  std::cout << "This is test GPU>FORWARD \nLAM_SHOT=" << l << " \nMU_SHOT=" << m << " \nRHO_SHOT=" << r << " \n\n";





            // -----------------------------------
            // 4.0. ADJOING MODELLING
            // ------------------------------------

            // Preparing adjoint shot to fire in the shot
            // Fire all adjoint sources in this shot

            int box1 = 32;
            dim3 threadsPerBlock(box1);
            
            dim3 blocksPerGrid(nrec / box1 + 1);
            forLoop<<<blocksPerGrid, threadsPerBlock>>>(rec_shot_to_fire, ishot, nrec);
            // Seismic adjoint kernel

            accu = false; // Accumulated storage for output
            grad = true;  // no gradient computation in forward kernel

            kernel_PSV_GPU(ishot, nt, nz, nx, dt, dx, dz, surf, isurf, hc, fdorder,
                           vz, vx, uz, ux, szz, szx, sxx, We_adj, dz_z, dx_z, dz_x, dx_x,
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
//    l=0;m=0;r=0;
//      dev_ptr11 = thrust::device_pointer_cast(grad_lam_shot);
//      dev_ptr22 = thrust::device_pointer_cast(grad_mu_shot);
//      dev_ptr33 = thrust::device_pointer_cast(grad_rho_shot);

//     l += thrust::reduce(dev_ptr11 , dev_ptr11 + snap_nz*snap_nx, 0.0, thrust::plus<real>());
//     m += thrust::reduce(dev_ptr22 , dev_ptr22 + snap_nz*snap_nx, 0.0, thrust::plus<real>());
//     r += thrust::reduce(dev_ptr33 , dev_ptr33 + snap_nz*snap_nx, 0.0, thrust::plus<real>());
    // std::cout << "This is test GPU> ADJOINT \nLAM_SHOT=" << l << " \nMU_SHOT=" << m << " \nRHO_SHOT=" << r << " \n\n";

            

            // Smooth gradients (Option 2 here)
            // ----------------------------------------------------
            // APPLY GAUSSIAN SMOOTHING (BLURRING) FUNCTIONS TO
            // to grad_lam_shot, grad_mu_shot, grad_rho_shot,
            // -----------------------------------------------------

            // Calculate Energy Weights
           energy_weights2_GPU(We, We_adj, snap_z1, snap_z2, snap_x1, snap_x2, nx);

          
            // [We_adj used as temporary gradient here after]

            // GRAD_LAM
            // ----------------------------------------
            // Interpolate gradients to temporary array
           interpol_grad2_GPU(We_adj, grad_lam_shot, snap_z1, snap_z2,
                              snap_x1, snap_x2, snap_dz, snap_dx, nx);

            // Scale to energy weight and add to global array
            scale_grad_E2_GPU(grad_lam, We_adj, scalar_lam, We,
                              snap_z1, snap_z2, snap_x1, snap_x2, nz, nx);

            // GRAD_MU
            // ----------------------------------------
            // Interpolate gradients to temporary array
           interpol_grad2_GPU(We_adj, grad_mu_shot, snap_z1, snap_z2,
                              snap_x1, snap_x2, snap_dz, snap_dx, nx);
            // Scale to energy weight and add to global array
            scale_grad_E2_GPU(grad_mu, We_adj, scalar_mu, We,
                              snap_z1, snap_z2, snap_x1, snap_x2, nz, nx);

            // GRAD_RHO
            // ----------------------------------------
            // Interpolate gradients to temporary array
           interpol_grad2_GPU(We_adj, grad_rho_shot, snap_z1, snap_z2,
                              snap_x1, snap_x2, snap_dz, snap_dx, nx);
            // Scale to energy weight and add to global array
            scale_grad_E2_GPU(grad_rho, We_adj, scalar_rho, We,
                              snap_z1, snap_z2, snap_x1, snap_x2, nz, nx);
        }

        // Smooth the global gradient with taper functions

        // Preconditioning of Gradients

        // -------------------
        // 5.0. OPTIMIZATION (Directly computed here)
        // ----------------------------------

        // Congugate Gradient Method
        // -------------------------------
        std::cout << "Applying Preconditioning" << std::endl;

        // ====================================
        // Applying Conjugate Gradient Method
        // ====================================
        //PCG_PSV(PCG_dir_lam, PCG_lam, grad_lam, nz, nx);
        //PCG_PSV(PCG_dir_mu, PCG_mu, grad_mu, nz, nx);
        //PCG_PSV(PCG_dir_rho, PCG_rho, grad_rho, nz, nx);
        //-------------------------------------------------

        //write_mat(grad_lam, grad_mu, grad_rho, nz, nx, 1000*(iterstep+1));

        // Applying taper function
        // Currently only Tukey Taper function available
        taper2_GPU(grad_lam, nz, nx, snap_z1, snap_z2, snap_x1, snap_x2,
                   taper_t1, taper_t2, taper_b1, taper_b2, taper_l1, taper_l2, taper_r1, taper_r2);

        taper2_GPU(grad_mu, nz, nx, snap_z1, snap_z2, snap_x1, snap_x2,
                   taper_t1, taper_t2, taper_b1, taper_b2, taper_l1, taper_l2, taper_r1, taper_r2);

        taper2_GPU(grad_rho, nz, nx, snap_z1, snap_z2, snap_x1, snap_x2,
                   taper_t1, taper_t2, taper_b1, taper_b2, taper_l1, taper_l2, taper_r1, taper_r2);


        //write_mat(grad_lam, grad_mu, grad_rho, nz, nx, 1000*(iterstep+1)+2);
        // ----------------------
        // 6.0. MATERIAL UPDATE
        // ---------------------

        // Step length estimation for wave parameters
      

        step_length = step_length_PSV_GPU(step_length, L2_norm[iterstep], nshot, nt, nz, nx, dt, dx, dz, surf, isurf, hc, fdorder,
                                          vz, vx, uz, ux, szz, szx, sxx, We, dz_z, dx_z, dz_x, dx_x,
                                          lam, mu, rho, lam_copy, mu_copy, rho_copy,
                                          mu_zx, rho_zp, rho_xp, grad, grad_lam, grad_mu, grad_rho,
                                          pml_z, a_z, b_z, K_z, a_half_z, b_half_z, K_half_z,
                                          pml_x, a_x, b_x, K_x, a_half_x, b_half_x, K_half_x,
                                          mem_vz_z, mem_vx_z, mem_szz_z, mem_szx_z,
                                          mem_vz_x, mem_vx_x, mem_szx_x, mem_sxx_x,
                                          nsrc, stf_type, stf_z, stf_x, z_src, x_src, src_shot_to_fire,
                                          nrec, rtf_type, rtf_uz, rtf_ux, z_rec, x_rec,
                                          rtf_z_true, rtf_x_true, accu, accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx,
                                          snap_z1, snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx, 0);
                                          std::cout<<"\n\n *****STEP LENGTH GPU ******"<<step_length<<"\n";

<<<<<<< HEAD
       //========================================================================
       // APPLY GAUSSIAN SMOOTHING / BLURRING TO grad_lam, grad_mu and grad_rho
       //======================================================================
=======
                                          

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
>>>>>>> cuda_fwi_integration

        // Update material parameters to the gradients !!
        update_mat2_GPU(lam, lam_copy, grad_lam, 4.8e+10, 0.0, step_length, nz, nx);
        update_mat2_GPU(mu, mu_copy, grad_mu, 2.7e+10, 0.0, step_length, nz, nx);

        step_length_rho = 0.5 * step_length;
        update_mat2_GPU(rho, rho_copy, grad_rho, 3000.0, 1.25, step_length_rho, nz, nx);


          

        //
        // Saving the Accumulative storage file to a binary file for every shots
        std::cout << "Iteration step: " << iterstep << ", " << mat_save_interval << ", " << iterstep % mat_save_interval << std::endl;
        if (mat_save_interval > 0 && !(iterstep % mat_save_interval))
        {
            // Writing the accumulation array
            std::cout << "Writing updated material to binary file for ITERATION " << iterstep;


        cudaCheckError(cudaMemcpy(h_lam[0],lam, nz*nx* sizeof(real), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(h_mu[0],mu,  nz*nx* sizeof(real), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(h_rho[0],rho, nz*nx* sizeof(real), cudaMemcpyDeviceToHost));

            write_mat(h_lam, h_mu, h_rho, nz, nx, iterstep);
            std::cout << " <DONE>" << std::endl;
        }

        // smooth model

        //
    
        return;

        iterstep++;
        iter = (iterstep < maxIter) ? true : false; // Temporary condition
        if (iterstep > 25)
        {
            iter = (abs((L2_norm[iterstep] - L2_norm[iterstep - 2]) / L2_norm[iterstep - 2]) < 0.001) ? false : true; // Temporary condition
            std::cout << "The change is less than minimal after " << iterstep << " iteration steps." << std::endl;
        }
    }

    // Saving the Accumulative storage file to a binary file for every shots
    if (mat_save_interval < 1)
    {
        // Writing the accumulation array
        std::cout << "Writing updated material to binary file <FINAL> ITERATION " << iterstep;

        cudaCheckError(cudaMemcpy(h_lam[0],lam, nz*nx* sizeof(real), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(h_mu[0],mu,  nz*nx* sizeof(real), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(h_rho[0],rho, nz*nx* sizeof(real), cudaMemcpyDeviceToHost));

        write_mat(h_lam, h_mu, h_rho, nz, nx, iterstep);
        //write_mat(lam, mu, rho, nz, nx, iterstep);
        std::cout << " <DONE>" << std::endl;
    }
}
