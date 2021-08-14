#include "d_preproc.cuh"
#include <math.h>




void alloc_varpre_PSV_GPU( real *&hc, int *&isurf, int *&npml, // holberg coefficients, surface indices and number pml in each side
    // Medium arguments
    real *&lam, real    *&mu, real *&rho,
    //PML arguments (z and x direction)
    real *&a_z, real *&b_z, real *&K_z, 
    real *&a_half_z, real *&b_half_z, real *&K_half_z,
    real *&a_x, real *&b_x, real *&K_x, 
    real *&a_half_x, real *&b_half_x, real *&K_half_x, 
    // Seismic sources
    int *&z_src, int *&x_src, // source locations
    int *&src_shot_to_fire, // which source to fire on which shot index
    real *&stf_z, real *&stf_x, // source time functions
    // Reciever seismograms
    int *&z_rec, int *&x_rec,
    real *&rtf_z_true, real *&rtf_x_true, // Field measurements for receivers
    // Scalar variables for allocation
    int fdorder, 
    bool pml_z, bool pml_x, int nsrc, int nrec, 
    int nt,int nshot, int nz, int nx)
    {
    // Allocates the variables that are to be...
    // Transferred from device to the host
    // These Arrays need to take array readings from (device to the host)
    // Every shots should have same number of time steps

    // Allocating holberg coefficient
    //hc = new real[6];
    cudaCheckError(cudaMalloc((void**)&hc, 6 * sizeof(real)));
    // Allocating surface indices & npml
   // isurf = new int[4]; 
    cudaCheckError(cudaMalloc((void**)&isurf, 4* sizeof(real)));// four sized in the surface

   // npml = new int[4];
     cudaCheckError(cudaMalloc((void**)&npml, 4 * sizeof(real)));// number of PMLs in each side
    // Allocating medium arrays
   
    
    cudaCheckError(cudaMalloc((void**)&lam, nz*nx* sizeof(real)));
    cudaCheckError(cudaMalloc((void**)&mu,  nz*nx* sizeof(real)));
    cudaCheckError(cudaMalloc((void**)&rho, nz*nx* sizeof(real)));

    // Allocating PML coeffieients
    if (pml_z){
        cudaCheckError(cudaMalloc((void**)&a_z, nz * sizeof(real)));//a_z = new real[nz];
        cudaCheckError(cudaMalloc((void**)&b_z, nz * sizeof(real)));// b_z = new real[nz];
        cudaCheckError(cudaMalloc((void**)&K_z, nz * sizeof(real)));//K_z = new real[nz];

        cudaCheckError(cudaMalloc((void**)&a_half_z, nz * sizeof(real)));// a_half_z = new real[nz];
        cudaCheckError(cudaMalloc((void**)&b_half_z, nz * sizeof(real)));//b_half_z = new real[nz];
        cudaCheckError(cudaMalloc((void**)&K_half_z, nz * sizeof(real)));//K_half_z = new real[nz];
    }

    if (pml_x){
        cudaCheckError(cudaMalloc((void**)&a_x, nx * sizeof(real)));//a_x = new real[nx];
        cudaCheckError(cudaMalloc((void**)&b_x, nx * sizeof(real)));// b_x = new real[nx];
        cudaCheckError(cudaMalloc((void**)&K_x, nx * sizeof(real)));// K_x = new real[nx];

        cudaCheckError(cudaMalloc((void**)&a_half_x, nx * sizeof(real)));//a_half_x = new real[nx];
        cudaCheckError(cudaMalloc((void**)&b_half_x, nx * sizeof(real)));//b_half_x = new real[nx];
        cudaCheckError(cudaMalloc((void**)&K_half_x, nx * sizeof(real))); //K_half_x = new real[nx];
    }

    // Allocating Source locations and time functions
    if (nsrc){
        // locations (grid index)
        cudaCheckError(cudaMalloc((void**)&z_src, nsrc * sizeof(real))); //z_src = new int[nsrc];
        cudaCheckError(cudaMalloc((void**)&x_src, nsrc * sizeof(real)));//x_src = new int[nsrc];
        cudaCheckError(cudaMalloc((void**)&src_shot_to_fire, nsrc * sizeof(real)));// src_shot_to_fire = new int[nsrc];

        // stf  nsrc * nt
       cudaCheckError(cudaMalloc((void**)&stf_z, nsrc * nt * sizeof(real)));// allocate_array(stf_z, nsrc, nt);
       cudaCheckError(cudaMalloc((void**)&stf_x,  nsrc * nt * sizeof(real)));// allocate_array(stf_x, nsrc, nt);
    }

    if (nrec){
        // locations (grid index)
        cudaCheckError(cudaMalloc((void**)&z_rec, nrec * sizeof(real))); //z_rec = new int[nrec];
        cudaCheckError(cudaMalloc((void**)&x_rec, nrec * sizeof(real)));// x_rec = new int[nrec];

       std::cout << "Receivers allocated on GPU" << std::endl;
        // rtf field measurements
        cudaCheckError(cudaMalloc((void**)&rtf_z_true,  nshot * nrec * nt * sizeof(real)));//allocate_array(rtf_z_true, nrec, nt);
        cudaCheckError(cudaMalloc((void**)&rtf_x_true,  nshot * nrec * nt * sizeof(real)));//allocate_array(rtf_x_true, nrec, nt);


    }

}


void copy_varpre_PSV_CPU_TO_GPU( 
    real *&hc, int *&isurf, int *&npml, // holberg coefficients, surface indices and number pml in each side
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
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    real *&d_hc, int *&d_isurf, int *&d_npml, // holberg coefficients, surface indices and number pml in each side
    // Medium arguments
    real *&d_lam, real    *&d_mu, real *&d_rho,
    //PML arguments (z and x direction)
    real *&d_a_z, real *&d_b_z, real *&d_K_z, 
    real *&d_a_half_z, real *&d_b_half_z, real *&d_K_half_z,
    real *&d_a_x, real *&d_b_x, real *&d_K_x, 
    real *&d_a_half_x, real *&d_b_half_x, real *&d_K_half_x, 
    // Seismic sources
    int *&d_z_src, int *&d_x_src, // source locations
    int *&d_src_shot_to_fire, // which source to fire on which shot index
    real *&d_stf_z, real *&d_stf_x, // source time functions
    // Reciever seismograms
    int *&d_z_rec, int *&d_x_rec,
    real *&d_rtf_z_true, real *&d_rtf_x_true, // Field measurements for receivers
    // Scalar variables for allocation
    int fdorder, 
    bool pml_z, bool pml_x, int nsrc, int nrec, 
    int nt,int nshot, int nz, int nx)
    {
    // Allocates the variables that are to be...
    // Transferred from device to the host
    // These Arrays need to take array readings from (device to the host)
    // Every shots should have same number of time steps

    // Allocating holberg coefficient
    //hc = new real[6];
    cudaCheckError(cudaMemcpy(d_hc,hc, 6 * sizeof(real), cudaMemcpyHostToDevice));
    // Allocating surface indices & npml
   // isurf = new int[4]; 
    cudaCheckError(cudaMemcpy(d_isurf,isurf, 4* sizeof(real), cudaMemcpyHostToDevice));// four sized in the surface

   // npml = new int[4];
     cudaCheckError(cudaMemcpy(d_npml,npml, 4 * sizeof(real), cudaMemcpyHostToDevice));// number of PMLs in each side
    // Allocating medium arrays
   
    
    cudaCheckError(cudaMemcpy(d_lam,lam[0], nz*nx* sizeof(real), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_mu,mu[0],  nz*nx* sizeof(real), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_rho,rho[0], nz*nx* sizeof(real), cudaMemcpyHostToDevice));

    cudaCheckError(cudaMemcpy(lam[0],d_lam, nz*nx* sizeof(real), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(mu[0],d_mu,  nz*nx* sizeof(real), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(rho[0],d_rho, nz*nx* sizeof(real), cudaMemcpyDeviceToHost));

    // Allocating PML coeffieients
    if (pml_z){
        cudaCheckError(cudaMemcpy(d_a_z,a_z, nz * sizeof(real), cudaMemcpyHostToDevice));//a_z = new real[nz];
        cudaCheckError(cudaMemcpy(d_b_z,b_z, nz * sizeof(real), cudaMemcpyHostToDevice));// b_z = new real[nz];
        cudaCheckError(cudaMemcpy(d_K_z,K_z, nz * sizeof(real), cudaMemcpyHostToDevice));//K_z = new real[nz];

        cudaCheckError(cudaMemcpy(d_a_half_z, a_half_z, nz * sizeof(real), cudaMemcpyHostToDevice));// a_half_z = new real[nz];
        cudaCheckError(cudaMemcpy(d_b_half_z, b_half_z, nz * sizeof(real), cudaMemcpyHostToDevice));//b_half_z = new real[nz];
        cudaCheckError(cudaMemcpy(d_K_half_z, K_half_z, nz * sizeof(real), cudaMemcpyHostToDevice));//K_half_z = new real[nz];
    }

    if (pml_x){
        cudaCheckError(cudaMemcpy(d_a_x, a_x, nx * sizeof(real), cudaMemcpyHostToDevice));//a_x = new real[nx];
        cudaCheckError(cudaMemcpy(d_b_x, b_x, nx * sizeof(real), cudaMemcpyHostToDevice));// b_x = new real[nx];
        cudaCheckError(cudaMemcpy(d_K_x, K_x, nx * sizeof(real), cudaMemcpyHostToDevice));// K_x = new real[nx];

        cudaCheckError(cudaMemcpy(d_a_half_x, a_half_x, nx * sizeof(real), cudaMemcpyHostToDevice));//a_half_x = new real[nx];
        cudaCheckError(cudaMemcpy(d_b_half_x, b_half_x, nx * sizeof(real), cudaMemcpyHostToDevice));//b_half_x = new real[nx];
        cudaCheckError(cudaMemcpy(d_K_half_x, K_half_x, nx * sizeof(real), cudaMemcpyHostToDevice)); //K_half_x = new real[nx];
    }

    // Allocating Source locations and time functions
    if (nsrc){
        // locations (grid index)
        cudaCheckError(cudaMemcpy(d_z_src, z_src, nsrc * sizeof(real), cudaMemcpyHostToDevice)); //z_src = new int[nsrc];
        cudaCheckError(cudaMemcpy(d_x_src, x_src, nsrc * sizeof(real), cudaMemcpyHostToDevice));//x_src = new int[nsrc];
        cudaCheckError(cudaMemcpy(d_src_shot_to_fire, src_shot_to_fire, nsrc * sizeof(real), cudaMemcpyHostToDevice));// src_shot_to_fire = new int[nsrc];

        // stf  nsrc * nt
       cudaCheckError(cudaMemcpy(d_stf_z,stf_z[0], nsrc * nt * sizeof(real), cudaMemcpyHostToDevice));// allocate_array(stf_z, nsrc, nt);
       cudaCheckError(cudaMemcpy(d_stf_x,stf_x[0],  nsrc * nt * sizeof(real), cudaMemcpyHostToDevice));// allocate_array(stf_x, nsrc, nt);
    }

    if (nrec){
        // locations (grid index)
        cudaCheckError(cudaMemcpy(d_z_rec,z_rec, nrec * sizeof(real), cudaMemcpyHostToDevice)); //z_rec = new int[nrec];
        cudaCheckError(cudaMemcpy(d_x_rec,x_rec, nrec * sizeof(real), cudaMemcpyHostToDevice));// x_rec = new int[nrec];

       std::cout << "Receivers Copied on GPU" << std::endl;
        // rtf field measurements
        cudaCheckError(cudaMemcpy(d_rtf_z_true,rtf_z_true[0][0],  nshot * nrec * nt * sizeof(real), cudaMemcpyHostToDevice));//allocate_array(rtf_z_true, nrec, nt);
        cudaCheckError(cudaMemcpy(d_rtf_x_true,rtf_x_true[0][0],  nshot * nrec * nt * sizeof(real), cudaMemcpyHostToDevice));//allocate_array(rtf_x_true, nrec, nt);
    }

}