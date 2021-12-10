//kernel_lib_PSV.cpp

/* 
* Created by: Min Basnet
* 2020.November.17
* Kathmandu, Nepal
*/

// Contains the functions for finite difference computation of
// Seismic wave propagation in time domain
// For stress velocity formulations
// Currently only for the order = 2

#include <iostream>
#include <math.h>
#include "d_kernel_lib_PSV.cuh"


__global__ void cuda_mat_grid2_GPU(real* lam, real* mu, real* rho, //remove the "&" in device func calls
    real lam_sc, real mu_sc, real rho_sc,
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, int nz, int nx)
{

    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = blockIdx.y * blockDim.y + threadIdx.y;

    if ( iz >= 0 && iz < nz && ix >= 0 && ix < nx)
    {

        lam[iz * nx + ix] = lam_sc;
        mu[iz * nx + ix] = mu_sc;
        rho[iz * nx +ix] = rho_sc;

       // printf("GPU i=%d j=%d ans=%lf %lf %lf \n",iz,ix, lam[iz * nx + ix], mu[iz * nx + ix], rho[iz * nx + ix] );
    }
}


void mat_grid2_GPU(
    // Gradients, material average and energy weights
    real* lam, real* mu, real * rho,
    real lam_sc, real mu_sc, real rho_sc,
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, int nz, int nx)
{

    //kernel configration
    int box1 = 32, box2 = 32;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid((nz) / box1 + 1, (nx) / box2 + 1);

//    auto start_GPU = high_resolution_clock::now();
    cuda_mat_grid2_GPU << <blocksPerGrid, threadsPerBlock >> > (lam, mu, rho, lam_sc, mu_sc, rho_sc,
        snap_z1, snap_z2, snap_x1, snap_x2, nz, nx);
    // auto stop_GPU = high_resolution_clock::now();
    // auto duration_GPU = duration_cast<microseconds>(stop_GPU - start_GPU);
    // cout << "Time taken by GPU: "
    //     << duration_GPU.count() << " microseconds" << endl;

    cudaCheckError(cudaDeviceSynchronize());
}

__global__ void cuda_copy_mat_GPU(real *lam_copy, real *mu_copy,  real *rho_copy,
        real *lam, real *mu,  real *rho, int nz, int nx)
{

    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz<nz&&ix<nx)
    {

        lam_copy[iz*nx+ix]=lam[iz*nx+ix];
       mu_copy[iz*nx+ix]=mu[iz*nx+ix];
        rho_copy[iz*nx+ix]=rho[iz*nx+ix];

        //printf("GPU i=%d j=%d ans=%lf %lf %lf \n",iz,ix, lam_copy[iz*nx+ ix], mu_copy[iz * nx + ix], rho_copy[iz * nx + ix] );
    }
}

//gpu kernel end

//gpu code function
void copy_mat_GPU( real *lam_copy, real *mu_copy,  real *rho_copy,
        real *lam, real *mu,  real *rho, int nz, int nx)
{

    //kernel configration
    int box1 = 32, box2 = 32;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid((nz) / box1 + 1, (nx) / box2 + 1);

    cuda_copy_mat_GPU<<<blocksPerGrid, threadsPerBlock>>>(lam_copy,mu_copy,rho_copy,lam,mu,rho,nz,nx);

    cudaCheckError(cudaDeviceSynchronize());
}

__global__ void cuda_scale_grad_E2_GPU(real *grad, real *grad_shot, //remove the "&" in device func calls
                                       real mat_av, real *We,
                                       // space snap parameters
                                       int snap_z1, int snap_z2, int snap_x1, int snap_x2, int nz, int nx)
{

    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = blockIdx.y * blockDim.y + threadIdx.y;

    if (mat_av > 0 && iz >= snap_z1 && iz < snap_z2 && ix >= snap_x1 && ix < snap_x2)
    {

        grad[iz * nx + ix] += grad_shot[iz * nx + ix] / (We[iz * nx + ix] * mat_av * mat_av);//grad_shot,we_adj,we

        //printf("GPU i=%d j=%d ans=%lf %lf %lf \n",iz,ix, grad[iz*nx+ ix], grad_shot[iz * nx + ix], We[iz * nx + ix] );
    }
}
void scale_grad_E2_GPU(
    // Gradients, material average and energy weights
    real *grad, real *grad_shot,
    real mat_av, real *We,
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, int nz, int nx)
{

    //kernel configration
    int box1 = 32, box2 = 32;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid((nz) / box1 + 1, (nx) / box2 + 1);

    cuda_scale_grad_E2_GPU<<<blocksPerGrid, threadsPerBlock>>>(grad, grad_shot, mat_av, We,
                                                               snap_z1, snap_z2, snap_x1, snap_x2, nz, nx);

    cudaCheckError(cudaDeviceSynchronize());
}






__global__ void cuda_mat_av2_GPU(real *lam, real *mu, real *rho,
                                 real *mu_zx, real *rho_zp, real *rho_xp, // inverse of densityint dimz, int dimx
                                 int nz, int nx)
{

    long int iz = blockIdx.x * blockDim.x + threadIdx.x;
    long int ix = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nx - 1 && ix >= 0 && iz >= 0 && iz < nz - 1)
    {
        // Harmonic average for mu
        mu_zx[iz * nx + ix] = 4.0 / ((1.0 / mu[iz * nx + ix]) + (1.0 / mu[iz * nx + ix + 1]) + (1.0 / mu[(iz + 1) * nx + ix]) + (1.0 / mu[(iz + 1) * nx + ix + 1]));

        if ((mu[iz * nx + ix] == 0.0) || (mu[iz * nx + ix + 1] == 0.0) || (mu[(iz + 1) * nx + ix] == 0.0) || (mu[(iz + 1) * nx + ix + 1] == 0.0))
        {
            mu_zx[iz * nx + ix] = 0.0;
        }

        // Arithmatic average of rho
        // the averages are inversed for computational efficiency
        rho_zp[iz * nx + ix] = 1.0 / (0.5 * (rho[iz * nx + ix] + rho[(iz + 1) * nx + ix]));
        rho_xp[iz * nx + ix] = 1.0 / (0.5 * (rho[iz * nx + ix] + rho[iz * nx + ix + 1]));

        if ((rho[iz * nx + ix] < 1e-4) && (rho[(iz + 1) * nx + ix] < 1e-4))
        {
            rho_zp[iz * nx + ix] = 0.0;
        }

        if ((rho[iz * nx + ix] < 1e-4) && (rho[iz * nx + ix + 1] < 1e-4))
        {
            rho_zp[iz * nx + ix] = 0.0;
        }
    }
  
}

void mat_av2_GPU(
    // Medium arguments
    real *&lam, real *&mu, real *&rho,
    real *&mu_zx, real *&rho_zp, real *&rho_xp, // inverse of densityint dimz, int dimx
    real &C_lam, real &C_mu, real &C_rho,       // scalar averages
    int nz, int nx)
{

    // Harmonic 2d average of mu and
    // Arithmatic 1d average of rho

    C_lam = 0.0;
    C_mu = 0.0;
    C_rho = 0.0;
    //Cuda config////
    int box1 = 32, box2 = 32;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid((nz) / box1 + 1, (nx) / box2 + 1);

    cuda_mat_av2_GPU<<<blocksPerGrid, threadsPerBlock>>>(lam, mu, rho,
                                                         mu_zx, rho_zp, rho_xp, // inverse of densityint dimz, int dimx
                                                         nz, nx);
    cudaCheckError(cudaDeviceSynchronize());

    thrust::device_ptr<real> dev_ptr1 = thrust::device_pointer_cast(mu_zx);
    thrust::device_ptr<real> dev_ptr2 = thrust::device_pointer_cast(rho_xp);
    thrust::device_ptr<real> dev_ptr3 = thrust::device_pointer_cast(rho_zp);

    for (int iz = 0; iz < nz - 1; iz++)
    {
        C_lam += thrust::reduce(dev_ptr1 + iz * nx, dev_ptr1 + iz * nx + (nx - 1), 0.0, thrust::plus<real>());
        C_mu += thrust::reduce(dev_ptr2 + iz * nx, dev_ptr2 + iz * nx + (nx - 1), 0.0, thrust::plus<real>());
        C_rho += thrust::reduce(dev_ptr3 + iz * nx, dev_ptr3 + iz * nx + (nx - 1), 0.0, thrust::plus<real>());
    }

 
    C_lam = C_lam / ((nz - 1) * (nx - 1));
    C_mu = C_mu / ((nz - 1) * (nx - 1));
    C_rho = C_rho / ((nz - 1) * (nx - 1));

    //TEST
    double l=0,m=0,r=0;
    thrust::device_ptr<real> dev_ptr11 = thrust::device_pointer_cast(lam);
    thrust::device_ptr<real> dev_ptr22 = thrust::device_pointer_cast(mu);
    thrust::device_ptr<real> dev_ptr33 = thrust::device_pointer_cast(rho);

    l += thrust::reduce(dev_ptr11 , dev_ptr11 + nz*nx, 0.0, thrust::plus<real>());
    m += thrust::reduce(dev_ptr22 , dev_ptr22 + nz*nx, 0.0, thrust::plus<real>());
    r += thrust::reduce(dev_ptr33 , dev_ptr33 + nz*nx, 0.0, thrust::plus<real>());

    


    std::cout << "This is test GPU \nC_lam=" << C_lam << " \nC_mu=" << C_mu << " \nC_rho=" << C_rho << " \n\n";

    std::cout << "This is test GPU II \nlam sum =" << l << " \nmu sum=" << m << " \nr sum=" << r << " \n\n";
}

void reset_sv2_GPU(
    // wave arguments (velocity) & Energy weights
    real *&vz, real *&vx, real *&uz, real *&ux,
    real *&szz, real *&szx, real *&sxx, real *&We,
    // time & space grids (size of the arrays)
    real nz, real nx)
{
    // reset the velocity and stresses to zero
    // generally applicable in the beginning of the time loop

    const size_t size = nz * nx * sizeof(real);
    cudaCheckError(cudaMemset(vz, 0, size));
    cudaCheckError(cudaMemset(vx, 0, size));
    cudaCheckError(cudaMemset(uz, 0, size));
    cudaCheckError(cudaMemset(ux, 0, size));
    cudaCheckError(cudaMemset(szz, 0, size));
    cudaCheckError(cudaMemset(szx, 0, size));
    cudaCheckError(cudaMemset(sxx, 0, size));
    cudaCheckError(cudaMemset(We, 0, size));
}

void reset_PML_memory2_GPU(
    // PML memory arrays
    real *&mem_vz_z, real *&mem_vx_z, real *&mem_vz_x, real *&mem_vx_x,
    // time & space grids (size of the arrays)
    real nz, real nx)
{
   
    // reset the velocity and stresses to zero
    // generally applicable in the beginning of the time loop
    const size_t size = nz * nx * sizeof(real);
    cudaCheckError(cudaMemset(mem_vz_z, 0, size));
    cudaCheckError(cudaMemset(mem_vx_z, 0, size));
    cudaCheckError(cudaMemset(mem_vz_x, 0, size));
    cudaCheckError(cudaMemset(mem_vx_x, 0, size));
}

void reset_grad_shot2_GPU(real *&grad_lam, real *&grad_mu, real *&grad_rho,
                          int snap_z1, int snap_z2, int snap_x1, int snap_x2,
                          int snap_dz, int snap_dx, int nx, int nz)
{

    int jz=0;

    


    

   // for (int iz = snap_z1; iz <= snap_z2; iz += snap_dz)
    //{
        int snap_nz = 1 + (snap_z2 - snap_z1)/snap_dz;
        int snap_nx = 1 + (snap_x2 - snap_x1)/snap_dx;

        const size_t size = snap_nz* snap_nx * sizeof(real);
        
        cudaCheckError(cudaMemset(grad_lam , 0.0,size));
        cudaCheckError(cudaMemset(grad_mu,  0.0,  size));
        cudaCheckError(cudaMemset(grad_rho, 0.0,  size));
        
        cudaCheckError(cudaDeviceSynchronize());
        //jz++;
   // }
}

__global__ void cuda_vdiff2_GPU(
    // spatial velocity derivatives
    real *vz_z, real *vx_z, real *vz_x, real *vx_x,
    // wave arguments (velocity)
    real *vz, real *vx,
    // holberg coefficient
    real *hc,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dz, real dx, int nx)
{

    real dxi = 1.0 / dx;
    real dzi = 1.0 / dz; // inverse of dx and dz

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz >= nz1 && iz < nz2 && ix < nx2 && ix >= nx1)
    {
        vz_z[iz * nx + ix] = dzi * hc[1] * (vz[iz * nx + ix] - vz[(iz - 1) * nx + ix]);
        vx_z[iz * nx + ix] = dzi * hc[1] * (vx[(iz + 1) * nx + ix] - vx[iz * nx + ix]);
        vz_x[iz * nx + ix] = dxi * hc[1] * (vz[iz * nx + ix + 1] - vz[iz * nx + ix]);
        vx_x[iz * nx + ix] = dxi * hc[1] * (vx[iz * nx + ix] - vx[iz * nx + ix - 1]);
    }
  
}

void vdiff2_GPU(
    // spatial velocity derivatives
    real *&vz_z, real *&vx_z, real *&vz_x, real *&vx_x,
    // wave arguments (velocity)
    real *&vz, real *&vx,
    // holberg coefficient
    real *&hc,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dz, real dx, int nx)
{
    // updates the stress kernels for each timestep in 2D grid

    //Cuda config
    int box1 = 32, box2 = 32;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid((nx2) / box1 + 1, (nz2) / box2 + 1);
    // 2D space grid in GPU

    cuda_vdiff2_GPU<<<blocksPerGrid, threadsPerBlock>>>( // spatial velocity derivatives
        vz_z, vx_z, vz_x, vx_x,
        // wave arguments (velocity)
        vz, vx,
        // holberg coefficient
        hc,
        // time space grids
        nz1, nz2, nx1, nx2, dz, dx, nx);

  // thrust::device_ptr<real> dev_ptr1 = thrust::device_pointer_cast(vz);
    //real k=     thrust::reduce(dev_ptr1 , dev_ptr1 + 401*nx, 0.0, thrust::plus<real>());
     //std::cout<<k<<"  << vz of vdiff2 \n";

    cudaCheckError(cudaDeviceSynchronize());

    // std::cout<<"***vdiff2_GPU*******\n"<<"\n";
}

__global__ void cuda_pml_diff2_GPU(bool pml_z, bool pml_x,
                                   // spatial derivatives
                                   real *dz_z, real *dx_z, real *dz_x, real *dx_x,
                                   //PML arguments (z and x direction)
                                   real *a_z, real *b_z, real *K_z,
                                   real *a_half_z, real *b_half_z, real *K_half_z,
                                   real *a_x, real *b_x, real *K_x,
                                   real *a_half_x, real *b_half_x, real *K_half_x,
                                   // PML memory arrays for spatial derivatives
                                   real *mem_z_z, real *mem_x_z,
                                   real *mem_z_x, real *mem_x_x,
                                   // time space grids
                                   int nz1, int nz2, int nx1, int nx2, int nx)
{

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz >= nz1 && iz < nz2 && ix >= nx1 && ix < nx2)
    {

        if (pml_z)
        {
            // CPML memory variables in z-direction
            mem_z_z[iz * nx + ix] = b_z[iz] * mem_z_z[iz * nx + ix] + a_z[iz] * dz_z[iz * nx + ix];
            mem_x_z[iz * nx + ix] = b_half_z[iz] * mem_x_z[iz * nx + ix] + a_half_z[iz] * dx_z[iz * nx + ix];

            dz_z[iz * nx + ix] = dz_z[iz * nx + ix] / K_z[iz] + mem_z_z[iz * nx + ix];
            dx_z[iz * nx + ix] = dx_z[iz * nx + ix] / K_half_z[iz] + mem_x_z[iz * nx + ix];
        }
        if (pml_x)
        {
            // CPML memory variables in x-direction
            mem_x_x[iz * nx + ix] = b_x[ix] * mem_x_x[iz * nx + ix] + a_x[ix] * dx_x[iz * nx + ix];
            mem_z_x[iz * nx + ix] = b_half_x[ix] * mem_z_x[iz * nx + ix] + a_half_x[ix] * dz_x[iz * nx + ix];

            dx_x[iz * nx + ix] = dx_x[iz * nx + ix] / K_x[ix] + mem_x_x[iz * nx + ix];
            dz_x[iz * nx + ix] = dz_x[iz * nx + ix] / K_half_x[ix] + mem_z_x[iz * nx + ix];
        }
    }
}

void pml_diff2_GPU(bool pml_z, bool pml_x,
                   // spatial derivatives
                   real *&dz_z, real *&dx_z, real *&dz_x, real *&dx_x,
                   //PML arguments (z and x direction)
                   real *&a_z, real *&b_z, real *&K_z,
                   real *&a_half_z, real *&b_half_z, real *&K_half_z,
                   real *&a_x, real *&b_x, real *&K_x,
                   real *&a_half_x, real *&b_half_x, real *&K_half_x,
                   // PML memory arrays for spatial derivatives
                   real *&mem_z_z, real *&mem_x_z,
                   real *&mem_z_x, real *&mem_x_x,
                   // time space grids
                   int nz1, int nz2, int nx1, int nx2, int nx)
{

    // updates PML memory variables for velicity derivatives
    // absorption coefficients are for the whole grids
    // 2D space grid
    //Cuda config
    int box1 = 32, box2 = 32;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid((nx2) / box1 + 1, (nz2) / box2 + 1);
    // 2D space grid in GPU
    cuda_pml_diff2_GPU<<<blocksPerGrid, threadsPerBlock>>>(pml_z, pml_x,
                                                           dz_z, dx_z, dz_x, dx_x, a_z, b_z, K_z, a_half_z, b_half_z,
                                                           K_half_z, a_x, b_x, K_x, a_half_x, b_half_x, K_half_x, mem_z_z, mem_x_z, mem_z_x, mem_x_x, nz1, nz2, nx1, nx2, nx);
    cudaCheckError(cudaDeviceSynchronize());
}

__global__ void cuda_update_s2_GPU( // Wave arguments (stress)
    real *szz, real *szx, real *sxx,
    // spatial velocity derivatives
    real *vz_z, real *vx_z, real *vz_x, real *vx_x,
    // Medium arguments
    real *lam, real *mu, real *mu_zx,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dt, int nx)
{

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz >= nz1 && iz < nz2 && ix >= nx1 && ix < nx2)
    {

        // updating stresses
        szz[iz * nx + ix] += dt * (lam[iz * nx + ix] * (vx_x[iz * nx + ix] + vz_z[iz * nx + ix]) + (2.0 * mu[iz * nx + ix] * vz_z[iz * nx + ix]));
        szx[iz * nx + ix] += dt * mu_zx[iz * nx + ix] * (vz_x[iz * nx + ix] + vx_z[iz * nx + ix]);
        sxx[iz * nx + ix] += dt * (lam[iz * nx + ix] * (vx_x[iz * nx + ix] + vz_z[iz * nx + ix]) + (2.0 * mu[iz * nx + ix] * vx_x[iz * nx + ix]));
    }
}

void update_s2_GPU(
    // Wave arguments (stress)
    real *&szz, real *&szx, real *&sxx,
    // spatial velocity derivatives
    real *&vz_z, real *&vx_z, real *&vz_x, real *&vx_x,
    // Medium arguments
    real *&lam, real *&mu, real *&mu_zx,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dt, int nx)
{
    // update stress from velocity derivatives

    //Cuda config
    int box1 = 32, box2 = 32;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid((nx2) / box1 + 1, (nz2) / box2 + 1);
    cuda_update_s2_GPU<<<blocksPerGrid, threadsPerBlock>>>(szz, szx, sxx, vz_z, vx_z, vz_x, vx_x, lam, mu,
                                                           mu_zx, nz1, nz2, nx1, nx2, dt, nx);

    // thrust::device_ptr<real> dev_ptr1 = thrust::device_pointer_cast(vz_z);
    // real k=     thrust::reduce(dev_ptr1 , dev_ptr1 + 401*nx, 0.0, thrust::plus<real>());
    //  std::cout<<k<<"  << szx of update_s2 \n";
    cudaCheckError(cudaDeviceSynchronize());
}

__global__ void cuda_sdiff2_GPU(
    // spatial stress derivatives
    real *szz_z, real *szx_z, real *szx_x, real *sxx_x,
    // Wave arguments (stress)
    real *szz, real *szx, real *sxx,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dz, real dx,
    // holberg coefficient
    real *hc, int nx)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    real dxi = 1.0 / dx;
    real dzi = 1.0 / dz; // inverse of dx and dz

    if (iz >= nz1 && iz < nz2 && ix >= nx1 && ix < nx2)
    {
        // compute spatial stress derivatives
        szz_z[iz * nx + ix] = dzi * hc[1] * (szz[(iz + 1) * nx + ix] - szz[iz * nx + ix]);
        szx_z[iz * nx + ix] = dzi * hc[1] * (szx[iz * nx + ix] - szx[(iz - 1) * nx + ix]);
        szx_x[iz * nx + ix] = dxi * hc[1] * (szx[iz * nx + ix] - szx[iz * nx + ix - 1]);
        sxx_x[iz * nx + ix] = dxi * hc[1] * (sxx[iz * nx + ix + 1] - sxx[iz * nx + ix]);
    }
}

void sdiff2_GPU(
    // spatial stress derivatives
    real *&szz_z, real *&szx_z, real *&szx_x, real *&sxx_x,
    // Wave arguments (stress)
    real *&szz, real *&szx, real *&sxx,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dz, real dx,
    // holberg coefficient
    real *&hc, int nx)
{
    // updates the stress kernels for each timestep in 2D grid

    //Cuda config
    int box1 = 32, box2 = 32;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid((nx2) / box1 + 1, (nz2) / box2 + 1);

    cuda_sdiff2_GPU<<<blocksPerGrid, threadsPerBlock>>>(
        // spatial stress derivatives
        szz_z, szx_z, szx_x, sxx_x,
        // Wave arguments (stress)
        szz, szx, sxx,
        // time space grids
        nz1, nz2, nx1, nx2, dz, dx,
        // holberg coefficient
        hc, nx);
    cudaCheckError(cudaDeviceSynchronize());
}

__global__ void cuda_update_v2_GPU(
    // wave arguments (velocity) & Energy weights
    real *vz, real *vx,
    // displacement and energy arrays
    real *uz, real *ux, real *We,
    // spatial stress derivatives
    real *szz_z, real *szx_z, real *szx_x, real *sxx_x,
    // Medium arguments
    real *rho_zp, real *rho_xp,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dt, int nx)
{

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz >= nz1 && iz < nz2 && ix >= nx1 && ix < nx2)
    {
        // Calculating displacement from previous velocity
        uz[iz * nx + ix] += dt * vz[iz * nx + ix];
        ux[iz * nx + ix] += dt * vx[iz * nx + ix];

        // update particle velocities
        vz[iz * nx + ix] += dt * rho_zp[iz * nx + ix] * (szx_x[iz * nx + ix] + szz_z[iz * nx + ix]);
        vx[iz * nx + ix] += dt * rho_xp[iz * nx + ix] * (sxx_x[iz * nx + ix] + szx_z[iz * nx + ix]);
        // Displacements and Energy weights
        We[iz * nx + ix] += vx[iz * nx + ix] * vx[iz * nx + ix] + vz[iz * nx + ix] * vz[iz * nx + ix];
    }
}

void update_v2_GPU(
    // wave arguments (velocity) & Energy weights
    real *&vz, real *&vx,
    // displacement and energy arrays
    real *&uz, real *&ux, real *&We,
    // spatial stress derivatives
    real *&szz_z, real *&szx_z, real *&szx_x, real *&sxx_x,
    // Medium arguments
    real *&rho_zp, real *&rho_xp,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dt, int nx)
{
    // update stress from velocity derivatives
    int box1 = 32, box2 = 32;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid((nx2) / box1 + 1, (nz2) / box2 + 1);

    cuda_update_v2_GPU<<<blocksPerGrid, threadsPerBlock>>>(vz, vx, uz, ux, We, szz_z, szx_z, szx_x, sxx_x,
                                                           rho_zp, rho_xp, nz1, nz2, nx1, nx2, dt, nx);
    cudaCheckError(cudaDeviceSynchronize());
}




__global__ void cuda_surf_mirror_GPU(
    // Wave arguments (stress & velocity derivatives)
    real* szz, real* szx, real* sxx, real* vz_z, real* vx_x,
    // Medium arguments
    real* lam, real* mu,
    // surface indices for four directions(0.top, 1.bottom, 2.left, 3.right)
    int* surf,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dt, int nx)
{
    // surface mirroring for free surface


    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int sz = blockIdx.y * blockDim.y + threadIdx.y;


    int sx = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;


    int isurf;

    // -----------------------------
    // 1. TOP SURFACE
    // -----------------------------
    if (surf[0] > 0)
    {
        isurf = surf[0];
        //std::cout << std::endl << "SURF INDEX: "<< isurf<<std::endl;


        if (ix >= nx1 && ix < nx2)
        {
            // Denise manual  page 13
            szz[isurf * nx + ix] = 0.0;
            szx[isurf * nx + ix] = 0.0;
            sxx[isurf * nx + ix] = 4.0 * dt * vx_x[isurf * nx + ix] * (lam[isurf * nx + ix] * mu[isurf * nx + ix] + mu[isurf * nx + ix] * mu[isurf * nx + ix]) / (lam[isurf * nx + ix] + 2.0 * mu[isurf * nx + ix]);


            if (sz >= 1 && sz < (isurf - nz1 + 1))
            { // mirroring
                szx[(isurf - sz) * nx + ix] = -szx[(isurf + sz) * nx + ix];
                szz[(isurf - sz) * nx + ix] = -szz[(isurf + sz) * nx + ix];
                //std::cout<<"surf: "<< isurf-sz <<", " << isurf+sz <<", ::" ;
            }
        }
        //printf("inside gpu %lf" ,sxx[isurf * nx + ix]);
    }


    // -----------------------------
    // 2. BOTTOM SURFACE
    // -----------------------------
    if (surf[1] > 0)
    {
        isurf = surf[1];


        if (ix >= nx1 && ix < nx2)
        {
            // Denise manual  page 13
            szz[isurf * nx + ix] = 0.0;
            szx[isurf * nx + ix] = 0.0;
            sxx[isurf * nx + ix] = 4.0 * dt * vx_x[isurf * nx + ix] * (lam[isurf * nx + ix] * mu[isurf * nx + ix] + mu[isurf * nx + ix] * mu[isurf * nx + ix]) / (lam[isurf * nx + ix] + 2.0 * mu[isurf * nx + ix]);


            if (sz >= 1 && sz <= nz2 - isurf)
            { // mirroring
                szx[(isurf + sz) * nx + ix] = -szx[(isurf - sz) * nx + ix];
                szz[(isurf + sz) * nx + ix] = -szz[(isurf - sz) * nx + ix];
            }
        }
    }
    // -----------------------------
    // 3. LEFT SURFACE
    // -----------------------------
    if (surf[2] > 0)
    {
        isurf = surf[2];
        if (iz >= nz1 && iz < nz2)
        {


            // Denise manual  page 13
            sxx[iz * nx + isurf] = 0.0;
            szx[iz * nx + isurf] = 0.0;
            szz[iz * nx + isurf] = 4.0 * dt * vz_z[iz * nx + isurf] * (lam[iz * nx + isurf] * mu[iz * nx + isurf] + mu[iz * nx + isurf] * mu[iz * nx + isurf]) / (lam[iz * nx + isurf] + 2.0 * mu[iz * nx + isurf]);


            if (sx >= 1 && sx < isurf - nx1 + 1)
            { // mirroring
                szx[iz * nx + isurf - sx] = -szx[iz * nx + isurf + sx];
                sxx[iz * nx + isurf - sx] = -sxx[iz * nx + isurf + sx];
            }
        }
    }


    // -----------------------------
    // 4. RIGHT SURFACE
    // -----------------------------
    if (surf[3] > 0)
    {
        isurf = surf[3];
        if (iz >= nz1 && iz < nz2)
        {


            // Denise manual  page 13
            sxx[iz * nx + isurf] = 0.0;
            szx[iz * nx + isurf] = 0.0;
            szz[iz * nx + isurf] = 4.0 * dt * vz_z[iz * nx + isurf] * (lam[iz * nx + isurf] * mu[iz * nx + isurf] + mu[iz * nx + isurf] * mu[iz * nx + isurf]) / (lam[iz * nx + isurf] + 2.0 * mu[iz * nx + isurf]);


            if (sx >= 1 && sx <= nx2 - isurf)
            { // mirroring
                szx[iz * nx + isurf + sx] = -szx[iz * nx + isurf - sx];
                sxx[iz * nx + isurf + sx] = -sxx[iz * nx + isurf - sx];
            }
        }
    }

    // printf("Done gpu /n");
}


void surf_mirror_GPU(
    // Wave arguments (stress & velocity derivatives)
    real*& szz, real*& szx, real*& sxx, real*& vz_z, real*& vx_x,
    // Medium arguments
    real*& lam, real*& mu,
    // surface indices for four directions(0.top, 1.bottom, 2.left, 3.right)
    int*& surf,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dt, int nx)
{
    // surface mirroring for free surface


    int box1 = 32, box2 = 32;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid((nx2) / box1 + 1, (nz2) / box2 + 1);
    
    //auto start_GPU = high_resolution_clock::now();
    // printf("Calling kernel/n");
    cuda_surf_mirror_GPU << <blocksPerGrid, threadsPerBlock >> > (szz, szx, sxx, vz_z, vx_x, lam,
        mu, surf, nz1, nz2, nx1, nx2, dt, nx);
    // auto stop_GPU = high_resolution_clock::now();
    // auto duration_GPU = duration_cast<microseconds>(stop_GPU - start_GPU);
    // cout << "Time taken by GPU: "
    //     << duration_GPU.count() << " microseconds" << endl;
    //printf("done with kernel/n");
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaPeekAtLastError());
}




__global__ void cuda_gard_fwd_storage2_GPU( // forward storage for full waveform inversion
    real *accu_vz, real *accu_vx,
    real *accu_szz, real *accu_szx, real *accu_sxx,
    // velocity and stress tensors
    real *vz, real *vx, real *szz, real *szx, real *sxx,
    // time and space parameters
    real dt, int itf, int snap_z1, int snap_z2,
    int snap_x1, int snap_x2, int snap_dz, int snap_dx, int nx)
{

    int snap_nz = 1 + (snap_z2 - snap_z1) / snap_dz;
    int snap_nx = 1 + (snap_x2 - snap_x1) / snap_dx;

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int jx = 0, jz = 0;
    if (iz >= snap_z1 && iz <= snap_z2 && (iz - snap_z1) % snap_dz == 0)
    {
        jz = (iz - snap_z1) / snap_dz;
        if (ix >= snap_x1 && ix <= snap_x2 && (ix - snap_x1) % snap_dx == 0)
        {
            jx = (ix - snap_x1) / snap_dx;
            accu_sxx[itf * snap_nz * snap_nx + jz * snap_nx + jx] = sxx[iz * nx + ix];
            accu_szx[itf * snap_nz * snap_nx + jz * snap_nx + jx] = szx[iz * nx + ix];
            accu_szz[itf * snap_nz * snap_nx + jz * snap_nx + jx] = szz[iz * nx + ix];

            accu_vx[itf * snap_nz * snap_nx + jz * snap_nx + jx] = vx[iz * nx + ix] / dt;
            accu_vz[itf * snap_nz * snap_nx + jz * snap_nx + jx] = vz[iz * nx + ix] / dt;
        }
    }
}

void gard_fwd_storage2_GPU(
    // forward storage for full waveform inversion
    real *&accu_vz, real *&accu_vx,
    real *&accu_szz, real *&accu_szx, real *&accu_sxx,
    // velocity and stress tensors
    real *&vz, real *&vx, real *&szz, real *&szx, real *&sxx,
    // time and space parameters
    real dt, int itf, int snap_z1, int snap_z2,
    int snap_x1, int snap_x2, int snap_dz, int snap_dx, int nx)
{

    // Stores forward velocity and stress for gradiant calculation in fwi
    // dt: the time step size
    // itf: reduced continuous time index after skipping the time steps in between
    // snap_z1, snap_z2, snap_x1, snap_z2: the indices for fwi storage
    // snap_dz, snap_dx: the grid interval for reduced (skipped) storage of tensors

    int box1 = 32, box2 = 32;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid(snap_x2 / box1 + 1, snap_z2 / box2 + 1);
    cuda_gard_fwd_storage2_GPU<<<blocksPerGrid, threadsPerBlock>>>(
        accu_vz, accu_vx,
        accu_szz, accu_szx, accu_sxx,
        vz, vx, szz, szx, sxx,
        dt, itf, snap_z1, snap_z2,
        snap_x1, snap_x2, snap_dz, snap_dx, nx);
    cudaCheckError(cudaPeekAtLastError());
}


__global__ void cuda_fwi_grad2_GPU(
    // Gradient of the materials
    real *grad_lam, real *grad_mu, real *grad_rho,
    // forward storage for full waveform inversion
    real *accu_vz, real *accu_vx,
    real *accu_szz, real *accu_szx, real *accu_sxx,
    // displacement and stress tensors
    real *uz, real *ux, real *szz, real *szx, real *sxx,
    // Medium arguments
    real *lam, real *mu,
    // time and space parameters
    real dt, int tf, int snap_dt, int snap_z1, int snap_z2,
    int snap_x1, int snap_x2, int snap_dz, int snap_dx, int nx)
{
    // Calculates the gradient of medium from stored forward tensors & current tensors
    real s1, s2, s3, s4; // Intermediate variables for gradient calculation
    //real lm;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    
    int jz, jx; // mapping for storage with intervals
    int snap_nz = 1 + (snap_z2 - snap_z1) / snap_dz;
    int snap_nx = 1 + (snap_x2 - snap_x1) / snap_dx;
    jz = 0;
    if (iz >= snap_z1 && iz <= snap_z2 && (iz - snap_z1) % snap_dz == 0)
    {
        jz = (iz - snap_z1) / snap_dz;
        if (ix >= snap_x1 && ix <= snap_x2 && (ix - snap_x1) % snap_dx == 0)
        {
            jx = (ix - snap_x1) / snap_dx;
            s1 = 0.25 * (accu_szz[tf * snap_nz * snap_nx + jz * snap_nx + jx] + accu_sxx[tf * snap_nz * snap_nx + jz * snap_nx + jx]) * (szz[iz * nx + ix] + sxx[iz * nx + ix]) / ((lam[iz * nx + ix] + mu[iz * nx + ix]) * (lam[iz * nx + ix] + mu[iz * nx + ix]));
            s2 = 0.25 * (accu_szz[tf * snap_nz * snap_nx + jz * snap_nx + jx] - accu_sxx[tf * snap_nz * snap_nx + jz * snap_nx + jx]) * (szz[iz * nx + ix] - sxx[iz * nx + ix]) / (mu[iz * nx + ix] * mu[iz * nx + ix]);
            s3 = (accu_szx[tf * snap_nz * snap_nx + jz * snap_nx + jx] * szx[iz * nx + ix]) / (mu[iz * nx + ix] * mu[iz * nx + ix]);
            // The time derivatives of the velocity may have to be computed differently
            s4 = ux[iz * nx + ix] * accu_vx[tf * snap_nz * snap_nx + jz * snap_nx + jx] + uz[iz * nx + ix] * accu_vz[tf * snap_nz * snap_nx + jz * snap_nx + jx];
            grad_lam[jz * snap_nx + jx] += snap_dt * dt * s1;
            grad_mu[jz * snap_nx + jx] += snap_dt * dt * (s3 + s1 + s2);
            grad_rho[jz * snap_nx + jx] += snap_dt * dt * s4;

           // printf(" tf=%d jx=%d Jz=%d s1=%lf accu_szz=%lf grad=%lf \n", tf, jx,jz,s1,accu_szz[tf * snap_nz * snap_nx + jz * snap_nx + jx], grad_lam[jz * nx + jx] );
        }
    }
}
///////////////////kernel//////////////////
void fwi_grad2_GPU(
    // Gradient of the materials
    real *&grad_lam, real *&grad_mu, real *&grad_rho,
    // forward storage for full waveform inversion
    real *&accu_vz, real *&accu_vx,
    real *&accu_szz, real *&accu_szx, real *&accu_sxx,
    // displacement and stress tensors
    real *&uz, real *&ux, real *&szz, real *&szx, real *&sxx,
    // Medium arguments
    real *&lam, real *&mu,
    // time and space parameters
    real dt, int tf, int snap_dt, int snap_z1, int snap_z2,
    int snap_x1, int snap_x2, int snap_dz, int snap_dx, int nx)
{
    // Calculates the gradient of medium from stored forward tensors & current tensors
    int box1 = 32, box2 = 32;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid(snap_x2 / box1 + 1, snap_z2 / box2 + 1);
    cuda_fwi_grad2_GPU<<<blocksPerGrid, threadsPerBlock>>>(
        grad_lam, grad_mu, grad_rho,
        accu_vz, accu_vx,
        accu_szz, accu_szx, accu_sxx,
        uz, ux, szz, szx, sxx,
        lam, mu,
        dt, tf, snap_dt, snap_z1, snap_z2,
        snap_x1, snap_x2, snap_dz, snap_dx, nx);

     cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaPeekAtLastError());
}

__global__ void cuda_vsrc2_GPU(
    // Velocity tensor arrays
    real* vz, real* vx,
    // inverse of density arrays
    real* rho_zp, real* rho_xp,
    // source parameters
    int nsrc, int stf_type, real* stf_z, real* stf_x,
    int* z_src, int* x_src, int* src_shot_to_fire,
    int ishot, int it, real dt, real dz, real dx, int nx, int nt)
{
    // firing the velocity source term
    // nsrc: number of sources
    // stf_type: type of source time function (0:displacement, 1:velocity currently implemented)
    // stf_z: source time function z component
    // stf_x: source time function x component
    // z_src: corresponding grid index along z direction
    // x_src: corresponding grid index along x direction
    // it: time step index

   // std::cout << "src: " << stf_type <<std::endl;
    int is = blockIdx.x * blockDim.x + threadIdx.x;
    switch (stf_type)
    {

    case (0): // Displacement stf
        if (is >= 0 && is < nsrc)
        {
            if (src_shot_to_fire[is] == ishot)
            {
               vz[z_src[is] * nx + x_src[is]] += dt * rho_zp[z_src[is] * nx + x_src[is]] * stf_z[is * nt + it] / (dz * dx);
               vx[z_src[is] * nx + x_src[is]] += dt * rho_xp[z_src[is] * nx + x_src[is]] * stf_x[is * nt + it] / (dz * dx);

              // printf(">>>>>>>>>>>is=%d  vz=%lf \n", is , vz[z_src[is]*nx+x_src[is]]);
            }
        }
        break;

    case (1): // velocity stf
        if (is >= 0 && is < nsrc)
        {
            if (src_shot_to_fire[is] == ishot)
            {
               // printf("it=%d is=%d vz=%lf z_src=%d x_src=%d \n",it,  is, vz[z_src[is] * nx + x_src[is]],z_src[is], x_src[is] );
                vz[z_src[is] * nx + x_src[is]] += stf_z[is * nt + it];
                vx[z_src[is] * nx + x_src[is]] += stf_x[is * nt + it];
                //std::cout << "v:" << vz[z_src[is]*nx+x_src[is]] <<", " << stf_z[is*nx+it]<<std::endl;
               // printf("after it=%d is=%d  vz=%lf stf_z=%lf\n",it,  is , vz[z_src[is]*nx+x_src[is]], stf_z[is * nx + it]);
            }
        }
        break;
    }
}

void vsrc2_GPU(
    // Velocity tensor arrays
    real*& vz, real*& vx,
    // inverse of density arrays
    real*& rho_zp, real*& rho_xp,
    // source parameters
    int nsrc, int stf_type, real*& stf_z, real*& stf_x,
    int*& z_src, int*& x_src, int*& src_shot_to_fire,
    int ishot, int it, real dt, real dz, real dx, int nx, int nt)
{
  // std::cout<<"  vsrc2_GPU  \n";
    int box1 = 32;
    dim3 threadsPerBlock(box1);
    dim3 blocksPerGrid(nsrc / box1 + 1);
    cuda_vsrc2_GPU << <blocksPerGrid, threadsPerBlock >> > (vz, vx, rho_zp, rho_xp, nsrc, stf_type, stf_z, stf_x,
        z_src, x_src, src_shot_to_fire, ishot, it, dt, dz, dx, nx, nt);
    cudaCheckError(cudaDeviceSynchronize());
}



__global__ void cuda_urec2_GPU(int rtf_type,
    // reciever time functions
    real* rtf_uz, real* rtf_ux,
    // velocity tensors
    real* vz, real* vx,
    // reciever
    int nrec, int* rz, int* rx,
    // time and space grids
    int it, real dt, real dz, real dx, int nt, int nx)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (rtf_type == 0)
    {
        // This module is only for rtf type as displacement
        if (ir >= 0 && ir < nrec)
        {
            if (it == 0)
            {
               rtf_uz[ir * nt + it] = dt * vz[rz[ir] * nx + rx[ir]] / (dz * dx);
                rtf_ux[ir * nt + it] = dt * vx[rz[ir] * nx + rx[ir]] / (dz * dx);
            }
            else
            {
                rtf_uz[ir * nt + it] = rtf_uz[ir * nt + it - 1] + dt * vz[rz[ir] * nx + rx[ir]] / (dz * dx);
                rtf_ux[ir * nt + it] = rtf_ux[ir * nt + it - 1] + dt * vx[rz[ir] * nx + rx[ir]] / (dz * dx);
            }
        }
    }

    //  if (rtf_type == 0) {
    //     // This module is only for rtf type as displacement
    //     for (int ir = 0; ir < nrec; ir++) {
    //         if (it == 0) {
    //             rtf_uz[ir][it] = dt * vz[rz[ir]][rx[ir]] / (dz * dx);
    //             rtf_ux[ir][it] = dt * vx[rz[ir]][rx[ir]] / (dz * dx);
    //         }
    //         else {
    //             rtf_uz[ir][it] = rtf_uz[ir][it - 1] + dt * vz[rz[ir]][rx[ir]] / (dz * dx);
    //             rtf_ux[ir][it] = rtf_ux[ir][it - 1] + dt * vx[rz[ir]][rx[ir]] / (dz * dx);
    //         }
    //     }

    // }
}

void urec2_GPU(int rtf_type,
    // reciever time functions
    real*& rtf_uz, real*& rtf_ux,
    // velocity tensors
    real*& vz, real*& vx,
    // reciever
    int nrec, int*& rz, int*& rx,
    // time and space grids
    int it, real dt, real dz, real dx, int nt, int nx)
{
    // recording the output seismograms
    // nrec: number of recievers
    // rtf_uz: reciever time function (displacement_z)
    // rtf_uz: reciever time function (displacement_x)
    // rec_signal: signal file for seismogram index and time index
    // rz: corresponding grid index along z direction
    // rx: corresponding grid index along x direction
    // it: time step index
    

     
     
    int box1 = 32;
    dim3 threadsPerBlock(box1);
    dim3 blocksPerGrid(nrec / box1 + 1);
   
    cuda_urec2_GPU << <blocksPerGrid, threadsPerBlock >> > (rtf_type,
        // reciever time functions
        rtf_uz, rtf_ux,
        // velocity tensors
        vz, vx,
        // reciever
        nrec, rz, rx,
        // time and space grids
        it, dt, dz, dx, nt, nx);
        
 
    cudaCheckError(cudaDeviceSynchronize());
}


////////////////////////// GPU //////////////////

class power_functor {

    double a;
    double dt;

    public:

        power_functor(real a_,real dt_) { a = a_; dt=dt_;}

        __host__ __device__ real operator()(real x) const 
        {
            return 0.5*dt*(x*x);
        }
};

__global__ void cuda_adjsrc2_GPU(int ishot, real *a_stf_uz, real *a_stf_ux, 
            int rtf_type, real *rtf_uz_true, real *rtf_ux_true, 
            real *rtf_uz_mod, real *rtf_ux_mod,             
            real dt, int nseis, int nt)
{

     int is = blockIdx.x * blockDim.x + threadIdx.x;
     int it = blockIdx.y * blockDim.y + threadIdx.y;
    if(is>=0 && is<nseis && it>=0 && it<nt)
    {
             a_stf_uz[is*nt+it] = rtf_uz_mod[is*nt+it] - rtf_uz_true[ishot*nt*nseis+is*nt+it];
             a_stf_ux[is*nt+it] = rtf_ux_mod[is*nt+it] - rtf_ux_true[ishot*nt*nseis+ is*nt + it];

    }

}    

real adjsrc2_GPU(int ishot, int *&a_stf_type, real *&a_stf_uz, real *&a_stf_ux, 
            int rtf_type, real *&rtf_uz_true, real *&rtf_ux_true, 
            real *&rtf_uz_mod, real *&rtf_ux_mod,             
            real dt, int nseis, int nt){
    // Calculates adjoint sources and L2 norm
    // a_stf: adjoint sources
    // rtf: reciever time function (mod: forward model, true: field measured)

    real L2;
    L2 = 0;
    
    int box1 = 32, box2 = 32;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid((nseis) / box1 + 1, (nt) / box2 + 1);

if (rtf_type == 0)
{

    cuda_adjsrc2_GPU<<<blocksPerGrid, threadsPerBlock>>>(ishot, a_stf_uz, a_stf_ux, 
           rtf_type, rtf_uz_true, rtf_ux_true,rtf_uz_mod, rtf_ux_mod,dt, nseis,  nt);

    cudaCheckError(cudaDeviceSynchronize());
    
    thrust::device_ptr<real> dev_ptr1 = thrust::device_pointer_cast(a_stf_uz);

    thrust::device_ptr<real> dev_ptr2 = thrust::device_pointer_cast(a_stf_ux);
    L2 = thrust::transform_reduce(thrust::device,dev_ptr1,dev_ptr1+nseis*nt,power_functor(2.,dt),0.0,thrust::plus<real>());
   
    //std::cout<<L2<<"  <<1 a_stf_uz sum \n";

    L2 += thrust::transform_reduce(thrust::device,dev_ptr2,dev_ptr2+nseis*nt,power_functor(2.,dt),0.0,thrust::plus<real>());
    // L2=     thrust::reduce(dev_ptr1 , dev_ptr1 + nx*nz, 0.0, thrust::plus<real>());
    // std::cout<<L2<<"  <<1 a_stf_ux sum \n";
        
}
 

        a_stf_type = &rtf_type; // Calculating displacement adjoint sources
    

    std::cout<< "Calculated norm: " << L2 << std::endl;
    //std::cout << a_stf_type << std::endl;
    return L2;

}
//////////////////////////////////////// GPU  //////////////


//cpu code ends
__global__ void cuda_interpol_grad2_GPU(
    // Global and shot gradient
    real *grad, real *grad_shot, 
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, 
    int snap_dz, int snap_dx,int nx){


    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    
    int jz, jx; // mapping for storage with intervals
    int snap_nz = 1 + (snap_z2 - snap_z1) / snap_dz;//no of time loopoccurs
    int snap_nx = 1 + (snap_x2 - snap_x1) / snap_dx;//no of time loop occurs
    real temp_grad;
    
  
    // --------------------------------------
    // FOR LOOP SET 1
    // -----------------------------------
    jz = 0;
    
    if (iz >= snap_z1 && iz <= snap_z2 && (iz - snap_z1) % snap_dz == 0)// to ensure multiples 
    {
        jz = (iz - snap_z1) / snap_dz;
        if (ix >= snap_x1 && ix <= snap_x2 && (ix - snap_x1) % snap_dx == 0){
            
            jx = (ix - snap_x1) / snap_dx;

            grad[iz * nx + ix]=grad_shot[jz * snap_nx+jx];
// printf("grad[%d]= %lu = grad_shot[%d] = %lu \n ",iz * nx + ix, grad[iz * nx + ix],jz*nx+jx,grad_shot[jz*nx+jx]);
            
        }
      // __syncthreads();
    }
  
  

    //gpu snippet
    if(snap_dx>1){  
        // now updating the snap rows only
        if(iz>=snap_z1&&iz<snap_z2&&(iz - snap_z1) % snap_dz == 0)
        {
            if(ix >= snap_x1 && ix < snap_x2 && (ix - snap_x1) % snap_dx == 0){
                temp_grad = (grad[iz*nx+(ix+snap_dx)] - grad[iz*nx+ix])/snap_dx;
                for(int kx=1;kx<snap_dx;kx++){
                    grad[iz*nx+(ix+kx)] = grad[iz*nx+ix] + temp_grad*kx;
                }
            }
        }
    }
    

 
    if(snap_dz>1){
        
        if(iz>=snap_z1&&iz<snap_z2&&(iz - snap_z1) % snap_dz == 0){
            if(ix >= snap_x1 && ix < snap_x2 ){

                temp_grad = (grad[(iz+snap_dz)*nx+ix] - grad[iz*nx+ix])/snap_dz;
                 for(int kz=1;kz<snap_dz;kz++){

                    grad[(iz+kz)*nx+ix] = grad[iz*nx+ix] + temp_grad*kz;
                }
            

            }

        } 

    }
  //  __syncthreads();
    
    
    
  }

void interpol_grad2_GPU( // Global and shot gradient
    real *grad, real *grad_shot, 
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, 
    int snap_dz, int snap_dx, int nx)
{
    // Calculates the gradient of medium from stored forward tensors & current tensors

    int box1 = 32, box2 = 32;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid(snap_x2 / box1 + 1, snap_z2 / box2 + 1);

    cuda_interpol_grad2_GPU<<<blocksPerGrid, threadsPerBlock>>>(
         grad, grad_shot, 
    // space snap parameters
    snap_z1, snap_z2, snap_x1,  snap_x2, 
     snap_dz,  snap_dx, nx);

    
    cudaCheckError(cudaDeviceSynchronize());
   cudaCheckError(cudaPeekAtLastError());
}




__global__ void cuda_energy_weights2_GPU(
    // Energy Weights (forward and reverse)
    real *We, real *We_adj, 
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, int nx)
    {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if( iz>=snap_z1 && iz<snap_z2 && ix>=snap_x1 && ix<snap_x2) 
    {
        We[iz*nx+ix] = sqrt(We[iz*nx + ix]*We_adj[iz*nx+ix]);
    }

    }



__global__ void cuda_energy_weights2_GPU2(
    // Energy Weights (forward and reverse)
    real *We, 
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, int nx, real epsilon_We, real max_We )
    {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if( iz>=snap_z1 && iz<snap_z2 && ix>=snap_x1 && ix<snap_x2) 
    {
       We[iz*nx+ix] += epsilon_We *  max_We;
    }

}

void energy_weights2_GPU(
    // Energy Weights (forward and reverse)
    real *&We, real *&We_adj, 
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, int nx){
    // Scale gradients to the Energy Weight
    // We: input as forward energy weight, and output as combined energy weight

    real max_We = 0;
    real max_w1 = 0, max_w2=0;
    real epsilon_We = 0.005; 
    
    int box1 = 32, box2 = 32;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid(snap_x2 / box1 + 1, snap_z2 / box2 + 1);
        cuda_energy_weights2_GPU<<<blocksPerGrid, threadsPerBlock>>>
        ( We,We_adj, snap_z1, snap_z2, snap_x1,snap_x2,  nx);

  cudaCheckError(cudaDeviceSynchronize());
    thrust::device_ptr<real> dev_ptr1 = thrust::device_pointer_cast(We);
   // thrust::device_ptr<real> dev_ptr2 = thrust::device_pointer_cast(We_adj);
    

    for (int iz = snap_z1; iz < snap_z2; iz++)
    {
        real ma = thrust::reduce(dev_ptr1 +iz*nx+snap_x1 , dev_ptr1 + iz * nx + snap_x2, 0.0, thrust::maximum<real>());
         if(ma>max_We)
         max_We=ma;
       
    }
       cuda_energy_weights2_GPU2<<<blocksPerGrid, threadsPerBlock>>>
       ( We, snap_z1, snap_z2, snap_x1,snap_x2,  nx, epsilon_We, max_We );
    cudaCheckError(cudaDeviceSynchronize());

    std::cout << "Max. Energy Weight = " << max_We << std::endl;
    //std::cout << "Max. Energy part = " << max_w1<<", "<< max_w2 << std::endl;
}


class power_functor2
{

    double a;
    double dt;

public:
    power_functor2() {}

    __host__ __device__ double operator()(double x) const
    {
        return abs(x);
    }
};
__global__ void cuda_update_mat2_GPU(real *mat, real *mat_old, real *grad_mat,
                                     real mat_max, real mat_min, real step_length, real step_factor, int nz, int nx)
{

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if(iz>=0 && iz<nz && ix<nx && ix>=0)
    {
            mat[iz*nx+ix] = mat_old[iz*nx+ix] + step_length * step_factor * grad_mat[iz*nx+ix];
            if (mat[iz*nx+ix] > mat_max){ mat[iz*nx+ix] = mat_max;}
            if (mat[iz*nx+ix] < mat_min){ mat[iz*nx+ix] = mat_min;}

    }

}

void update_mat2_GPU(real *&mat, real *&mat_old, real *&grad_mat,//rho,rho_copy,grad_rho
                     real mat_max, real mat_min, real step_length, int nz, int nx)
{
    // update gradients to the material
    real mat_av = 0, mat_av_old = 0, mat_av_grad = 0;

    // Scale factors for gradients
    real grad_max = 0.0, mat_array_max = 0.0, step_factor;

    thrust::device_ptr<real> dev_ptr1 = thrust::device_pointer_cast(grad_mat);
    thrust::device_ptr<real> dev_ptr2 = thrust::device_pointer_cast(mat_old);

    grad_max = thrust::transform_reduce(thrust::device, dev_ptr1, dev_ptr1 + nx * nz, power_functor2(), 0.0, thrust::maximum<real>());

    mat_array_max = thrust::transform_reduce(thrust::device, dev_ptr2, dev_ptr2 + nx * nz, power_functor2(), 0.0, thrust::maximum<real>());

    if (mat_array_max < mat_max)
        mat_array_max = mat_max;


    // for (int iz=0;iz<nz;iz++){
    //     for (int ix=0;ix<nx;ix++){

    //         grad_max = std::max(grad_max, abs(grad_mat[iz][ix]));
    //         mat_array_max = std::max(mat_max, abs(mat_old[iz][ix]));

    //     }
    // }
    step_factor = mat_array_max / grad_max;
    std::cout << "GPU Update factor: " << step_factor << ", " << mat_max << ", " << grad_max << ", " << mat_array_max << std::endl;//grad max error

    int box1 = 32, box2 = 32;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid(nx / box1 + 1, nz / box2 + 1);
cuda_update_mat2_GPU<<<blocksPerGrid, threadsPerBlock>>>(mat, mat_old, grad_mat,
                                      mat_max,  mat_min,  step_length,  step_factor,  nz,  nx);
     cudaCheckError(cudaDeviceSynchronize());

}


__global__ void cuda_taper2_GPU(real* A, int nz, int nx,
    int snap_z1, int snap_z2, int snap_x1, int snap_x2,
    int taper_t1, int taper_t2, int taper_b1, int taper_b2,
    int taper_l1, int taper_l2, int taper_r1, int taper_r2)
{
    double PIE = 3.14159265;

    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = blockIdx.y * blockDim.y + threadIdx.y;

    int taper_l = taper_l2 - taper_l1;
    int taper_r = taper_r1 - taper_r2;
    int taper_t = taper_t2 - taper_t1;
    int taper_b = taper_b1 - taper_b2;

    if (ix >= 0 && ix < nx && iz >= 0 && iz < nz)// for loop decomposition
    {
        if (ix >= snap_x1 && ix < taper_l1) {
            A[iz * nx + ix] *= 0.0;
        }

        else if (ix >= taper_l1 && ix < taper_l2) {
            A[iz * nx + ix] *= 0.5 * (1.0 - cos(PIE * (ix - taper_l1) / taper_l));
        }

        else if (ix > taper_r2 && ix < taper_r1) {
            A[iz * nx + ix] *= 0.5 * (1.0 - cos(PIE * (taper_r1 - ix) / taper_r));
        }

        else if (ix >= taper_r1 && ix <= snap_x2) {
            A[iz * nx + ix] *= 0.0;
        }
    }

    if (ix >= 0 && ix < nx && iz >= 0 && iz < nz)
    {
        if (iz >= snap_z1 && iz < taper_t1) {
            A[iz * nx + ix] *= 0.0;
        }

        else if (iz >= taper_t1 && iz < taper_t2) {
            A[iz * nx + ix] *= 0.5 * (1.0 - cos(PIE * (iz - taper_t1) / taper_t));
        }

        else if (iz > taper_b2 && iz < taper_b1) {
            A[iz * nx + ix] *= 0.5 * (1.0 - cos(PIE * (taper_b1 - iz) / taper_b));
        }

        else if (iz >= taper_b1 && iz <= snap_z2) {
            A[iz * nx + ix] *= 0.0;
        }
    }

    //printf("GPU i=%d j=%d ans=%lf %lf %lf \n", iz, ix, lam[iz * nx + ix], mu[iz * nx + ix], rho[iz * nx + ix]);

}



void taper2_GPU(
    // Gradients, material average and energy weights
    real* A, int nz, int nx,
    int snap_z1, int snap_z2, int snap_x1, int snap_x2,
    int& taper_t1, int& taper_t2, int& taper_b1, int& taper_b2,
    int& taper_l1, int& taper_l2, int& taper_r1, int& taper_r2)
{

    //kernel configration
    int box1 = 32, box2 = 32;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid((nz) / box1 + 1, (nx) / box2 + 1);
   // auto start_GPU = high_resolution_clock::now();
     
    cuda_taper2_GPU << <blocksPerGrid, threadsPerBlock >> > (A, nz, nx,
        snap_z1, snap_z2, snap_x1, snap_x2,
        taper_t1, taper_t2, taper_b1, taper_b2,
        taper_l1, taper_l2, taper_r1, taper_r2);
    //auto stop_GPU = high_resolution_clock::now();
    //auto duration_GPU = duration_cast<microseconds>(stop_GPU - start_GPU);
    // cout << "Time taken by GPU: "
    //     << duration_GPU.count() << " microseconds" << endl;

    cudaCheckError(cudaDeviceSynchronize());
}