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

__global__ void cuda_mat_av2_GPU(real *lam, real *mu, real *rho,
                                 real *mu_zx, real *rho_zp, real *rho_xp, // inverse of densityint dimz, int dimx
                                 int nz, int nx)
{

    long int iz = blockIdx.x * blockDim.x + threadIdx.x;
    long int ix = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nx - 1 && ix >= 0 && iz >= 0 && iz < nz - 1)
    {
        // Harmonic average for mu
        mu[iz * nx + ix] = 4.0 / ((1.0 / mu[iz * nx + ix]) + (1.0 / mu[iz * nx + ix + 1]) + (1.0 / mu[(iz + 1) * nx + ix]) + (1.0 / mu[(iz + 1) * nx + ix + 1]));

        if ((mu[iz * nx + ix] == 0.0) || (mu[iz * nx + ix + 1] == 0.0) || (mu[(iz + 1) * nx + ix] == 0.0) || (mu[(iz + 1) * nx + ix + 1] == 0.0))
        {
            mu_zx[iz * nx + ix] = 0.0;
        }

        // Arithmatic average of rho
        // the averages are inversed for computational efficiency
        rho_zp[iz * nx + ix] = 1.0 / (0.5 * (rho[iz * nx + ix] + rho[(iz + 1) * nx + ix]));
        rho_xp[iz * nx + ix] = 1.0 / (0.5 * (rho[iz * nx + ix] + rho[iz * nx + ix + 1]));

        if ((rho[iz * nx + ix] < 1e-4) && (rho[iz + 1 * nx + ix] < 1e-4))
        {
            rho_zp[iz * nx + ix] = 0.0;
        }

        if ((rho[iz * nx + ix] < 1e-4) && (rho[iz * nx + ix + 1] < 1e-4))
        {
            rho_zp[iz * nx + ix] = 0.0;
        }
    }
    else
    {
        return;
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

    thrust::device_ptr<real> dev_ptr1 = thrust::device_pointer_cast(lam);
    thrust::device_ptr<real> dev_ptr2 = thrust::device_pointer_cast(mu);
    thrust::device_ptr<real> dev_ptr3 = thrust::device_pointer_cast(rho);

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
    std::cout << "This is test GPU \nC_lam=" << C_lam << " \nC_mu=" << C_mu << " \nC_rho=" << C_rho << " \n\n";
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
                          int snap_dz, int snap_dx, int nx)
{

    int jz=0;

    int k = ceil((snap_x2 - snap_x1) / snap_dz) + 1;

    const size_t size = k * sizeof(real);

    for (int iz = snap_z1; iz <= snap_z2; iz += snap_dz)
    {
        cudaCheckError(cudaMemset(grad_lam + jz * nx, 0, size));
        cudaCheckError(cudaMemset(grad_mu + jz * nx, 0, size));
        cudaCheckError(cudaMemset(grad_rho + jz * nx, 0, size));
        jz++;
    }
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

    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz >= nz1 && iz < nz2 && ix < nx2 && ix >= nx1)
    {
        vz_z[iz * nx + ix] = dzi * hc[1] * (vz[iz * nx + ix] - vz[(iz - 1) * nx + ix]);
        vx_z[iz * nx + ix] = dzi * hc[1] * (vx[(iz + 1) * nx + ix] - vx[iz * nx + ix]);
        vz_x[iz * nx + ix] = dxi * hc[1] * (vz[iz * nx + ix + 1] - vz[iz * nx + ix]);
        vx_x[iz * nx + ix] = dxi * hc[1] * (vx[iz * nx + ix] - vx[iz * nx + ix - 1]);
    }
    else
    {
        return;
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
    real *szz, real *szx, real *sxx, real *vz_z, real *vx_x,
    // Medium arguments
    real *lam, real *mu,
    // surface indices for four directions(0.top, 1.bottom, 2.left, 3.right)
    int *&surf,
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
}

void surf_mirror_GPU(
    // Wave arguments (stress & velocity derivatives)
    real *&szz, real *&szx, real *&sxx, real *&vz_z, real *&vx_x,
    // Medium arguments
    real *&lam, real *&mu,
    // surface indices for four directions(0.top, 1.bottom, 2.left, 3.right)
    int *&surf,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dt, int nx)
{
    // surface mirroring for free surface

    int box1 = 32, box2 = 32;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid((nx2) / box1 + 1, (nz2) / box2 + 1);

    cuda_surf_mirror_GPU<<<blocksPerGrid, threadsPerBlock>>>(szz, szx, sxx, vz_z, vx_x, lam,
                                                             mu, surf, nz1, nz2, nx1, nx2, dt, nx);
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

            grad_lam[jz * nx + jx] += snap_dt * dt * s1;
            grad_mu[jz * nx + jx] += snap_dt * dt * (s3 + s1 + s2);
            grad_rho[jz * nx + jx] -= snap_dt * dt * s4;
        }
    }
}
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

    cudaCheckError(cudaPeekAtLastError());
}
__global__ void cuda_vsrc2_GPU(
    // Velocity tensor arrays
    real *vz, real *vx,
    // inverse of density arrays
    real *rho_zp, real *rho_xp,
    // source parameters
    int nsrc, int stf_type, real *stf_z, real *stf_x,
    int *z_src, int *x_src, int *src_shot_to_fire,
    int ishot, int it, real dt, real dz, real dx, int nx)
{
    // firing the velocity source term
    // nsrc: number of sources
    // stf_type: type of source time function (0:displacement, 1:velocity currently implemented)
    // stf_z: source time function z component
    // stf_x: source time function x component
    // z_src: corresponding grid index along z direction
    // x_src: corresponding grid index along x direction
    // it: time step index

    //std::cout << "src: " << stf_type <<std::endl;
    int is = blockIdx.x * blockDim.x + threadIdx.x;
    switch (stf_type)
    {

    case (0): // Displacement stf
        if (is >= 0 && is < nsrc)
        {
            if (src_shot_to_fire[is] == ishot)
            {
                vz[z_src[is] * nx + x_src[is]] += dt * rho_zp[z_src[is] * nx + x_src[is]] * stf_z[is * nx + it] / (dz * dx);
                vx[z_src[is] * nx + x_src[is]] += dt * rho_xp[z_src[is] * nx + x_src[is]] * stf_x[is * nx + it] / (dz * dx);
            }
        }
        break;

    case (1): // velocity stf
        if (is >= 0 && is < nsrc)
        {
            if (src_shot_to_fire[is] == ishot)
            {
                vz[z_src[is] * nx + x_src[is]] += stf_z[is * nx + it];
                vx[z_src[is] * nx + x_src[is]] += stf_x[is * nx + it];
                //std::cout << "v:" << vz[z_src[is]*nx+x_src[is]] <<", " << stf_z[is*nx+it]<<std::endl;
            }
        }
        break;
    }
}

void vsrc2_GPU(
    // Velocity tensor arrays
    real *&vz, real *&vx,
    // inverse of density arrays
    real *&rho_zp, real *&rho_xp,
    // source parameters
    int nsrc, int stf_type, real *&stf_z, real *&stf_x,
    int *&z_src, int *&x_src, int *&src_shot_to_fire,
    int ishot, int it, real dt, real dz, real dx, int nx)
{

    int box1 = 32;
    dim3 threadsPerBlock(box1);
    dim3 blocksPerGrid(nsrc / box1 + 1);
    cuda_vsrc2_GPU<<<blocksPerGrid, threadsPerBlock>>>(vz, vx, rho_zp, rho_xp, nsrc, stf_type, stf_z, stf_x,
                                                       z_src, x_src, src_shot_to_fire, ishot, it, dt, dz, dx, nx);
    cudaCheckError(cudaPeekAtLastError());
}

__global__ void cuda_urec2_GPU(int rtf_type,
                               // reciever time functions
                               real *rtf_uz, real *rtf_ux,
                               // velocity tensors
                               real *vz, real *vx,
                               // reciever
                               int nrec, int *rz, int *rx,
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
}

void urec2_GPU(int rtf_type,
               // reciever time functions
               real *&rtf_uz, real *&rtf_ux,
               // velocity tensors
               real *&vz, real *&vx,
               // reciever
               int nrec, int *&rz, int *&rx,
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
    cuda_urec2_GPU<<<blocksPerGrid, threadsPerBlock>>>(rtf_type,
                                                       // reciever time functions
                                                       rtf_uz, rtf_ux,
                                                       // velocity tensors
                                                       vz, vx,
                                                       // reciever
                                                       nrec, rz, rx,
                                                       // time and space grids
                                                       it, dt, dz, dx, nt, nx);
    rtf_type = 0; // Displacement rtf computed
}
