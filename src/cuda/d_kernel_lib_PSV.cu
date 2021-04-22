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


__global__ void cuda_mat_av2_GPU  (real *lam, real *mu, real *rho,
    real *mu_zx, real *rho_zp, real *rho_xp, // inverse of densityint dimz, int dimx 
    int nz, int nx){

       long int iz = blockIdx.x * blockDim.x + threadIdx.x;
       long int ix = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix < nx-1 && ix >= 0 && iz >= 0 && iz < nz-1){
                // Harmonic average for mu
                mu[iz*nx+ix]=4.0/((1.0/mu[iz*nx+ix])+(1.0/mu[iz*nx+ix+1])
                               +(1.0/mu[(iz+1)*nx+ix])+(1.0/mu[(iz+1)*nx+ix+1])); 

                if((mu[iz*nx+ix]==0.0)||(mu[iz*nx+ix+1]==0.0)||(mu[(iz+1)*nx+ix]==0.0)||(mu[(iz+1)*nx+ix+1]==0.0)){ 
                mu_zx[iz*nx+ix]=0.0;
                }

               // Arithmatic average of rho
               // the averages are inversed for computational efficiency
                rho_zp[iz*nx + ix] = 1.0/(0.5*(rho[iz*nx + ix]+rho[(iz+1)*nx + ix]));
                rho_xp[iz*nx + ix] = 1.0/(0.5*(rho[iz*nx + ix]+rho[iz*nx + ix+1]));


                if((rho[iz*nx+ix]<1e-4)&&(rho[iz+1*nx+ix]<1e-4)){
                rho_zp[iz*nx+ix] = 0.0;
                }

                if((rho[iz*nx+ix]<1e-4)&&(rho[iz*nx+ix+1]<1e-4)){
                rho_zp[iz*nx+ix] = 0.0;
                } 

        }else{
            return;
        }
    }


void mat_av2_GPU(
    // Medium arguments
    real *&lam, real *&mu, real *&rho,
    real *&mu_zx, real *&rho_zp, real *&rho_xp, // inverse of densityint dimz, int dimx
    real &C_lam, real &C_mu, real &C_rho, // scalar averages
    int nz, int nx){
       
      // Harmonic 2d average of mu and
      // Arithmatic 1d average of rho


            C_lam = 0.0; C_mu = 0.0; C_rho = 0.0;
             //Cuda config////
            int box1 = 32, box2 = 32;
            dim3 threadsPerBlock(box1, box2);
            dim3 blocksPerGrid((nz)/box1+1, (nx)/box2+1);
  
          cuda_mat_av2_GPU << < blocksPerGrid, threadsPerBlock >> > (lam, mu, rho,
          mu_zx, rho_zp, rho_xp, // inverse of densityint dimz, int dimx
          nz, nx);
            cudaCheckError(cudaDeviceSynchronize());

        thrust::device_ptr<real> dev_ptr1 = thrust::device_pointer_cast(lam);
        thrust::device_ptr<real> dev_ptr2 = thrust::device_pointer_cast(mu);
        thrust::device_ptr<real> dev_ptr3 = thrust::device_pointer_cast(rho);

        for (int iz=0; iz<nz-1; iz++){
         C_lam += thrust::reduce(dev_ptr1+iz*nx, dev_ptr1+iz*nx+(nx-1), 0.0, thrust::plus<real>());
         C_mu += thrust::reduce(dev_ptr2+iz*nx, dev_ptr2+iz*nx+(nx-1), 0.0, thrust::plus<real>());
         C_rho += thrust::reduce(dev_ptr3+iz*nx, dev_ptr3+iz*nx+(nx-1), 0.0, thrust::plus<real>());
           
        }

        C_lam = C_lam/((nz-1)*(nx-1));
        C_mu = C_mu/((nz-1)*(nx-1));
        C_rho = C_rho/((nz-1)*(nx-1));
        
        //TEST
        std::cout<<"This is test GPU \nC_lam="<<C_lam<<" \nC_mu="<<C_mu<<" \nC_rho="<<C_rho<<" \n\n";
    }


