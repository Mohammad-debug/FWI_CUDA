#include <cuda_runtime_api.h>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
//#define nz 412 //
//#define nx 402 //

using namespace std;
using real = double;

void allocate_array_2d(real **&preal, const int dim1, const int dim2)
{
    // Contiguous allocation of 2D arrays

    preal = new real *[dim1];
    preal[0] = new real[dim1 * dim2];
    for (int i = 1; i < dim1; i++)
        preal[i] = preal[i - 1] + dim2;

    for (int i = 0; i < dim1; i++)
    {
        for (int j = 0; j < dim2; j++)
        {
            preal[i][j] = 0;
        }
    }
}
#define cudaCheckError(code)                                                   \
    {                                                                          \
        if ((code) != cudaSuccess)                                             \
        {                                                                      \
            fprintf(stderr, "Cuda failure %s:%d: '%s' \n", __FILE__, __LINE__, \
                    cudaGetErrorString(code));                                 \
        }                                                                      \
    }
////////////////////////////////////////

__global__ void cuda_scale_grad_E2_GPU(real *grad, real *grad_shot, //remove the "&" in device func calls
                                       real mat_av, real *We,
                                       // space snap parameters
                                       int snap_z1, int snap_z2, int snap_x1, int snap_x2, int nz, int nx)
{

    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = blockIdx.y * blockDim.y + threadIdx.y;

    if (mat_av > 0 && iz >= snap_z1 && iz < snap_z2 && ix >= snap_x1 && ix < snap_x2)
    {

        grad[iz * nx + ix] += grad_shot[iz * nx + ix] / (We[iz * nx + ix] * mat_av * mat_av);

        printf("GPU i=%d j=%d ans=%lf %lf %lf \n",iz,ix, grad[iz*nx+ ix], grad_shot[iz * nx + ix], We[iz * nx + ix] );
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
    dim3 blocksPerGrid((nz) / box1 + 1, (nx) / box2 + 1);//nz=first loop, 

    cuda_scale_grad_E2_GPU<<<blocksPerGrid, threadsPerBlock>>>(grad, grad_shot, mat_av, We,
                                                               snap_z1, snap_z2, snap_x1, snap_x2, nz, nx);

    cudaCheckError(cudaDeviceSynchronize());
}
/////////////////


void scale_grad_E2(
    // Gradients, material average and energy weights
    real **&grad, real **&grad_shot,
    real mat_av, real **&We,
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2)
{
    // Scale gradients to the Energy Weight
    // We: input as forward energy weight, and output as combined energy weight
    // grad and grad shot here have same dimensions (grad_shot = temp grad from each shot)
    // Scale gradients to the energy weight

    if (mat_av > 0)
    {
        for (int iz = snap_z1; iz < snap_z2; iz++)
        {
            for (int ix = snap_x1; ix < snap_x2; ix++)
            {
                grad[iz][ix] += grad_shot[iz][ix] / (We[iz][ix] * mat_av * mat_av);

                // cout<<"CPU at"<<iz<<" "<<ix<<" "<<grad[iz][ix]<<" "<<grad_shot[iz][ix]<<" "<<We[iz][ix]<<"\n";
            }
        }
    }
}

int main()

{
    real **grad_shot;
    real **grad;
    real **We;
    
    int nz = 10, nx = 20;

    std::cout.precision(30);
    allocate_array_2d(grad, nz, nx);      //input array
    allocate_array_2d(grad_shot, nz, nx); //input array
    allocate_array_2d(We, nz, nx);        //input array

    //initilization
    for (int i = 0; i < nz; i++)
    {

        for (int j = 0; j < nx; j++)
        {
            grad[i][j] = 123;
            grad_shot[i][j] = 23343; // just put any value
            We[i][j] = 231;
        }
    }

    // Scale to energy weight and add to global array
    int snap_z1 = 2, snap_z2 = 5, snap_x1 = 1, snap_x2 = 5;
    real scalar_mu = 2;


    // DATA COPY TO GPU AFTER VALUES INITIALIZED

    real *d_grad;
    real *d_grad_shot;
    real *d_We;
    cudaCheckError(cudaMalloc(&d_grad, nz * nx * sizeof(real)));//device allocation
    cudaCheckError(cudaMalloc(&d_grad_shot, nz * nx * sizeof(real)));
    cudaCheckError(cudaMalloc(&d_We, nz * nx * sizeof(real)));

    cudaCheckError(cudaMemcpy(d_grad, grad[0], nz * nx * sizeof(real), cudaMemcpyHostToDevice)); //grad[0]=*grad
    cudaCheckError(cudaMemcpy(d_grad_shot, grad_shot[0], nz * nx * sizeof(real), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_We, We[0], nz * nx * sizeof(real), cudaMemcpyHostToDevice));

    //CPU CALL OF SAME FUNCTION
    scale_grad_E2(grad, grad_shot, scalar_mu, We,
                  snap_z1, snap_z2, snap_x1, snap_x2);

    ///************************
    //CUDA KERNELS ARE HERE
    scale_grad_E2_GPU(d_grad, d_grad_shot, scalar_mu, d_We,
                      snap_z1, snap_z2, snap_x1, snap_x2, nz, nx);

    ///     TESTING //
    real **grad_test;
    allocate_array_2d(grad_test, nz, nx); //input array

    cudaCheckError(cudaMemcpy(grad_test[0], d_grad, nz * nx * sizeof(real), cudaMemcpyDeviceToHost));

    for (int i = snap_z1; i < snap_z2; i++)
    {
        for (int j = snap_x1; j < snap_x2; j++)
        {
            if (grad[i][j] != grad_test[i][j])
            {
                cout << "Failed at" << i << " " << j << " " << grad[i][j] << " " << grad_test[i][j] << "\n";
            }
        }
    }

    cout << "\ndone\n";
}