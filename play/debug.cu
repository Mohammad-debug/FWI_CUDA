#include <cuda_runtime_api.h>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <math.h>
#define   NZ  512//
#define   NX  512//
using namespace std;
using real =double;


real adjsrc2(int ishot, int *&a_stf_type, real **&a_stf_uz, real **&a_stf_ux, 
            int rtf_type, real ***&rtf_uz_true, real ***&rtf_ux_true, 
            real **&rtf_uz_mod, real **&rtf_ux_mod,             
            real dt, int nseis, int nt){
    // Calculates adjoint sources and L2 norm
    // a_stf: adjoint sources
    // rtf: reciever time function (mod: forward model, true: field measured)

    real L2;
    L2 = 0;
    
    if (rtf_type == 0){
        // RTF type is displacement
        for(int is=0; is<nseis; is++){ // for all seismograms
            for(int it=0;it<nt;it++){ // for all time steps

                
                // calculating adjoint sources

               //! a_stf_uz[is][it] = rtf_uz_mod[is][it] - rtf_uz_true[ishot][is][it];
               //! a_stf_ux[is][it] = rtf_ux_mod[is][it] - rtf_ux_true[ishot][is][it];

                //if (!(abs(a_stf_uz[is][it])<1000.0 || abs(a_stf_uz[is][it])<1000.0)){
                //    std::cout << rtf_uz_mod[is][it] <<"," << rtf_uz_true[ishot][is][it] << "::";
                //}
                

                // Calculating L2 norm
                L2 += 0.5 * dt * pow(a_stf_uz[is][it], 2); 
                L2 += 0.5 * dt * pow(a_stf_ux[is][it], 2);
                //std::cout<< rtf_uz_mod[is][it] <<", "<<rtf_ux_mod[is][it];
                
            }
            
        }

        a_stf_type = &rtf_type; // Calculating displacement adjoint sources
    
    }
    std::cout<< "Calculated norm: " << L2 << std::endl;
    //std::cout << a_stf_type << std::endl;
    return L2;

}

void allocate_array_2d(real**& preal, const int dim1, const int dim2) {
    // Contiguous allocation of 2D arrays

    preal = new real * [dim1];
    preal[0] = new real[dim1 * dim2];
    for (int i = 1; i < dim1; i++) preal[i] = preal[i - 1] + dim2;

    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            preal[i][j] = 0;
        }
    }
}
#define cudaCheckError(code)                                             \
  {                                                                      \
    if ((code) != cudaSuccess) {                                         \
      fprintf(stderr, "Cuda failure %s:%d: '%s' \n", __FILE__, __LINE__, \
              cudaGetErrorString(code));                                 \
    }                                                                    \
  }


template <class T> void allocate_array(T ***&pointer, int dim1, int dim2, int dim3) {

  pointer = new T **[dim1];
  pointer[0] = new T *[dim1 * dim2];
  pointer[0][0] = new T[dim1 * dim2 * dim3];

  for (int i = 0; i < dim1; i++) {
    if (i > 0) {
      pointer[i] = pointer[i - 1] + dim2;
      pointer[i][0] = pointer[i - 1][0] + dim2 * dim3;
    }

    for (int j = 1; j < dim2; j++) {
      pointer[i][j] = pointer[i][j - 1] + dim3;
    }
  }
}


class power_functor {

    double a;
    double dt;

    public:

        power_functor(double a_,double dt_) { a = a_; dt=dt_;}

        __host__ __device__ double operator()(double x) const 
        {
            return 0.5*dt*pow(x,a);
        }
};

__global__ void cuda_adjsrc2_GPU(int ishot, real *a_stf_uz, real *a_stf_ux, 
            int rtf_type, real *rtf_uz_true, real *rtf_ux_true, 
            real *rtf_uz_mod, real *rtf_ux_mod,             
            real dt, int nseis, int nt)
{

     int is = blockIdx.x * blockDim.x + threadIdx.x;
     int it = blockIdx.y * blockDim.y + threadIdx.y;
    if(is>=0 && nseis && it>=0 && it<nseis)
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

    //cuda_adjsrc2_GPU<<<blocksPerGrid, threadsPerBlock>>>(ishot, a_stf_uz, a_stf_ux, 
     //       rtf_type, rtf_uz_true, rtf_ux_true,rtf_uz_mod, rtf_ux_mod,dt, nseis,  nt);

    //cudaCheckError(cudaDeviceSynchronize());
    
    thrust::device_ptr<real> dev_ptr1 = thrust::device_pointer_cast(a_stf_uz);
    thrust::device_ptr<real> dev_ptr2 = thrust::device_pointer_cast(a_stf_ux);
    L2 = thrust::transform_reduce(thrust::device,dev_ptr1,dev_ptr1+nseis*nt,power_functor(2.,dt),0.0,thrust::plus<real>());

    L2 += thrust::transform_reduce(thrust::device,dev_ptr2,dev_ptr2+nseis*nt,power_functor(2.,dt),0.0,thrust::plus<real>());
        
}
 

        a_stf_type = &rtf_type; // Calculating displacement adjoint sources
    

    std::cout<< "Calculated norm: " << L2 << std::endl;
    //std::cout << a_stf_type << std::endl;
    return L2;

}

int main()

{
    
    std::cout.precision(30);

    int *a_stf_type ;//
  
    int rtf_type=0;
    real ***rtf_z_true;
    real ***rtf_x_true;
    real **rtf_uz;
    real **rtf_ux;
    int nrec=10;
    int nt=10;
    int nshot=10;
    int ishot=1;
    real dt=2.5;

    allocate_array_2d(rtf_uz,  nrec, nt);
    allocate_array_2d(rtf_ux, nrec, nt);

    allocate_array_2d(rtf_uz,  nrec, nt);
    allocate_array_2d(rtf_ux, nrec, nt);

  

    allocate_array(rtf_z_true, nshot, nrec, nt);
    allocate_array(rtf_x_true, nshot, nrec, nt);
   
    for (int i = 0; i <nrec; i++) {
        for (int j = 0; j < nt; j++) {
            rtf_uz[i][j] = rand();
            rtf_ux[i][j] = rand();
          
          
        }
    }

    for (int i = 0; i <nshot; i++) {
        for (int j = 0; j < nrec; j++) {
            for (int k = 0; k < nt; k++)
            {
               rtf_z_true[i][j][k]=rand();
               rtf_x_true[i][j][k]=rand();

            }
        }
    }

   
    rtf_type=0;
    real *d_rtf_z_true;
    real *d_rtf_x_true;
    real *d_rtf_uz;
    real *d_rtf_ux; 

    cudaCheckError(cudaMalloc(&d_rtf_uz, nrec* nt  * sizeof(real)));
    cudaCheckError(cudaMalloc(&d_rtf_ux, nrec* nt  * sizeof(real)));

    cudaCheckError(cudaMemcpy(d_rtf_uz, rtf_uz[0],  nrec* nt  * sizeof(real), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_rtf_ux, rtf_ux[0], nrec* nt * sizeof(real), cudaMemcpyHostToDevice));


    cudaCheckError(cudaMalloc(&d_rtf_z_true, nshot * nrec * nt  * sizeof(real)));
    cudaCheckError(cudaMalloc(&d_rtf_x_true, nshot * nrec * nt  * sizeof(real)));

    cudaCheckError(cudaMemcpy(d_rtf_z_true, rtf_z_true[0][0], nshot* nrec* nt  * sizeof(real), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_rtf_x_true, rtf_x_true[0][0],  nshot* nrec* nt * sizeof(real), cudaMemcpyHostToDevice));



  real L=adjsrc2(ishot, a_stf_type, rtf_uz, rtf_ux, rtf_type, rtf_z_true, rtf_x_true,
                            rtf_uz, rtf_ux, dt, nrec, nt);
        cout<<"CPU= "<<L<<"\n\n";
        

 L= adjsrc2_GPU(ishot, a_stf_type, d_rtf_uz, d_rtf_ux, rtf_type, d_rtf_z_true, d_rtf_x_true,
                            d_rtf_uz, d_rtf_ux, dt, nrec, nt);
       cout<<"GPU= "<<L<<"\n\n";
   
}



//  real* da;
//         double dt=0.5;
//         cudaCheckError(cudaMalloc(&da, NZ * NX  * sizeof(real)));
//         cudaCheckError(cudaMemcpy(da,a[0], NZ * NX  * sizeof(real),cudaMemcpyHostToDevice));

//         ///************************
//         //CUDA KERNELS ARE HERE
//         // REMOVED FOR CLEAR QUESTION
//         ///*************************
  
//         real L1=0;
//         // L2 += 0.5 * dt * pow(a_stf_uz[is][it], 2);

//         thrust::device_ptr<real> dev_ptr = thrust::device_pointer_cast(da);
//         L1 = thrust::transform_reduce(thrust::device,dev_ptr,dev_ptr+NX*NZ,power_functor(2.,dt),0.0,thrust::plus<real>());
        
//         cout<<" \nsum gpu "<< L1<<"\n";

//         real L2=0;

//         ////////CPU PART DOING SAME THING//////
//         for (int i = 0; i < NZ; i++) {

//             for (int j = 0; j < NX; j++) {
//                L2+=0.5 * dt * pow(a[i][j], 2);
                
//             }
//         }


//         cout<<"\nsum cpu "<< L2<<"\n";
//         if(abs(L2-L1)<0.001)
//         std::cout << "\nSUCESS "<< "\n";
//         else
//         std::cout << "\nFailure & by "<<L2-L1<< "\n";


