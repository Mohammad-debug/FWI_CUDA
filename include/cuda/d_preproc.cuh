
#include "d_globvar.cuh"
#include "d_contiguous_arrays.cuh"



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
    int nt,int nshot, int nz, int nx);

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
    int nt,int nshot, int nz, int nx);   



template <class T>
void allocate_array_cpu(T ****&pointer, int dim1, int dim2, int dim3, int dim4) {

  pointer = new T ***[dim1];
  pointer[0] = new T **[dim1 * dim2];
  pointer[0][0] = new T *[dim1 * dim2 * dim3];
  pointer[0][0][0] = new T[dim1 * dim2 * dim3 * dim4];

  // Pointer arithmetic
  for (int h = 0; h < dim1; h++) {
    if (h > 0) {
      pointer[h] = pointer[h - 1] + dim2;
      pointer[h][0] = pointer[h - 1][0] + (dim2 * dim3);
      pointer[h][0][0] = pointer[h - 1][0][0] + (dim2 * dim3 * dim4);
    }

    for (int i = 0; i < dim2; i++) {
      if (i > 0) {
        pointer[h][i] = pointer[h][i - 1] + dim3;
        pointer[h][i][0] = pointer[h][i - 1][0] + (dim3 * dim4);
      }

      for (int j = 1; j < dim3; j++) {
        pointer[h][i][j] = pointer[h][i][j - 1] + dim4;
      }
    }
  }
}

template <class T> void allocate_array_cpu(T ***&pointer, int dim1, int dim2, int dim3) {

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

template <class T> void allocate_array_cpu(T **&pointer, int dim1, int dim2) {
  pointer = new T *[dim1];

  int size = dim1 * dim2;

  pointer[0] = new T[size];
  
  for (int ir = 1; ir < dim1; ir++) {
    pointer[ir] = pointer[ir - 1] + dim2;
  }
}

template <class T> void allocate_array_cpu(T *&pointer, int dim1) { pointer = new T[dim1]; }
template <class T> void deallocate_array_cpu(T *&pointer) { delete[] pointer; }
template <class T> void deallocate_array_cpu(T **&pointer) {
  delete[] pointer[0];
  delete[] pointer;
}
template <class T> void deallocate_array_cpu(T ***&pointer) {
  delete[] pointer[0][0];
  delete[] pointer[0];
  delete[] pointer;
}
template <class T> void deallocate_array_cpu(T ****&pointer) {
  delete[] pointer[0][0][0];
  delete[] pointer[0][0];
  delete[] pointer[0];
  delete[] pointer;
}

