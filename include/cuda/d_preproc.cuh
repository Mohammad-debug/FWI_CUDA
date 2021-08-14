
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