// h_simulate_PSV.hpp

/* 
* Created by: Min Basnet
* 2020.December.21
* Kathmandu, Nepal
*/

#ifndef H_GLOBVAR_H		
#define H_GLOBVAR_H	

using real = double;
#define PI 3.14159265

/*
// Device global variables to be copied from the host
// --------------------------
// GEOMETRIC PARAMETERS
// ------------------------
// Geometric grid
int nt, nz, nx;
float dt, dz, dx;

// Snap boundary and intervals for saving memory
int snap_t1, snap_t2, snap_dt;
int snap_z1, snap_z2, snap_dz;
int snap_x1, snap_x2, snap_dx;

// Surface conditions
bool surf; // if surface exists in any side
int *isurf; // surface indices on 0.top, 1.bottom, 2.left, 3.right

// PML conditions
bool pml_z, pml_x; // if pml exists in respective directions

// ------------------
// SEISMOGRAMS
// ------------------
// Numbers of seismograms
int nshot, nsrc, nrec; // number of shots, sources & recievers

int *z_src, *x_src; // source coordiante indices
int *src_shot_to_fire; // which source to fire in which shot
int *z_rec, *x_rec; // source coordinate indices

// stf and rtf 
int stf_type, rtf_type; // time function types 1.displacement
real **stf_z, **stf_x; // source time functions
real **rtf_z_true, **rtf_x_true; // rtf (measured (true) values)

// simulation parameters
bool fwinv; // 0.fwd, 1.fwu
int fdorder, fpad; // FD order and the numerical padding in each side
real *hc; // Holberg coefficients
bool accu_save; // if to write accumulated tensor storage to disk

// --------------------------
// MATERIAL PARAMETERS
// -----------------------------
real scalar_lam, scalar_mu, scalar_rho; // Scalar elastic materials
real **lam, **mu, **rho; // Initial material arrays

// ------------------------
// PML COEFFICIENTS
// -------------------------
real *a_z, *b_z, *a_half_z, *b_half_z, k_half_z; // PML coefficient z-dir
real *a_x, *b_x, *a_half_x, *b_half_x, k_half_x; // PML coefficient x-dir

// PML parameters not to be transferred to the device
*/
#endif