//seismic_kernel_lib2.cpp

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
#include "seismic_kernel_lib2.hpp"

void reset_sv2(
    // wave arguments (velocity) & Energy weights
    real **vz, real **vx, real ** sxx, real ** szx, real ** szz, real **We, 
    // time & space grids (size of the arrays)
    real nz, real nx){
    // reset the velocity and stresses to zero
    // generally applicable in the beginning of the time loop

    for (int iz = 0; iz<nz; iz++){
        for (int ix = 0; ix<nx; ix++){
            // Wave velocity and stress tensor arrays
            vx[iz][ix] = 0.0;
            vz[iz][ix] = 0.0; 
            sxx[iz][ix] = 0.0;
            szx[iz][ix] = 0.0;
            szz[iz][ix] = 0.0;
            We[iz][ix] = 0.0;
        }
    }

}


void vdiff2(
    // spatial velocity derivatives
    real **vz_z, real **vx_z, real **vz_x, real **vx_x,
    // wave arguments (velocity)
    real **vz, real **vx,
    // holberg coefficient
    real *hc,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, int dz, int dx){
    // updates the stress kernels for each timestep in 2D grid

    real dxi = 1.0/dx; real dzi = 1.0/dz; // inverse of dx and dz

    // 2D space grid
    for(int iz=nz1; iz<nz2; iz++){
        for(int ix=nx1; ix<nx2; ix++){

            // Calculating the spatial velocity derivatives
            vz_z[iz][ix] = dzi * hc[1] * ( vz[iz][ix] - vz[iz-1][ix] );
            vx_z[iz][ix] = dzi * hc[1] * ( vx[iz+1][ix] - vx[iz][ix] );   
            vz_x[iz][ix] = dxi * hc[1] * ( vz[iz][ix+1] - vz[iz][ix] );
            vx_x[iz][ix] = dxi * hc[1] * ( vx[iz][ix] - vx[iz][ix-1] );
            
        }
    }

}


void pml_vdiff2(
    // spatial velocity derivatives
    real **vz_z, real **vx_z, real **vz_x, real **vx_x,
    // PML memory arrays
    real** mem_vz_z, real** mem_vx_z, real** mem_vz_x, real ** mem_vx_x, 
    //PML arguments
    real ** a, real ** b, real ** K, 
    real ** a_half, real ** b_half, real ** K_half,
    // time space grids
    int nz1, int nz2, int nx1, int nx2){
    
    // updates PML memory variables for velicity derivatives
    // absorption coefficients are for the whole grids
    // 2D space grid
    for(int iz=nz1; iz<nz2; iz++){
        for(int ix=nx1; ix<nx2; ix++){

            // CPML memory variables in z-direction
            mem_vz_z[iz][ix] = b[iz][ix] * mem_vz_z[iz][ix] 
                                + a[iz][ix] * vz_z[iz][ix];                                            
            mem_vx_z[iz][ix] = b_half[iz][ix] * mem_vx_z[iz][ix] 
                                + a_half[iz][ix] * vx_z[iz][ix];
                     
            vz_z[iz][ix] = vz_z[iz][ix] / K[iz][ix] + mem_vz_z[iz][ix];
            vx_z[iz][ix] = vx_z[iz][ix] / K_half[iz][ix] + mem_vx_z[iz][ix];


            // CPML memory variables in x-direction
            mem_vx_x[iz][ix] = b[iz][ix] * mem_vx_x[iz][ix] 
                                + a[iz][ix] * vx_x[iz][ix];
            mem_vz_x[iz][ix] = b_half[iz][ix] * mem_vz_x[iz][ix] 
                                + a_half[iz][ix] * vz_x[iz][ix];

            vx_x[iz][ix] = vx_x[iz][ix] / K[iz][ix] + mem_vx_x[iz][ix];                
            vz_x[iz][ix] = vz_x[iz][ix] / K_half[iz][ix] + mem_vz_x[iz][ix]; 
            
        }

    }

}


void update_s2(
    // Wave arguments (stress)
    real ** sxx, real ** szx, real ** szz,
    // spatial velocity derivatives
    real **vz_z, real **vx_z, real **vz_x, real **vx_x,
    // Medium arguments
    real ** lam, real ** mu, real ** mu_zx,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, int dt){
    // update stress from velocity derivatives

    // 2D space grid
    for(int iz=nz1; iz<nz2; iz++){
        for(int ix=nx1; ix<nx2; ix++){

            // updating stresses
            szx[iz][ix] += dt * mu_zx[iz][ix] * (vz_x[iz][ix]+vx_z[iz][ix]);
            sxx[iz][ix] += dt * ( lam[iz][ix] * (vx_x[iz][ix]+vz_z[iz][ix]) 
                            + (2.0*mu[iz][ix]*vx_x[iz][ix]) );
            szz[iz][ix] += dt * ( lam[iz][ix] * (vx_x[iz][ix]+vz_z[iz][ix]) 
                            + (2.0*mu[iz][ix]*vz_z[iz][ix]) );

        }
    }

}


void sdiff2(
    // spatial stress derivatives
    real **szz_z, real **szx_z, real **szx_x, real **sxx_x,
    // Wave arguments (stress)
    real ** sxx, real ** szx, real ** szz,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, int dz, int dx,
    // holberg coefficient
    real *hc){
    // updates the stress kernels for each timestep in 2D grid

    real dxi = 1.0/dx; real dzi = 1.0/dz; // inverse of dx and dz

    // 2D space grid
    for(int iz=nz1; iz<nz2; iz++){
        for(int ix=nx1; ix<nx2; ix++){

            // compute spatial stress derivatives
            szz_z[iz][ix] = dzi * hc[1] * (szz[iz+1][ix] - szz[iz][ix]);  
            szx_z[iz][ix] = dzi * hc[1] * (szx[iz][ix] - szx[iz-1][ix]);
            szx_x[iz][ix] = dxi * hc[1] * (szx[iz][ix] - szx[iz][ix-1]);
            sxx_x[iz][ix] = dxi * hc[1] * (sxx[iz][ix+1] - sxx[iz][ix]);
            
        }
    }

}


void pml_sdiff2(){
    // updates PML memory variables for stress derivatives

}


void update_v2(
    // wave arguments (velocity) & Energy weights
    real **vz, real **vx, 
    // displacement and energy arrays 
    real **uz, real **ux, real **We,
    // spatial stress derivatives
    real **szz_z, real **szx_z, real **szx_x, real **sxx_x,
    // Medium arguments
    real ** rho_zp, real ** rho_xp,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, int dt){
    // update stress from velocity derivatives


    // 2D space grid
    for(int iz=nz1; iz<nz2; iz++){
        for(int ix=nx1; ix<nx2; ix++){

            // update particle velocities
            vz[iz][ix] += dt * rho_zp[iz][ix]*(szx_x[iz][ix]+szz_z[iz][ix]);
            vx[iz][ix] += dt * rho_xp[iz][ix]*(sxx_x[iz][ix]+szx_z[iz][ix]);

            // Displacements and Energy weights
            uz[iz][ix] += dt * vz[iz][ix];
            ux[iz][ix] += dt * vx[iz][ix];
            We[iz][ix] += vx[iz][ix] * vx[iz][ix] + vz[iz][ix] * vz[iz][ix];
            
        }
    }

}

void surf_mirror(
    // Wave arguments (stress & velocity derivatives)
    real ** sxx, real ** szx, real ** szz, real **vz_z, real **vx_x,
    // Medium arguments
    real ** lam, real ** mu,
    // surface indices for four directions(0.top, 1.bottom, 2.left, 3.right)
    int *surf,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, int dt){
    // surface mirroring for free surface
    

    int isurf;
    // -----------------------------
    // 1. TOP SURFACE
    // -----------------------------
    if (surf[0]>0){
        isurf = surf[0];
        for(int ix=nx1; ix<nx2; ix++){
            // Denise manual  page 13
            szz[isurf][ix] = 0.0;
            szx[isurf][ix] = 0.0;
            sxx[isurf][ix] = 4.0 * dt * vx_x[isurf][ix] *(lam[isurf][ix] * mu[isurf][ix] 
                                + mu[isurf][ix] * mu[isurf][ix])
                                / (lam[isurf][ix] + 2.0 * mu[isurf][ix]);

            for (int sz=1; sz<isurf-nz1+1; sz++){ // mirroring 
                szx[isurf-sz][ix] = -szx[isurf+sz][ix];
                szz[isurf-sz][ix] = -szz[isurf+sz][ix];
            }
        }

    }

    // -----------------------------
    // 2. BOTTOM SURFACE
    // -----------------------------
    if (surf[1]>0){
        isurf = surf[1];
        for(int ix=nx1; ix<nx2; ix++){
            // Denise manual  page 13
            szz[isurf][ix] = 0.0;
            szx[isurf][ix] = 0.0;
            sxx[isurf][ix] = 4.0 * dt * vx_x[isurf][ix] *(lam[isurf][ix] * mu[isurf][ix] 
                                + mu[isurf][ix] * mu[isurf][ix])
                                / (lam[isurf][ix] + 2.0 * mu[isurf][ix]);

            for (int sz=1; sz<=nz2-isurf; sz++){ // mirroring 
                szx[isurf+sz][ix] = -szx[isurf-sz][ix];
                szz[isurf+sz][ix] = -szz[isurf-sz][ix];
            }
        }

    }

    // -----------------------------
    // 3. LEFT SURFACE
    // -----------------------------
    if (surf[2]>0){
        isurf = surf[0];
        for(int iz=nz1; iz<nz2; iz++){
            // Denise manual  page 13
            sxx[iz][isurf] = 0.0;
            szx[iz][isurf] = 0.0;
            szz[iz][isurf] = 4.0 * dt * vz_z[iz][isurf] *(lam[iz][isurf] * mu[iz][isurf] 
                                + mu[iz][isurf] * mu[iz][isurf])
                                / (lam[iz][isurf] + 2.0 * mu[iz][isurf]);

            for (int sx=1; sx<isurf-nx1+1; sx++){ // mirroring 
                szx[iz][isurf-sx] = -szx[iz][isurf+sx];
                sxx[iz][isurf-sx] = -szz[iz][isurf+sx];
            }
        }

    }



    // -----------------------------
    // 4. RIGHT SURFACE
    // -----------------------------
    if (surf[3]>0){
        isurf = surf[0];
        for(int iz=nz1; iz<nz2; iz++){
            // Denise manual  page 13
            sxx[iz][isurf] = 0.0;
            szx[iz][isurf] = 0.0;
            szz[iz][isurf] = 4.0 * dt * vz_z[iz][isurf] *(lam[iz][isurf] * mu[iz][isurf] 
                                + mu[iz][isurf] * mu[iz][isurf])
                                / (lam[iz][isurf] + 2.0 * mu[iz][isurf]);

            for (int sx=1; sx<=nx2-isurf; sx++){ // mirroring 
                szx[iz][isurf+sx] = -szx[iz][isurf-sx];
                sxx[iz][isurf+sx] = -szz[iz][isurf-sx];
            }
        }

    }
    
}


void gard_fwd_storage2(
    // forward storage for full waveform inversion 
    real ***accu_vz, real ***accu_vx, 
    real ***accu_szz, real ***accu_szx, real ***accu_sxx,
    // velocity and stress tensors
    real **vz, real **vx, real **szz, real **szx, real **sxx,
    // time and space parameters
    int dt, int itf, int snap_z1, int snap_z2, 
    int snap_x1, int snap_x2, int snap_dz, int snap_dx){
    
    
    // Stores forward velocity and stress for gradiant calculation in fwi
    // dt: the time step size
    // itf: reduced continuous time index after skipping the time steps in between 
    // snap_z1, snap_z2, snap_x1, snap_z2: the indices for fwi storage
    // snap_dz, snap_dx: the grid interval for reduced (skipped) storage of tensors
    
    
    
    int jz, jx; // mapping for storage with intervals
    jz = 0; jx = 0;
    for(int iz=snap_z1;iz<snap_z2;iz+=snap_dz){
        for(int ix=snap_x1;ix<snap_x2;ix+=snap_dx){
            accu_sxx[itf][jz][jx]  = sxx[iz][ix];
            accu_szx[itf][jz][jx]  = szx[iz][ix];
            accu_szz[itf][jz][jx]  = szz[iz][ix];

            accu_vx[itf][jz][jx] = vx[iz][ix]/dt;
            accu_vz[itf][jz][jx] = vz[iz][ix]/dt;
            
            jx++;
        }
        jz++;
    }
    
}

void fwi_grad2(
    // Gradient of the materials
    real **grad_lam, real ** grad_mu, real ** grad_rho,
    // forward storage for full waveform inversion 
    real ***accu_vz, real ***accu_vx, real ***accu_szz, real ***accu_szx, real ***accu_sxx,
    // displacement and stress tensors
    real **uz, real **ux, real **szz, real **szx, real **sxx,
    // Medium arguments
    real ** lam, real ** mu,
    // time and space parameters
    int dt, int tf, int snap_dt, int snap_z1, int snap_z2, int snap_x1, int snap_x2, int snap_dz, int snap_dx){
    // Calculates the gradient of medium from stored forward tensors & current tensors

    real s1, s2, s3, s4; // Intermediate variables for gradient calculation
    int jz, jx; // mapping for storage with intervals
    
    jz = 0; jx = 0;
    for(int iz=snap_z1;iz<snap_z2;iz+=snap_dz){
        for(int ix=snap_x1;ix<snap_x2;ix+=snap_dx){

            s1 = (accu_sxx[tf][jz][jx] + accu_szz[tf][jz][jx]) * (sxx[iz][ix] + szz[iz][ix])
               *0.25 /((lam[iz][ix] + mu[iz][ix])*(lam[iz][ix] + mu[iz][ix]));

            s2 = (accu_sxx[tf][jz][jx] - accu_szz[tf][jz][jx]) * (sxx[iz][ix] - szz[iz][ix])
                / (mu[iz][ix]*mu[iz][ix]) ;

            s3 = (accu_szx[tf][jz][jx] * szx[iz][ix] )/ (mu[iz][ix]*mu[iz][ix]);

            // The time derivatives of the velocity may have to be computed differently
            s4 = ux[iz][ix] * accu_vx[tf][jz][jx] + uz[iz][ix] * accu_vz[tf][jz][jx];

            grad_lam[jz][jx] += snap_dt * dt *  s1 ; 
            grad_mu[jz][jx] +=  snap_dt * dt * (s3 + s1 + s2) ;
            grad_rho[jz][jx] -= snap_dt * dt * s4 ;

            // ----------------------------------------------------------------------
            // Interpolating the gradients of missing values (Linear Interpolation)
            // ----------------------------------------------------------------------
            // To be added later


        }
    }
}

void vsrc2(
    // Velocity tensor arrays
    real **vz, real **vx, 
    // inverse of density arrays
    real **rho_zp, real **rho_xp,
    // source parameters
    int nsrc, int stf_type, real **stf_z, real **stf_x, 
    int *z_src, int *x_src, int it,
    int dt, int dz, int dx){
    // firing the velocity source term
    // nsrc: number of sources
    // stf_type: type of source time function (0:displacement, 1:velocity currently implemented)
    // stf_z: source time function z component
    // stf_x: source time function x component
    // z_src: corresponding grid index along z direction
    // x_src: corresponding grid index along x direction
    // it: time step index

    switch(stf_type){
    
        case(0): // Displacement stf
            for(int is=0; is<nsrc; is++){
                vz[z_src[is]][x_src[is]] += dt*rho_zp[z_src[is]][x_src[is]]*stf_z[is][it]/(dz*dx);
                vx[z_src[is]][x_src[is]] += dt*rho_xp[z_src[is]][x_src[is]]*stf_x[is][it]/(dz*dx);
            }
            break;
        
        case(1): // velocity stf
            for(int is=0; is<nsrc; is++){
                vz[z_src[is]][x_src[is]] += stf_z[is][it];
                vx[z_src[is]][x_src[is]] += stf_x[is][it];
            }
            break;
    }
    
}

void urec2(int rtf_type,
    // reciever time functions
    real **rtf_uz, real **rtf_ux, 
    // velocity tensors
    real **vz, real **vx,
    // reciever time and space grids
    int nrec, int *rz, int *rx, int it, real dt, real dz, real dx){
    // recording the output seismograms
    // nrec: number of recievers
    // rtf_uz: reciever time function (displacement_z)
    // rtf_uz: reciever time function (displacement_x)
    // rec_signal: signal file for seismogram index and time index
    // rz: corresponding grid index along z direction
    // rx: corresponding grid index along x direction
    // it: time step index

    if (rtf_type == 0){
        // This module is only for rtf type as displacement
        for(int ir=0; ir<nrec; ir++){
            if (it ==0){
                rtf_uz[ir][it] = dt * vz[rz[ir]][rx[ir]] / (dz*dx);
                rtf_ux[ir][it] = dt * vx[rz[ir]][rx[ir]] / (dz*dx);
            }
            else{
                rtf_uz[ir][it] = rtf_uz[ir][it-1] + dt * vz[rz[ir]][rx[ir]] / (dz*dx);
                rtf_ux[ir][it] = rtf_ux[ir][it-1] + dt * vx[rz[ir]][rx[ir]] / (dz*dx);
            }
        }

    } 
    rtf_type = 0; // Displacement rtf computed
}

real adjsrc2(int a_stf_type, real ** a_stf_uz, real ** a_stf_ux, 
            int rtf_type, real ** rtf_uz_true, real ** rtf_ux_true, 
            real ** rtf_uz_mod, real ** rtf_ux_mod,             
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
                a_stf_uz[is][it] = rtf_uz_mod[is][it] - rtf_uz_true[is][it];
                a_stf_ux[is][it] = rtf_ux_mod[is][it] - rtf_ux_true[is][it];

                // Calculating L2 norm
                L2 += 0.5 * dt * pow(a_stf_uz[is][it], 2);
                L2 += 0.5 * dt * pow(a_stf_ux[is][it], 2);

            }
        }
    a_stf_type = 0; // Calculating displacement adjoint sources
    }
    
    return L2;

}