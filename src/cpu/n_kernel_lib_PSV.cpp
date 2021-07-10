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
#include "n_kernel_lib_PSV.hpp"
#include<omp.h>

void reset_sv2(
    // wave arguments (velocity) & Energy weights
    real **&vz, real **&vx, real **&uz, real **&ux, 
    real **&szz, real **&szx, real **&sxx, real **&We, 
    // time & space grids (size of the arrays)
    real nz, real nx)
    {
    // reset the velocity and stresses to zero
    // generally applicable in the beginning of the time loop

    for (int iz = 0; iz<nz; iz++){
        for (int ix = 0; ix<nx; ix++){
            // Wave velocity and stress tensor arrays
            vz[iz][ix] = 0.0;
            vx[iz][ix] = 0.0; 
            uz[iz][ix] = 0.0; 
            ux[iz][ix] = 0.0;
            szz[iz][ix] = 0.0;
            szx[iz][ix] = 0.0;
            sxx[iz][ix] = 0.0;
            We[iz][ix] = 0.0;
        }
    }

}


void reset_PML_memory2(
    // PML memory arrays
    real **&mem_vz_z, real **&mem_vx_z, real **&mem_vz_x, real **&mem_vx_x, 
    // time & space grids (size of the arrays)
    real nz, real nx){
    // reset the velocity and stresses to zero
    // generally applicable in the beginning of the time loop

    for (int iz = 0; iz<nz; iz++){
        for (int ix = 0; ix<nx; ix++){
            // Wave velocity and stress tensor arrays
            mem_vz_z[iz][ix] = 0.0;
            mem_vx_z[iz][ix] = 0.0; 
            mem_vz_x[iz][ix] = 0.0;
            mem_vx_x[iz][ix] = 0.0;
        }
    }

}

void reset_grad_shot2(real **&grad_lam, real **&grad_mu, real **&grad_rho,
					int snap_z1, int snap_z2, int snap_x1, int snap_x2,
					int snap_dz, int snap_dx){
	
	int jz , jx ;
    jz = 0;
	for(int iz=snap_z1;iz<=snap_z2;iz+=snap_dz){
        jx = 0;
        for(int ix=snap_x1;ix<=snap_x2;ix+=snap_dx){

            grad_lam[jz][jx] = 0.0; 
            grad_mu[jz][jx]  = 0.0; 
            grad_rho[jz][jx] = 0.0; 
			
			jx++;
        }
		
		jz++;
    }
}


void vdiff2(
    // spatial velocity derivatives
    real **&vz_z, real **&vx_z, real **&vz_x, real **&vx_x,
    // wave arguments (velocity)
    real **&vz, real **&vx,
    // holberg coefficient
    real *&hc,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dz, real dx){
    // updates the stress kernels for each timestep in 2D grid

    real dxi = 1.0/dx; real dzi = 1.0/dz; //local variables
    // inverse of dx and dz

    // 2D space grid
   
   #pragma omp parallel for collapse(2)
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

//parallelising
void pml_diff2(bool pml_z, bool pml_x,
    // spatial derivatives
    real **&dz_z, real **&dx_z, real **&dz_x, real **&dx_x,
    //PML arguments (z and x direction)
    real *&a_z, real *&b_z, real *&K_z, 
    real *&a_half_z, real *&b_half_z, real *&K_half_z,
    real *&a_x, real *&b_x, real *&K_x, 
    real *&a_half_x, real *&b_half_x, real *&K_half_x, 
    // PML memory arrays for spatial derivatives
    real **&mem_z_z, real **&mem_x_z, 
    real **&mem_z_x, real **&mem_x_x,  
    // time space grids
    int nz1, int nz2, int nx1, int nx2){
    
    // updates PML memory variables for velicity derivatives
    // absorption coefficients are for the whole grids
    // 2D space grid
   #pragma omp parallel for collapse(2)
    for(int iz=nz1; iz<nz2; iz++){
        for(int ix=nx1; ix<nx2; ix++){
            if (pml_z){
                // CPML memory variables in z-direction
                mem_z_z[iz][ix] = b_z[iz] * mem_z_z[iz][ix] 
                                + a_z[iz] * dz_z[iz][ix];                                            
                mem_x_z[iz][ix] = b_half_z[iz] * mem_x_z[iz][ix] 
                                + a_half_z[iz] * dx_z[iz][ix];
                     
                dz_z[iz][ix] = dz_z[iz][ix] / K_z[iz] + mem_z_z[iz][ix];
                dx_z[iz][ix] = dx_z[iz][ix] / K_half_z[iz] + mem_x_z[iz][ix];
            }
            
            if (pml_x){
                // CPML memory variables in x-direction
                mem_x_x[iz][ix] = b_x[ix] * mem_x_x[iz][ix] 
                                + a_x[ix] * dx_x[iz][ix];
                mem_z_x[iz][ix] = b_half_x[ix] * mem_z_x[iz][ix] 
                                + a_half_x[ix] * dz_x[iz][ix];

                dx_x[iz][ix] = dx_x[iz][ix] / K_x[ix] + mem_x_x[iz][ix];                
                dz_x[iz][ix] = dz_x[iz][ix] / K_half_x[ix] + mem_z_x[iz][ix]; 
            }

            
            
        }

    }

    

}

//parallelising
void update_s2(
    // Wave arguments (stress)
    real **&szz, real **&szx, real **&sxx, 
    // spatial velocity derivatives
    real **&vz_z, real **&vx_z, real **&vz_x, real **&vx_x,
    // Medium arguments
    real **&lam, real **&mu, real **&mu_zx,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dt){
    // update stress from velocity derivatives

   
   #pragma omp parallel for collapse(2)
    for(int iz=nz1; iz<nz2; iz++){
        for(int ix=nx1; ix<nx2; ix++){
            

            // updating stresses
            szz[iz][ix] += dt * ( lam[iz][ix] * (vx_x[iz][ix]+vz_z[iz][ix]) 
                            + (2.0*mu[iz][ix]*vz_z[iz][ix]) );
            szx[iz][ix] += dt * mu_zx[iz][ix] * (vz_x[iz][ix]+vx_z[iz][ix]);
            sxx[iz][ix] += dt * ( lam[iz][ix] * (vx_x[iz][ix]+vz_z[iz][ix]) 
                            + (2.0*mu[iz][ix]*vx_x[iz][ix]) );
            
        }
    }
    
}


void sdiff2(
    // spatial stress derivatives
    real **&szz_z, real **&szx_z, real **&szx_x, real **&sxx_x,
    // Wave arguments (stress)
    real **&szz, real **&szx, real **&sxx, 
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dz, real dx,
    // holberg coefficient
    real *&hc){
    // updates the stress kernels for each timestep in 2D grid

    real dxi = 1.0/dx; real dzi = 1.0/dz; // inverse of dx and dz

    // 2D space grid
    #pragma omp parallel for collapse(2)
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


//void pml_sdiff2(){
    // updates PML memory variables for stress derivatives
//}


void update_v2(
    // wave arguments (velocity) & Energy weights
    real **&vz, real **&vx, 
    // displacement and energy arrays 
    real **&uz, real **&ux, real **&We,
    // spatial stress derivatives
    real **&szz_z, real **&szx_z, real **&szx_x, real **&sxx_x,
    // Medium arguments
    real **&rho_zp, real **&rho_xp,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dt){
    // update stress from velocity derivatives

    #pragma omp parallel for collapse(2)
    for(int iz=nz1; iz<nz2; iz++){
        for(int ix=nx1; ix<nx2; ix++){
           // printf("Hello World from thread %d\n", omp_get_thread_num());
            // Calculating displacement from previous velocity
            uz[iz][ix] += dt * vz[iz][ix];
            ux[iz][ix] += dt * vx[iz][ix];

            // update particle velocities
            vz[iz][ix] += dt * rho_zp[iz][ix]*(szx_x[iz][ix]+szz_z[iz][ix]);
            vx[iz][ix] += dt * rho_xp[iz][ix]*(sxx_x[iz][ix]+szx_z[iz][ix]);

            // Displacements and Energy weights      
            We[iz][ix] += vx[iz][ix] * vx[iz][ix] + vz[iz][ix] * vz[iz][ix];
            
        }
    }
    

}


void surf_mirror(
    // Wave arguments (stress & velocity derivatives)
    real **&szz, real **&szx, real **&sxx, real **&vz_z, real **&vx_x,
    // Medium arguments
    real **&lam, real **&mu,
    // surface indices for four directions(0.top, 1.bottom, 2.left, 3.right)
    int *&surf,
    // time space grids
    int nz1, int nz2, int nx1, int nx2, real dt){
    // surface mirroring for free surface
    

    int isurf;
    // -----------------------------
    // 1. TOP SURFACE
    // -----------------------------
    if (surf[0]>0){
        isurf = surf[0];
        //std::cout << std::endl << "SURF INDEX: "<< isurf<<std::endl;
        
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
                //std::cout<<"surf: "<< isurf-sz <<", " << isurf+sz <<", ::" ;
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
        isurf = surf[2];
        
        for(int iz=nz1; iz<nz2; iz++){
            // Denise manual  page 13
            sxx[iz][isurf] = 0.0;
            szx[iz][isurf] = 0.0;
            szz[iz][isurf] = 4.0 * dt * vz_z[iz][isurf] *(lam[iz][isurf] * mu[iz][isurf] 
                                + mu[iz][isurf] * mu[iz][isurf])
                                / (lam[iz][isurf] + 2.0 * mu[iz][isurf]);

            
            for (int sx=1; sx<isurf-nx1+1; sx++){ // mirroring 
                szx[iz][isurf-sx] = -szx[iz][isurf+sx];
                sxx[iz][isurf-sx] = -sxx[iz][isurf+sx];
            }
        }
        

    }



    // -----------------------------
    // 4. RIGHT SURFACE
    // -----------------------------
    if (surf[3]>0){
        isurf = surf[3];

       
        for(int iz=nz1; iz<nz2; iz++){
            // Denise manual  page 13
            sxx[iz][isurf] = 0.0;
            szx[iz][isurf] = 0.0;
            szz[iz][isurf] = 4.0 * dt * vz_z[iz][isurf] *(lam[iz][isurf] * mu[iz][isurf] 
                                + mu[iz][isurf] * mu[iz][isurf])
                                / (lam[iz][isurf] + 2.0 * mu[iz][isurf]);

        
            for (int sx=1; sx<=nx2-isurf; sx++){ // mirroring 
                szx[iz][isurf+sx] = -szx[iz][isurf-sx];
                sxx[iz][isurf+sx] = -sxx[iz][isurf-sx];
            }
        }
        

    }
    
}



void gard_fwd_storage2(
    // forward storage for full waveform inversion 
    real ***&accu_vz, real ***&accu_vx, 
    real ***&accu_szz, real ***&accu_szx, real ***&accu_sxx,
    // velocity and stress tensors
    real **&vz, real **&vx, real **&szz, real **&szx, real **&sxx,
    // time and space parameters
    real dt, int itf, int snap_z1, int snap_z2, 
    int snap_x1, int snap_x2, int snap_dz, int snap_dx) {
    
    // Stores forward velocity and stress for gradiant calculation in fwi
    // dt: the time step size
    // itf: reduced continuous time index after skipping the time steps in between 
    // snap_z1, snap_z2, snap_x1, snap_z2: the indices for fwi storage
    // snap_dz, snap_dx: the grid interval for reduced (skipped) storage of tensors
    
    
    #pragma omp parallel //parallel region starts
    {
    int jz=0, jx=0; // mapping for storage with intervals
    //#pragma omp for
    for(int iz=snap_z1;iz<=snap_z2;iz+=snap_dz)
    {
        jz = (iz-snap_z1)/snap_dz;
        jx=0;
        for(int ix=snap_x1;ix<=snap_x2;ix+=snap_dx){
            //printf("the thread id is %d and jx is %d and iz is %d \n",omp_get_thread_num(),jz,iz);
            accu_sxx[itf][jz][jx]  = sxx[iz][ix];
            accu_szx[itf][jz][jx]  = szx[iz][ix];
            accu_szz[itf][jz][jx]  = szz[iz][ix];

            accu_vx[itf][jz][jx] = vx[iz][ix]/dt;
            accu_vz[itf][jz][jx] = vz[iz][ix]/dt;
            
            jx++;
            }
        }
   }//parallel region ends
}

void fwi_grad2(
    // Gradient of the materials
    real **&grad_lam, real **&grad_mu, real **&grad_rho,
    // forward storage for full waveform inversion 
    real ***&accu_vz, real ***&accu_vx, 
    real ***&accu_szz, real ***&accu_szx, real ***&accu_sxx,
    // displacement and stress tensors
    real **&uz, real **&ux, real **&szz, real **&szx, real **&sxx,
    // Medium arguments
    real **&lam, real **&mu,
    // time and space parameters
    real dt, int tf, int snap_dt, int snap_z1, int snap_z2, 
    int snap_x1, int snap_x2, int snap_dz, int snap_dx){
    // Calculates the gradient of medium from stored forward tensors & current tensors

    real s1, s2, s3, s4; // Intermediate variables for gradient calculation
    //real lm;
    int jz, jx; // mapping for storage with intervals
    
    jz = 0; 
    for(int iz=snap_z1;iz<=snap_z2;iz+=snap_dz){
        jx = 0;
        for(int ix=snap_x1;ix<=snap_x2;ix+=snap_dx){
            
            s1 = 0.25 * (accu_szz[tf][jz][jx] + accu_sxx[tf][jz][jx]) * (szz[iz][ix] + sxx[iz][ix])
                / ((lam[iz][ix] + mu[iz][ix])*(lam[iz][ix] + mu[iz][ix]));

            s2 = 0.25 * (accu_szz[tf][jz][jx] - accu_sxx[tf][jz][jx]) * (szz[iz][ix] - sxx[iz][ix])
                / (mu[iz][ix]*mu[iz][ix]) ;

            s3 = (accu_szx[tf][jz][jx] * szx[iz][ix] )/ (mu[iz][ix]*mu[iz][ix]);

            // The time derivatives of the velocity may have to be computed differently
            s4 = ux[iz][ix] * accu_vx[tf][jz][jx] + uz[iz][ix] * accu_vz[tf][jz][jx];

            grad_lam[jz][jx] += snap_dt * dt * s1 ; 
            grad_mu[jz][jx]  += snap_dt * dt  *(s3 + s1 + s2) ;
            grad_rho[jz][jx] += snap_dt * dt * s4 ;
            
            /*
            lm = lam[iz][ix] + 2.0 *mu[iz][ix];
            grad_rho[jz][jx] -=
              snap_dt * dt *
              (accu_vx[tf][jz][jx] * ux[iz][ix] + accu_vz[tf][jz][jx] * uz[iz][ix]);
           
            grad_lam[jz][jx] +=
              snap_dt * dt *
              (((accu_sxx[tf][jz][jx] - (accu_szz[tf][jz][jx] * lam[iz][ix]) / lm) +
                (accu_szz[tf][jz][jx] - (accu_sxx[tf][jz][jx] * lam[iz][ix]) / lm)) *
               ((sxx[iz][ix] - (szz[iz][ix] * lam[iz][ix]) / lm) +
                (szz[iz][ix] - (sxx[iz][ix] * lam[iz][ix]) / lm))) /
              ((lm - ((lam[iz][ix] * lam[iz][ix]) / (lm))) *
               (lm - ((lam[iz][ix] * lam[iz][ix]) / (lm))));
           
            grad_mu[jz][jx] +=
              snap_dt * dt * 2 *
              ((((sxx[iz][ix] - (szz[iz][ix] * lam[iz][ix]) / lm) *
                 (accu_sxx[tf][jz][jx] - (accu_szz[tf][jz][jx] * lam[iz][ix]) / lm)) +
                ((szz[iz][ix] - (sxx[iz][ix] * lam[iz][ix]) / lm) *
                 (accu_szz[tf][jz][jx] - (accu_sxx[tf][jz][jx] * lam[iz][ix]) / lm))) /
                   ((lm - ((lam[iz][ix] * lam[iz][ix]) / (lm))) *
                    (lm - ((lam[iz][ix] * lam[iz][ix]) / (lm)))) +
               2.0 * (szx[iz][ix] * accu_szx[tf][jz][jx] / (4.0 * mu[iz][ix] * mu[iz][ix])));
			*/
			jx++;
        }
		
		jz++;
    }
}

void vsrc2(
    // Velocity tensor arrays
    real **&vz, real **&vx, 
    // inverse of density arrays
    real **&rho_zp, real **&rho_xp,
    // source parameters
    int nsrc, int stf_type, real **&stf_z, real **&stf_x, 
    int *&z_src, int *&x_src, int *&src_shot_to_fire,
    int ishot, int it, real dt, real dz, real dx){
    // firing the velocity source term
    // nsrc: number of sources
    // stf_type: type of source time function (0:displacement, 1:velocity currently implemented)
    // stf_z: source time function z component
    // stf_x: source time function x component
    // z_src: corresponding grid index along z direction
    // x_src: corresponding grid index along x direction
    // it: time step index
    
    //std::cout << "src: " << stf_type <<std::endl;
    switch(stf_type){
    
        case(0): // Displacement stf
            for(int is=0; is<nsrc; is++){
                if (src_shot_to_fire[is] == ishot){

                    //std::cout << "firing shot " << ishot << "::" << stf_z[is][it] <<"::" << stf_x[is][it] << std::endl;;
                    vz[z_src[is]][x_src[is]] += dt*rho_zp[z_src[is]][x_src[is]]*stf_z[is][it]/(dz*dx);
                    vx[z_src[is]][x_src[is]] += dt*rho_xp[z_src[is]][x_src[is]]*stf_x[is][it]/(dz*dx);
                }
                
            }
            break;
        
        case(1): // velocity stf
            for(int is=0; is<nsrc; is++){
                if (src_shot_to_fire[is] == ishot){
                   // std::cout << "firing shot " << ishot << "::" << stf_z[is][it] <<"::" << stf_x[is][it];
                    vz[z_src[is]][x_src[is]] += stf_z[is][it];
                    vx[z_src[is]][x_src[is]] += stf_x[is][it];
                    //std::cout << "v:" << vz[z_src[is]][x_src[is]] <<", " << stf_z[is][it]<<std::endl;
                }
                
            }
            break;
    }
    
}

void urec2(int rtf_type,
    // reciever time functions
    real **&rtf_uz, real **&rtf_ux, 
    // velocity tensors
    real **&vz, real **&vx,
    // reciever 
    int nrec, int *&rz, int *&rx, 
    // time and space grids
    int it, real dt, real dz, real dx){
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
        #pragma omp parallel for
        for(int ir=0; ir<nrec; ir++){//when function is called only one of the case would get executed
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
        //parallel region starts

           #pragma omp parallel for collapse(2) reduction(+: L2)
            for( int is=0; is<nseis; is++){ // for all seismograms
                for(int it=0;it<nt;it++){ // for all time steps

                    
                    // calculating adjoint sources
                    a_stf_uz[is][it] = rtf_uz_mod[is][it] - rtf_uz_true[ishot][is][it];
                  
                    a_stf_ux[is][it] = rtf_ux_mod[is][it] - rtf_ux_true[ishot][is][it];

                    //if (!(abs(a_stf_uz[is][it])<1000.0 || abs(a_stf_uz[is][it])<1000.0)){
                    //    std::cout << rtf_uz_mod[is][it] <<"," << rtf_uz_true[ishot][is][it] << "::";
                    //}
                    

                    // Calculating L2 norm
                   
                    L2 += 0.5 * dt * pow(a_stf_uz[is][it], 2)+ 0.5 * dt * pow(a_stf_ux[is][it], 2); 
                   
                  //  L2 += 0.5 * dt * pow(a_stf_ux[is][it], 2);
                   
                    
                }

                
            }
          
        //paralell region ends
        a_stf_type = &rtf_type; // Calculating displacement adjoint sources
    
    }
    std::cout<< "Calculated norm : " << L2 << std::endl;
    //std::cout << a_stf_type << std::endl;
    return L2;

}

void interpol_grad2(
    // Global and shot gradient
    real **&grad, real **&grad_shot, 
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2, 
    int snap_dz, int snap_dx){
    // Interpolates the gradients to the missing material variables
    int jz, jx;
    real temp_grad; // temporary scalar gradiant

    // --------------------------------------
    // FOR LOOP SET 1
    // -----------------------------------
    jz = 0; 
    for(int iz=snap_z1;iz<=snap_z2;iz+=snap_dz){
        //std::cout<< "[iz: " << iz << ", jz: " << jz << "] ::";
        // Fist filling only the snap grids and along the x-axis
        jx = 0;
        for(int ix=snap_x1;ix<=snap_x2;ix+=snap_dx){
            
            grad[iz][ix] = grad_shot[jz][jx];
            
			jx++;
        }
        
		jz++;

    }

    if(snap_dx>1){
        // now updating the snap rows only
        for(int iz=snap_z1; iz<snap_z2; iz+=snap_dz){
            for(int ix=snap_x1; ix<snap_x2; ix+=snap_dx){
                temp_grad = (grad[iz][ix+snap_dx] - grad[iz][ix])/snap_dx;
                for(int kx=1;kx<snap_dx;kx++){
                    grad[iz][ix+kx] = grad[iz][ix] + temp_grad*kx;
                }
            }
        }
    }
   
    
    if(snap_dz>1){
        // now updating all the columns
        for(int iz=snap_z1; iz<snap_z2; iz+=snap_dz){
            for(int ix=snap_x1; ix<snap_x2; ix++){
                temp_grad = (grad[iz+snap_dz][ix] - grad[iz][ix])/snap_dz;
                for(int kz=1;kz<snap_dz;kz++){
                
                    grad[iz+kz][ix] = grad[iz][ix] + temp_grad*kz;
                }
            
            }
        }
    }   
}


void energy_weights2(
    // Energy Weights (forward and reverse)
    real **&We, real **&We_adj, 
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2){
    // Scale gradients to the Energy Weight
    // We: input as forward energy weight, and output as combined energy weight
    //changes I made 
   //real temp_max_We=0;
    real max_We = 0;
    real max_w1 = 0, max_w2=0;
    real epsilon_We = 0.005; 
   //#pragma omp parallel 
       //parallel region begins
      // #pragma omp parallel 
       //{
        #pragma omp parallel for collapse(2)
            for (int iz=snap_z1;iz<snap_z2;iz++){
                for (int ix=snap_x1;ix<snap_x2;ix++){
                    if (We[iz][ix] > max_w1){
                        max_w1 = We[iz][ix];
                    }
                    if (We_adj[iz][ix] > max_w2){
                        max_w2 = We_adj[iz][ix];
                    }
                    We[iz][ix] = sqrt(We[iz][ix]*We_adj[iz][ix]);
                    
                }
            }
   
            // Finding maximum of the energy weight
            #pragma omp parallel for collapse(2) reduction(max: max_We)
            for (int iz=snap_z1;iz<snap_z2;iz++){
                for (int ix=snap_x1;ix<snap_x2;ix++){
                    
                    // Estimate maximum energy weight in CPU
                    if (We[iz][ix] > max_We){
                        max_We = We[iz][ix];
                    }

                }
            
            }
            //parallel region ends

           // Regularize energy weight to avoid division by zero
           #pragma parallel omp for collapse(2)
            for (int iz=snap_z1;iz<snap_z2;iz++){
            for (int ix=snap_x1;ix<snap_x2;ix++){
                
                We[iz][ix] += epsilon_We *  max_We;
            }
            
        }
       //}
        std::cout << "Max. Energy Weight = " << max_We << std::endl;
        std::cout << "Max. Energy part = " << max_w1<<", "<< max_w2 << std::endl;
    
}


void scale_grad_E2(
    // Gradients, material average and energy weights
    real **&grad, real **&grad_shot, 
    real mat_av, real **&We,
    // space snap parameters
    int snap_z1, int snap_z2, int snap_x1, int snap_x2){
    // Scale gradients to the Energy Weight
    // We: input as forward energy weight, and output as combined energy weight
    // grad and grad shot here have same dimensions (grad_shot = temp grad from each shot)
    // Scale gradients to the energy weight

    if(mat_av>0){
        #pragma omp parallel for collapse(2)
        for (int iz=snap_z1;iz<snap_z2;iz++){
            for (int ix=snap_x1;ix<snap_x2;ix++){      
                grad[iz][ix] += grad_shot[iz][ix] / (We[iz][ix] * mat_av * mat_av);

            }   
        }
    }


}

void update_mat2(real **&mat, real **&mat_old,  real **&grad_mat, 
            real mat_max, real mat_min, real step_length, int nz, int nx){
    // update gradients to the material
    real mat_av=0, mat_av_old=0, mat_av_grad=0;

    // Scale factors for gradients
    real grad_max = 0.0, mat_array_max = 0.0, step_factor;
    #pragma omp parallel for collapse(2) reduction(max: grad_max,mat_array_max)
    for (int iz=0;iz<nz;iz++){
        for (int ix=0;ix<nx;ix++){
            
            grad_max = std::max(grad_max, abs(grad_mat[iz][ix]));
            mat_array_max = std::max(mat_max, abs(mat_old[iz][ix]));

        } 
    }
    step_factor = mat_array_max/grad_max;
    //std::cout << "Update factor: " << step_factor << ", " << mat_max << ", " << grad_max << std::endl;

    // Material update to whole array
    #pragma omp parallel for collapse(2)
    for (int iz=0;iz<nz;iz++){
        for (int ix=0;ix<nx;ix++){
            
            mat[iz][ix] = mat_old[iz][ix] + step_length * step_factor * grad_mat[iz][ix];
            if (mat[iz][ix] > mat_max){ mat[iz][ix] = mat_max;}
            if (mat[iz][ix] < mat_min){ mat[iz][ix] = mat_min;}

            mat_av += mat[iz][ix];
            mat_av_old += mat_old[iz][ix];
            mat_av_grad += grad_mat[iz][ix];

        }
        
    }
    //std::cout << "Mat update: SL = " <<step_length <<", new = " << mat_av <<", old = " << mat_av_old <<", grad = " << mat_av_grad << std::endl;;
}
void copy_mat(real **&lam_copy, real **&mu_copy,  real **&rho_copy,
        real **&lam, real **&mu,  real **&rho, int nz, int nx){

    // Copy material values for storage
    for (int iz=0;iz<nz;iz++){
        for (int ix=0;ix<nx;ix++){
            
            lam_copy[iz][ix] = lam[iz][ix];
            mu_copy[iz][ix] = mu[iz][ix];
            rho_copy[iz][ix] = rho[iz][ix];

        }
        
    }
    

}

void mat_av2(
    // Medium arguments
    real **&lam, real **&mu, real **&rho,
    real **&mu_zx, real **&rho_zp, real **&rho_xp, // inverse of densityint dimz, int dimx
    real &C_lam, real &C_mu, real &C_rho, // scalar averages
    int nz, int nx){
    // Harmonic 2d average of mu and
    // Arithmatic 1d average of rho

    C_lam = 0.0; C_mu = 0.0; C_rho = 0.0;
    for (int iz=0; iz<nz-1; iz++){
        for (int ix=0; ix<nx-1; ix++){
            // Harmonic average for mu
            mu_zx[iz][ix]= 4.0/((1.0/mu[iz][ix])+(1.0/mu[iz][ix+1])
            +(1.0/mu[iz+1][ix])+(1.0/mu[iz+1][ix+1])); 
          
            if((mu[iz][ix]==0.0)||(mu[iz][ix+1]==0.0)||(mu[iz+1][ix]==0.0)||(mu[iz+1][ix+1]==0.0)){ 
                mu_zx[iz][ix]=0.0;
            }
            
            // Arithmatic average of rho
            // the averages are inversed for computational efficiency
            rho_zp[iz][ix] = 1.0/(0.5*(rho[iz][ix]+rho[iz+1][ix]));
            rho_xp[iz][ix] = 1.0/(0.5*(rho[iz][ix]+rho[iz][ix+1]));
          
            if((rho[iz][ix]<1e-4)&&(rho[iz+1][ix]<1e-4)){
                rho_zp[iz][ix] = 0.0;
            }
          
            if((rho[iz][ix]<1e-4)&&(rho[iz][ix+1]<1e-4)){
              rho_zp[iz][ix] = 0.0;
            } 
            // Scalar averages
            C_lam += lam[iz][ix];
            C_mu += mu[iz][ix];
            C_rho += rho[iz][ix];
     
        }

    }

    C_lam = C_lam/((nz-1)*(nx-1));
    C_mu = C_mu/((nz-1)*(nx-1));
    C_rho = C_rho/((nz-1)*(nx-1));

}

void mat_grid2(real **&lam, real **&mu, real **&rho, 
    real lam_sc, real mu_sc, real rho_sc, int nz, int nx){
    // Scalar material value is distributed over the grid

    for (int iz=0;iz<nz;iz++){
        for (int ix=0;ix<nx;ix++){
            lam[iz][ix] = lam_sc;
            mu[iz][ix] = mu_sc;
            rho[iz][ix] = rho_sc;
        }
    }
}


void taper2(real **&A, int nz, int nx,  
    int snap_z1, int snap_z2, int snap_x1, int snap_x2,
    int &taper_t1, int &taper_t2, int &taper_b1, int &taper_b2, 
    int &taper_l1, int &taper_l2, int &taper_r1, int &taper_r2){
    // Applying taper function to the matrix A
    
    int taper_l = taper_l2 - taper_l1;
    int taper_r = taper_r1 - taper_r2;
    int taper_t = taper_t2 - taper_t1;
    int taper_b = taper_b1 - taper_b2;


    // Horizontal taper
    for (int iz=0;iz<nz;iz++){
        for (int ix=0;ix<nx;ix++){
            
            if (ix>=snap_x1 && ix<taper_l1){
                A[iz][ix] *= 0.0;
            }

            else if (ix>=taper_l1 && ix<taper_l2){
                A[iz][ix] *= 0.5*(1.0-cos(PI*(ix-taper_l1)/taper_l));
            }

            else if (ix>taper_r2 && ix<taper_r1){
                A[iz][ix] *= 0.5*(1.0-cos(PI*(taper_r1-ix)/taper_r));
            }

            else if(ix>=taper_r1 && ix<=snap_x2){
                A[iz][ix] *= 0.0;
            }

        }
    }


    // Vertical taper
    for (int ix=0;ix<nx;ix++){
        for (int iz=0;iz<nz;iz++){

            if (iz>=snap_z1 && iz<taper_t1){
                A[iz][ix] *= 0.0;
            }

            else if (iz>=taper_t1 && iz<taper_t2){
                A[iz][ix] *= 0.5*(1.0-cos(PI*(iz-taper_t1)/taper_t));
            }

            else if (iz>taper_b2 && iz<taper_b1){
                A[iz][ix] *= 0.5*(1.0-cos(PI*(taper_b1-iz)/taper_b));
            }

            else if(iz>=taper_b1 && iz<=snap_z2){
                A[iz][ix] *= 0.0;
            }

        }
    }

}


