//h_preproc_PSV.hpp

/* 
* Created by: Min Basnet
* 2020.December.09
* Kathmandu, Nepal
*/

// preprocessing in host
// staggering material & PML coefficients computation
#include "h_preproc_PSV.hpp"
#include <math.h>
#include <iostream>
#include <fstream>


void alloc_varpre_PSV( real *&hc, int *&isurf, int *&npml, // holberg coefficients, surface indices and number pml in each side
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
    real **&rtf_z_true, real **&rtf_x_true, // Field measurements for receivers
    // Scalar variables for allocation
    int fdorder, 
    bool pml_z, bool pml_x, int nsrc, int nrec, 
    int nt, int nz, int nx){
    // Allocates the variables that are to be...
    // Transferred from device to the host
    // These Arrays need to take array readings from (device to the host)
    // Every shots should have same number of time steps

    // Allocating holberg coefficient
    hc = new real[fdorder];
    // Allocating surface indices & npml
    isurf = new int[4]; // four sized in the surface
    npml = new int[4]; // number of PMLs in each side
    // Allocating medium arrays
    allocate_array(lam, nz, nx);
    allocate_array(mu, nz, nx);
    allocate_array(rho, nz, nx);
    
    // Allocating PML coeffieients
    if (pml_z){
        a_z = new real[nz];
        b_z = new real[nz];
        K_z = new real[nz];

        a_half_z = new real[nz];
        b_half_z = new real[nz];
        K_half_z = new real[nz];
    }

    if (pml_x){
        a_x = new real[nx];
        b_x = new real[nx];
        K_x = new real[nx];

        a_half_x = new real[nx];
        b_half_x = new real[nx];
        K_half_x = new real[nx];
    }

    // Allocating Source locations and time functions
    if (nsrc){
        // locations (grid index)
        z_src = new int[nsrc];
        x_src = new int[nsrc];
        src_shot_to_fire = new int[nsrc];

        // stf
        allocate_array(stf_z, nsrc, nt);
        allocate_array(stf_x, nsrc, nt);
    }

    if (nrec){
        // locations (grid index)
        z_rec = new int[nrec];
        x_rec = new int[nrec];
       std::cout << "Receivers allocated" << std::endl;
        // rtf field measurements
        allocate_array(rtf_z_true, nrec, nt);
        allocate_array(rtf_x_true, nrec, nt);


    }

}


void mat_stag_PSV(real **&lam, real **&mu, real **&rho, real scalar_lam, real scalar_mu, real scalar_rho, 
                int nz, int nx){
    // Stagger scalar material uniformly the grid
    for (int iz=0; iz<nz; iz++){
        for (int ix=0; ix<nx; ix++){
            // The scalar material variable is staggered over the whole grid
            rho[iz][ix] = scalar_rho;
            lam[iz][ix] = scalar_lam;
            mu[iz][ix] = scalar_mu;
        }
    }
}

void cpml_PSV( real *&a, real *&b, real *&K, 
        real *&a_half, real *&b_half, real *&K_half,
        real npower, real damp_v_PML, real rcoef, real k_max_PML,
        real freq, int npml_h1, int npml_h2, int fpad, int nh, real dt, real dh){
    // Calculates the Convolutional PML factors for the given grid

    // Local CPML variable
    const real alpha_max_PML = 2.0 * PI * (freq/2.0); // from festa and Vilotte 
    real thickness_PML, xorigin_PML, xval;
    real abscissa_in_PML, abscissa_normalized;
    real d0, d, d_half, alpha_prime, alpha_prime_half;
    
    // Initialize the arrays to respective default values
    
    for (int ih=0;ih<nh;ih++){
        a[ih] = 0.0; a_half[ih] = 0.0;
        b[ih] = 1.0; b_half[ih] = 1.0;
        K[ih] = 1.0; K_half[ih] = 1.0;
    }
    

    // --------------------------------------------------------------------------
    // Left/ top / h1 side of the PML layer
    if (npml_h1){
        thickness_PML = npml_h1*dh; // Cartesian thickness of CPML layer in one side

        // zero index in the start of PML Boundary and advancing towards the start of absorption
        xorigin_PML = thickness_PML;

        // compute d0 from INRIA report section 6.1 
        d0 = - (npower + 1) * damp_v_PML * log(rcoef) / (2.0 * thickness_PML);

        // Negative or left side of the boundary
        for(int i=fpad;i<=npml_h1+fpad;i++){

            // Initialize and reset the variables
            abscissa_normalized =0.0;
            d = 0; d_half = 0;
            alpha_prime = 0.0; alpha_prime_half = 0.0;

            xval = dh*real(i); // value of the absissa

            // define damping profile at the grid points 
            abscissa_in_PML = xorigin_PML - xval;
            if (abscissa_in_PML >=0){
                abscissa_normalized = abscissa_in_PML/thickness_PML;
                d = d0 * pow(abscissa_normalized,npower);

                // this taken from Gedney page 8.2 
                K[i] = 1.0 + (k_max_PML - 1.0) * pow(abscissa_normalized,npower);
                alpha_prime = alpha_max_PML * (1.0 - abscissa_normalized);
            }

            // define damping profile at half the grid points 
            abscissa_in_PML =  xorigin_PML - (xval + dh/2.0);
            if (abscissa_in_PML >=0){
                abscissa_normalized = abscissa_in_PML/ thickness_PML;
                d_half = d0 * pow(abscissa_normalized,npower);

                // this taken from Gedney page 8.2 
                K_half[i] = 1.0 + (k_max_PML - 1.0) * pow(abscissa_normalized,npower);
                alpha_prime_half = alpha_max_PML * (1.0 - abscissa_normalized);
            }

            // just in case, for -0.5 at the end 
            if(alpha_prime < 0.0) alpha_prime = 0.0;
            if(alpha_prime_half < 0.0) alpha_prime_half = 0.0;
       
            b[i] = exp(- (d / K[i] + alpha_prime) * dt);
            b_half[i] = exp(- (d_half / K_half[i] + alpha_prime_half) * dt);

            // avoid division by zero outside the PML 
            if(fabs(d) > 1.0e-6){ 
                a[i] = d * (b[i] - 1.0) / (K[i] * (d + K[i] * alpha_prime));
            }

            if(fabs(d_half) > 1.0e-6){ 
                a_half[i] = d_half * (b_half[i] - 1.0) / 
                    (K_half[i] * (d_half + K_half[i] * alpha_prime_half));
            }

        } 

    }// Negative side of the boundary completed

    // ---------------------------------------------------
    // ---------------------------------------------------

    if (npml_h2){
        // Positive or right side of the boundary
        thickness_PML = npml_h2*dh; // Cartesian thickness of CPML layer in one side
       
        // zero index in the start of PML decay and advancing towards the end
        xorigin_PML = dh*(nh-fpad-npml_h2-1);
        for(int i=nh-fpad-npml_h2-1; i<nh-fpad; i++){

            // Initialize and reset the variables
            abscissa_normalized =0.0;
            d = 0; d_half = 0;
            alpha_prime = 0.0; alpha_prime_half = 0.0;

            xval = dh*real(i); // value of the absissa

            // define damping profile at the grid points 
            abscissa_in_PML = xval -xorigin_PML ;
            if (abscissa_in_PML >=0){
                abscissa_normalized = abscissa_in_PML/thickness_PML;
                d = d0 * pow(abscissa_normalized,npower);

                // this taken from Gedney page 8.2 
                K[i] = 1.0 + (k_max_PML - 1.0) * pow(abscissa_normalized,npower);
                alpha_prime = alpha_max_PML * (1.0 - abscissa_normalized);
            }

            // define damping profile at half the grid points 
            abscissa_in_PML =  (xval + dh/2.0) - xorigin_PML;
            if (abscissa_in_PML >=0){
                abscissa_normalized = abscissa_in_PML/ thickness_PML;
                d_half = d0 * pow(abscissa_normalized,npower);

                // this taken from Gedney page 8.2 
                K_half[i] = 1.0 + (k_max_PML - 1.0) * pow(abscissa_normalized,npower);
                alpha_prime_half = alpha_max_PML * (1.0 - abscissa_normalized);
            }

            // just in case, for -0.5 at the end 
            if(alpha_prime < 0.0) alpha_prime = 0.0;
            if(alpha_prime_half < 0.0) alpha_prime_half = 0.0;
       
            b[i] = exp(- (d / K[i] + alpha_prime) * dt);
            b_half[i] = exp(- (d_half / K_half[i] + alpha_prime_half) * dt);

            // avoid division by zero outside the PML 
            if(fabs(d) > 1.0e-6){ 
                a[i] = d * (b[i] - 1.0) / (K[i] * (d + K[i] * alpha_prime));
            }

            if(fabs(d_half) > 1.0e-6){ 
                a_half[i] = d_half * (b_half[i] - 1.0) / 
                    (K_half[i] * (d_half + K_half[i] * alpha_prime_half));
            }

        }

    } // Positive side of the boundary completed

}


// Create Seismic source
void wavelet(real *&signal, int nt, real dt, real amp, real fc, real ts, int shape){
  // Create signal
  // **signal: The array in which signal is to be written
  // nt: number of time steps, dt: time step size, ts: time shift
  // fc: peak frequency, amp: amplitude of the signal

  real t, tau; // time 
  const real fci = 1.0/fc;

  switch(shape){
    case(1): // Ricker wavelet
      for (int it = 0; it<nt; it++){
        t = it * dt;
        tau = PI * (t - 1.5 * fci - ts) / (1.5 * fci);
        signal[it] = amp*(1.0 - 2.0 * tau * tau) * exp(-2.0 * tau * tau);
        //std::cout<<it<<", "<<t<<", "<<signal[it] << std::endl;
      }
  }

} 


