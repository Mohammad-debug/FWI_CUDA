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



void read_input_metaint(bool &surf, bool &pml_z, bool &pml_x, bool &accu_save, bool &fwinv, bool &rtf_meas_true, 
    int &nt, int &nz, int &nx, int &snap_t1, int &snap_t2, int &snap_z1, int &snap_z2, int &snap_x1, int &snap_x2, 
	int &snap_dt, int &snap_dz, int &snap_dx,int &nsrc, int &nrec, int &nshot, int &stf_type, int &rtf_type, 
	int &fdorder, int &fpad){
	// ------------------------------------------------------------
    // Reads the input data from the binary file created in python
    // ---------------------------------------------------------------------
    // integer values for boolen
    int surf_inp, pml_z_inp, pml_x_inp, accu_save_inp, fwinv_inp, rtf_meas_true_inp;


    // Loading the integer input file "metaint"
    std::ifstream metaint("./bin/metaint.bin", std::ios::in | std::ios::binary);
    if(!metaint) {
        std::cout << "Cannot open input file <METAINT>.";
        return;
    }
    // -----------------------------------------------------------------
    // Boolen values (Temp: as int)
    metaint.read(reinterpret_cast<char*> (&surf_inp), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&pml_z_inp), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&pml_x_inp), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&accu_save_inp), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&fwinv_inp), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&rtf_meas_true_inp), sizeof(int32_t));


    // Assign the actual boolen values from integers
    surf = (surf_inp == 0) ? false : true;
    pml_z = (pml_z_inp == 0) ? false : true;
    pml_x = (pml_x_inp == 0) ? false : true;
    accu_save = (accu_save_inp == 0) ? false : true;
    fwinv = (fwinv_inp == 0) ? false : true;
    rtf_meas_true = (rtf_meas_true_inp == 0) ? false : true;
    // ------------------------------------------------------------------------

    // Geomerty: number of grids
    metaint.read(reinterpret_cast<char*> (&nt), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&nz), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&nx), sizeof(int32_t));
    
    // Geometry: snap grid indices
    metaint.read(reinterpret_cast<char*> (&snap_t1), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&snap_t2), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&snap_z1), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&snap_z2), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&snap_x1), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&snap_x2), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&snap_dt), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&snap_dz), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&snap_dx), sizeof(int32_t));
    
    // Source and reciever parameters
    metaint.read(reinterpret_cast<char*> (&nsrc), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&nrec), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&nshot), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&stf_type), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&rtf_type), sizeof(int32_t));
    
    // Finite difference order and respective padding
    metaint.read(reinterpret_cast<char*> (&fdorder), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&fpad), sizeof(int32_t));
    
    metaint.close();
}

void read_input_int_array(int *&npml, int *&isurf, int *&z_src, int *&x_src, int *&src_shot_to_fire, 
    int *&z_rec, int *&x_rec, int nsrc, int nrec){
	// ------------------------------------------------------------
    // Reads the input data from the binary file created in python
    // ---------------------------------------------------------------------

    // Loading the integer input file "metaint"
    std::ifstream metaint("./bin/intarray.bin", std::ios::in | std::ios::binary);
    if(!metaint) {
        std::cout << "Cannot open input file <METAINTARRAY>.";
        return;
    }
        
    // Number of PML layers in each side
    metaint.read(reinterpret_cast<char*> (&npml[0]), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&npml[1]), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&npml[2]), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&npml[3]), sizeof(int32_t));
    
    // Grid indices for the surface in each side
    metaint.read(reinterpret_cast<char*> (&isurf[0]), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&isurf[1]), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&isurf[2]), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&isurf[3]), sizeof(int32_t));
    std::cout << "check 1."<<std::endl;
    // Reading Source and Receiver Locations
    for (int is = 0; is<nsrc; is++){
        metaint.read(reinterpret_cast<char*> (&z_src[is]), sizeof(int32_t));
    }
    std::cout << "check 2."<<std::endl;
    for (int is = 0; is<nsrc; is++){
        metaint.read(reinterpret_cast<char*> (&x_src[is]), sizeof(int32_t));
    }
    std::cout << "check 3."<<std::endl;
    for (int is = 0; is<nsrc; is++){
        metaint.read(reinterpret_cast<char*> (&src_shot_to_fire[is]), sizeof(int32_t));
    }
    std::cout << "check 4."<<nrec<<z_rec[1]<<std::endl;
    for (int ir = 0; ir<nrec; ir++){
        metaint.read(reinterpret_cast<char*> (&z_rec[ir]), sizeof(int32_t));
    }
    std::cout << "check 5."<<std::endl;
    for (int ir = 0; ir<nrec; ir++){
        metaint.read(reinterpret_cast<char*> (&x_rec[ir]), sizeof(int32_t));
    }
    
    //
    
    // Geometry: Output snap intervals

    // see how many bytes have been read

    std::cout << "The read data is: "<<std::endl;
    std::cout << "The data are: "<< z_src[1] <<", "<< x_src[1]<<", "<< src_shot_to_fire[0] <<", "<< x_rec[1]<<std::endl;
    metaint.close();
}



void read_inp_metafloat(real &dt, real &dz, real &dx, real &npower_pml, real &damp_v_pml, 
    real &rcoef_pml, real &k_max_pml, real &freq_pml, real &scalar_lam, real &scalar_mu, real &scalar_rho){
	// ------------------------------------------------------------
    	// Reads the input data from the binary file created in python (BOOLENS)
    	// ---------------------------------------------------------------------
	
    // Loading the integer input file "metaint"
    std::ifstream metafloat("./bin/metafloat.bin", std::ios::in | std::ios::binary);
    if(!metafloat) {
        std::cout << "Cannot open input file <METAFLOAT>.";
        return;
    }
       
    // Boolen values (Temp: as int)
    metafloat.read(reinterpret_cast<char*> (&dt), sizeof(real));
    metafloat.read(reinterpret_cast<char*> (&dz), sizeof(real));
    metafloat.read(reinterpret_cast<char*> (&dx), sizeof(real));
    // PML variables
    metafloat.read(reinterpret_cast<char*> (&npower_pml), sizeof(real));
    metafloat.read(reinterpret_cast<char*> (&damp_v_pml), sizeof(real));
    metafloat.read(reinterpret_cast<char*> (&rcoef_pml), sizeof(real));
    metafloat.read(reinterpret_cast<char*> (&k_max_pml), sizeof(real));
    metafloat.read(reinterpret_cast<char*> (&freq_pml), sizeof(real));
    // Scalar elastic materials
    metafloat.read(reinterpret_cast<char*> (&scalar_lam), sizeof(real));
    metafloat.read(reinterpret_cast<char*> (&scalar_mu), sizeof(real));
    metafloat.read(reinterpret_cast<char*> (&scalar_rho), sizeof(real));

    std::cout << "The read data is: "<<std::endl;
    std::cout << "The data are: "<< scalar_lam<<", "<< scalar_mu<<", "<< scalar_rho <<std::endl;
    metafloat.close();
    
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


