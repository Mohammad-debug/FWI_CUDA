//h_simulate_PSV.hpp

/* 
* Created by: Min Basnet
* 2020.December.09
* Kathmandu, Nepal
*/

#include "h_simulate_PSV.hpp"
#include <iostream>
#include <math.h>
#include <omp.h>

// ----------------------------------
// HOST GLOBAL SIMULATION VARIABLES
// ---------------------------------
bool GPU_CODE;
// Device global variables to be copied from the host
// --------------------------
// GEOMETRIC PARAMETERS
// ------------------------
// Geometric grid
int NT, NZ, NX;
real DT, DZ, DX;

// Snap boundary and intervals for saving memory
int SNAP_T1, SNAP_T2, SNAP_DT;
int SNAP_Z1, SNAP_Z2, SNAP_DZ;
int SNAP_X1, SNAP_X2, SNAP_DX;

// Taper locations
int TAPER_T1, TAPER_T2, TAPER_B1, TAPER_B2;
int TAPER_L1, TAPER_L2, TAPER_R1, TAPER_R2;

// Surface conditions
bool SURF;  // if surface exists in any side
int *ISURF; // surface indices on 0.top, 1.bottom, 2.left, 3.right

// PML conditions
bool PML_Z, PML_X; // if pml exists in respective directions

// ------------------
// SEISMOGRAMS
// ------------------
// Numbers of seismograms
int NSHOT, NSRC, NREC; // number of shots, sources & recievers

int *Z_SRC, *X_SRC;    // source coordiante indices
int *SRC_SHOT_TO_FIRE; // which source to fire in which shot
int *Z_REC, *X_REC;    // source coordinate indices

// stf and rtf
int STF_TYPE, RTF_TYPE;            // time function types 1.displacement
real **STF_Z, **STF_X;             // source time functions
real ***RTF_Z_TRUE, ***RTF_X_TRUE; // rtf (measured (true) values)
bool RTF_MEAS_TRUE;                // if the true measurement exists

// simulation parameters
bool FWINV;            // 0.fwd, 1.fwu
int FDORDER, FPAD;     // FD order and the numerical padding in each side
real *HC;              // Holberg coefficients
bool ACCU_SAVE;        // if to write accumulated tensor storage to disk
bool SEISMO_SAVE;      // Saving the recorder seismograms
int MAT_SAVE_INTERVAL; // The iteration step interval for material save
int MAT_GRID;          // if the grid material is available 0: scalar only 1: material grid

// --------------------------
// MATERIAL PARAMETERS
// -----------------------------
real SCALAR_LAM, SCALAR_MU, SCALAR_RHO; // Scalar elastic materials
real **LAM, **MU, **RHO;                // Initial material arrays

// ------------------------
// PML COEFFICIENTS
// -------------------------

real *A_Z, *B_Z, *K_Z, *A_HALF_Z, *B_HALF_Z, *K_HALF_Z; // PML coefficient z-dir
real *A_X, *B_X, *K_X, *A_HALF_X, *B_HALF_X, *K_HALF_X; // PML coefficient x-dir

// ------------------------
// HOST ONLY PARAMETERS
// -------------------------

// PML parameters not to be transferred to the device
// ------------------------------------------------------------
real h_NPOWER, h_DAMP_V_PML, h_RCOEF, h_K_MAX_PML, h_FREQ_PML; // pml FACTORS
int *h_NPML;                                                   // number of PML grids in each side

// -----
bool ACCU = true, GRAD = false;

void simulate_PSV()
{
    //
    // ------------------------------------------------------------------
    // A. PREPROCESSING IN HOST
    // ----------------------------------------------------------------

    // Reading input integer parameters
    std::cout << "Reading input parameters <INT>." << std::endl;
    read_input_metaint(GPU_CODE, SURF, PML_Z, PML_X, ACCU_SAVE, SEISMO_SAVE, FWINV, RTF_MEAS_TRUE, NT, NZ, NX,
                       SNAP_T1, SNAP_T2, SNAP_Z1, SNAP_Z2, SNAP_X1, SNAP_X2, SNAP_DT, SNAP_DZ, SNAP_DX,
                       TAPER_T1, TAPER_T2, TAPER_B1, TAPER_B2, TAPER_L1, TAPER_L2, TAPER_R1, TAPER_R2,
                       NSRC, NREC, NSHOT, STF_TYPE, RTF_TYPE, FDORDER, FPAD, MAT_SAVE_INTERVAL, MAT_GRID);

    // Allocate the arrays for preprocessing in host
    std::cout << "Allocating memory for the variables." << std::endl;
    alloc_varpre_PSV(HC, ISURF, h_NPML, LAM, MU, RHO, A_Z, B_Z, K_Z, A_HALF_Z, B_HALF_Z, K_HALF_Z,
                     A_X, B_X, K_X, A_HALF_X, B_HALF_X, K_HALF_X, Z_SRC, X_SRC, SRC_SHOT_TO_FIRE, STF_Z, STF_X,
                     Z_REC, X_REC, RTF_Z_TRUE, RTF_X_TRUE, FDORDER, PML_Z, PML_X, NSHOT, NSRC, NREC, NT, NZ, NX);

    // Reading integer arrays from input
    std::cout << "Reading input parameters <INTEGER ARAYS>." << std::endl;
    read_input_int_array(h_NPML, ISURF, Z_SRC, X_SRC, SRC_SHOT_TO_FIRE, Z_REC, X_REC, NSRC, NREC);

    std::cout << "Reading input parameters <FLOAT>." << std::endl;
    read_inp_metafloat(DT, DZ, DX, h_NPOWER, h_DAMP_V_PML, h_RCOEF, h_K_MAX_PML, h_FREQ_PML, SCALAR_LAM, SCALAR_MU, SCALAR_RHO);

    if (FWINV)
    {
        if (RTF_MEAS_TRUE)
        {
            std::cout << "Reading field measurements <FLOAT>." << std::endl;
            for (int ishot = 0; ishot < NSHOT; ishot++)
            {

                read_seismo(RTF_Z_TRUE, RTF_X_TRUE, NREC, NT, ishot); // reading for each shot at a time
            }
        }
        else
        {
            std::cout << "No field measurement exists. UNABLE to run FWI simulation." << std::endl;
            exit(0);
        }
    }

    if (MAT_GRID)
    {
        // Reading material over grid
        std::cout << "Reading material grids <FLOAT>." << std::endl;
        read_material_array(LAM, MU, RHO, NZ, NX);
    }
    else
    {
        // Staggering the scalar material to the grid
        std::cout << "Staggering scalar material over grids." << std::endl;
        mat_stag_PSV(LAM, MU, RHO, SCALAR_LAM, SCALAR_MU, SCALAR_RHO, NZ, NX);
    }

    std::cout << "Input parameters loaded." << std::endl;

    // MATERIAL INPUT

    // Computation of PML Coefficients in z and x directions
    if (PML_Z)
    {

        cpml_PSV(A_Z, B_Z, K_Z, A_HALF_Z, B_HALF_Z, K_HALF_Z, h_NPOWER, h_DAMP_V_PML, h_RCOEF, h_K_MAX_PML,
                 h_FREQ_PML, h_NPML[0], h_NPML[1], FPAD, NZ, DT, DZ);
    }
    if (PML_X)
    {

        cpml_PSV(A_X, B_X, K_X, A_HALF_X, B_HALF_X, K_HALF_X, h_NPOWER, h_DAMP_V_PML, h_RCOEF, h_K_MAX_PML,
                 h_FREQ_PML, h_NPML[2], h_NPML[3], FPAD, NX, DT, DX);
    }

    // Value of Holberg coefficient
    const int MAXRELERROR = 1;
    holbergcoeff(FDORDER, MAXRELERROR, HC);

    std::cout << "nz = " << NZ << ", nx = " << NX << ", dz = " << DZ << ", dt = " << DT
              << ", freq = " << h_FREQ_PML << ", npml = " << h_NPML[1]  << std::endl;

    // STABILITY AND DISPERSION CHECK
    checkfd_ssg_elastic(NZ, NX, DZ, DT, h_FREQ_PML, h_NPML[1], RHO, LAM, MU, HC);

   
    // Creating source wavelet
    //read_seismo_src(STF_Z, STF_X, NSRC, NT);
    
    for (int is=0;is<NSRC;is++){
        wavelet(STF_Z[is], NT, DT, 0.0, h_FREQ_PML, 0.0, 1); // Creating one Ricker wavelet 
        wavelet(STF_X[is], NT, DT, 1.0, h_FREQ_PML, 0.0, 1); // zero amplitude wavelet
    }
    
    //std::cout << "Error here 4"<<std::endl;
    // --------------------------------------------------------------------------
    // B. MEMORY COPY TO THE DEVICE
    // -------------------------------------------------------------------------
    // For the CPU only code the host variables (ALL CAPS) is used to call
    // For the device code the memory has to be copied from host varibales (ALL CAPS) to lowercase device variables

    // --------------------------------------------------------------------------
    // C. MEMORY COPY TO THE DEVICE
    // -------------------------------------------------------------------------
    
    if (GPU_CODE)
    {
        // Calling now the device codes
        std::cout << "THE COMPUTATION STARTS IN GPU" << std::endl;

        //
/*
        g_simulate_PSV(h_NPML, NT, NZ, NX, DT, DZ, DX, SNAP_Z1, SNAP_Z2, SNAP_X1, SNAP_X2, SNAP_DT, SNAP_DZ, SNAP_DX,
                       SURF, PML_Z, PML_X, NSRC, NREC, NSHOT, STF_TYPE, RTF_TYPE, FDORDER, SCALAR_LAM, SCALAR_MU, SCALAR_RHO,
                       HC, ISURF, LAM, MU, RHO, A_Z, B_Z, K_Z, A_HALF_Z, B_HALF_Z, K_HALF_Z, A_X, B_X, K_X, A_HALF_X, B_HALF_X,
                       K_HALF_X, Z_SRC, X_SRC, Z_REC, X_REC, SRC_SHOT_TO_FIRE, STF_Z, STF_X,
                       ACCU_SAVE, SEISMO_SAVE,
                       RTF_Z_TRUE, RTF_X_TRUE, MAT_SAVE_INTERVAL, TAPER_T1, TAPER_T2, TAPER_B1, TAPER_B2,
                       TAPER_L1, TAPER_L2, TAPER_R1, TAPER_R2, FWINV);
*/
    }
    else
    {
        // Calling the codes in the host only
        std::cout << "THE COMPUTATION STARTS IN CPU" << std::endl;
//warm up;
#pragma omp parallel
        ;

        printf("**OpenMP running with %d threads**\n", omp_get_max_threads());

        if (FWINV)
        {
            // Full Waveform Inversion
            std::cout << "Full Waveform Inversion...." << std::endl;
            //
            double dif;
            double start = omp_get_wtime();

            simulate_fwi_PSV(NT, NZ, NX, DT, DZ, DX, SNAP_Z1, SNAP_Z2, SNAP_X1, SNAP_X2, SNAP_DT, SNAP_DZ, SNAP_DX,
                             SURF, PML_Z, PML_X, NSRC, NREC, NSHOT, STF_TYPE, RTF_TYPE, FDORDER, SCALAR_LAM, SCALAR_MU, SCALAR_RHO,
                             HC, ISURF, LAM, MU, RHO, A_Z, B_Z, K_Z, A_HALF_Z, B_HALF_Z, K_HALF_Z, A_X, B_X, K_X, A_HALF_X, B_HALF_X,
                             K_HALF_X, Z_SRC, X_SRC, Z_REC, X_REC, SRC_SHOT_TO_FIRE, STF_Z, STF_X,
                             RTF_Z_TRUE, RTF_X_TRUE, MAT_SAVE_INTERVAL, TAPER_T1, TAPER_T2, TAPER_B1, TAPER_B2,
                             TAPER_L1, TAPER_L2, TAPER_R1, TAPER_R2);

            double end = omp_get_wtime(); // end the timer
            dif = end - start;          // stores the difference in dif
            std::cout << "the time of CPU = " << dif  << " us\n";
        }

        else
        {
            // Forward Modelling
            std::cout << "Forward Modelling only...." << std::endl;
            //
            simulate_fwd_PSV(NT, NZ, NX, DT, DZ, DX, SNAP_Z1, SNAP_Z2, SNAP_X1, SNAP_X2, SNAP_DT, SNAP_DZ, SNAP_DX,
                             SURF, PML_Z, PML_X, NSRC, NREC, NSHOT, STF_TYPE, RTF_TYPE, FDORDER, SCALAR_LAM, SCALAR_MU, SCALAR_RHO,
                             HC, ISURF, LAM, MU, RHO, A_Z, B_Z, K_Z, A_HALF_Z, B_HALF_Z, K_HALF_Z, A_X, B_X, K_X, A_HALF_X, B_HALF_X,
                             K_HALF_X, Z_SRC, X_SRC, Z_REC, X_REC, SRC_SHOT_TO_FIRE, STF_Z, STF_X,
                             ACCU_SAVE, SEISMO_SAVE);
        }
    }

    std::cout << "Simulation Complete" << std::endl;
}
