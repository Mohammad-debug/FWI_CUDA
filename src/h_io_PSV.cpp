//h_io_PSV.cpp

/* 
* Created by: Min Basnet
* 2020.December.09
* Kathmandu, Nepal
*/

// preprocessing in host
// staggering material & PML coefficients computation

#include "h_io_PSV.hpp"
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>


void read_input_metaint(bool &gpu_code, bool &surf, bool &pml_z, bool &pml_x, bool &accu_save, bool &seismo_save, bool &fwinv, bool &rtf_meas_true, 
    int &nt, int &nz, int &nx, int &snap_t1, int &snap_t2, int &snap_z1, int &snap_z2, int &snap_x1, int &snap_x2, 
	int &snap_dt, int &snap_dz, int &snap_dx, int &taper_t1, int &taper_t2, int &taper_b1, int &taper_b2, 
    int &taper_l1, int &taper_l2, int &taper_r1, int &taper_r2,
    int &nsrc, int &nrec, int &nshot, int &stf_type, int &rtf_type, 
	int &fdorder, int &fpad, int &mat_save_interval, int &mat_grid){
	// ------------------------------------------------------------
    // Reads the input data from the binary file created in python
    // ---------------------------------------------------------------------
    // integer values for boolen
    int gpu_code_inp, surf_inp, pml_z_inp, pml_x_inp, accu_save_inp, seismo_save_inp, fwinv_inp, rtf_meas_true_inp;


    // Loading the integer input file "metaint"
    std::ifstream metaint("./bin/metaint.bin", std::ios::in | std::ios::binary);
    if(!metaint) {
        std::cout << "Cannot open input file <METAINT>.";
        return;
    }
    // -----------------------------------------------------------------
    // Boolen values (Temp: as int)
    metaint.read(reinterpret_cast<char*> (&gpu_code_inp), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&surf_inp), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&pml_z_inp), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&pml_x_inp), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&accu_save_inp), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&seismo_save_inp), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&fwinv_inp), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&rtf_meas_true_inp), sizeof(int32_t));


    // Assign the actual boolen values from integers
    gpu_code = (gpu_code_inp == 0) ? false : true;
    surf = (surf_inp == 0) ? false : true;
    pml_z = (pml_z_inp == 0) ? false : true;
    pml_x = (pml_x_inp == 0) ? false : true;
    accu_save = (accu_save_inp == 0) ? false : true;
    seismo_save = (seismo_save_inp == 0) ? false : true;
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

    // Geometry: taper locations
    metaint.read(reinterpret_cast<char*> (&taper_t1), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&taper_t2), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&taper_b1), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&taper_b2), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&taper_l1), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&taper_l2), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&taper_r1), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&taper_r2), sizeof(int32_t));
    
    // Source and reciever parameters
    metaint.read(reinterpret_cast<char*> (&nsrc), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&nrec), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&nshot), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&stf_type), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&rtf_type), sizeof(int32_t));
    
    // Finite difference order and respective padding
    metaint.read(reinterpret_cast<char*> (&fdorder), sizeof(int32_t));
    metaint.read(reinterpret_cast<char*> (&fpad), sizeof(int32_t));

    // The iteration intervals to save the updated material in fwi
    metaint.read(reinterpret_cast<char*> (&mat_save_interval), sizeof(int32_t));

    // The material grid available or not 0:scalar material 1:material grid
    metaint.read(reinterpret_cast<char*> (&mat_grid), sizeof(int32_t));

    
    
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
    
    // Reading Source and Receiver Locations
    for (int is = 0; is<nsrc; is++){
        metaint.read(reinterpret_cast<char*> (&z_src[is]), sizeof(int32_t));
    }
    
    for (int is = 0; is<nsrc; is++){
        metaint.read(reinterpret_cast<char*> (&x_src[is]), sizeof(int32_t));
    }
    
    for (int is = 0; is<nsrc; is++){
        metaint.read(reinterpret_cast<char*> (&src_shot_to_fire[is]), sizeof(int32_t));
    }
    
    for (int ir = 0; ir<nrec; ir++){
        metaint.read(reinterpret_cast<char*> (&z_rec[ir]), sizeof(int32_t));
    }
    
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

void read_material_array(real **&lam, real **&mu, real **&rho,  int nz, int nx){
	// ------------------------------------------------------------
    	// Reads the input data from the binary file created in python (BOOLENS)
    	// ---------------------------------------------------------------------
	
    // Loading the integer input file "metaint"
    std::ifstream matfile("./bin/mat.bin", std::ios::in | std::ios::binary);
    if(!matfile) {
        std::cout << "Cannot open input file <MATERIAL>.";
        return;
    }
       
    // Reading  material parameters

    // First parameter
    std::cout << "LAM <FLOAT>."<<std::endl;
    for(int iz=0; iz<nz; iz++){
        for(int ix=0; ix<nx; ix++){
            //std::cout << "LAM <FLOAT>."<<iz <<"," << ix<< std::endl;
            matfile.read(reinterpret_cast<char*> (&lam[iz][ix]), sizeof(real));
        }
    }

    // Second parameter
    std::cout << "MU <FLOAT>."<<std::endl;
    for(int iz=0; iz<nz; iz++){
        for(int ix=0; ix<nx; ix++){
            matfile.read(reinterpret_cast<char*> (&mu[iz][ix]), sizeof(real));
        }
    }

    // Third parameter
    std::cout << "RHO <FLOAT>."<<std::endl;
    for(int iz=0; iz<nz; iz++){
        for(int ix=0; ix<nx; ix++){
            matfile.read(reinterpret_cast<char*> (&rho[iz][ix]), sizeof(real));
        }
    }

    matfile.close();
    
}


// Saving Accumulation Array to hard disk binary file
void write_accu(real ***&accu_vz, real ***&accu_vx, 
            real ***&accu_szz, real ***&accu_szx, real ***&accu_sxx, 
            int nt, int snap_z1, int snap_z2, int snap_x1, 
            int snap_x2, int snap_dt, int snap_dz, int snap_dx, int ishot){
    // Saves data to bin folder

    int snap_nt = 1+(nt-1)/snap_dt;
    int snap_nz = 1 + (snap_z2 - snap_z1)/snap_dz;
    int snap_nx = 1 + (snap_x2 - snap_x1)/snap_dx;

    std::string fpath = "./bin/shot";

    // saving accumulated tensors
    std::ofstream outfile_vz(fpath+std::to_string(ishot)+"_vz.bin", std::ios::out | std::ios::binary);
    std::ofstream outfile_vx(fpath+std::to_string(ishot)+"_vx.bin", std::ios::out | std::ios::binary);
    //std::ofstream outfile_szz(fpath+std::to_string(ishot)+"_szz.bin", std::ios::out | std::ios::binary);
    //std::ofstream outfile_szx(fpath+std::to_string(ishot)+"_szx.bin", std::ios::out | std::ios::binary);
    //std::ofstream outfile_sxx(fpath+std::to_string(ishot)+"_sxx.bin", std::ios::out | std::ios::binary);
    
    if(!outfile_vz || !outfile_vx){ // || !outfile_szz || !outfile_szx || !outfile_sxx){
        std::cout << "Cannot open output files.";
        return;
    }

    for (int it=0; it<snap_nt; it++){
        for (int iz=0; iz<snap_nz; iz++){
            for(int ix = 0; ix<snap_nx; ix++){
                outfile_vz.write(reinterpret_cast<const char*> (&accu_vz[it][iz][ix]), sizeof(real));
                outfile_vx.write(reinterpret_cast<const char*> (&accu_vx[it][iz][ix]), sizeof(real));
                //outfile_szz.write(reinterpret_cast<const char*> (&accu_szz[it][iz][ix]), sizeof(real));
                //outfile_szx.write(reinterpret_cast<const char*> (&accu_szx[it][iz][ix]), sizeof(real));
                //outfile_sxx.write(reinterpret_cast<const char*> (&accu_sxx[it][iz][ix]), sizeof(real));
            }
        }
    }
   
    outfile_vz.close();
    outfile_vx.close();
    //outfile_szz.close();
    //outfile_szx.close();
    //outfile_sxx.close();

}


// Saving Receiver Seismogram to hard disk binary file
void write_seismo(real **&rtf_uz, real **&rtf_ux, 
            int nrec, int nt, int ishot){
    // Saves data to bin folder

    std::string fpath = "./bin/shot";

    // saving accumulated tensors
    std::ofstream outfile_rtf_uz(fpath+std::to_string(ishot)+"_rtf_uz.bin", std::ios::out | std::ios::binary);
    std::ofstream outfile_rtf_ux(fpath+std::to_string(ishot)+"_rtf_ux.bin", std::ios::out | std::ios::binary);
   
    if(!outfile_rtf_uz || !outfile_rtf_ux ){
        std::cout << "Cannot open output files.";
        return;
    }

    for (int ir=0; ir<nrec; ir++){
        for (int it=0; it<nt; it++){
            outfile_rtf_uz.write(reinterpret_cast<const char*> (&rtf_uz[ir][it]), sizeof(real));
            outfile_rtf_ux.write(reinterpret_cast<const char*> (&rtf_ux[ir][it]), sizeof(real));
        }
    }
   
    outfile_rtf_uz.close();
    outfile_rtf_ux.close();

}

// Reading Receiver Seismogram to hard disk binary file
void read_seismo(real ***&rtf_uz, real ***&rtf_ux, 
            int nrec, int nt, int ishot){
    // Saves data to bin folder

    std::string fpath = "./io/shot";
    
    // saving accumulated tensors
    std::ifstream infile_rtf_uz(fpath+std::to_string(ishot)+"_rtf_uz.bin", std::ios::in | std::ios::binary);
    std::ifstream infile_rtf_ux(fpath+std::to_string(ishot)+"_rtf_ux.bin", std::ios::in | std::ios::binary);
  
    if(!infile_rtf_uz || !infile_rtf_ux ){
        std::cout << "Cannot open output files.";
        return;
    }
    
    for (int ir=0; ir<nrec; ir++){
        for (int it=0; it<nt; it++){
            infile_rtf_uz.read(reinterpret_cast<char*> (&rtf_uz[ishot][ir][it]), sizeof(real));
            infile_rtf_ux.read(reinterpret_cast<char*> (&rtf_ux[ishot][ir][it]), sizeof(real));
        }
    }
   
    infile_rtf_uz.close();
    infile_rtf_ux.close();

}

// Reading source Seismogram to hard disk binary file
void read_seismo_src(real **&stf_uz, real **&stf_ux, 
            int nsrc, int nt){
    // Saves data to bin folder

    std::string fpath = "./io/";
    
    // saving accumulated tensors
    std::ifstream infile_stf_uz(fpath+"stf_vz.bin", std::ios::in | std::ios::binary);
    std::ifstream infile_stf_ux(fpath+"stf_vx.bin", std::ios::in | std::ios::binary);
  
    if(!infile_stf_uz || !infile_stf_ux ){
        std::cout << "Cannot open output files.";
        return;
    }
    
    for (int ir=0; ir<nsrc; ir++){
        for (int it=0; it<nt; it++){
            infile_stf_uz.read(reinterpret_cast<char*> (&stf_uz[ir][it]), sizeof(real));
            infile_stf_ux.read(reinterpret_cast<char*> (&stf_ux[ir][it]), sizeof(real));
        }
    }
   
    infile_stf_uz.close();
    infile_stf_ux.close();

}


// Saving Material Arrays to the hard disk binary file
void write_mat(real **&lam, real **&mu, real **&rho, int nz, int nx, int iterstep){
    // Saves data to bin folder


    std::string fpath = "./bin/iter";

    // saving accumulated tensors
    std::ofstream outfile_mat(fpath+std::to_string(iterstep)+"_mat.bin", std::ios::out | std::ios::binary);
    
    if(!outfile_mat){
        std::cout << "Cannot open output files.";
        return;
    }


    for (int iz=0; iz<nz; iz++){
        for(int ix = 0; ix<nx; ix++){
            outfile_mat.write(reinterpret_cast<const char*> (&lam[iz][ix]), sizeof(real)); 
        }
    }

    for (int iz=0; iz<nz; iz++){
        for(int ix = 0; ix<nx; ix++){
            outfile_mat.write(reinterpret_cast<const char*> (&mu[iz][ix]), sizeof(real)); 
        }
    }

    for (int iz=0; iz<nz; iz++){
        for(int ix = 0; ix<nx; ix++){
            outfile_mat.write(reinterpret_cast<const char*> (&rho[iz][ix]), sizeof(real)); 
        }
    }
   
    outfile_mat.close();

}