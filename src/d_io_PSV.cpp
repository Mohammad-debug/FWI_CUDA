//io_PSV.cpp

/* 
* Created by: Min Basnet
* 2020.December.09
* Kathmandu, Nepal
*/

// preprocessing in host
// staggering material & PML coefficients computation

#include "d_io_PSV.hpp"
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>

// Saving Accumulation Array to hard disk binary file
void write_accu(real ***&accu_vz, real ***&accu_vx, 
            real ***&accu_szz, real ***&accu_szx, real ***&accu_sxx, 
            int nt, int nz, int nx, int snap_z1, int snap_z2, int snap_x1, 
            int snap_x2, int snap_dt, int snap_dz, int snap_dx, int ishot){
    // Saves data to bin folder

    int snap_nt = 1+(nt-1)/snap_dt;
    int snap_nz = 1 + (snap_z2 - snap_z1)/snap_dz;
    int snap_nx = 1 + (snap_x2 - snap_x1)/snap_dx;

    std::string fpath = "./bin/shot";

    // saving accumulated tensors
    std::ofstream outfile_vz(fpath+std::to_string(ishot)+"_vz.bin", std::ios::out | std::ios::binary);
    std::ofstream outfile_vx(fpath+std::to_string(ishot)+"_vx.bin", std::ios::out | std::ios::binary);
    std::ofstream outfile_szz(fpath+std::to_string(ishot)+"_szz.bin", std::ios::out | std::ios::binary);
    std::ofstream outfile_szx(fpath+std::to_string(ishot)+"_szx.bin", std::ios::out | std::ios::binary);
    std::ofstream outfile_sxx(fpath+std::to_string(ishot)+"_sxx.bin", std::ios::out | std::ios::binary);
    
    if(!outfile_vz || !outfile_vx || !outfile_szz || !outfile_szx || !outfile_sxx){
        std::cout << "Cannot open output files.";
        return;
    }

    for (int it=0; it<snap_nt; it++){
        for (int iz=0; iz<snap_nz; iz++){
            for(int ix = 0; ix<snap_nx; ix++){
                outfile_vz.write(reinterpret_cast<const char*> (&accu_vz[it][iz][ix]), sizeof(real));
                outfile_vx.write(reinterpret_cast<const char*> (&accu_vx[it][iz][ix]), sizeof(real));
                outfile_szz.write(reinterpret_cast<const char*> (&accu_szz[it][iz][ix]), sizeof(real));
                outfile_szx.write(reinterpret_cast<const char*> (&accu_szx[it][iz][ix]), sizeof(real));
                outfile_sxx.write(reinterpret_cast<const char*> (&accu_sxx[it][iz][ix]), sizeof(real));
            }
        }
    }
   
    outfile_vz.close();
    outfile_vx.close();
    outfile_szz.close();
    outfile_szx.close();
    outfile_sxx.close();



}
