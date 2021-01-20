//h_io_PSV.hpp

/* 
* Created by: Min Basnet
* 2020.December.09.13
* Kathmandu, Nepal
*/

#ifndef IO_PSV_H		
#define IO_PSV_H	


#include "h_globvar.hpp"

using real = double;
#define PI 3.14159265


void read_input_metaint(bool &surf, bool &pml_z, bool &pml_x, bool &accu_save, bool &seismo_save, bool &fwinv, bool &rtf_meas_true, 
    int &nt, int &nz, int &nx, int &snap_t1, int &snap_t2, int &snap_z1, int &snap_z2, int &snap_x1, int &snap_x2, 
	int &snap_dt, int &snap_dz, int &snap_dx,int &nsrc, int &nrec, int &nshot, int &stf_type, int &rtf_type, 
	int &fdorder, int &fpad, int &mat_save_interval, int &mat_grid);

void read_input_int_array(int *&npml, int *&isurf, int *&z_src, int *&x_src, int *&src_shot_to_fire, 
    int *&z_rec, int *&x_rec, int nsrc, int nrec);

void read_inp_metafloat(real &dt, real &dz, real &dx, real &npower_pml, real &damp_v_pml, 
    real &rcoef_pml, real &k_max_pml, real &freq_pml, real &scalar_lam, real &scalar_mu, real &scalar_rho);

void read_material_array(real **&lam, real **&mu, real **&rho,  int nz, int nx);

void read_seismo(real **&rtf_uz, real **&rtf_ux, int nrec, int nt, int ishot);


// Writing the accumulative data for velocity and stress tensor to the disk
//void write_accu(real ***&accu_vz, real ***&accu_vx, 
void write_accu(real ***&accu_vz, real ***&accu_vx, 
            real ***&accu_szz, real ***&accu_szx, real ***&accu_sxx, 
            int nt, int snap_z1, int snap_z2, int snap_x1, 
            int snap_x2, int snap_dt, int snap_dz, int snap_dx, int ishot);

// Saving Receiver Seismogram to hard disk binary file
void write_seismo(real **&rtf_uz, real **&rtf_ux, 
            int nrec, int nt, int ishot);

void write_mat(real **&lam, real **&mu, real **&rho, int nz, int nx, int iterstep);
                
#endif
