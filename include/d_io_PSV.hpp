//io_PSV.hpp

/* 
* Created by: Min Basnet
* 2020.December.09.13
* Kathmandu, Nepal
*/

#ifndef IO_PSV_H		
#define IO_PSV_H	


#include "d_globvar.hpp"

using real = double;
#define PI 3.14159265


// Writing the accumulative data for velocity and stress tensor to the disk
void write_accu(real ***&accu_vz, real ***&accu_vx, 
            real ***&accu_szz, real ***&accu_szx, real ***&accu_sxx, 
            int nt, int nz, int nx, int snap_z1, int snap_z2, int snap_x1, 
            int snap_x2, int snap_dt, int snap_dz, int snap_dx, int ishot);
            


                
#endif
