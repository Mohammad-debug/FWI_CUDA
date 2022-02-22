// h_simulate_PSV.hpp

/* 
* Created by: Min Basnet
* 2020.December.21
* Kathmandu, Nepal
*/

#ifndef H_SIMULATE_PSV_H		
#define H_SIMULATE_PSV_H	

#include "h_globvar.hpp"
#include "h_preproc_PSV.hpp"
#include "h_io_PSV.hpp"
#include "h_checkfd_ssg_elastic.hpp"
#include "h_holbergcoeff.hpp"

#include "n_simulate_PSV.hpp"
#include "d_simulate_PSV.cuh"


// The function call for PSV simulation
void simulate_PSV();


#endif
