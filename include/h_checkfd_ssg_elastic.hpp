/*-------------------------------------------------------------
 *  Check FD-Grid for stability and grid dispersion.
 *  If the stability criterion is not fullfilled the program will
 *  terminate.                   
 *  last update  03/04/2004 
 *
 *  ----------------------------------------------------------*/
#ifndef H_STABILITY_CPP		
#define H_STABILITY_CPP

//#include "fd.h"
#include <cmath>
#include <iostream>
#include "h_globvar.hpp"

void checkfd_ssg_elastic(int NX, int NY, real DH, real DT, real TS, int FW, 
	real ** prho, real ** ppi, real ** pu, real *hc);

#endif