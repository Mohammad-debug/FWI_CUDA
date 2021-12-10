/*
   Holberg coefficients for a certain FD order and a margin of error E
   (MAXRELERROR)

   MAXRELERROR = 0 -> Taylor-coeff.
   MAXRELERROR = 1 -> Holberg-coeff.: E = 0.1 %
   MAXRELERROR = 2 ->                 E = 0.5 %
   MAXRELERROR = 3 ->                 E = 1.0 %
   MAXRELERROR = 4 ->                 E = 3.0 %
 
  hc: column 0 = minimum number of grid points per shortest wavelength
      columns 1-6 = Holberg coefficients
     
*/
#ifndef H_HOLBERG_CPP		
#define H_HOLBERG_CPP

//#include "fd.h"           /* general include file for viscoelastic FD programs */
#include <cmath>
#include <iostream>
#include "h_globvar.hpp"

void holbergcoeff(const int FDORDER, const int MAXRELERROR, real *hc) ;

#endif