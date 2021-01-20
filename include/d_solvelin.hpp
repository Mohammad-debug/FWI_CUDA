/*
 *------------------------------------------------------------------------
 *
 *   Solve linear equation systems
 *
 *------------------------------------------------------------------------
*/

#ifndef SOLVELIN_H	
#define SOLVELIN_H
#include <iostream>

template <class T>
void solvelin(T  A[3][3], T b[3], T x[3], int e, int method){

	/* local variables */
	int k, m, n, rows, columns;
	T a, c;

	
	rows = e;
	columns = e;

	switch (method)
	{
	case 1:	/* Gau� algorithm */
	{
		for (k=0;k<rows-1;k++)
			for (n=k;n<rows-1;n++)
			{
				a = A[n+1][k]/A[k][k];
				for (m=0;m<columns;m++) A[n+1][m] = A[n+1][m] - a*A[k][m];
				b[n+1] = b[n+1] - a*b[k];
			 //std::cout <<std::endl << std::endl << "CHK:" <<a <<", " << b[k] << std::endl;
			}
		
		for (k=rows;k>=0;k--)
		{
			c = b[k];
			for (m=columns;m>=k;m--) c = c - A[k][m]*x[m];
			x[k] = c/A[k][k];
		}
		break;
	} /* END of case Gau� */
		
	} /* END of switch (method) */
	

	return;
}

#endif




