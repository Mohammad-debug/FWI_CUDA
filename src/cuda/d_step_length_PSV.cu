//d_step_length_PSV.cpp

/* 
* Created by: Min Basnet
* 2021.January.09
* Kathmandu, Nepal
*/

// Step length estimation for the computation of optimum step length for gradiant update
#include "d_step_length_PSV.cuh"
#include <iostream>



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



real step_length_PSV_GPU(real est_step_length, real L2_norm_0, int nshot, // shot index
                int nt, int nz, int nx, // Time and space grid arguments
                real dt, real dx, real dz, 
                // surface incides (0.top, 1.bottom, 2.left, 3.right)
                bool surf, int *&isurf,
                // computationsl arguments
                real *&hc, int fdorder, 
                // Wave arguments (velocity, displacement and stress tensors)
                real *&vz, real *&vx,  real *&uz, real *&ux, 
                real *&szz, real *&szx, real *&sxx,  real *&We,
                // Spatial derivatives (for internal computations)
                real *&dz_z, real *&dx_z, real *&dz_x, real *&dx_x, 
                // Medium arguments
                real *&lam, real *&mu, real *&rho,
                // Medium arguments old copy
                real *&lam_copy, real *&mu_copy, real *&rho_copy,
                real *&mu_zx, real *&rho_zp, real *&rho_xp, // inverse of density
                // Gradients of the medium
                bool grad, real *&grad_lam, real *&grad_mu, real *&grad_rho,
                //PML arguments
                bool pml_z, real *&a_z, real *&b_z, real *&K_z, 
                real *&a_half_z, real *&b_half_z, real *&K_half_z,
                bool pml_x, real *&a_x, real *&b_x, real *&K_x, 
                real *&a_half_x, real *&b_half_x, real *&K_half_x, 
                // PML memory arrays
                real *&mem_vz_z, real *&mem_vx_z, real *&mem_szz_z, real *&mem_szx_z, 
                real *&mem_vz_x, real *&mem_vx_x, real *&mem_szx_x, real *&mem_sxx_x,
                // Seismic sources
                int nsrc, int stf_type, real *&stf_z, real *&stf_x, 
                int *&z_src, int *&x_src, int *&src_shot_to_fire,
                // Reciever seismograms
                int nrec, int rtf_type, real *&rtf_uz, real *&rtf_ux, int *&z_rec, int *&x_rec,
                real *&rtf_z_true, real *&rtf_x_true,
                // Accumulate the snap of forward wavefield parameters
                bool accu, real *&accu_vz, real *&accu_vx, //accumulated velocity memory over time
                real *&accu_szz, real *&accu_szx, real *&accu_sxx, //accumulated velocity memory over time
                int snap_z1, int snap_z2, int snap_x1, int snap_x2, // grid boundaries for fwi
                int snap_dt, int snap_dz, int snap_dx, // time n space grid intervals to save storage
                int update_param){

	real step_factor_rho = 0.5; // Scale factor for updating density
    real scalefac = 2.0; // Step factor in next approximation
    real stepmax_1 = 15; // The maximum number of steps to find optimum step length
    real stepmax_2 = 4;
    
    int *a_stf_type;
    real scalar_lam_local, scalar_mu_local, scalar_rho_local;

	real L2_test[3];
	real step_length[3];
    L2_test[0] = L2_norm_0; // Save for step length approximateion
    //L2_test[2] = L2_norm_0; // Save for step length approximation
    //step_length[0] = est_step_length;
    //est_step_length = 0.01;

    real L2_tmp = 0;

    // controllers for steps in approximation
    unsigned int step1 = 0, step2 = 0;
    //unsigned int step3;
    unsigned int itests = 1, iteste = 1; // initialize start and end of test
    unsigned int countstep = 0;

	// multiple material test checks to calculate L2 norms for these changes
	// three tests performed currently
    int nshot_rep = 2;
    // number of representative shots (use maximum 2 shots)
    //if (nshot < nshot_rep) 
    nshot_rep = nshot;

    while ((step2!=1) || (step1!=1)){
	   for (unsigned int itest = itests; itest <= iteste; itest++){
		    //
		    // Material update test
            if(update_param == 0 || update_param == 1){
                update_mat2_GPU(lam, lam_copy, grad_lam, 4.8e+10, 0.0, est_step_length, nz, nx);
                update_mat2_GPU(mu, mu_copy, grad_mu, 2.7e+10, 0.0, est_step_length, nz, nx);
            }

            if(update_param == 0 || update_param == 2){
                update_mat2_GPU(rho, rho_copy, grad_rho, 3000.0, 1.5, step_factor_rho*est_step_length, nz, nx);
            }
            

            // calculate material average
            mat_av2_GPU(lam, mu, rho, mu_zx, rho_zp, rho_xp, 
                scalar_lam_local, scalar_mu_local, scalar_rho_local, nz, nx);
            L2_tmp = 0;
            for (int ishot=0;ishot<nshot_rep;ishot++){
                // Now calling forward kernel with updated material
                accu = true; // Accumulated storage for output
                grad = false; // no gradient computation in forward kernel
                kernel_PSV_GPU(ishot, nt, nz, nx, dt, dx, dz, surf, isurf, hc, fdorder, 
                    vz, vx,  uz, ux, szz, szx, sxx, We, dz_z, dx_z, dz_x, dx_x, 
                    lam, mu, mu_zx, rho_zp, rho_xp, grad, grad_lam, grad_mu, grad_rho,
                    pml_z, a_z, b_z, K_z, a_half_z, b_half_z, K_half_z,
                    pml_x, a_x, b_x, K_x, a_half_x, b_half_x, K_half_x, 
                    mem_vz_z, mem_vx_z, mem_szz_z, mem_szx_z,
                    mem_vz_x, mem_vx_x, mem_szx_x, mem_sxx_x,
                    nsrc, stf_type, stf_z, stf_x, z_src, x_src, src_shot_to_fire,
                    nrec, rtf_type, rtf_uz, rtf_ux, z_rec, x_rec,
                    accu, accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx, 
                    snap_z1, snap_z2, snap_x1, snap_x2, 
                    snap_dt, snap_dz, snap_dx);

                // calculating L2 norm and adjoint sources
                L2_tmp += adjsrc2_GPU(ishot, a_stf_type, rtf_uz, rtf_ux, rtf_type, rtf_z_true, rtf_x_true,
                        rtf_uz, rtf_ux, dt, nrec, nt);

            }   
            L2_test[itest] = L2_tmp;
            step_length[itest] = est_step_length;
	        step_length[0] = 0.0;
        } 
        std::cout << "Step Length: " << est_step_length << ", L2 = " << L2_tmp <<", counter = "<< countstep << std::endl;
        std::cout << "L2_test = [" << L2_test[0] << ", " << L2_test[1] << ", " << L2_test[2] << " ]" <<std::endl;
        // multiple tests performed

        // Different conditions arise here, which need to be calculated

        /* Did not found a step size which reduces the misfit function */
	    if((step1==0)&&(L2_test[0] <= L2_test[1])){
	        est_step_length = est_step_length/scalefac; 
	        countstep++;
            std::cout << "CASE 1.1" << std::endl;
	    }

        /* Found a step size with L2t[2] < L2t[3]*/
	    if((step1==1)&&(L2_test[1]<L2_test[2])){
	        step_length[2]=est_step_length;
	        step2=1;
            std::cout << "CASE 2.2" << std::endl;
	    }

        /* Could not found a step size with L2t[2] < L2t[3]*/
	    if((step1==1)&&(L2_test[1]>=L2_test[2])){
	        step_length[2]=est_step_length;
	        /* increase step length to find  a larger misfit function than L2t[2]*/
	        est_step_length = est_step_length + (est_step_length/scalefac);
	        countstep++;     
            std::cout << "CASE 2.1" << std::endl;                  
	    }   

        

        /* found a step size which reduces the misfit function */
	    if((step1==0)&&(L2_test[0]>L2_test[1])){
	        step_length[1]=est_step_length; 
	        step1=1;
	        iteste=2;
	        itests=2;
	        countstep=0;
	        /* find a second step length with a larger misfit function than L2t[2]*/
	        est_step_length = est_step_length + (est_step_length/scalefac);
            std::cout << "CASE 1.2" << std::endl;
	    }


        // *step3=0;

	    if((step1==0)&&(countstep>stepmax_1)){
            std::cout << " Steplength estimation failed!"<<std::endl; 
	        //*step3=1;
	        //break;
            exit(0);
	    }

        if((step1==1)&&(countstep>stepmax_2)){
	        std::cout << "Could not found a proper 3rd step length which brackets the minimum" <<std::endl;
	        step1=1;
	        step2=1;
	    }
        std::cout << "-------------------------------------------------------" << std::endl;
        std::cout << "Step Length: " << est_step_length << ", L2 = " << L2_tmp <<", counter = "<< countstep << std::endl;
        std::cout << "step1 = " << step1 <<", step2 = " << step2 <<" itests = " << itests <<" iteste = " << iteste  <<std::endl;

        break;/////temp

    }
        
    if(step1==1){ /* only find an optimal step length if step1==1 */

        if(update_param == 0 || update_param == 1){
		    std::cout << "=================================================" <<std::endl;
		    std::cout << "calculate optimal step length epsilon for Vp and Vs" <<std::endl;
		    std::cout << "================================================="  <<std::endl;
        }
        if(update_param == 2){
            std::cout << "=================================================" <<std::endl;
		    std::cout << "calculate optimal step length epsilon for rho" <<std::endl;
		    std::cout << "================================================="  <<std::endl;
        }
	
	    est_step_length=calc_opt_step_GPU(L2_test,step_length);
    }
    
	return est_step_length;
}


/*
real step_length_PSV1(real est_step_length, real L2_norm_0, int nshot, // shot index
                int nt, int nz, int nx, // Time and space grid arguments
                real dt, real dx, real dz, 
                // surface incides (0.top, 1.bottom, 2.left, 3.right)
                bool surf, int *&isurf,
                // computationsl arguments
                real *&hc, int fdorder, 
                // Wave arguments (velocity, displacement and stress tensors)
                real **&vz, real **&vx,  real **&uz, real **&ux, 
                real **&szz, real **&szx, real **&sxx,  real **&We,
                // Spatial derivatives (for internal computations)
                real **&dz_z, real **&dx_z, real **&dz_x, real **&dx_x, 
                // Medium arguments
                real **&lam, real **&mu, real **&rho,
                // Medium arguments old copy
                real **&lam_copy, real **&mu_copy, real **&rho_copy,
                real **&mu_zx, real **&rho_zp, real **&rho_xp, // inverse of density
                // Gradients of the medium
                bool grad, real **&grad_lam, real **&grad_mu, real **&grad_rho,
                //PML arguments
                bool pml_z, real *&a_z, real *&b_z, real *&K_z, 
                real *&a_half_z, real *&b_half_z, real *&K_half_z,
                bool pml_x, real *&a_x, real *&b_x, real *&K_x, 
                real *&a_half_x, real *&b_half_x, real *&K_half_x, 
                // PML memory arrays
                real **&mem_vz_z, real **&mem_vx_z, real **&mem_szz_z, real **&mem_szx_z, 
                real **&mem_vz_x, real **&mem_vx_x, real **&mem_szx_x, real **&mem_sxx_x,
                // Seismic sources
                int nsrc, int stf_type, real **&stf_z, real **&stf_x, 
                int *&z_src, int *&x_src, int *&src_shot_to_fire,
                // Reciever seismograms
                int nrec, int rtf_type, real **&rtf_uz, real **&rtf_ux, int *&z_rec, int *&x_rec,
                real **&rtf_z_true, real **&rtf_x_true,
                // Accumulate the snap of forward wavefield parameters
                bool accu, real ***&accu_vz, real ***&accu_vx, //accumulated velocity memory over time
                real ***&accu_szz, real ***&accu_szx, real ***&accu_sxx, //accumulated velocity memory over time
                int snap_z1, int snap_z2, int snap_x1, int snap_x2, // grid boundaries for fwi
                int snap_dt, int snap_dz, int snap_dx // time n space grid intervals to save storage
                ){
    // Calculates the optimum step length for material update after gradient computation
    
    real step_factor_rho=0.5, est_scale_factor=2.0;
    real L2_test[4] = {0.0, 0.0, 0.0, 0.0};
    real step_length[4] = {0.0, 0.0, 0.0, 0.0};
    real L2_temp;
    int *a_stf_type; // adjoint source type
    real scalar_lam_local, scalar_mu_local, scalar_rho_local;

    L2_test[0] = L2_norm_0;
    step_length[0] = 0;

    // controllers for steps in approximation
    bool step1 = false, step2 = false, step3=false;
    unsigned int countstep = 0, maxstep1 = 20, maxstep2 = 3, maxstep3 = 5; 
    bool opt_step; // The optimum step length can be calculated
    std::cout << "Step length Estimation" << std::endl;
    // ----------------------------------------------------
    // PART 1: GETTING THE OPTIMIZATION PARAMETERS
    // ----------------------------------------------
    while (!((step1 && step2) && step3)){
        //std::cout << "new iteration " << std::endl;
        // Updatinng material gradient
        // Update material parameters to the gradients !!
		//update_mat2(lam, lam_copy, grad_lam, est_step_length, nz, nx);
        //update_mat2(mu, mu_copy, grad_mu, est_step_length, nz, nx);
        update_mat2(rho, rho_copy, grad_rho, 3000.0, 1.5, step_factor_rho*est_step_length, nz, nx);

        // calculate material average
        mat_av2(lam, mu, rho, mu_zx, rho_zp, rho_xp, 
            scalar_lam_local, scalar_mu_local, scalar_rho_local, nz, nx);

        for (int ishot=0;ishot<nshot;ishot++){
            // Now calling forward kernel with updated material
            accu = true; // Accumulated storage for output
            grad = false; // no gradient computation in forward kernel
            kernel_PSV(ishot, nt, nz, nx, dt, dx, dz, surf, isurf, hc, fdorder, 
                vz, vx,  uz, ux, szz, szx, sxx, We, dz_z, dx_z, dz_x, dx_x, 
                lam, mu, mu_zx, rho_zp, rho_xp, grad, grad_lam, grad_mu, grad_rho,
                pml_z, a_z, b_z, K_z, a_half_z, b_half_z, K_half_z,
                pml_x, a_x, b_x, K_x, a_half_x, b_half_x, K_half_x, 
                mem_vz_z, mem_vx_z, mem_szz_z, mem_szx_z,
                mem_vz_x, mem_vx_x, mem_szx_x, mem_sxx_x,
                nsrc, stf_type, stf_z, stf_x, z_src, x_src, src_shot_to_fire,
                nrec, rtf_type, rtf_uz, rtf_ux, z_rec, x_rec,
                accu, accu_vz, accu_vx, accu_szz, accu_szx, accu_sxx, 
                snap_z1, snap_z2, snap_x1, snap_x2, 
                snap_dt, snap_dz, snap_dx);

            // calculating L2 norm and adjoint sources
            L2_temp= adjsrc2(a_stf_type, rtf_uz, rtf_ux, rtf_type, rtf_z_true, rtf_x_true,
                    rtf_uz, rtf_ux, dt, nrec, nt);

        }

        // For different cases
        std::cout << "Step Length Cases: " << est_step_length << ", " << L2_temp <<", " << step1 << step2 << step3 << std::endl;    
        // Different conditions arise upto this point
        if (step1 == false){ // CASE 1
            //std::cout << "CASE 1.0: ";
            // The first step still not found
            if (L2_test[0]<=L2_temp){ // CASE 1.1
                // The update increases the error
                
                // Thus try with reduced step length
                step3 = true; // the third step accounted for
                L2_test[3] = L2_temp; // third step can be loaded now
                step_length[3] = est_step_length;


                std::cout << "CASE 1.1: " << est_step_length << ", " << L2_temp <<", " << step1 << step2 << step3 << std::endl;
                est_step_length = est_step_length/est_scale_factor;
                
                countstep++;
                if (countstep>maxstep1){
                    opt_step = false;
                    est_step_length = 0;
                    break;
                }
                
            }

            else if (L2_test[0]>L2_temp){
                // The update has decreased the error
                
                step1 = true; // The first step achieved
                L2_test[1] = L2_temp;
                step_length[1] = est_step_length;

                std::cout << "CASE 1.2: " << est_step_length << ", " << L2_temp <<", " << step1 << step2 << step3 << std::endl;
                est_step_length += (est_step_length/est_scale_factor);
                countstep = 0; // reset the counter after getting first stepsize  
  
            }
        }

        else {
            // The first step is already found
            //std::cout << "CASE 2.0: ";
            if (step2 == false){
                //std::cout << "CASE 2.1: ";
                if (L2_test[1]<=L2_temp){
                    // This update has increased the  error
                    
                    step3 = true; // The third step is true, irrespective if it is calculated before or not
                    L2_test[3] = L2_temp; // third step can be reupdated now
                    step_length[3] = est_step_length;
                    
                    std::cout << "CASE 2.1.1: " << est_step_length << ", " << L2_temp <<", " << step1 << step2 << step3 << std::endl;
                    est_step_length = 0.7 * step_length[1] + 0.3 * est_step_length ; // taking between two points

                    countstep++;
                    if (countstep>maxstep2){
                        opt_step = true;
                        // readjusting the data to use zero step size as well
                        step2 = true;
                        L2_test[2] = L2_test[1];
                        step_length[2] = step_length[1];

                        L2_test[1] = L2_test[0];
                        step_length[1] = step_length[0];

                    }

                    
                }

                else if (L2_test[1]>L2_temp){
                    // This update has decreased the error
                    //std::cout << "CASE 2.12: ";
                    step2 = true;
                    L2_test[2] = L2_temp; // second step can be added now
                    step_length[2] = est_step_length;

                    if (step3){
                        std::cout << "CASE 2.1.2.1: " << est_step_length << ", " << L2_temp <<", " << step1 << step2 << step3 << std::endl;
                        est_step_length += (est_step_length/est_scale_factor);
                        
                    }
                    else{
                        // A bigger step move
                        std::cout << "CASE 2.1.2.2: " << est_step_length << ", " << L2_temp <<", " << step1 << step2 << step3 << std::endl;
                        est_step_length += (est_step_length*est_scale_factor);            
                    }
                    
                }

            }

            else{
                // Step 1 and step 2 are already done
                //std::cout << "CASE 2.2: ";
                if (L2_test[2]<=L2_temp){
                    // This update has increased the  error
                    
                    step3 = true; // The third step is true, irrespective if it is calculated before or not
                    L2_test[3] = L2_temp; // third step can be reupdated now
                    step_length[3] = est_step_length;
                    // now all steps are true and we got all the parameters
                    std::cout << "CASE 2.2.1: " << est_step_length << ", " << L2_temp <<", " << step1 << step2 << step3 << std::endl;

                    opt_step = true;
                    
                }

                else{
                    // This update has decreased the error
                    // Now we update the second step & first step further
                    //std::cout << "CASE 2.2.2: ";
                    L2_test[1] = L2_test[2];
                    step_length[1] = step_length[2];

                    L2_test[2] = L2_temp; // second step can be added now
                    step_length[2] = est_step_length;

                    if (step3){
                        std::cout << "CASE 2.2.2.1: " << est_step_length << ", " << L2_temp <<", " << step1 << step2 << step3 << std::endl;
                        est_step_length += (est_step_length/est_scale_factor);
                        
                    }
                    else{
                        // A bigger step move
                        std::cout << "CASE 2.2.2.2: " << est_step_length << ", " << L2_temp <<", " << step1 << step2 << step3 << std::endl;
                        est_step_length += (est_step_length*est_scale_factor);            
                    }

                    countstep++;
                    if (countstep>maxstep3){
                        opt_step = false;
                        break;
                    }
                    
                }
            }
                
        }
  
    }

    // ---------------------------------------------------------
    // PART 2: OPTIMIZING THE PARAMETERS IF REQUIRED & POSSIBLE
    // ---------------------------------------------------------]
    if (opt_step){
        est_step_length = calc_opt_step(L2_test, step_length);
    }
    return est_step_length;
}
*/

real calc_opt_step_GPU(real L2[3], real sl[3]){
    // Calculates the optimum step length from
    // Array of L2 norms with respect to step_lengths
    // L2 = L2 norm, sl = steplength
    
    int n = 3; // size of the system
    real A[3][3]; // Defining coefficient matrix
    real b[3]; // Defining RHS vector
    real x[3]; // The variable to be solved vector

    real opteps; // optimum step

    /* calculate optimal step size by line search */
    /* ------------------------------------------ */
    /* fit parabola function to L2 norm */

    std::cout << "L2: " << L2[0] <<", " << L2[1] <<", " << L2[2]  << std::endl;
    std::cout << "SL: " << sl[0] <<", " << sl[1] <<", " << sl[2]  << std::endl;
    
    //define coefficient matrix A 
    for (int i=0;i<n;i++){
        A[i][2]=(sl[i]*sl[i]);
        A[i][1]=(sl[i]);
        A[i][0]=(1.0);
    }

    
    //define RHS vector b
    for (int i=0;i<n;i++){
        b[i]=L2[i];
    }

    /* solve matrix equation using LU decomposition */
    /*LU_decomp(A,x,b,n);*/
    solvelin(A,b,x,n,1);

    // calculate optimal step length -> extremum of the parabola */
    opteps = -x[1]/(2.0*x[2]);

    std::cout << "Optimum Step Length: <calculated>" << opteps ;
    /* if opteps < 50.0 set opteps=50.0 */
    if (opteps > sl[2]){
        opteps = sl[2];
    }

    if (opteps < 0.0){
        opteps = sl[1];
    }
    
    if (L2[1]<L2[0] && L2[2]<L2[1]){
        opteps = sl[2];
    }

    std::cout << ", <adopted> " << opteps <<std::endl;

    return opteps;

}