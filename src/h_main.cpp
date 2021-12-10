

#include "h_globvar.hpp"
#include "h_simulate_PSV.hpp"
#include<omp.h>
int main(){
    double t1,t2;
    t1=omp_get_wtime();
    simulate_PSV();
    t2=omp_get_wtime();
    std::cout<<"total time taken "<<t2-t1;
}