#ifndef __GDD_LOG_CU__
#define __GDD_LOG_CU__

//#include "common.cu"
#include "gdd_real.h"


/* Logarithm.  Computes log(x) in double-double precision.
   This is a natural logarithm (i.e., base e).            */
__device__
gdd_real log(const gdd_real &a) {

	if (is_one(a)) {	
		return _dd_zero;
	}

	if (a.dd.x <= 0.0) {
		return _dd_qnan;
	}

	gdd_real x(std::log(a.dd.x));   // Initial approximation 

	x = x + a * exp(negative(x)) - 1.0;

	return x;
}



__device__
gdd_real log10(const gdd_real &a) {
	return log(a) / _dd_log10;
}


#endif /* __GDD_LOG_CU__ */


