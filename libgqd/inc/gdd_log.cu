#ifndef __GDD_LOG_CU__
#define __GDD_LOG_CU__

//#include "common.cu"
#include "gdd_real.h"


// Natural logarithm.  Computes log(x) in double-double precision.
__device__
gdd_real log(const gdd_real &a) {
/* 
	Strategy.  The Taylor series for log converges much more
	slowly than that of exp, due to the lack of the factorial
	term in the denominator.  Hence this routine instead tries
	to determine the root of the function

	f(x) = exp(x) - a

	using Newton iteration.  The iteration is given by

	x' = x - f(x)/f'(x)
	= x - (1 - a * exp(-x))
	= x + a * exp(-x) - 1.

	Only one iteration is needed, since Newton's iteration
	approximately doubles the number of digits per iteration. 
*/

	if (isnan(a)) {
		return _dd_qnan;
	}

	if (is_one(a)) {	
		return _dd_zero;
	}

	if (is_zero(a)) {
		return negative(_dd_inf);
	}

	if (is_negative(a)) {
		return _dd_qnan;
	}

	if (is_pinf(a)) {
		return _dd_inf;
	}

	gdd_real x(std::log(a.dd.x));   // Initial approximation 

	x = x + a * exp(negative(x)) - 1.0;

	return x;
}


// Binary logarithm.
__device__
gdd_real log2(const gdd_real &a) {
	return log(a) / _dd_log2;
}


// Common logarithm.
__device__
gdd_real log10(const gdd_real &a) {
	return log(a) / _dd_log10;
}


#endif /* __GDD_LOG_CU__ */


