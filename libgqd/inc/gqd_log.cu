#ifndef __GQD_LOG_CU__
#define __GQD_LOG_CU__

#include "gqd_real.h"
//#include "common.cu"


// Natural logarithm.
__device__
gqd_real log(const gqd_real &a) {
/*	Strategy:
	The Taylor series for log converges much more slowly than
	that of exp, due to the lack of the factorial term in the
	denominator.  Hence this routine instead tries to determine 
	the root of the function

		f(x) = exp(x) - a

	using Newton iteration.  The iteration is given by

		x' = x - f(x)/f'(x)
		   = x - (1 - a * exp(-x))
		   = x + a * exp(-x) - 1.

	Two iteration is needed, since Newton's iteration
	approximately doubles the number of digits per iteration.
*/

	if (isnan(a)) {
		return _qd_qnan;
	}

	if (is_one(a)) {
		return _qd_zero;
	}

	if (is_zero(a)) {
		return negative(_qd_inf);
	}

	if (is_negative(a)) {
		return _qd_qnan;
	}

	if (is_pinf(a)) {
		return _qd_inf;
	}

	gqd_real x = gqd_real(std::log(a[0]));

	x = x + a * exp(negative(x)) - 1.0;
	x = x + a * exp(negative(x)) - 1.0;
	x = x + a * exp(negative(x)) - 1.0;

	return x;
}


// Binary logarithm.
__device__
gqd_real log2(const gqd_real &a) {
	return log(a) / _qd_log2;
}


// Common logarithm.
__device__
gqd_real log10(const gqd_real &a) {
	return log(a) / _qd_log10;
}


#endif /* __GQD_LOG_CU__ */
