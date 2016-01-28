#ifndef __GQD_SQRT_CU__
#define __GQD_SQRT_CU__


#include "gqd_real.h"
//#include "common.cu"

__device__
gqd_real sqrt(const gqd_real &a) {
/*	Strategy:
	Perform the following Newton iteration:

		x' = x + (1 - a * x^2) * x / 2;

	which converges to 1/sqrt(a), starting with the
	double precision approximation to 1/sqrt(a).
	Since Newton's iteration more or less doubles the
	number of correct digits, we only need to perform it twice.
*/

	if (is_zero(a)) {
		return a;
	}

	if (is_pinf(a)) {
		return _qd_inf;
	}

	if (is_negative(a)) {
		return _qd_qnan;
	}

	gqd_real r = gqd_real((1.0 / std::sqrt(a[0])));
	gqd_real h = mul_pwr2(a, 0.5);

	r = r + ((0.5 - h * sqr(r)) * r);
	r = r + ((0.5 - h * sqr(r)) * r);
	r = r + ((0.5 - h * sqr(r)) * r);

	r = r * a;

	return r;
}

#endif /* __GQD_SQRT_CU__ */


