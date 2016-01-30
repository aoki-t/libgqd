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


// Computes the n-th root of a
__device__
gqd_real nroot(const gqd_real &a, int n) {
/*  Strategy:
	Use Newton's iteration to solve

	    1/(x^n) - a = 0

	Newton iteration becomes

	    x' = x + x * (1 - a * x^n) / n

	Since Newton's iteration converges quadratically,
	we only need to perform it twice.
*/
	if (n <= 0) {
		//gqd_real::error("(gqd_real::nroot): N must be positive.");
		//return gqd_real::_nan;
		return _qd_qnan;
	}

	if (n % 2 == 0 && is_negative(a)) {
		//gqd_real::error("(gqd_real::nroot): Negative argument.");
		//return gqd_real::_nan;
		return _qd_qnan;
	}

	if (n == 1) {
		return a;
	}
	if (n == 2) {
		return sqrt(a);
	}
	if (is_zero(a)) {
		return _qd_zero;
	}


	// Note  a^{-1/n} = exp(-log(a)/n)
	gqd_real r = abs(a);
	gqd_real x = std::exp(-std::log(r[0]) / n);

	// Perform Newton's iteration.
	double dbl_n = static_cast<double>(n);
	x += x * (1.0 - r * npwr(x, n)) / dbl_n;
	x += x * (1.0 - r * npwr(x, n)) / dbl_n;
	x += x * (1.0 - r * npwr(x, n)) / dbl_n;

	if (a[0] < 0.0) {
		x = -x;
	}
	return 1.0 / x;
}


#endif /* __GQD_SQRT_CU__ */


