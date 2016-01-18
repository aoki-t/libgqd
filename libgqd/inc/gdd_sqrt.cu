#ifndef __GDD_SQRT_CU__
#define __GDD_SQRT_CU__

//#include "common.cu"
#include "gdd_real.h"

/* Computes the square root of the double-double number dd.
   NOTE: dd must be a non-negative number.                   */
__device__
gdd_real sqrt(const gdd_real &a) {
	if (is_zero(a)) {
		return 0.0;
	}

	//TODO: should make an error
	if (is_negative(a)) {
		//return _nan;
		return _dd_qnan;
	}

	double x = 1.0 / std::sqrt(a.dd.x);
	double ax = a.dd.x * x;

	return dd_add(ax, (a - dd_sqr(ax)).dd.x * (x * 0.5));
	//return a - sqr(ax);
}

/* Computes the n-th root of the double-double number a.
NOTE: n must be a positive integer.
NOTE: If n is even, then a must not be negative.       */
__device__
gdd_real nroot(const gdd_real &a, int n) {
	/* Strategy:  Use Newton iteration for the function

		f(x) = x^(-n) - a

		to find its root a^{-1/n}.  The iteration is thus

		x' = x + x * (1 - a * x^n) / n

		which converges quadratically.  We can then find
		a^{1/n} by taking the reciprocal.
	*/
	if (n == 0) {
		return _dd_qnan;
	}
	if (n <= 0) {
		//gdd_real::error("(dd_real::nroot): N must be positive.");
		//return gdd_real::_nan;
		return _dd_qnan;
	}

	if (n % 2 == 0 && is_negative(a)) {
		//gdd_real::error("(dd_real::nroot): Negative argument.");
		//return gdd_real::_nan;
		return _dd_qnan;
	}

	if (n == 1) {
		return a;
	}
	if (n == 2) {
		return sqrt(a);
	}

	if (is_zero(a)){
		return 0.0;
	}

	/* Note  a^{-1/n} = exp(-log(a)/n) */
	gdd_real r = abs(a);
	gdd_real x = std::exp(-std::log(r.dd.x) / n);

	/* Perform Newton's iteration. */
	x += x * (1.0 - r * npwr(x, n)) / static_cast<double>(n);
	if (a.dd.x < 0.0){
		x = -x;
	}
	return 1.0 / x;
}


#endif /* __GDD_SQRT_CU__ */


