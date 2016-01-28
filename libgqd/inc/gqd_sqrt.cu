#ifndef __GQD_SQRT_CU__
#define __GQD_SQRT_CU__


#include "gqd_real.h"
//#include "common.cu"

__device__
gqd_real sqrt(const gqd_real &a) {
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


