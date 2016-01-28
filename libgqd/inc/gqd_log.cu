#ifndef __GQD_LOG_CU__
#define __GQD_LOG_CU__

#include "gqd_real.h"
//#include "common.cu"


// Natural logarithm.
__device__
gqd_real log(const gqd_real &a) {
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
