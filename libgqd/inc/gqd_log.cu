#ifndef __GQD_LOG_CU__
#define __GQD_LOG_CU__

#include "gqd_real.h"
//#include "common.cu"


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


#endif /* __GQD_LOG_CU__ */
