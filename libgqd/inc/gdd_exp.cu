#ifndef __GDD_EXP_CU__
#define __GDD_EXP_CU__


#include "gdd_real.h"
//#include "common.cu"

//#define INV_K (1.0/512.0) 


//the completed version with additional branches for parameter checking
__device__
gdd_real exp(const gdd_real &a) {

	const double k = 512.0;
	const double inv_k = 1.0 / k;

	if (a.dd.x <= -709.0){
		return 0.0;
	}

	if (a.dd.x >= 709.0){
		return _dd_inf;
		//TODO: return dd_real::_inf;
	}

	if (is_zero(a)){
		return _dd_one;
	}

	if (is_one(a)){
		return _dd_e;
	}

	double m = floor(a.dd.x / _dd_log2.dd.x + 0.5);
	gdd_real r = mul_pwr2(a - _dd_log2 * m, inv_k);
	gdd_real s, t, p;

	p = sqr(r);
	s = r + mul_pwr2(p, 0.5);
	p = p * r;
	t = p * gdd_real(dd_inv_fact[0][0], dd_inv_fact[0][1]); 
	int i = 0;
	do {
		s = s + t;
		p = p * r;
		t = p * gdd_real(dd_inv_fact[++i][0], dd_inv_fact[++i][1]);
	} while ((fabs(to_double(t)) > inv_k * _dd_eps) && (i < 5));

	s = s + t;

	#pragma unroll 9
	for( int i = 0; i < 9; i++ ) {
		s = mul_pwr2(s, 2.0) + sqr(s);
	}

	s = s + 1.0;

	return ldexp(s, int(m));
}


#endif /* __GDD_EXP_CU__ */


