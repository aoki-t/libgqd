#ifndef __GDD_EXP_CU__
#define __GDD_EXP_CU__


#include "gdd_real.h"
//#include "common.cu"

// Inverse of factorial
static __device__ __constant__ double dd_inv_fact[n_dd_inv_fact][2] = {
	{ 1.66666666666666657e-01,  9.25185853854297066e-18 },	// 1/3!
	{ 4.16666666666666644e-02,  2.31296463463574266e-18 },	// 1/4!
	{ 8.33333333333333322e-03,  1.15648231731787138e-19 },	// 1/5!
	{ 1.38888888888888894e-03, -5.30054395437357706e-20 },	// 1/6!
	{ 1.98412698412698413e-04,  1.72095582934207053e-22 },	// 1/7!
	{ 2.48015873015873016e-05,  2.15119478667758816e-23 },	// 1/8!
	{ 2.75573192239858925e-06, -1.85839327404647208e-22 },	// 1/9!
	{ 2.75573192239858883e-07,  2.37677146222502973e-23 },	// 1/10!
	{ 2.50521083854417202e-08, -1.44881407093591197e-24 },	// 1/11!
	{ 2.08767569878681002e-09, -1.20734505911325997e-25 },	// 1/12!
	{ 1.60590438368216133e-10,  1.25852945887520981e-26 },	// 1/13!
	{ 1.14707455977297245e-11,  2.06555127528307454e-28 },	// 1/14!
	{ 7.64716373181981641e-13,  7.03872877733453001e-30 },	// 1/15!
	{ 4.77947733238738525e-14,  4.39920548583408126e-31 },	// 1/16!
	{ 2.81145725434552060e-15,  1.65088427308614326e-31 }	// 1/17!
};


// Exponential.  Computes exp(x) in double-double precision.
__device__
gdd_real exp(const gdd_real &a) {
/* 
	Strategy:  We first reduce the size of x by noting that

	exp(kr + m * log(2)) = 2^m * exp(r)^k

	where m and k are integers.  By choosing m appropriately
	we can make |kr| <= log(2) / 2 = 0.347.  Then exp(r) is
	evaluated using the familiar Taylor series.  Reducing the
	argument substantially speeds up the convergence.       
*/

	const double k = 512.0;
	const double inv_k = 1.0 / k;
	const double thresh = inv_k * _dd_eps;

	if (a.dd.x <= -709.0){
		return _dd_zero;
	}

	if (a.dd.x >= 709.0){
		return _dd_inf;
	}

	if (is_zero(a)){
		return _dd_one;
	}

	if (is_one(a)){
		return _dd_e;
	}

	if (is_ninf(a)) {
		return _dd_zero;
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
		++i;
		t = p * gdd_real(dd_inv_fact[i][0], dd_inv_fact[i][1]);
	} while ( (fabs(to_double(t)) > thresh) && (i < 5) );

	s = s + t;

	#pragma unroll 9
	for( int i = 0; i < 9; i++ ) {
		s = mul_pwr2(s, 2.0) + sqr(s);
	}

	s = s + 1.0;

	return ldexp(s, int(m));
}


#endif /* __GDD_EXP_CU__ */


