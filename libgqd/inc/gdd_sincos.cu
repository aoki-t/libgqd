#ifndef __GDD_SIN_COS_CU__
#define __GDD_SIN_COS_CU__

#include "gdd_real.h"

//#include "common.cu"

extern __device__ __constant__ double dd_inv_fact[n_dd_inv_fact][2];

// Table of sin(k * pi/16) and cos(k * pi/16).
static __device__ __constant__ double dd_sin_table[4][2] = {
	{ 1.950903220161282758e-01, -7.991079068461731263e-18 },
	{ 3.826834323650897818e-01, -1.005077269646158761e-17 },
	{ 5.555702330196021776e-01,  4.709410940561676821e-17 },
	{ 7.071067811865475727e-01, -4.833646656726456726e-17 }
};

static __device__ __constant__ double dd_cos_table[4][2] = {
	{ 9.807852804032304306e-01,  1.854693999782500573e-17 },
	{ 9.238795325112867385e-01,  1.764504708433667706e-17 },
	{ 8.314696123025452357e-01,  1.407385698472802389e-18 },
	{ 7.071067811865475727e-01, -4.833646656726456726e-17 }
};




// Computes sin(a) using Taylor series.
// Assumes |a| <= pi/32.
__device__
gdd_real sin_taylor(const gdd_real &a) {
	const double thresh = 0.5 * fabs(to_double(a)) * _dd_eps;
	gdd_real r, s, t, x;

	if (is_zero(a)) {
		return _dd_zero;
	}

	int i = 0;
	x = negative(sqr(a)); //-sqr(a);
	s = a;
	r = a;
	do {
		r = r*x;
		t = r * gdd_real(dd_inv_fact[i][0], dd_inv_fact[i][1]);
		s = s + t;
		i += 2;
	} while (i < n_dd_inv_fact && fabs(to_double(t)) > thresh);

	return s;
}

__device__
gdd_real cos_taylor(const gdd_real &a) {
	const double thresh = 0.5 * _dd_eps;
	gdd_real r, s, t, x;
	int i = 1;

	if (is_zero(a)) {
		return gdd_real(1.0);
	}

	x = negative(sqr(a));
	r = x;
	s = 1.0 + mul_pwr2(r, 0.5);
	do {
		r = r*x;
		t = r * gdd_real(dd_inv_fact[i][0], dd_inv_fact[i][1]);
		s = s + t;
		i += 2;
	} while (i < n_dd_inv_fact && fabs(to_double(t)) > thresh);

	return s;
}

__device__
void sincos_taylor(const gdd_real &a, gdd_real &sin_a, gdd_real &cos_a) {
	if (is_zero(a)) {
		sin_a.dd.x = 0.0; sin_a.dd.y = 0.0;
		cos_a.dd.x = 1.0; cos_a.dd.y = 0.0;
		return;
	}

	sin_a = sin_taylor(a);
	cos_a = sqrt(1.0 - sqr(sin_a));
}



__device__
gdd_real sin(const gdd_real &a) {  
/*
	Strategy.  To compute sin(x), we choose integers a, b so that

		x = s + a * (pi/2) + b * (pi/16)

	and |s| <= pi/32.  Using the fact that

		sin(pi/16) = 0.5 * sqrt(2 - sqrt(2 + sqrt(2)))

	we can compute sin(x) from sin(s), cos(s).  This greatly
	increases the convergence of the sine Taylor series.
*/

	if (is_zero(a)) {
		return gdd_real(0.0);
	}

	// approximately reduce modulo 2*pi
	gdd_real z = nint(a / _dd_2pi);
	gdd_real r = a - _dd_2pi * z;

	// approximately reduce modulo pi/2 and then modulo pi/16.
	gdd_real t;
	double q = floor(r.dd.x / _dd_pi2.dd.x + 0.5);
	t = r - _dd_pi2 * q;
	int j = (int)(q);
	q = floor(t.dd.x / _dd_pi16.dd.x + 0.5);
	t = t - _dd_pi16 * q;
	int k = (int)(q);
	int abs_k = abs(k);

	if (j < -2 || j > 2) {
		//dd_real::error("(dd_real::sin): Cannot reduce modulo pi/2.");
		//r.dd.x = r.dd.y = 0.0;
		//return r;
		return _dd_qnan;
	}

	if (abs_k > 4) {
		//dd_real::error("(dd_real::sin): Cannot reduce modulo pi/16.");
		//r.dd.x = r.dd.y = 0.0;
		//return r;
		return _dd_qnan;
	}

	if (k == 0) {
		switch (j) {
			case 0:
				return sin_taylor(t);
			case 1:
				return cos_taylor(t);
			case -1:
				return negative(cos_taylor(t));
			default:
				return negative(sin_taylor(t));
		}
	}

	gdd_real u(dd_cos_table[abs_k - 1][0], dd_cos_table[abs_k - 1][1]);
	gdd_real v(dd_sin_table[abs_k - 1][0], dd_sin_table[abs_k - 1][0]);
	gdd_real sin_t, cos_t;
	sincos_taylor(t, sin_t, cos_t);
	if (j == 0) {
		if (k > 0) {
			r = u * sin_t + v * cos_t;
		} else {
			r = u * sin_t - v * cos_t;
		}
	} else if (j == 1) {
		if (k > 0) {
			r = u * cos_t - v * sin_t;
		} else {
			r = u * cos_t + v * sin_t;
		}
	} else if (j == -1) {
		if (k > 0) {
			r = v * sin_t - u * cos_t;
		} else if (k < 0) {
			//r = -u * cos_t - v * sin_t;
			r = negative(u * cos_t) - v * sin_t;
		}
	} else {
		if (k > 0) {
			//r = -u * sin_t - v * cos_t;
			r = negative(u * sin_t) - v * cos_t;
		} else {
			r = v * cos_t - u * sin_t;
		}
	}

	return r;
}

__device__
gdd_real cos(const gdd_real &a) {

	if (is_zero(a)) {
		return gdd_real(1.0);
	}

	// approximately reduce modulo 2*pi
	gdd_real z = nint(a / _dd_2pi);
	gdd_real r = a - z * _dd_2pi;

	// approximately reduce modulo pi/2 and then modulo pi/16
	gdd_real t;
	double q = floor(r.dd.x / _dd_pi2.dd.x + 0.5);
	t = r - _dd_pi2 * q;
	int j = (int)(q);
	q = floor(t.dd.x / _dd_pi16.dd.x + 0.5);
	t = t - _dd_pi16 * q;
	int k = (int)(q);
	int abs_k = abs(k);

	if (j < -2 || j > 2) {
		//dd_real::error("(dd_real::cos): Cannot reduce modulo pi/2.");
		//return dd_real::_nan;
		return _dd_qnan;
	}

	if (abs_k > 4) {
		//dd_real::error("(dd_real::cos): Cannot reduce modulo pi/16.");
		//return dd_real::_nan;
		return _dd_qnan;
	}

	if (k == 0) {
		switch (j) {
			case 0:
				return cos_taylor(t);
			case 1:
				return negative(sin_taylor(t));
			case -1:
				return sin_taylor(t);
			default:
				return negative(cos_taylor(t));
		}
	}

	gdd_real sin_t, cos_t;
	sincos_taylor(t, sin_t, cos_t);
	gdd_real u(dd_cos_table[abs_k - 1][0], dd_cos_table[abs_k - 1][1]);
	gdd_real v(dd_sin_table[abs_k - 1][0], dd_sin_table[abs_k - 1][0]);

	if (j == 0) {
		if (k > 0) {
			r = u * cos_t - v * sin_t;
		} else {
			r = u * cos_t + v * sin_t;
		}
	} else if (j == 1) {
		if (k > 0) {
			r = negative(u * sin_t) - v * cos_t;
		} else {
			r = v * cos_t - u * sin_t;
		}
	} else if (j == -1) {
		if (k > 0) {
			r = u * sin_t + v * cos_t;
		} else {
			r = u * sin_t - v * cos_t;
		}
	} else {
		if (k > 0) {
			r = v * sin_t - u * cos_t;
		} else {
			r = negative(u * cos_t) - v * sin_t;
		}
	}

	return r;
}


__device__
void sincos(const gdd_real &a, gdd_real &sin_a, gdd_real &cos_a) {

	if (is_zero(a)) {
		sin_a = gdd_real(0.0);
		cos_a = gdd_real(1.0);
		return;
	}

	// approximately reduce modulo 2*pi
	gdd_real z = nint(a / _dd_2pi);
	gdd_real r = a - _dd_2pi * z;

	// approximately reduce module pi/2 and pi/16
	gdd_real t;
	double q = floor(r.dd.x / _dd_pi2.dd.x + 0.5);
	t = r - _dd_pi2 * q;
	int j = (int)(q);
	int abs_j = abs(j);
	q = floor(t.dd.x / _dd_pi16.dd.x + 0.5);
	t = t - _dd_pi16 * q;
	int k = (int)(q);
	int abs_k = abs(k);

	if (abs_j > 2) {
		//dd_real::error("(dd_real::sincos): Cannot reduce modulo pi/2.");
		//cos_a = sin_a = dd_real::_nan;
		cos_a = sin_a = _dd_qnan;
		return;
	}

	if (abs_k > 4) {
		//dd_real::error("(dd_real::sincos): Cannot reduce modulo pi/16.");
		//cos_a = sin_a = dd_real::_nan;
		cos_a = sin_a = _dd_qnan;
		return;
	}

	gdd_real sin_t, cos_t;
	gdd_real s, c;

	sincos_taylor(t, sin_t, cos_t);

	if (abs_k == 0) {
		s = sin_t;
		c = cos_t;
	} else {
		gdd_real u(dd_cos_table[abs_k - 1][0], dd_cos_table[abs_k - 1][1]);
		gdd_real v(dd_sin_table[abs_k - 1][0], dd_sin_table[abs_k - 1][0]);

		if (k > 0) {
			s = u * sin_t + v * cos_t;
			c = u * cos_t - v * sin_t;
		} else {
			s = u * sin_t - v * cos_t;
			c = u * cos_t + v * sin_t;
		}
	}

	if (abs_j == 0) {
		sin_a = s;
		cos_a = c;
	} else if (j == 1) {
		sin_a = c;
		cos_a = negative(s);
	} else if (j == -1) {
		sin_a = negative(c);
		cos_a = s;
	} else {
		sin_a = negative(s);
		cos_a = negative(c);
	}
}


__device__
gdd_real tan(const gdd_real &a) {
	gdd_real s, c;
	sincos(a, s, c);
	return s/c;
}


#ifndef ALL_MATH

__device__
gdd_real atan2(const gdd_real &y, const gdd_real &x) {
/* 
	Strategy: Instead of using Taylor series to compute
	arctan, we instead use Newton's iteration to solve the equation

		sin(z) = y/r    or    cos(z) = x/r

	where r = sqrt(x^2 + y^2).
	The iteration is given by

		z' = z + (y - sin(z)) / cos(z)          (for equation 1)
		z' = z - (x - cos(z)) / sin(z)          (for equation 2)

	Here, x and y are normalized so that x^2 + y^2 = 1.
	If |x| > |y|, then first iteration is used since the
	denominator is larger.  Otherwise, the second is used.
*/

	if (is_zero(x)) {

		if (is_zero(y)) {
			/* Both x and y is zero. */
			//dd_real::error("(dd_real::atan2): Both arguments zero.");
			//return dd_real::_nan;
			
			//return gdd_real(0.0);
			return _dd_qnan;
		}
		return (is_positive(y)) ? _dd_pi2 : negative(_dd_pi2);

	} else if (is_zero(y)) {
		return (is_positive(x)) ? gdd_real(0.0) : _dd_pi;
	}

	if (x == y) {
		return (is_positive(y)) ? _dd_pi4 : negative(_dd_3pi4);
	}

	if (x == negative(y)) {
		return (is_positive(y)) ? _dd_3pi4 : negative(_dd_pi4);
	}

	gdd_real r = sqrt(sqr(x) + sqr(y));
	gdd_real xx = x / r;
	gdd_real yy = y / r;

	/* Compute double precision approximation to atan. */
	gdd_real z = gdd_real(atan2(to_double(y), to_double(x)));
	gdd_real sin_z, cos_z;

	if (abs(xx.dd.x) > abs(yy.dd.x)) {
		/* Use Newton iteration 1.  z' = z + (y - sin(z)) / cos(z)  */
		sincos(z, sin_z, cos_z);
		z = z + (yy - sin_z)/cos_z;
	} else {
		/* Use Newton iteration 2.  z' = z - (x - cos(z)) / sin(z)  */
		sincos(z, sin_z, cos_z);
		z = z - (xx - cos_z) / sin_z;
	}

	return z;
}


__device__
gdd_real atan(const gdd_real &a) {
	return atan2(a, gdd_real(1.0));
}


__device__
gdd_real asin(const gdd_real &a) {
	gdd_real abs_a = abs(a);

	if (abs_a > 1.0) {
		//dd_real::error("(dd_real::asin): Argument out of domain.");
		//return dd_real::_nan;
		return _dd_qnan;
	}

	if (is_one(abs_a)) {
		return (is_positive(a)) ? _dd_pi2 : negative(_dd_pi2);
	}

	return atan2(a, sqrt(1.0 - sqr(a)));
}


__device__
gdd_real acos(const gdd_real &a) {
	gdd_real abs_a = abs(a);

	if (abs_a > 1.0) {
		//dd_real::error("(dd_real::acos): Argument out of domain.");
		//return dd_real::_nan;
		return _dd_qnan;
	}

	if (is_one(abs_a)) {
		return (is_positive(a)) ? gdd_real(0.0) : _dd_pi;
	}

	return atan2(sqrt(1.0 - sqr(a)), a);
}


__device__
gdd_real sinh(const gdd_real &a) {
	if (is_zero(a)) {
		return gdd_real(0.0);
	}

	if (abs(a) > 0.05) {
		gdd_real ea = exp(a);
		return mul_pwr2(ea - inv(ea), 0.5);
	}

	/* since a is small, using the above formula gives
	a lot of cancellation.  So use Taylor series.   */
	gdd_real s = a;
	gdd_real t = a;
	gdd_real r = sqr(t);
	double m = 1.0;
	double thresh = abs((to_double(a)) * _dd_eps);

	do {
		m = m +  2.0;
		t = (t*r);
		t = t/((m-1) * m);

		s = s + t;
	} while (abs(t) > thresh);

	return s;

}


__device__
gdd_real cosh(const gdd_real &a) {
	if (is_zero(a)) {
		return gdd_real(1.0);
	}

	gdd_real ea = exp(a);
	return mul_pwr2(ea + inv(ea), 0.5);
}


__device__
gdd_real tanh(const gdd_real &a) {
	if (is_zero(a)) {
		return gdd_real(0.0);
	}

	gdd_real ea = exp(a);
	gdd_real inv_ea = inv(ea);
	return (ea - inv_ea) / (ea + inv_ea);
}


__device__
void sincosh(const gdd_real &a, gdd_real &sinh_a, gdd_real &cosh_a) {
	sinh_a = sinh(a);
	cosh_a = cosh(a);
}


__device__
gdd_real asinh(const gdd_real &a) {
	return log(a + sqrt(sqr(a) + 1.0));
}


__device__
gdd_real acosh(const gdd_real &a) {
	if (a < 1.0) {
		//dd_real::error("(dd_real::acosh): Argument out of domain.");
		//return dd_real::_nan;
		return _dd_qnan;
	}

	return log(a + sqrt(sqr(a) - 1.0));
}


__device__
gdd_real atanh(const gdd_real &a) {
	if (abs(a) >= 1.0) {
		//dd_real::error("(dd_real::atanh): Argument out of domain.");
		//return dd_real::_nan;
		return _dd_qnan;
	}

	return mul_pwr2(log((1.0 + a) / (1.0 - a)), 0.5);
}


#endif /* ALL_MATH */


#endif /* __GDD_SIN_COS_CU__ */


