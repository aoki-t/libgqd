#ifndef __GDD_BASIC_CU__
#define __GDD_BASIC_CU__


#include <ostream>
#include <string>
#include <cstring>
#include <stdio.h>
#include <algorithm>
//#include <cmath>
#include <math.h>

#include "cuda_header.h"

#include "gdd_real.h"
#include "inline.cu"		//basic functions used by both gdd_real and gqd_real

//#include "common.cu"


__device__ __constant__ double _dd_eps = (4.930380657631323783e-32);	// 2^-104

__device__ __constant__ double __dd_zero = 0.0;
__device__ __constant__ double __dd_one = 1.0;
__device__ __constant__ double  __dd_inf[2];	// h_inf, CUDART_INF;
__device__ __constant__ double __dd_qnan[2];	// h_qnan, CUDART_NAN;

__device__ __constant__ double     __dd_e[2] = { 2.718281828459045091e+00, 1.445646891729250158e-16 };		// __dd_e
__device__ __constant__ double  __dd_log2[2] = { 6.931471805599452862e-01, 2.319046813846299558e-17 };		// __dd_log2
__device__ __constant__ double __dd_log10[2] = { 2.302585092994045901e+00, -2.170756223382249351e-16 };		// __dd_log10
__device__ __constant__ double   __dd_2pi[2] = { 6.283185307179586232e+00, 2.449293598294706414e-16 };		// __dd_2pi
__device__ __constant__ double    __dd_pi[2] = { 3.141592653589793116e+00, 1.224646799147353207e-16 };		// __dd_pi
__device__ __constant__ double   __dd_pi2[2] = { 1.570796326794896558e+00, 6.123233995736766036e-17 };		// __dd_pi2
__device__ __constant__ double  __dd_pi16[2] = { 1.963495408493620697e-01, 7.654042494670957545e-18 };		// __dd_pi16
__device__ __constant__ double   __dd_pi4[2] = { 7.853981633974482790e-01, 3.061616997868383018e-17 };		// __dd_pi4
__device__ __constant__ double  __dd_3pi4[2] = { 2.356194490192344837e+00, 9.1848509936051484375e-17 };		// __dd_3pi4


// Constructors ================================================================

// default constructor
__device__
gdd_real::gdd_real()
:dd({ 0.0, 0.0 }){}


__device__
gdd_real::gdd_real(double hi, double lo){
	dd.x = hi;
	dd.y = lo;

	// with renormize
	// dd.x = two_sum(hi, lo, dd.y);
}

__device__
gdd_real::gdd_real(double d){
	dd.x = d;
	dd.y = 0.0;
}

__device__
gdd_real::gdd_real(int i) {
	dd.x = (static_cast<double>(i));
	dd.y = 0.0;
}


__device__
gdd_real::gdd_real(const double *d) {
	dd.x = d[0];
	dd.y = d[1];
}

// copy constructor
__device__
gdd_real::gdd_real(const gdd_real &a){
	dd.x = a.dd.x;
	dd.y = a.dd.y;
}

// destructor
__device__
gdd_real::~gdd_real(){}



// Assignments =================================================================

__device__
gdd_real &gdd_real::operator=(const gdd_real &a) {
	if (this == &a) {
		return *this;
	}

	dd.x = a.dd.x;
	dd.y = a.dd.y;
	return *this;
}


__device__
gdd_real &gdd_real::operator=(double a) {
	dd.x = a;
	dd.y = 0.0;
	return *this;
}


__device__
gdd_real gdd_real::operator-() const {
	return gdd_real(-dd.x, -dd.y);
}


__device__
gdd_real negative(const gdd_real &a) {
	return gdd_real(-a.dd.x, -a.dd.y);
}



// Additions ===================================================================

// double-double + double
__device__
gdd_real operator+(const gdd_real &a, double b) {
	if (isnan(a) || isnan(b)) {
		return _dd_qnan;
	}
	if ((is_pinf(a) && b == -CUDART_INF) || (is_ninf(a) && b == CUDART_INF)) {
		return _dd_qnan;
	}
	if (is_zero(a)) {
		return gdd_real(b);
	}
	if (b == 0.0) {
		return a;
	}

	double s1, s2;
	s1 = two_sum(a.dd.x, b, s2);
	s2 += a.dd.y;
	s1 = quick_two_sum(s1, s2, s2);
	return gdd_real(s1, s2);
}


// double + double-double
__device__
gdd_real operator+(double a, const gdd_real &b) {
	return b + a;
}


// double-double + double-double
#ifdef SLOPPY_ADD
__forceinline__ __device__
gdd_real sloppy_add(const gdd_real &a, const gdd_real &b) {
	double s, e;

	s = two_sum(a.dd.x, b.dd.x, e);
	e += (a.dd.y + b.dd.y);
	s = quick_two_sum(s, e, e);
	return gdd_real(s, e);
}

#else
__forceinline__ __device__
gdd_real ieee_add(const gdd_real &a, const gdd_real &b) {
	/* This one satisfies IEEE style error bound,
	due to K. Briggs and W. Kahan.                   */
	double s1, s2, t1, t2;

	s1 = two_sum(a.dd.x, b.dd.x, s2);
	t1 = two_sum(a.dd.y, b.dd.y, t2);
	s2 += t1;
	s1 = quick_two_sum(s1, s2, s2);
	s2 += t2;
	s1 = quick_two_sum(s1, s2, s2);
	return gdd_real(s1, s2);
}

#endif


__inline__ __device__
gdd_real operator+(const gdd_real &a, const gdd_real &b) {
	if (isnan(a) || isnan(b)) {
		return _dd_qnan;
	}
	if ((is_pinf(a) && is_ninf(b)) || (is_ninf(a) && is_pinf(b))) {
		return _dd_qnan;
	}
	if (a == negative(b)) {
		return _dd_zero;
	}
	if (is_zero(a)) {
		return b;
	}
	if (is_zero(b)) {
		return a;
	}

#ifdef SLOPPY_ADD
	return sloppy_add(a, b);
#else
	return ieee_add(a, b);
#endif
}



// Self-Additions ==============================================================

// double-double += double
__inline__ __device__
gdd_real &gdd_real::operator+=(double a) {
	if (isnan(*this) || isnan(a)) {
		dd.x = __dd_qnan[1];
		dd.y = __dd_qnan[1];
		return *this;
	}
	if ((is_pinf(*this) && a == -CUDART_INF) || (is_ninf(*this) && a == CUDART_INF)) {
		dd.x = __dd_qnan[1];
		dd.y = __dd_qnan[1];
		return *this;
	}
	if (is_zero(*this)) {
		dd.x = a;
		return *this;
	}
	if (a == 0.0) {
		return *this;
	}

	double s1, s2;
	s1 = two_sum(dd.x, a, s2);
	s2 += dd.y;
	dd.x = quick_two_sum(s1, s2, dd.y);
	return *this;
}


// double-double += double-double
__inline__ __device__
gdd_real &gdd_real::operator+=(const gdd_real &a) {
	if (isnan(*this) || isnan(a)) {
		dd.x = __dd_qnan[1];
		dd.y = __dd_qnan[1];
		return *this;
	}
	if ((is_pinf(*this) && is_ninf(a)) || (is_ninf(*this) && is_pinf(a))) {
		dd.x = __dd_qnan[1];
		dd.y = __dd_qnan[1];
		return *this;
	}
	if (*this == negative(a)) {
		dd.x = 0.0;
		dd.y = 0.0;
		return *this;
	}
	if (is_zero(*this)) {
		dd.x = a.dd.x;
		dd.y = a.dd.y;
		return *this;
	}
	if (is_zero(a)) {
		return *this;
	}

#ifdef SLOPPY_ADD
	double s, e;
	s = two_sum(dd.x, a.dd.x, e);
	e += dd.y;
	e += a.dd.y;
	dd.x = quick_two_sum(s, e, dd.y);
	return *this;
#else
	double s1, s2, t1, t2;
	s1 = two_sum(dd.x, a.dd.x, s2);
	t1 = two_sum(dd.y, a.dd.y, t2);
	s2 += t1;
	s1 = quick_two_sum(s1, s2, s2);
	s2 += t2;
	dd.x = quick_two_sum(s1, s2, dd.y);
	return *this;
#endif
}



// Subtractions ================================================================

// double-double - double-double
__device__
gdd_real operator-(const gdd_real &a, const gdd_real &b) {
	if (isnan(a) || isnan(b)) {
		return _dd_qnan;
	}
	if ((is_pinf(a) && is_pinf(b)) || (is_ninf(a) && is_ninf(b))) {
		return _dd_qnan;
	}
	if (a == b) {
		return _dd_zero;
	}
	if (is_zero(a)) {
		return -b;
	}
	if (is_zero(b)) {
		return a;
	}

#ifdef SLOPPY_ADD
	double s, e;
	s = two_diff(a.dd.x, b.dd.x, e);
	e += a.dd.y;
	e -= b.dd.y;
	s = quick_two_sum(s, e, e);
	return gdd_real(s, e);

#else
	double s1, s2, t1, t2;
	s1 = two_diff(a.dd.x, b.dd.x, s2);
	t1 = two_diff(a.dd.y, b.dd.y, t2);
	s2 += t1;
	s1 = quick_two_sum(s1, s2, s2);
	s2 += t2;
	s1 = quick_two_sum(s1, s2, s2);
	return gdd_real(s1, s2);
#endif
}


// double-double - double
__device__
gdd_real operator-(const gdd_real &a, double b) {
	if (isnan(a) || isnan(b)) {
		return _dd_qnan;
	}
	if ((is_pinf(a) && b == CUDART_INF) || (is_ninf(a) && b == -CUDART_INF)) {
		return _dd_qnan;
	}
	if (is_zero(a)) {
		return gdd_real(-b);
	}
	if (b == 0.0) {
		return a;
	}

	double s1, s2;
	s1 = two_diff(a.dd.x, b, s2);
	s2 += a.dd.y;
	s1 = quick_two_sum(s1, s2, s2);
	return gdd_real(s1, s2);
}


// double - double-double
__device__
gdd_real operator-(double a, const gdd_real &b) {
	if (isnan(a) || isnan(b)) {
		return _dd_qnan;
	}
	if ((a == CUDART_INF && is_pinf(b)) || (a == -CUDART_INF && is_ninf(b))) {
		return _dd_qnan;
	}
	if (a == 0.0) {
		return -b;
	}
	if (is_zero(b)) {
		return gdd_real(a);
	}

	double s1, s2;
	s1 = two_diff(a, b.dd.x, s2);
	s2 -= b.dd.y;
	s1 = quick_two_sum(s1, s2, s2);
	return gdd_real(s1, s2);
}




// Self-Subtractions ===========================================================
// double-double -= double-double
__device__
gdd_real &gdd_real::operator-=(const gdd_real &b) {
	if (isnan(*this) || isnan(b)) {
		dd.x = __dd_qnan[1];
		dd.y = __dd_qnan[1];
		return *this;
	}
	if ((is_pinf(*this) && is_pinf(b)) || (is_ninf(*this) && is_ninf(b))) {
		dd.x = __dd_qnan[1];
		dd.y = __dd_qnan[1];
		return *this;
	}
	if (*this == b) {
		dd.x = 0.0;
		dd.y = 0.0;
		return *this;
	}
	if (is_zero(*this)) {
		dd.x = -b.dd.x;
		dd.y = -b.dd.y;
		return *this;
	}
	if (is_zero(b)) {
		return *this;
	}

#ifdef SLOPPY_ADD
	double s, e;
	s = two_diff(dd.x, b.dd.x, e);
	e += dd.y;
	e -= b.dd.y;
	dd.x = quick_two_sum(s, e, dd.y);
	return *this;

#else
	double s1, s2, t1, t2;
	s1 = two_diff(dd.x, b.dd.x, s2);
	t1 = two_diff(dd.y, b.dd.y, t2);
	s2 += t1;
	s1 = quick_two_sum(s1, s2, s2);
	s2 += t2;
	dd.x = quick_two_sum(s1, s2, dd.y);
	return *this;
#endif
}


// double-double -= double
__device__
gdd_real &gdd_real::operator-=(double b) {
	if (isnan(*this) || isnan(b)) {
		dd.x = __dd_qnan[1];
		dd.y = __dd_qnan[1];
		return *this;
	}
	if ((is_pinf(*this) && b == CUDART_INF) || (is_ninf(*this) && b == -CUDART_INF)) {
		dd.x = __dd_qnan[1];
		dd.y = __dd_qnan[1];
		return *this;
	}
	if (is_zero(*this)) {
		dd.x = -b;
		dd.y = 0.0;
		return *this;
	}
	if (b == 0.0) {
		return *this;
	}


	double s1, s2;
	s1 = two_diff(dd.x, b, s2);
	s2 += dd.y;
	dd.x = quick_two_sum(s1, s2, dd.y);
	return *this;
}



// Squaring ====================================================================

// double-double ^ 2
__device__
gdd_real sqr(const gdd_real &a) {
	double p1, p2;
	double s1, s2;
	p1 = two_sqr(a.dd.x, p2);
	p2 += (2.0 * a.dd.x * a.dd.y);
	//p2 = __dadd_rn(p2,__dmul_rn(__dmul_rn(2.0, a.dd.x), a.dd.y));

	p2 += (a.dd.y * a.dd.y);
	//p2 = __dadd_rn(p2, __dmul_rn(a.dd.y, a.dd.y));
	s1 = quick_two_sum(p1, p2, s2);
	return gdd_real(s1, s2);
}



// Multiplications =============================================================

// double-double * (2.0 ^ exp)
__device__
gdd_real ldexp(const gdd_real &a, int exp) {
	return gdd_real(std::ldexp(a.dd.x, exp), std::ldexp(a.dd.y, exp));
}


// double-double * double,  where double is a power of 2.
__device__
gdd_real mul_pwr2(const gdd_real &a, double b) {
	return gdd_real(a.dd.x * b, a.dd.y * b);
}


// double-double * double-double
__device__
gdd_real operator*(const gdd_real &a, const gdd_real &b) {
	if (isnan(a) || isnan(b)) {
		return _dd_qnan;
	}
	if ((isinf(a) && is_zero(b)) || (is_zero(a) && isinf(b))) {
		return _dd_qnan;
	}
	if (is_zero(a) || is_zero(b)) {
		return _dd_zero;
	}
	if (is_one(a)) {
		return b;
	}
	if (is_one(-a)) {
		return -b;
	}
	if (is_one(b)) {
		return a;
	}
	if (is_one(-b)) {
		return -a;
	}

	double p1, p2;

	p1 = two_prod(a.dd.x, b.dd.x, p2);
	p2 += (a.dd.x * b.dd.y + a.dd.y * b.dd.x);
	//p2 = p2 + (__dmul_rn(a.dd.x,b.dd.y) + __dmul_rn(a.dd.y,b.dd.x));

	p1 = quick_two_sum(p1, p2, p2);
	return gdd_real(p1, p2);
}


// double-double * double
__device__
gdd_real operator*(const gdd_real &a, double b) {
	if (isnan(a) || isnan(b)) {
		return _dd_qnan;
	}
	if ((isinf(a) && b == 0.0) || (is_zero(a) && isinf(b))) {
		return _dd_qnan;
	}
	if (is_zero(a) || b == 0.0) {
		return _dd_zero;
	}
	if (is_one(a)) {
		return b;
	}
	if (is_one(-a)) {
		return -b;
	}
	if (b == 1.0) {
		return a;
	}
	if (b == -1.0) {
		return -a;
	}

	double p1, p2;

	p1 = two_prod(a.dd.x, b, p2);
	p2 += a.dd.y * b;
	//p2 = __dadd_rn(p2, __dmul_rn(a.dd.y, b) );

	p1 = quick_two_sum(p1, p2, p2);
	return gdd_real(p1, p2);
}


// double * double-double
__device__
gdd_real operator*(double a, const gdd_real &b) {
	return (b * a);
}



// Self-Multiplications ========================================================

// double-double *= double-double
__device__
gdd_real &gdd_real::operator*=(const gdd_real &b) {
	if (isnan(*this) || isnan(b)) {
		dd.x = __dd_qnan[1];
		dd.y = __dd_qnan[1];
		return *this;
	}
	if ((isinf(*this) && is_zero(b)) || (is_zero(*this) && isinf(b))) {
		dd.x = __dd_qnan[1];
		dd.y = __dd_qnan[1];
		return *this;
	}
	if (is_zero(*this) || is_zero(b)) {
		dd.x = __dd_zero;
		dd.y = __dd_zero;
		return *this;
	}
	if (is_one(*this)) {
		dd.x = b.dd.x;
		dd.y = b.dd.y;
		return *this;
	}
	if (is_one(-(*this))) {
		dd.x = -b.dd.x;
		dd.y = -b.dd.y;
		return *this;
	}
	if (is_one(b)) {
		return *this;
	}
	if (is_one(-b)) {
		dd.x = -dd.x;
		dd.y = -dd.y;
		return *this;
	}

	double p1, p2;

	p1 = two_prod(dd.x, b.dd.x, p2);
	p2 += (dd.x * b.dd.y + dd.y * b.dd.x);
	//p2 = p2 + ( __dmul_rn(dd.x, b.dd.y) + __dmul_rn(dd.y, b.dd.x) );

	dd.x = quick_two_sum(p1, p2, dd.y);
	return *this;
}


// double-double *= double
__device__
gdd_real &gdd_real::operator*=(double b) {
	if (isnan(*this) || isnan(b)) {
		dd.x = __dd_qnan[1];
		dd.y = __dd_qnan[1];
		return *this;
	}
	if ((isinf(*this) && b == 0.0) || (is_zero(*this) && isinf(b))) {
		dd.x = __dd_qnan[1];
		dd.y = __dd_qnan[1];
		return *this;
	}
	if (is_zero(*this) || b == 0.0) {
		dd.x = __dd_zero;
		dd.y = __dd_zero;
		return *this;
	}
	if (is_one(*this)) {
		dd.x = b;
		dd.y = 0.0;
		return *this;
	}
	if (is_one(-(*this))) {
		dd.x = -b;
		dd.y = 0.0;
		return *this;
	}
	if (b == 1.0) {
		return *this;
	}
	if (b == -1.0) {
		dd.x = -dd.x;
		dd.y = -dd.y;
		return *this;
	}

	double p1, p2;

	p1 = two_prod(dd.x, b, p2);
	p2 += dd.y * b;
	//p2 = __dadd_rn(p2, __dmul_rn(dd.y, b) );

	dd.x = quick_two_sum(p1, p2, dd.y);
	return *this;
}



// Divisions ===================================================================

// double-double / double-double
#ifdef SLOPPY_DIV
__forceinline__ __device__
gdd_real sloppy_div(const gdd_real &a, const gdd_real &b) {
	double s1, s2;
	double q1, q2;
	gdd_real r;

	q1 = a.dd.x / b.dd.x;  /* approximate quotient */

	// compute  this - q1 * dd
	r = b * q1;
	s1 = two_diff(a.dd.x, r.dd.x, s2);
	s2 -= r.dd.y;
	s2 += a.dd.y;

	/ get next approximation
	q2 = (s1 + s2) / b.dd.x;

	// renormalize
	r.dd.x = quick_two_sum(q1, q2, r.dd.y);
	return r;
}

#else
__forceinline__ __device__
gdd_real accurate_div(const gdd_real &a, const gdd_real &b) {
	double q1, q2, q3;
	gdd_real r;

	q1 = a.dd.x / b.dd.x;	// approximate quotient

	r = a - q1 * b;

	q2 = r.dd.x / b.dd.x;
	r -= (q2 * b);

	q3 = r.dd.x / b.dd.x;

	q1 = quick_two_sum(q1, q2, q2);
	r = gdd_real(q1, q2) + q3;
	return r;
}

#endif


// double-double / double-double
__device__
gdd_real operator/(const gdd_real &a, const gdd_real &b) {
	if (isnan(a) || isnan(b)) {
		return _dd_qnan;
	}
	if ((is_zero(a) && is_zero(b)) || (isinf(a) && isinf(b))) {
		// (0/0), (inf/inf)
		return _dd_qnan;
	}
	if (is_zero(b)) {
		if((is_positive(a) && is_positive(b))
			|| (is_negative(a) && is_negative(b))) {
			// (x/0), (-x/-0)
			return _dd_inf;
		} else {
			// (x/-0), (-x/0)
			return -(_dd_inf);
		}
	}
	if (is_zero(a)) {
		if ((is_positive(a) && is_positive(b))
			|| is_negative(a) && is_negative(b)) {
			// (0/x), (-0/-x)
			return _dd_zero;
		} else {
			// (0/-x), (-0/x)
			return -(_dd_zero);
		}
	}
#ifdef SLOPPY_DIV
	return sloppy_div(a, b);
#else
	return accurate_div(a, b);
#endif
}


// double-double / double
__device__
gdd_real operator/(const gdd_real &a, double b) {
	if (isnan(a) || isnan(b)) {
		return _dd_qnan;
	}
	if ((is_zero(a) && b == 0.0) || (isinf(a) && isinf(b))) {
		// (0/0), (inf/inf)
		return _dd_qnan;
	}
	if (b == 0.0) {
		if ((is_positive(a) && is_positive(b))
			|| (is_negative(a) && is_negative(b))) {
			// (x/0), (-x/-0)
			return _dd_inf;
		} else {
			// (x/-0), (-x/0)
			return -(_dd_inf);
		}
	}
	if (is_zero(a)) {
		if ((is_positive(a) && is_positive(b))
			|| is_negative(a) && is_negative(b)) {
			// (0/x), (-0/-x)
			return _dd_zero;
		} else {
			// (0/-x), (-0/x)
			return -(_dd_zero);
		}
	}

	double q1, q2;
	double p1, p2;
	double s, e;
	gdd_real r;

	q1 = a.dd.x / b;   // approximate quotient.

	// Compute  this - q1 * d
	p1 = two_prod(q1, b, p2);
	s = two_diff(a.dd.x, p1, e);
	e = e + a.dd.y;
	e = e - p2;

	// get next approximation.
	q2 = (s + e) / b;

	// renormalize
	r.dd.x = quick_two_sum(q1, q2, r.dd.y);
	return r;
}


// double / double-double
__device__
gdd_real operator/(double a, const gdd_real &b) {
	return gdd_real(a) / b;
}



// Self-Divisions ==============================================================

// double-double /= double-double
__device__
gdd_real &gdd_real::operator/=(const gdd_real &b) {
	*this = *this / b;
	return *this;
}


// double-double /= double
__device__
gdd_real &gdd_real::operator/=(double b) {
	*this = *this / b;
	return *this;
}



// Comparisons =================================================================
// Equality Comparisons ---------------------------------

// double-double == double
__device__
bool operator==(const gdd_real &a, double b) {
	return (a.dd.x == b && a.dd.y == 0.0);
}

// double == double-double
__device__
bool operator==(double a, const gdd_real &b) {
	return (a == b.dd.x && b.dd.y == 0.0);
}

// double-double == double-double
__device__
bool operator==(const gdd_real &a, const gdd_real &b) {
	return (a.dd.x == b.dd.x && a.dd.y == b.dd.y);
}



// Not-Equal-To Comparisons -----------------------------

// double-double != double
__device__
bool operator!=(const gdd_real &a, double b) {
	return (a.dd.x != b || a.dd.y != 0.0);
}

// double != double-double
__device__
bool operator!=(double a, const gdd_real &b) {
	return (a != b.dd.x || b.dd.y != 0.0);
}

// double-double != double-double
__device__
bool operator!=(const gdd_real &a, const gdd_real &b) {
	return (a.dd.x != b.dd.x || a.dd.y != b.dd.y);
}



// Greater-Than Comparisons -----------------------------

// double-double > double
__device__
bool operator>(const gdd_real &a, double b) {
	return (a.dd.x > b || (a.dd.x == b && a.dd.y > 0.0));
}

// double > double-double
__device__
bool operator>(double a, const gdd_real &b) {
	return (a > b.dd.x || (a == b.dd.x && b.dd.y < 0.0));
}

// double-double > double-double
__device__
bool operator>(const gdd_real &a, const gdd_real &b) {
	return (a.dd.x > b.dd.x || (a.dd.x == b.dd.x && a.dd.y > b.dd.y));
}



// Less-Than Comparisons --------------------------------

// double-double < double
__device__
bool operator<(const gdd_real &a, double b) {
	return (a.dd.x < b || (a.dd.x == b && a.dd.y < 0.0));
}

// double < double-double
__device__
bool operator<(double a, const gdd_real &b) {
	return (a < b.dd.x || (a == b.dd.x && b.dd.y > 0.0));
}

// double-double < double-double
__device__
bool operator<(const gdd_real &a, const gdd_real &b) {
	return (a.dd.x < b.dd.x || (a.dd.x == b.dd.x && a.dd.y < b.dd.y));
}



// Greater-Than-Or-Equal-To Comparisons -----------------

// double-double >= double
__device__
bool operator>=(const gdd_real &a, double b) {
	return (a.dd.x > b || (a.dd.x == b && a.dd.y >= 0.0));
}

// double >= double-double
__device__
bool operator>=(double a, const gdd_real &b) {
	return (b <= a);
}

// double-double >= double-double
__device__
bool operator>=(const gdd_real &a, const gdd_real &b) {
	return (a.dd.x > b.dd.x || (a.dd.x == b.dd.x && a.dd.y >= b.dd.y));
}



// Less-Than-Or-Equal-To Comparisons --------------------

// double-double <= double
__device__
bool operator<=(const gdd_real &a, double b) {
	return (a.dd.x < b || (a.dd.x == b && a.dd.y <= 0.0));
}

// double <= double-double
__device__
bool operator<=(double a, const gdd_real &b) {
	return (b >= a);
}

// double-double <= double-double
__device__
bool operator<=(const gdd_real &a, const gdd_real &b) {
	return (a.dd.x < b.dd.x || (a.dd.x == b.dd.x && a.dd.y <= b.dd.y));
}



// Is functions ================================================================

__device__
bool is_zero(const gdd_real &a) {
	return (a.dd.x == 0.0);
}

__device__
bool is_pzero(const gdd_real &a) {
	return (a.dd.x == 0.0) && is_positive(a);
}

__device__
bool is_nzero(const gdd_real &a) {
	return (a.dd.x == 0.0) && is_negative(a);
}

__device__
bool is_one(const gdd_real &a) {
	return (a.dd.x == 1.0 && a.dd.y == 0.0);
}


__device__
bool is_positive(const gdd_real &a) {
	return !is_negative(a);
}

__device__
bool is_negative(const gdd_real &a) {
	const unsigned __int64 cons = 0x8000000000000000ULL;
	trans t;
	t.asDouble = a.dd.x;
	bool result = ((t.asInt64 & cons) == cons);
	return result;
}

__device__
bool isnan(const gdd_real &a) {
	return (isnan(a.dd.x) || isnan(a.dd.y));
}

__device__
bool isfinite(const gdd_real &a) {
	// Not infinity and not NaN
	return isfinite(a.dd.x);
}

__device__
bool isinf(const gdd_real &a) {
	return isinf(a.dd.x);
}

__device__
bool is_pinf(const gdd_real &a) {
	return (a.dd.x == __dd_inf[1]);
}

__device__
bool is_ninf(const gdd_real &a) {
	return (a.dd.x == -__dd_inf[1]);
}


// Cast functions ==============================================================

__device__
double to_double(const gdd_real &a) {
	return a.dd.x;
}

__device__
int to_int(const gdd_real &a) {
	return static_cast<int>(a.dd.x);
}

__device__
int to_int(double a) {
	return static_cast<int>(a);
}




/********** Remainder **********/
/* return remainder of a/b */
//__device__
//gdd_real drem(const gdd_real &a, const gdd_real &b) {
//	gdd_real n = nint(a / b);
//	return (a - n * b);
//}
//
///* return integer of the quotient, r = remainder of a/b */
//__device__
//gdd_real divrem(const gdd_real &a, const gdd_real &b, gdd_real &r) {
//	gdd_real n = nint(a / b);
//	r = a - n * b;
//	return n;
//}


// Miscellaneous ===============================================================

// Nearest Integer
__device__
gdd_real nint(const gdd_real &a) {
	double hi = nint(a.dd.x);
	double lo;

	if (hi == a.dd.x) {
		// High word is an integer already.  Round the low word.
		lo = nint(a.dd.y);

		// Renormalize. This is needed if x[0] = some integer, x[1] = 1/2.
		hi = quick_two_sum(hi, lo, lo);
	} else {
		// High word is not an integer.
		lo = 0.0;
		if (fabs(hi-a.dd.x) == 0.5 && a.dd.y < 0.0) {
			// There is a tie in the high word, consult the low word to break the tie.
			hi -= 1.0;      // NOTE: This does not cause INEXACT.
		}
	}

	return gdd_real(hi, lo);
}


__device__
gdd_real floor(const gdd_real &a) {
	double hi = std::floor(a.dd.x);
	double lo = 0.0;

	if (hi == a.dd.x) {
		// High word is integer already.  Round the low word.
		lo = std::floor(a.dd.y);
		hi = quick_two_sum(hi, lo, lo);
	}

	return gdd_real(hi, lo);
}


__device__
gdd_real ceil(const gdd_real &a) {
	double hi = std::ceil(a.dd.x);
	double lo = 0.0;

	if (hi == a.dd.x) {
		// High word is integer already.  Round the low word.
		lo = std::ceil(a.dd.y);
		hi = quick_two_sum(hi, lo, lo);
	}

	return gdd_real(hi, lo);
}


__device__
gdd_real aint(const gdd_real &a) {
	return (a.dd.x >= 0.0) ? floor(a) : ceil(a);
}


__device__
gdd_real abs(const gdd_real &a) {
	return (a.dd.x < 0.0) ? negative(a) : a;
}

__device__
gdd_real fabs(const gdd_real &a) {
	return abs(a);
}


// Inverse number
__device__
gdd_real inv(const gdd_real &a) {
	return 1.0 / a;
}


__device__
gdd_real fmod(const gdd_real &a, const gdd_real &b) {
	gdd_real n = aint(a / b);
	return (a - b * n);
}



// Power functions =============================================================

// Computes the n-th power of a double-double number.
// NOTE:  0^0 causes an error.
__device__
gdd_real npwr(const gdd_real &a, int n) {
	if (n == 0) {
		if (is_zero(a)){
			return _dd_qnan;
		}else{
			return gdd_real(1.0);
		}
	}
	if (n == 1) return gdd_real(a);

	gdd_real r = a;
	gdd_real s(1.0);

	for (int i = std::abs(n); i > 0;){
		if (i % 2){	// odd
			s *= r;
			--i;
		}else{		// even
			r = sqr(r);
			i /= 2;
		}
	}

	// Inverse if n is negative.
	if (n < 0.0){
		s = 1.0 / s;
	}
	return s;
}


// Computes double-double ^ int
__device__
gdd_real pow(const gdd_real &a, int n) {
	return npwr(a, n);
}


// Computes double-double ^ double-double
__device__
gdd_real pow(const gdd_real &a, const gdd_real &b) {
	return exp(b * log(a));
}


// Computes double-double ^ int
__device__
gdd_real operator^(const gdd_real &a, int n) {
	return npwr(a, n);
}


// Prohibit: Computes double-double ^ double 
// gdd_real::operator^(double) function is not supported. 
// To avoid incorrect calling when gdd_real::operator^(int) passed double parameter.
// parameter is assumed as int(narrow cast).
__device__
gdd_real gdd_real::operator^(double n){
	return _dd_qnan;
}



// Max Min functions ===========================================================

__device__
gdd_real max(const gdd_real &a, const gdd_real &b) {
	if (isnan(a)) {
		return b;
	}
	if (isnan(b)) {
		return a;
	}

	return (a > b) ? a : b;
}


__device__
gdd_real min(const gdd_real &a, const gdd_real &b) {
	if (isnan(a)) {
		return b;
	}
	if (isnan(b)) {
		return a;
	}

	return (a < b) ? a : b;
}


//__device__
//gdd_real ddrand() {
//	const double m_const = 4.6566128730773926e-10;  /* = 2^{-31} */
//	double m = m_const;
//	gdd_real r = 0.0;
//	double d;
//
//	/* Strategy:  Generate 31 bits at a time, using lrand48
//	random number generator.  Shift the bits, and reapeat
//	4 times. */
//
//	for (int i = 0; i < 4; i++, m *= m_const) {
//		//    d = lrand48() * m;
//		d = std::rand() * m;	// compile error with nvcc in device_code
//		r += d;
//	}
//
//	return r;
//}
//
//__device__
//gdd_real gdd_real::rand(void) {
//	return ddrand();
//}
//
//__device__
//gdd_real gdd_real::debug_rand(void) {
//
//	if (std::rand() % 2 == 0)
//		return ddrand();
//
//	int expn = 0;
//	gdd_real a = 0.0;
//	double d;
//	for (int i = 0; i < 2; i++) {
//		d = std::ldexp(static_cast<double>(std::rand()) / RAND_MAX, -expn);
//		a += d;
//		expn = expn + 54 + std::rand() % 200;
//	}
//	return a;
//}


/* Outputs the double-double number dd. */
//
//std::ostream &operator<<(std::ostream &os, const gdd_real &dd) {
//	bool showpos = (os.flags() & std::ios_base::showpos) != 0;
//	bool uppercase = (os.flags() & std::ios_base::uppercase) != 0;
//	return os << dd.to_string((int)os.precision(), (int)os.width(), os.flags(),
//		showpos, uppercase, os.fill());
//}


void gdd_real::error(const char *msg) {
	if (msg) { std::cerr << "ERROR " << msg <<std::endl; }
}


void round_string_dd(char *s, int precision, int *offset){
/*
	Input string must be all digits or errors will occur.
*/

	int i;
	int D = precision;

	/* Round, handle carry */
	if (s[D - 1] >= '5') {
		s[D - 2]++;

		i = D - 2;
		while (i > 0 && s[i] > '9') {
			s[i] -= 10;
			s[--i]++;
		}
	}

	/* If first digit is 10, shift everything. */
	if (s[0] > '9') {
		// e++; // don't modify exponent here
		for (i = precision; i >= 2; i--) {
			s[i] = s[i - 1];
		}
		s[0] = '1';
		s[1] = '0';

		(*offset)++; // now offset needs to be increased by one
		precision++;
	}

	s[precision] = 0; // add terminator for array
}

__host__
void append_expn(std::string &str, int expn) {
	int k;

	str += (expn < 0 ? '-' : '+');
	expn = std::abs(expn);

	if (expn >= 100) {
		k = (expn / 100);
		str += '0' + k;
		expn -= 100 * k;
	}

	k = (expn / 10);
	str += '0' + k;
	expn -= 10 * k;

	str += '0' + expn;
}


//__host__
//std::string gdd_real::to_string(int precision, int width, std::ios_base::fmtflags fmt,
//	bool showpos, bool uppercase, char fill) const {
//	std::string s;
//	bool fixed = (fmt & std::ios_base::fixed) != 0;
//	bool sgn = true;
//	int i, e = 0;
//
//	if (isnan()) {
//		s = uppercase ? "NAN" : "nan";
//		sgn = false;
//	}
//	else {
//		if (*this < 0.0) {
//			s += '-';
//		}
//		else if (showpos) {
//			s += '+';
//		}
//		else {
//			sgn = false;
//		}
//
//		if (isinf()) {
//			s += uppercase ? "INF" : "inf";
//		}
//		else if (*this == 0.0) {
//			/* Zero case */
//			s += '0';
//			if (precision > 0) {
//				s += '.';
//				s.append(precision, '0');
//			}
//		}
//		else {
//			/* Non-zero case */
//			int off = (fixed ? (1 + ::to_int(floor(log10(abs(*this))))) : 1);
//			int d = precision + off;
//
//			int d_with_extra = d;
//			if (fixed) {
//				d_with_extra = std::max(60, d); // longer than the max accuracy for DD
//			}
//			// highly special case - fixed mode, precision is zero, abs(*this) < 1.0
//			// without this trap a number like 0.9 printed fixed with 0 precision prints as 0
//			// should be rounded to 1.
//			if (fixed && (precision == 0) && (abs(*this) < 1.0)) {
//				if (abs(*this) >= 0.5) {
//					s += '1';
//				}
//				else {
//					s += '0';
//				}
//				return s;
//			}
//
//			// handle near zero to working precision (but not exactly zero)
//			if (fixed && d <= 0) {
//				s += '0';
//				if (precision > 0) {
//					s += '.';
//					s.append(precision, '0');
//				}
//			}
//			else { // default
//
//				char *t; //  = new char[d+1];
//				int j;
//
//				if (fixed) {
//					t = new char[d_with_extra + 1];
//					to_digits(t, e, d_with_extra);
//				}
//				else {
//					t = new char[d + 1];
//					to_digits(t, e, d);
//				}
//
//				if (fixed) {
//					// fix the std::string if it's been computed incorrectly
//					// round here in the decimal std::string if required
//					round_string_dd(t, d + 1, &off);
//
//					if (off > 0) {
//						for (i = 0; i < off; i++) {
//							s += t[i];
//						}
//						if (precision > 0) {
//							s += '.';
//							for (j = 0; j < precision; j++, i++) {
//								s += t[i];
//							}
//						}
//					}
//					else {
//						s += "0.";
//						if (off < 0) {
//							s.append(-off, '0');
//						}
//						for (i = 0; i < d; i++) {
//							s += t[i];
//						}
//					}
//				}
//				else {
//					s += t[0];
//					if (precision > 0) { s += '.'; }
//
//					for (i = 1; (i <= precision) & (i<d + 1); i++) {
//						s += t[i];
//					}
//				}
//				delete[] t;
//			}
//		}
//
//		// trap for improper offset with large values
//		// without this trap, output of values of the for 10^j - 1 fail for j > 28
//		// and are output with the point in the wrong place, leading to a dramatically off value
//		if (fixed && (precision > 0)) {
//			// make sure that the value isn't dramatically larger
//			double fromstring = atof(s.c_str());
//
//			// if this ratio is large, then we've got problems
//			if (fabs(fromstring / dd.x) > 3.0) {
//
//				// loop on the std::string, find the point, move it up one
//				// don't act on the first character
//				for (i = 1; (unsigned int)i < s.length(); i++) {
//					if (s[i] == '.') {
//						s[i] = s[i - 1];
//						s[i - 1] = '.';
//						break;
//					}
//				}
//
//				fromstring = atof(s.c_str());
//				// if this ratio is large, then the std::string has not been fixed
//				if (fabs(fromstring / dd.x) > 3.0) {
//					gdd_real::error("Re-rounding unsuccessful in large number fixed point trap.");
//				}
//			}
//		}
//
//
//		if (!fixed && !isinf()) {
//			/* Fill in exponent part */
//			s += uppercase ? 'E' : 'e';
//			append_expn(s, e);
//		}
//	}
//
//	/* Fill in the blanks */
//	int len = (int)s.length();
//	if (len < width) {
//		int delta = width - len;
//		if (fmt & std::ios_base::internal) {
//			if (sgn) {
//				s.insert(static_cast<std::string::size_type>(1), delta, fill);
//			}
//			else {
//				s.insert(static_cast<std::string::size_type>(0), delta, fill);
//			}
//		}
//		else if (fmt & std::ios_base::left) {
//			s.append(delta, fill);
//		}
//		else {
//			s.insert(static_cast<std::string::size_type>(0), delta, fill);
//		}
//	}
//
//	return s;
//}

//__host__
//void gdd_real::to_digits(char *s, int &expn, int precision) const {
//	int D = precision + 1;	/* number of digits to compute */
//
//	gdd_real r = abs(*this);
//	int e;	/* exponent */
//	int i, d;
//
//	if (dd.x == 0.0) {
//		/* this == 0.0 */
//		expn = 0;
//		for (i = 0; i < precision; i++) {
//			s[i] = '0';
//		}
//		return;
//	}
//
//	/* First determine the (approximate) exponent. */
//	e = ::to_int(std::floor(std::log10(std::abs(dd.x))));
//
//	if (e < -300) {
//		r *= gdd_real(10.0) ^ 300;
//		r /= gdd_real(10.0) ^ (e + 300);
//	}
//	else if (e > 300) {
//		r = ldexp(r, -53);
//		r /= gdd_real(10.0) ^ e;
//		r = ldexp(r, 53);
//	}
//	else {
//		r /= gdd_real(10.0) ^ e;
//	}
//
//	/* Fix exponent if we are off by one */
//	if (r >= 10.0) {
//		r /= 10.0;
//		e++;
//	}
//	else if (r < 1.0) {
//		r *= 10.0;
//		e--;
//	}
//
//	if (r >= 10.0 || r < 1.0) {
//		//gdd_real::error("(gdd_real::to_digits): can't compute exponent.");
//		return;
//	}
//
//	/* Extract the digits */
//	for (i = 0; i < D; i++) {
//		d = static_cast<int>(r.dd.x);
//		r -= d;
//		r *= 10.0;
//
//		s[i] = static_cast<char>(d + '0');
//	}
//
//	/* Fix out of range digits. */
//	for (i = D - 1; i > 0; i--) {
//		if (s[i] < '0') {
//			s[i - 1]--;
//			s[i] += 10;
//		}
//		else if (s[i] > '9') {
//			s[i - 1]++;
//			s[i] -= 10;
//		}
//	}
//
//	if (s[0] <= '0') {
//		//gdd_real::error("(gdd_real::to_digits): non-positive leading digit.");
//		return;
//	}
//
//	/* Round, handle carry */
//	if (s[D - 1] >= '5') {
//		s[D - 2]++;
//
//		i = D - 2;
//		while (i > 0 && s[i] > '9') {
//			s[i] -= 10;
//			s[--i]++;
//		}
//	}
//
//	/* If first digit is 10, shift everything. */
//	if (s[0] > '9') {
//		e++;
//		for (i = precision; i >= 2; i--) {
//			s[i] = s[i - 1];
//		}
//		s[0] = '1';
//		s[1] = '0';
//	}
//
//	s[precision] = 0;
//	expn = e;
//}



// return double-double = double + double
__device__
gdd_real dd_add(double a, double b) {
	double s, e;
	s = two_sum(a, b, e);
	return gdd_real(s, e);
}

// return double-double = double - double
__device__
gdd_real dd_sub(double a, double b) {
	double s, e;
	s = two_diff(a, b, e);
	return gdd_real(s, e);
}

// return double-double = double * double
__device__
gdd_real dd_mul(double a, double b) {
	double p, e;
	p = two_prod(a, b, e);
	return gdd_real(p, e);
}

// return double-double = double / double
__device__
gdd_real dd_div(double a, double b) {
	double q1, q2;
	double p1, p2;
	double s, e;

	q1 = a / b;

	// Compute  a - q1 * b
	p1 = two_prod(q1, b, p2);
	s = two_diff(a, p1, e);
	e -= p2;

	// get next approximation
	q2 = (s + e) / b;

	s = quick_two_sum(q1, q2, e);

	return gdd_real(s, e);
}

// return double-double = sqr(double)
__device__
gdd_real dd_sqr(double a) {
	double p1, p2;
	p1 = two_sqr(a, p2);
	return gdd_real(p1, p2);
}


#endif /* __GDD_BASIC_CU__ */

