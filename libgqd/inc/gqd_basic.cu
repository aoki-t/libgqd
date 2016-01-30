#ifndef __GQD_BASIC_CU__
#define __GQD_BASIC_CU__

#include "gqd_real.h"
#include "inline.cu"

//#include "common.cu"

//__device__ __constant__ double _qd_eps = (1.21543267145725e-63); // = 2^-209
__device__ __constant__ double _qd_eps = (1.215432671457254239e-63); // = 2^-209


__device__ __constant__ double __qd_zero = 0.0;
__device__ __constant__ double __qd_one = 1.0;
__device__ __constant__ double  __qd_inf[2];	// h_inf, CUDART_INF;
__device__ __constant__ double __qd_qnan[2];	// h_qnan, CUDART_NAN;

__device__ __constant__ double      __qd_e[4] = { 2.718281828459045091e+00, 1.445646891729250158e-16, -2.127717108038176765e-33, 1.515630159841218954e-49 };
__device__ __constant__ double   __qd_log2[4] = { 6.931471805599452862e-01, 2.319046813846299558e-17, 5.707708438416212066e-34, -3.582432210601811423e-50 };
__device__ __constant__ double  __qd_log10[4] = { 2.302585092994045901e+00, -2.170756223382249351e-16, -9.984262454465776570e-33, -4.023357454450206379e-49 };
__device__ __constant__ double    __qd_2pi[4] = { 6.283185307179586232e+00, 2.449293598294706414e-16, -5.989539619436679332e-33, 2.224908441726730563e-49 };
__device__ __constant__ double     __qd_pi[4] = { 3.141592653589793116e+00, 1.224646799147353207e-16, -2.994769809718339666e-33, 1.112454220863365282e-49 };
__device__ __constant__ double    __qd_pi2[4] = { 1.570796326794896558e+00, 6.123233995736766036e-17, -1.497384904859169833e-33, 5.562271104316826408e-50 };
__device__ __constant__ double __qd_pi1024[4] = { 3.067961575771282340e-03, 1.195944139792337116e-19, -2.924579892303066080e-36, 1.086381075061880158e-52 };
__device__ __constant__ double    __qd_pi4[4] = { 7.853981633974482790e-01, 3.061616997868383018e-17, -7.486924524295849165e-34, 2.781135552158413204e-50 };
__device__ __constant__ double   __qd_3pi4[4] = { 2.356194490192344837e+00, 9.1848509936051484375e-17, 3.9168984647504003225e-33, -2.5867981632704860386e-49 };


// Constructors ================================================================
// default constructor
__device__ __host__
gqd_real::gqd_real()
: qd({ 0.0, 0.0, 0.0, 0.0 }) {
}


// destructor
__device__ __host__
gqd_real::~gqd_real(){
}


__device__ __host__ 
gqd_real::gqd_real(double x0, double x1, double x2, double x3) {
	qd.x = x0;
	qd.y = x1;
	qd.z = x2;
	qd.w = x3;
}


__device__ __host__
gqd_real::gqd_real(double d) {
	qd.x = d;
	qd.y = qd.z = qd.w = 0.0;
}


__device__ __host__
gqd_real::gqd_real(const double *d4) {
	qd.x = d4[0];
	qd.y = d4[1];
	qd.z = d4[2];
	qd.w = d4[3];
}


__device__ __host__
gqd_real::gqd_real(const gqd_real &a) {
	qd.x = a.qd.x;
	qd.y = a.qd.y;
	qd.z = a.qd.z;
	qd.w = a.qd.w;

}

__device__ __host__
gqd_real::gqd_real(const gdd_real &a) {
	qd.x = a.dd.x;
	qd.y = a.dd.y;
	qd.z = qd.w = 0.0;
}


__device__ __host__
gqd_real::gqd_real(int i) {
	qd.x = static_cast<double>(i);
	qd.y = qd.z = qd.w = 0.0;
}



// Accessors ===================================================================
__device__ __host__
double gqd_real::operator[](int i) const {
	assert(i < 4);

	i = i % 4;
	switch (i) {
	case 0:
		return qd.x;
	case 1:
		return qd.y;
	case 2:
		return qd.z;
	case 3:
		return qd.w;
	default:
#ifdef __CUDACC__
		return __qd_qnan[1];
#else
		return std::numeric_limits<double>::quiet_NaN();
#endif
	}
}


__device__ __host__
double &gqd_real::operator[](int i) {
	assert(i < 4);

	switch (i) {
	case 0:
		return qd.x;
	case 1:
		return qd.y;
	case 2:
		return qd.z;
	case 3:
		return qd.w;
	default:
		return qd.x;	// adohoc
	}
}



// Assignments =================================================================
// quad-double = double
__device__ __host__
gqd_real &gqd_real::operator=(double a) {
	qd.x = a;
	qd.y = qd.z = qd.w = 0.0;
	return *this;
}


// quad-double = double-double
__device__ __host__
gqd_real &gqd_real::operator=(const gdd_real &a) {
	qd.x = a.dd.x;
	qd.y = a.dd.y;
	qd.z = qd.w = 0.0;
	return *this;
}


// quad-double = quad-double
__device__ __host__
gqd_real &gqd_real::operator=(const gqd_real &a) {
	qd.x = a[0];
	qd.y = a[1];
	qd.z = a[2];
	qd.w = a[3];
	return *this;
}



__device__ __host__
gqd_real gqd_real::operator-() const {
	return gqd_real(-qd.x, -qd.y, -qd.z, -qd.w);
}


__device__ __host__
gqd_real negative(const gqd_real &a) {
	return gqd_real(-a.qd.x, -a.qd.y, -a.qd.z, -a.qd.w);
}



// normalization functions
__device__ __host__
void quick_renorm(double &c0, double &c1, 
				  double &c2, double &c3, double &c4) {
	double t0, t1, t2, t3;
	double s;
	s  = quick_two_sum(c3, c4, t3);
	s  = quick_two_sum(c2, s , t2);
	s  = quick_two_sum(c1, s , t1);
	c0 = quick_two_sum(c0, s , t0);

	s  = quick_two_sum(t2, t3, t2);
	s  = quick_two_sum(t1, s , t1);
	c1 = quick_two_sum(t0, s , t0);

	s  = quick_two_sum(t1, t2, t1);
	c2 = quick_two_sum(t0, s , t0);

	c3 = t0 + t1;
}


__device__ __host__
void renorm(double &c0, double &c1, 
			double &c2, double &c3) {
	double s0, s1, s2 = 0.0, s3 = 0.0;

	//if (isinf(c0)) return;

	s0 = quick_two_sum(c2, c3, c3);
	s0 = quick_two_sum(c1, s0, c2);
	c0 = quick_two_sum(c0, s0, c1);

	s0 = c0;
	s1 = c1;
	if (s1 != 0.0) {
		s1 = quick_two_sum(s1, c2, s2);
		if (s2 != 0.0)
			s2 = quick_two_sum(s2, c3, s3);
		else
			s1 = quick_two_sum(s1, c3, s2);
	} else {
		s0 = quick_two_sum(s0, c2, s1);
		if (s1 != 0.0)
			s1 = quick_two_sum(s1, c3, s2);
		else
			s0 = quick_two_sum(s0, c3, s1);
	}

	c0 = s0;
	c1 = s1;
	c2 = s2;
	c3 = s3;
}


__device__ __host__
void renorm(double &c0, double &c1, 
			double &c2, double &c3, double &c4) {
	double s0, s1, s2 = 0.0, s3 = 0.0;

	//if (isinf(c0)) return;

	s0 = quick_two_sum(c3, c4, c4);
	s0 = quick_two_sum(c2, s0, c3);
	s0 = quick_two_sum(c1, s0, c2);
	c0 = quick_two_sum(c0, s0, c1);

	s0 = c0;
	s1 = c1;

	s0 = quick_two_sum(c0, c1, s1);
	if (s1 != 0.0) 
	{
		s1 = quick_two_sum(s1, c2, s2);
		if (s2 != 0.0) {
			s2 = quick_two_sum(s2, c3, s3);
			if (s3 != 0.0)
				s3 += c4;
			else
				s2 += c4;
		} else {
			s1 = quick_two_sum(s1, c3, s2);
			if (s2 != 0.0)
				s2 = quick_two_sum(s2, c4, s3);
			else
				s1 = quick_two_sum(s1, c4, s2);
		}
	} else {
		s0 = quick_two_sum(s0, c2, s1);
		if (s1 != 0.0) {
			s1 = quick_two_sum(s1, c3, s2);
			if (s2 != 0.0)
				s2 = quick_two_sum(s2, c4, s3);
			else
				s1 = quick_two_sum(s1, c4, s2);
		} else {
			s0 = quick_two_sum(s0, c3, s1);
			if (s1 != 0.0)
				s1 = quick_two_sum(s1, c4, s2);
			else
				s0 = quick_two_sum(s0, c4, s1);
		}
	}

	c0 = s0;
	c1 = s1;
	c2 = s2;
	c3 = s3;
}


__device__ __host__
void gqd_real::renorm() {
	::renorm(qd.x, qd.y, qd.z, qd.w);
}


__device__ __host__
void gqd_real::renorm(double &e) {
	::renorm(qd.x, qd.y, qd.z, qd.w, e);
}



// Inline funciotns ============================================================
__forceinline__ __device__ __host__
void three_sum(double &a, double &b, double &c) {
	double t1, t2, t3;
	t1 = two_sum(a, b, t2);
	a  = two_sum(c, t1, t3);
	b  = two_sum(t2, t3, c);
}


__forceinline__ __device__ __host__
void three_sum2(double &a, double &b, double &c) {
	double t1, t2, t3;
	t1 = two_sum(a, b, t2);
	a  = two_sum(c, t1, t3);
	b = (t2 + t3);
}


// s = quick_three_accum(a, b, c) adds c to the dd-pair (a, b).
// If the result does not fit in two doubles, then the sum is
// output into s and (a,b) contains the remainder.  Otherwise
// s is zero and (a,b) contains the sum.
__forceinline__ __device__ __host__
double quick_three_accum(double &a, double &b, double c) {
	double s;
	bool za, zb;

	s = two_sum(b, c, b);
	s = two_sum(a, s, a);

	za = (a != 0.0);
	zb = (b != 0.0);

	if (za && zb)
		return s;

	if (!zb) {
		b = a;
		a = s;
	} else {
		a = s;
	}

	return 0.0;
}



// Additions ===================================================================

// quad-double + double
__device__ __host__
gqd_real operator+(const gqd_real &a, double b) {
	double c0, c1, c2, c3;
	double e;

	c0 = two_sum(a[0], b, e);
	c1 = two_sum(a[1], e, e);
	c2 = two_sum(a[2], e, e);
	c3 = two_sum(a[3], e, e);

	renorm(c0, c1, c2, c3, e);

	return gqd_real(c0, c1, c2, c3);
}


// double + quad-double
__device__ __host__
gqd_real operator+( double a, const gqd_real &b ) {
	return ( b + a );
}


// quad-double + quad-double
#ifdef SLOPPY_ADD
__forceinline__ __device__ __host__
gqd_real sloppy_add(const gqd_real &a, const gqd_real &b) {
	double s0, s1, s2, s3;
	double t0, t1, t2, t3;

	double v0, v1, v2, v3;
	double u0, u1, u2, u3;
	double w0, w1, w2, w3;

	s0 = a[0] + b[0];
	s1 = a[1] + b[1];
	s2 = a[2] + b[2];
	s3 = a[3] + b[3];

	v0 = s0 - a[0];
	v1 = s1 - a[1];
	v2 = s2 - a[2];
	v3 = s3 - a[3];

	u0 = s0 - v0;
	u1 = s1 - v1;
	u2 = s2 - v2;
	u3 = s3 - v3;

	w0 = a[0] - u0;
	w1 = a[1] - u1;
	w2 = a[2] - u2;
	w3 = a[3] - u3;

	u0 = b[0] - v0;
	u1 = b[1] - v1;
	u2 = b[2] - v2;
	u3 = b[3] - v3;

	t0 = w0 + u0;
	t1 = w1 + u1;
	t2 = w2 + u2;
	t3 = w3 + u3;

	s1 = two_sum(s1, t0, t0);
	three_sum(s2, t0, t1);
	three_sum2(s3, t0, t2);
	t0 = t0 + t1 + t3;

	renorm(s0, s1, s2, s3, t0);

	return gqd_real(s0, s1, s2, s3);
}

#else
__forceinline__ __device__ __host__
gqd_real ieee_add(const gqd_real &a, const gqd_real &b) {
	int i, j, k;
	double s, t;
	double u, v;	 // double-length accumulator
	double x[4] = { 0.0, 0.0, 0.0, 0.0 };
	double aa[4];
	double bb[4];

	aa[0] = a[0];
	aa[1] = a[1];
	aa[2] = a[2];
	aa[3] = a[3];

	bb[0] = b[0];
	bb[1] = b[1];
	bb[2] = b[2];
	bb[3] = b[3];

	i = j = k = 0;
	if (std::abs(aa[i]) > std::abs(bb[j])) {
		u = aa[i++];
	} else {
		u = bb[j++];
	}
	if (std::abs(aa[i]) > std::abs(bb[j])) {
		v = aa[i++];
	} else {
		v = bb[j++];
	}
	u = quick_two_sum(u, v, v);

	//if (std::abs(a.x) > std::abs(b.x)) {
	//	u = a.x;
	//	if (std::abs(a.y) > std::abs(b.x)) {
	//		v = a.y;
	//	}else {
	//		v = b.x;
	//	}

	//} else {
	//	u = b.x;
	//	if (std::abs(a.x) > std::abs(b.y)) {
	//		v = a.x;
	//	} else {
	//		v = b.y;
	//	}

	//}
	//u = quick_two_sum(u, v, v);

	while (k < 4) {
		if (i >= 4 && j >= 4) {
			x[k] = u;
			if (k < 3) {
				x[++k] = v;
			}
			break;
		}

		if (i >= 4) {
			t = bb[j++];
		} else if (j >= 4) {
			t = aa[i++];
		} else if (std::abs(aa[i]) > std::abs(bb[j])) {
			t = aa[i++];
		} else {
			t = bb[j++];
		}
		s = quick_three_accum(u, v, t);

		if (s != 0.0) {
			x[k++] = s;
		}
	}

	// add the rest.
	for (k = i; k < 4; k++) {
		x[3] += aa[k];
	}
	for (k = j; k < 4; k++) {
		x[3] += bb[k];
	}

	renorm(x[0], x[1], x[2], x[3]);
	return gqd_real(x[0], x[1], x[2], x[3]);
}
#endif


__device__ __host__
gqd_real operator+(const gqd_real &a, const gqd_real &b) {
#ifdef SLOPPY_ADD
	return sloppy_add(a, b);
#else
	return ieee_add(a, b);
#endif
}


// quad-double + double-double
__device__ __host__
gqd_real operator+(const gqd_real &a, const gdd_real &b) {

	double s0, s1, s2, s3;
	double t0, t1;

	s0 = two_sum(a[0], b.dd.x, t0);
	s1 = two_sum(a[1], b.dd.y, t1);

	s1 = two_sum(s1, t0, t0);

	s2 = a[2];
	three_sum(s2, t0, t1);

	s3 = two_sum(t0, a[3], t0);
	t0 += t1;

	renorm(s0, s1, s2, s3, t0);
	return gqd_real(s0, s1, s2, s3);
}


// double-double + quad-double
__device__ __host__
gqd_real operator+(const gdd_real &a, const gqd_real &b) {
	return (b + a);
}



// Self-Additions ==============================================================

// quad-double += double
__device__ __host__
gqd_real &gqd_real::operator+=(double b) {
	double c0, c1, c2, c3;
	double e;

	c0 = two_sum(qd.x, b, e);
	c1 = two_sum(qd.y, e, e);
	c2 = two_sum(qd.z, e, e);
	c3 = two_sum(qd.w, e, e);

	::renorm(c0, c1, c2, c3, e);

	qd.x = c0;
	qd.y = c1;
	qd.z = c2;
	qd.w = c3;

	return *this;
}


// quad-double += double-double
__device__ __host__
gqd_real &gqd_real::operator+=(const gdd_real &a) {
	*this = *this + a;
	return *this;
}


// quad-double += quad-double
__device__ __host__
gqd_real &gqd_real::operator+=(const gqd_real &a) {
	*this = *this + a;
	return *this;
}



// Subtractions ================================================================
// quad-double - double
__device__ __host__
gqd_real operator-(const gqd_real &a, double b) {
	return (a + (-b));
}


// double - quad-double
__device__ __host__
gqd_real operator-(double a, const gqd_real &b) {
	return (a + negative(b));
}


// quad-double - quad-double
__device__ __host__
gqd_real operator-(const gqd_real &a, const gqd_real &b) {
	return (a + negative(b));
}


// quad-double - double-double
__device__ __host__
gqd_real operator-(const gqd_real &a, const gdd_real &b) {
	return (a + negative(b));
}


// double-double - quad-double
__device__ __host__
gqd_real operator-(const gdd_real &a, const gqd_real &b) {
	return (a + negative(b));
}


// Self-Subtractions ===========================================================
// quad-double -= double
__device__ __host__
gqd_real &gqd_real::operator-=(double a) {
	return ((*this) += (-a));
}


// quad-double -= double-double
__device__ __host__
gqd_real &gqd_real::operator-=(const gdd_real &a) {
	return ((*this) += (-a));
}


// quad-double -= quad-double
__device__ __host__
gqd_real &gqd_real::operator-=(const gqd_real &a) {
	return ((*this) += (-a));
}



// Multiplications =============================================================

// quad-double * (2.0 ^ exp)
__device__ __host__
gqd_real ldexp(const gqd_real &a, int n) {
	return gqd_real(std::ldexp(a[0], n), std::ldexp(a[1], n), 
					std::ldexp(a[2], n), std::ldexp(a[3], n));
}

// quad-double * double,  where double is a power of 2.
__device__ __host__
gqd_real mul_pwr2(const gqd_real &a, double b) {
	return gqd_real(a[0] * b, a[1] * b, a[2] * b, a[3] * b);
}


// quad_double * double
__device__ __host__
gqd_real operator*(const gqd_real &a, double b) {
	double p0, p1, p2, p3;
	double q0, q1, q2;
	double s0, s1, s2, s3, s4;

	p0 = two_prod(a[0], b, q0);
	p1 = two_prod(a[1], b, q1);
	p2 = two_prod(a[2], b, q2);
	p3 = a[3] * b;

	s0 = p0;

	s1 = two_sum(q0, p1, s2);

	three_sum(s2, q1, p2);

	three_sum2(q1, q2, p3);
	s3 = q1;

	s4 = q2 + p2;

	renorm(s0, s1, s2, s3, s4);
	return gqd_real(s0, s1, s2, s3);
}


// double * quad-double
__device__ __host__
gqd_real operator*(double a, const gqd_real &b) {
	return b*a;
}


// quad-double * quad-double
// a0 * b0                    0
//      a0 * b1               1
//      a1 * b0               2
//           a0 * b2          3
//           a1 * b1          4
//           a2 * b0          5
//                a0 * b3     6
//                a1 * b2     7
//                a2 * b1     8
//                a3 * b0     9
#ifdef SLOPPY_MUL
__device__ __host__
static gqd_real sloppy_mul(const gqd_real &a, const gqd_real &b) {
	double p0, p1, p2, p3, p4, p5;
	double q0, q1, q2, q3, q4, q5;
	double t0, t1;
	double s0, s1, s2;

	p0 = two_prod(a[0], b[0], q0);

	p1 = two_prod(a[0], b[1], q1);
	p2 = two_prod(a[1], b[0], q2);

	p3 = two_prod(a[0], b[2], q3);
	p4 = two_prod(a[1], b[1], q4);
	p5 = two_prod(a[2], b[0], q5);


	// Start Accumulation
	three_sum(p1, p2, q0);

	//return gqd_real(p1, p2, q0, 0.0);

	// Six-Three Sum  of p2, q1, q2, p3, p4, p5.
	three_sum(p2, q1, q2);
	three_sum(p3, p4, p5);
	// compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5).
	s0 = two_sum(p2, p3, t0);
	s1 = two_sum(q1, p4, t1);
	s2 = q2 + p5;
	s1 = two_sum(s1, t0, t0);
	s2 += (t0 + t1);

	//return gqd_real(s0, s1, t0, t1);

	// O(eps^3) order terms
	s1 = s1 + (a[0]*b[3] + a[1]*b[2] + a[2]*b[1] + a[3]*b[0] + q0 + q3 + q4 + q5);
	
	//s1 = s1 + (__dmul_rn(a[0],b[3]) + __dmul_rn(a[1],b[2]) + 
	//		__dmul_rn(a[2],b[1]) + __dmul_rn(a[3],b[0]) + q0 + q3 + q4 + q5);
	renorm(p0, p1, s0, s1, s2);

	return gqd_real(p0, p1, s0, s1);
	
}
#else
__device__ __host__
static gqd_real accurate_mul(const gqd_real &a, const gqd_real &b) {
	double p0, p1, p2, p3, p4, p5;
	double q0, q1, q2, q3, q4, q5;
	double p6, p7, p8, p9;
	double q6, q7, q8, q9;
	double r0, r1;
	double t0, t1;
	double s0, s1, s2;

	p0 = two_prod(a[0], b[0], q0);

	p1 = two_prod(a[0], b[1], q1);
	p2 = two_prod(a[1], b[0], q2);

	p3 = two_prod(a[0], b[2], q3);
	p4 = two_prod(a[1], b[1], q4);
	p5 = two_prod(a[2], b[0], q5);

	// Start Accumulation
	three_sum(p1, p2, q0);

	// Six-Three Sum  of p2, q1, q2, p3, p4, p5.
	three_sum(p2, q1, q2);
	three_sum(p3, p4, p5);
	// compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5).
	s0 = two_sum(p2, p3, t0);
	s1 = two_sum(q1, p4, t1);
	s2 = q2 + p5;
	s1 = two_sum(s1, t0, t0);
	s2 += (t0 + t1);

	// O(eps^3) order terms
	p6 = two_prod(a[0], b[3], q6);
	p7 = two_prod(a[1], b[2], q7);
	p8 = two_prod(a[2], b[1], q8);
	p9 = two_prod(a[3], b[0], q9);

	// Nine-Two-Sum of q0, s1, q3, q4, q5, p6, p7, p8, p9.
	q0 = two_sum(q0, q3, q3);
	q4 = two_sum(q4, q5, q5);
	p6 = two_sum(p6, p7, p7);
	p8 = two_sum(p8, p9, p9);
	// Compute (t0, t1) = (q0, q3) + (q4, q5).
	t0 = two_sum(q0, q4, t1);
	t1 += (q3 + q5);
	// Compute (r0, r1) = (p6, p7) + (p8, p9).
	r0 = two_sum(p6, p8, r1);
	r1 += (p7 + p9);
	// Compute (q3, q4) = (t0, t1) + (r0, r1).
	q3 = two_sum(t0, r0, q4);
	q4 += (t1 + r1);
	// Compute (t0, t1) = (q3, q4) + s1.
	t0 = two_sum(q3, s1, t1);
	t1 += q4;

	// O(eps^4) terms -- Nine-One-Sum
	t1 += a[1] * b[3] + a[2] * b[2] + a[3] * b[1] + q6 + q7 + q8 + q9 + s2;

	renorm(p0, p1, s0, t0, t1);
	return gqd_real(p0, p1, s0, t0);
}
#endif


__device__ __host__
gqd_real operator*(const gqd_real &a, const gqd_real &b) {
#ifdef SLOPPY_MUL
	return sloppy_mul(a, b);
#else
	return accurate_mul(a, b);
#endif
}


// quad-double * double-double
// a0 * b0                        0
//      a0 * b1                   1
//      a1 * b0                   2
//           a1 * b1              3
//           a2 * b0              4
//                a2 * b1         5
//                a3 * b0         6
//                     a3 * b1    7
__device__ __host__
gqd_real operator*(const gqd_real &a, const gdd_real &b) {
	double p0, p1, p2, p3, p4;
	double q0, q1, q2, q3, q4;
	double s0, s1, s2;
	double t0, t1;

	p0 = two_prod(a[0], b.dd.x, q0);
	p1 = two_prod(a[0], b.dd.y, q1);
	p2 = two_prod(a[1], b.dd.x, q2);
	p3 = two_prod(a[1], b.dd.y, q3);
	p4 = two_prod(a[2], b.dd.x, q4);

	three_sum(p1, p2, q0);

	// Five-Three-Sum
	three_sum(p2, p3, p4);
	q1 = two_sum(q1, q2, q2);
	s0 = two_sum(p2, q1, t0);
	s1 = two_sum(p3, q2, t1);
	s1 = two_sum(s1, t0, t0);
	s2 = t0 + t1 + p4;
	p2 = s0;

	p3 = a[2] * b.dd.x + a[3] * b.dd.y + q3 + q4;
	three_sum2(p3, q0, s1);
	p4 = q0 + s2;

	renorm(p0, p1, p2, p3, p4);
	return gqd_real(p0, p1, p2, p3);
}


// double-double * quad-double
__device__ __host__
gqd_real operator*(const gdd_real &a, const gqd_real &b) {
	return (b * a);
}


// Squaring ====================================================================
__device__ __host__
gqd_real sqr(const gqd_real &a) {
	double p0, p1, p2, p3, p4, p5;
	double q0, q1, q2, q3;
	double s0, s1;
	double t0, t1;

	p0 = two_sqr(a[0], q0);
	p1 = two_prod(2.0 * a[0], a[1], q1);
	p2 = two_prod(2.0 * a[0], a[2], q2);
	p3 = two_sqr(a[1], q3);

	p1 = two_sum(q0, p1, q0);

	q0 = two_sum(q0, q1, q1);
	p2 = two_sum(p2, p3, p3);

	s0 = two_sum(q0, p2, t0);
	s1 = two_sum(q1, p3, t1);

	s1 = two_sum(s1, t0, t0);
	t0 += t1;

	s1 = quick_two_sum(s1, t0, t0);
	p2 = quick_two_sum(s0, s1, t1);
	p3 = quick_two_sum(t1, t0, q0);

	p4 = 2.0 * a[0] * a[3];
	p5 = 2.0 * a[1] * a[2];

	p4 = two_sum(p4, p5, p5);
	q2 = two_sum(q2, q3, q3);

	t0 = two_sum(p4, q2, t1);
	t1 = t1 + p5 + q3;

	p3 = two_sum(p3, t0, p4);
	p4 = p4 + q0 + t1;

	renorm(p0, p1, p2, p3, p4);
	return gqd_real(p0, p1, p2, p3);
}



// Self-Multiplications ========================================================
// quad-double *= double
__device__ __host__
gqd_real &gqd_real::operator*=(double a) {
	*this = (*this * a);
	return *this;
}


// quad-double *= double-double
__device__ __host__
gqd_real &gqd_real::operator*=(const gdd_real &a) {
	*this = (*this * a);
	return *this;
}


// quad-double *= quad-double
__device__ __host__
gqd_real &gqd_real::operator*=(const gqd_real &a) {
	*this = *this * a;
	return *this;
}



// Divisions ===================================================================
// quad-double / double
__device__ __host__
gqd_real operator/(const gqd_real &a, double b) {
/*  Strategy:
	compute approximate quotient using high order
	doubles, and then correct it 3 times using the remainder.
	(Analogous to long division.)
*/
	double t0, t1;
	double q0, q1, q2, q3;
	gqd_real r;

	q0 = a[0] / b;  // approximate quotient

	// Compute the remainder  a - q0 * b
	t0 = two_prod(q0, b, t1);
	r = a - gdd_real(t0, t1);

	// Compute the first correction
	q1 = r[0] / b;
	t0 = two_prod(q1, b, t1);
	r -= gdd_real(t0, t1);

	// Second correction to the quotient.
	q2 = r[0] / b;
	t0 = two_prod(q2, b, t1);
	r -= gdd_real(t0, t1);

	// Final correction to the quotient.
	q3 = r[0] / b;

	renorm(q0, q1, q2, q3);
	return gqd_real(q0, q1, q2, q3);
}


// double / quad-double
__device__ __host__
gqd_real operator/(double a, const gqd_real &b) {
	return gqd_real(a) / b;
}


// quad-double / quad-double
#ifdef SLOPPY_DIV
__forceinline__ __device__ __host__
static gqd_real sloppy_div(const gqd_real &a, const gqd_real &b) {
	double q0, q1, q2, q3;

	gqd_real r;

	q0 = a[0] / b[0];
	r = a - (b * q0);

	q1 = r[0] / b[0];
	r = r - (b * q1);

	q2 = r[0] / b[0];
	r = r - (b * q2);

	q3 = r[0] / b[0];

	renorm(q0, q1, q2, q3);

	return gqd_real(q0, q1, q2, q3);
}
#else
__forceinline__ __device__ __host__
static gqd_real accurate_div(const gqd_real &a, const gqd_real &b) {
	double q0, q1, q2, q3;

	gqd_real r;

	q0 = a[0] / b[0];
	r = a - (b * q0);

	q1 = r[0] / b[0];
	r -= (b * q1);

	q2 = r[0] / b[0];
	r -= (b * q2);

	q3 = r[0] / b[0];

	r -= (b * q3);
	double q4 = r[0] / b[0];

	renorm(q0, q1, q2, q3, q4);

	return gqd_real(q0, q1, q2, q3);
}

#endif


__device__ __host__
gqd_real operator/(const gqd_real &a, const gqd_real &b) {
#ifdef SLOPPY_DIV
	return sloppy_div(a, b);
#else
	return accurate_div(a, b);
#endif
}


// quad-double / double-double
#ifdef SLOPPY_DIV
__forceinline__ __device__ __host__
gqd_real sloppy_div(const gqd_real &a, const gdd_real &b) {
	double q0, q1, q2, q3;
	gqd_real r;
	gqd_real qd_b(b);

	q0 = a[0] / b.dd.x;
	r = a - q0 * qd_b;

	q1 = r[0] / b.dd.x;
	r -= (q1 * qd_b);

	q2 = r[0] / b.dd.x;
	r -= (q2 * qd_b);

	q3 = r[0] / b.dd.x;

	renorm(q0, q1, q2, q3);
	return gqd_real(q0, q1, q2, q3);
}
#else

__forceinline__ __device__ __host__
gqd_real accurate_div(const gqd_real &a, const gdd_real &b) {
	double q0, q1, q2, q3, q4;
	gqd_real r;
	gqd_real qd_b(b);

	q0 = a[0] / b.dd.x;
	r = a - q0 * qd_b;

	q1 = r[0] / b.dd.x;
	r -= (q1 * qd_b);

	q2 = r[0] / b.dd.x;
	r -= (q2 * qd_b);

	q3 = r[0] / b.dd.x;
	r -= (q3 * qd_b);

	q4 = r[0] / b.dd.x;

	renorm(q0, q1, q2, q3, q4);
	return gqd_real(q0, q1, q2, q3);
}
#endif


__device__ __host__
gqd_real operator/(const gqd_real &a, const gdd_real &b) {
#ifdef SLOPPY_DIV
	return sloppy_div(a, b);
#else
	return accurate_div(a, b);
#endif
}


// double-double / quad-double
__device__ __host__
gqd_real operator/(const gdd_real &a, const gqd_real &b) {
	return gqd_real(a) / b;
}



// Self-Divisions ==============================================================
// quad-double /= double
gqd_real &gqd_real::operator/=(double a) {
	*this = (*this / a);
	return *this;
}


// quad-double /= double-double
gqd_real &gqd_real::operator/=(const gdd_real &a) {
	*this = (*this / a);
	return *this;
}


// quad-double /= quad-double
gqd_real &gqd_real::operator/=(const gqd_real &a) {
	*this = (*this / a);
	return *this;
}



// Comparisons =================================================================
// Equality Comparisons ---------------------------------
// Not-Equal-To Comparisons -----------------------------

// Greater-Than Comparisons -----------------------------
// Greater-Than-Or-Equal-To Comparisons -----------------

// Less-Than Comparisons --------------------------------
// Less-Than-Or-Equal-To Comparisons --------------------



// Is functions ================================================================

__device__
bool is_zero( const gqd_real &x ) {
	return (x[0] == 0.0 || x[0] == CUDART_NEG_ZERO);
}

__device__
bool is_one( const gqd_real &x ) {
	return (x[0] == 1.0 && x[1] == 0.0 && x[2] == 0.0 && x[3] == 0.0);
}

__device__
bool is_positive( const gqd_real &x ) {
	return (x[0] > 0.0);
}

__device__
bool is_negative( const gqd_real &x ) {
	return (x[0] < 0.0 || x[0] == CUDART_NEG_ZERO);
}

__device__ __host__
bool isnan(const gqd_real &a) {
	return (isnan(a[0]) || isnan(a[1]) || isnan(a[2]) || isnan(a[3]));
}

__device__ __host__
bool isfinite(const gqd_real &a) {
	return isfinite(a[0]);
}

__device__ __host__
bool isinf(const gqd_real &a) {
	return isinf(a[0]);
}

__device__
bool is_pinf(const gqd_real &a) {
	return (a[0] == __qd_inf[1]);
}

__device__
bool is_ninf(const gqd_real &a) {
	return (a[0] == -__qd_inf[1]);
}



// Cast functions ==============================================================

__device__ __host__
gdd_real to_gdd_real(const gqd_real &a) {
	return gdd_real(a[0], a[1]);
}

__device__ __host__
double to_double(const gqd_real &a) {
	return a[0];
}

__device__ __host__
int to_int(const gqd_real &a) {
	return static_cast<int>(a[0]);
}


// Miscellaneous ===============================================================

// Nearest Integer
__device__
gqd_real nint(const gqd_real &a) {
	double x0, x1, x2, x3;

	x0 = nint(a[0]);
	x1 = x2 = x3 = 0.0;

	if (x0 == a[0]) {
		// First double is already an integer.
		x1 = nint(a[1]);

		if (x1 == a[1]) {
			// Second double is already an integer.
			x2 = nint(a[2]);

			if (x2 == a[2]) {
				// Third double is already an integer.
				x3 = nint(a[3]);
			} else {
				if (fabs(x2 - a[2]) == 0.5 && a[3] < 0.0) {
					x2 -= 1.0;
				}
			}

		} else {
			if (fabs(x1 - a[1]) == 0.5 && a[2] < 0.0) {
				x1 -= 1.0;
			}
		}

	} else {
		// First double is not an integer.
		if (fabs(x0 - a[0]) == 0.5 && a[1] < 0.0) {
		x0 -= 1.0;
		}
	}

	renorm(x0, x1, x2, x3);
	return gqd_real(x0, x1, x2, x3);
}


__device__
gqd_real abs(const gqd_real &a) {
	return is_negative(a) ? negative(a) : a;
}


__device__
gqd_real fabs(const gqd_real &a) {
	return abs(a);
}


__device__
gqd_real inv(const gqd_real &a) {
	return 1.0 / a;
}


#endif


