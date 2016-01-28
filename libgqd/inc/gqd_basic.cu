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
		return qd.x;
	}
}

// Assignments =================================================================
/* quad-double = double */
__device__ __host__
gqd_real &gqd_real::operator=(double a) {
	qd.x = a;
	qd.y = qd.z = qd.w = 0.0;
	return *this;
}

/* quad-double = double-double */
__device__ __host__
gqd_real &gqd_real::operator=(const gdd_real &a) {
	qd.x = a.dd.x;
	qd.y = a.dd.y;
	qd.z = qd.w = 0.0;
	return *this;
}

/* quad-double = quad-double */
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



/** normalization functions */
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
void renorm( gqd_real &x ) {
	renorm(x.x, x.y, x.z, x.w);
}

__device__ __host__
void renorm( gqd_real &x, double &e) {
	renorm(x.x, x.y, x.z, x.w, e);
}

/** additions */
__device__ __host__
void three_sum(double &a, double &b, double &c) {
	double t1, t2, t3;
	t1 = two_sum(a, b, t2);
	a  = two_sum(c, t1, t3);
	b  = two_sum(t2, t3, c);
}

__device__ __host__
void three_sum2(double &a, double &b, double &c) {
	double t1, t2, t3;
	t1 = two_sum(a, b, t2);
	a  = two_sum(c, t1, t3);
	b = (t2 + t3);
}

///qd = qd + double
__device__ __host__
gqd_real operator+(const gqd_real &a, double b) {
	double c0, c1, c2, c3;
	double e;

	c0 = two_sum(a.x, b, e);
	c1 = two_sum(a.y, e, e);
	c2 = two_sum(a.z, e, e);
	c3 = two_sum(a.w, e, e);

	renorm(c0, c1, c2, c3, e);

	return make_qd(c0, c1, c2, c3);
}

///qd = double + qd
__device__ __host__
gqd_real operator+( double a, const gqd_real &b ) {
	return ( b + a );
}

///qd = qd + qd
__device__ __host__
gqd_real sloppy_add(const gqd_real &a, const gqd_real &b) {
	double s0, s1, s2, s3;
	double t0, t1, t2, t3;

	double v0, v1, v2, v3;
	double u0, u1, u2, u3;
	double w0, w1, w2, w3;

	s0 = a.x + b.x;
	s1 = a.y + b.y;
	s2 = a.z + b.z;
	s3 = a.w + b.w;

	v0 = s0 - a.x;
	v1 = s1 - a.y;
	v2 = s2 - a.z;
	v3 = s3 - a.w;

	u0 = s0 - v0;
	u1 = s1 - v1;
	u2 = s2 - v2;
	u3 = s3 - v3;

	w0 = a.x - u0;
	w1 = a.y - u1;
	w2 = a.z - u2;
	w3 = a.w - u3;

	u0 = b.x - v0;
	u1 = b.y - v1;
	u2 = b.z - v2;
	u3 = b.w - v3;

	t0 = w0 + u0;
	t1 = w1 + u1;
	t2 = w2 + u2;
	t3 = w3 + u3;

	s1 = two_sum(s1, t0, t0);
	three_sum(s2, t0, t1);
	three_sum2(s3, t0, t2);
	t0 = t0 + t1 + t3;

	renorm(s0, s1, s2, s3, t0);

	return make_qd(s0, s1, s2, s3);
}

__device__ __host__
gqd_real operator+(const gqd_real &a, const gqd_real &b) {
	return sloppy_add(a, b);
}


/** subtractions */
__device__ __host__
gqd_real negative( const gqd_real &a ) {
	return make_qd( -a.x, -a.y, -a.z, -a.w );
}

__device__ __host__
gqd_real operator-(const gqd_real &a, double b) {
	return (a + (-b));
}

__device__ __host__
gqd_real operator-(double a, const gqd_real &b) {
	return (a + negative(b));
}

__device__ __host__
gqd_real operator-(const gqd_real &a, const gqd_real &b) {
	return (a + negative(b));
}

/** multiplications */
__device__ __host__
gqd_real mul_pwr2(const gqd_real &a, double b) {
	return make_qd(a.x * b, a.y * b, a.z * b, a.w * b);
}


//quad_double * double
 __device__
gqd_real operator*(const gqd_real &a, double b) {
	double p0, p1, p2, p3;
	double q0, q1, q2;
	double s0, s1, s2, s3, s4;

	p0 = two_prod(a.x, b, q0);
	p1 = two_prod(a.y, b, q1);
	p2 = two_prod(a.z, b, q2);
	p3 = a.w * b;

	s0 = p0;

	s1 = two_sum(q0, p1, s2);

	three_sum(s2, q1, p2);

	three_sum2(q1, q2, p3);
	s3 = q1;

	s4 = q2 + p2;

	renorm(s0, s1, s2, s3, s4);
	return make_qd(s0, s1, s2, s3);
}
//quad_double = double*quad_double
__device__
gqd_real operator*( double a, const gqd_real &b ) {
	return b*a;
}

__device__
gqd_real sloppy_mul(const gqd_real &a, const gqd_real &b) {
	double p0, p1, p2, p3, p4, p5;
	double q0, q1, q2, q3, q4, q5;
	double t0, t1;
	double s0, s1, s2;

	p0 = two_prod(a.x, b.x, q0);

	p1 = two_prod(a.x, b.y, q1);
	p2 = two_prod(a.y, b.x, q2);

	p3 = two_prod(a.x, b.z, q3);
	p4 = two_prod(a.y, b.y, q4);
	p5 = two_prod(a.z, b.x, q5);


	/* Start Accumulation */
	three_sum(p1, p2, q0);

	//return make_qd(p1, p2, q0, 0.0);

	/* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
	three_sum(p2, q1, q2);
	three_sum(p3, p4, p5);
	/* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
	s0 = two_sum(p2, p3, t0);
	s1 = two_sum(q1, p4, t1);
	s2 = q2 + p5;
	s1 = two_sum(s1, t0, t0);
	s2 += (t0 + t1);

	//return make_qd(s0, s1, t0, t1);

	/* O(eps^3) order terms */
	//!!!s1 = s1 + (a.x*b.w + a.y*b.z + a.z*b.y + a.w*b.x + q0 + q3 + q4 + q5);
	
	s1 = s1 + (__dmul_rn(a.x,b.w) + __dmul_rn(a.y,b.z) + 
			__dmul_rn(a.z,b.y) + __dmul_rn(a.w,b.x) + q0 + q3 + q4 + q5);
	renorm(p0, p1, s0, s1, s2);

	return make_qd(p0, p1, s0, s1);
	
}

 __device__
gqd_real operator*(const gqd_real &a, const gqd_real &b) {
	return sloppy_mul(a, b);
}

 __device__
gqd_real sqr(const gqd_real &a) {
	double p0, p1, p2, p3, p4, p5;
	double q0, q1, q2, q3;
	double s0, s1;
	double t0, t1;

	p0 = two_sqr(a.x, q0);
	p1 = two_prod(2.0 * a.x, a.y, q1);
	p2 = two_prod(2.0 * a.x, a.z, q2);
	p3 = two_sqr(a.y, q3);

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

	p4 = 2.0 * a.x * a.w;
	p5 = 2.0 * a.y * a.z;

	p4 = two_sum(p4, p5, p5);
	q2 = two_sum(q2, q3, q3);

	t0 = two_sum(p4, q2, t1);
	t1 = t1 + p5 + q3;

	p3 = two_sum(p3, t0, p4);
	p4 = p4 + q0 + t1;

	renorm(p0, p1, p2, p3, p4);
	return make_qd(p0, p1, p2, p3);
}

/** divisions */
__device__
gqd_real sloppy_div(const gqd_real &a, const gqd_real &b) {
	double q0, q1, q2, q3;

	gqd_real r;

	q0 = a.x / b.x;
	r = a - (b * q0);

	q1 = r.x / b.x;
	r = r - (b * q1);

	q2 = r.x / b.x;
	r = r - (b * q2);

	q3 = r.x / b.x;

	renorm(q0, q1, q2, q3);

	return make_qd(q0, q1, q2, q3);
}

__device__
gqd_real operator/(const gqd_real &a, const gqd_real &b) {
	return sloppy_div(a, b);
}

/* double / quad-double */
__device__
gqd_real operator/(double a, const gqd_real &b) {
	return make_qd(a) / b;
}

/* quad-double / double */
__device__
gqd_real operator/( const gqd_real &a, double b ) {
	return a/make_qd(b);
}

/********** Miscellaneous **********/
__device__ __host__
gqd_real abs(const gqd_real &a) {
	return (a.x < 0.0) ? (negative(a)) : (a);
}

/********************** Simple Conversion ********************/
__device__ __host__
double to_double(const gqd_real &a) {
	return a.x;
}

__device__ __host__
gqd_real ldexp(const gqd_real &a, int n) {
	return make_qd(ldexp(a.x, n), ldexp(a.y, n), 
		ldexp(a.z, n), ldexp(a.w, n));
}

__device__
gqd_real inv(const gqd_real &qd) {
	return 1.0 / qd;
}


/********** Greater-Than Comparison ***********/

__device__ __host__
bool operator>=(const gqd_real &a, const gqd_real &b) {
	return (a.x > b.x || 
		(a.x == b.x && (a.y > b.y ||
		(a.y == b.y && (a.z > b.z ||
		(a.z == b.z && a.w >= b.w))))));
}

/********** Greater-Than-Or-Equal-To Comparison **********/
/*
__device__
bool operator>=(const gqd_real &a, double b) {
	return (a.x > b || (a.x == b && a.y >= 0.0));
}

__device__
bool operator>=(double a, const gqd_real &b) {
	return (b <= a);
}

__device__
bool operator>=(const gqd_real &a, const gqd_real &b) {
	return (a.x > b.x || 
			(a.x == b.x && (a.y > b.y ||
			(a.y == b.y && (a.z > b.z ||
			(a.z == b.z && a.w >= b.w))))));
}
*/

/********** Less-Than Comparison ***********/
__device__ __host__
bool operator<(const gqd_real &a, double b) {
	return (a.x < b || (a.x == b && a.y < 0.0));
}

__device__ __host__
bool operator<(const gqd_real &a, const gqd_real &b) {
	return (a.x < b.x ||
			(a.x == b.x && (a.y < b.y ||
			(a.y == b.y && (a.z < b.z ||
			(a.z == b.z && a.w < b.w))))));
}

__device__ __host__
bool operator<=(const gqd_real &a, const gqd_real &b) {
	return (a.x < b.x || 
			(a.x == b.x && (a.y < b.y ||
			(a.y == b.y && (a.z < b.z ||
			(a.z == b.z && a.w <= b.w))))));
}

__device__ __host__
bool operator==(const gqd_real &a, const gqd_real &b) {
	return (a.x == b.x && a.y == b.y && 
			a.z == b.z && a.w == b.w);
}



/********** Less-Than-Or-Equal-To Comparison **********/
__device__
bool operator<=(const gqd_real &a, double b) {
	return (a.x < b || (a.x == b && a.y <= 0.0));
}

/*
__device__
bool operator<=(double a, const gqd_real &b) {
	return (b >= a);
}
*/

/*
__device__
bool operator<=(const gqd_real &a, const gqd_real &b) {
	return (a.x < b.x || 
			(a.x == b.x && (a.y < b.y ||
			(a.y == b.y && (a.z < b.z ||
			(a.z == b.z && a.w <= b.w))))));
}
*/

/********** Greater-Than-Or-Equal-To Comparison **********/
__device__
bool operator>=(const gqd_real &a, double b) {
	return (a.x > b || (a.x == b && a.y >= 0.0));
}

__device__
bool operator<=(double a, const gqd_real &b) {
	return (b >= a);
}


__device__
bool operator>=(double a, const gqd_real &b) {
	return (b <= a);
}


/*
__device__
bool operator>=(const gqd_real &a, const gqd_real &b) {
	return (a.x > b.x ||
			(a.x == b.x && (a.y > b.y ||
			(a.y == b.y && (a.z > b.z ||
			(a.z == b.z && a.w >= b.w))))));
}

*/

/********** Greater-Than Comparison ***********/
__device__ __host__
bool operator>(const gqd_real &a, double b) {
	return (a.x > b || (a.x == b && a.y > 0.0));
}

__device__ __host__
bool operator<(double a, const gqd_real &b) {
	return (b > a);
}

__device__ __host__
bool operator>(double a, const gqd_real &b) {
	return (b < a);
}

__device__ __host__ 
bool operator>(const gqd_real &a, const gqd_real &b) {
	return (a.x > b.x ||
			(a.x == b.x && (a.y > b.y ||
			(a.y == b.y && (a.z > b.z ||
			(a.z == b.z && a.w > b.w))))));
}


__device__ __host__
bool is_zero( const gqd_real &x ) {
	return (x.x == 0.0);
}

__device__ __host__
bool is_one( const gqd_real &x ) {
	return (x.x == 1.0 && x.y == 0.0 && x.z == 0.0 && x.w == 0.0);
}

__device__ __host__
bool is_positive( const gqd_real &x ) {
	return (x.x > 0.0);
}

__device__ __host__
bool is_negative( const gqd_real &x ) {
	return (x.x < 0.0);
}

__device__
gqd_real nint(const gqd_real &a) {
	double x0, x1, x2, x3;

	x0 = nint(a.x);
	x1 = x2 = x3 = 0.0;

	if (x0 == a.x) {
		/* First double is already an integer. */
		x1 = nint(a.y);

		if (x1 == a.y) {
			/* Second double is already an integer. */
			x2 = nint(a.z);

			if (x2 == a.z) {
				/* Third double is already an integer. */
				x3 = nint(a.w);
			} else {
				if (fabs(x2 - a.z) == 0.5 && a.w < 0.0) {
					x2 -= 1.0;
				}
			}

		} else {
			if (fabs(x1 - a.y) == 0.5 && a.z < 0.0) {
				x1 -= 1.0;
			}
		}

	} else {
		/* First double is not an integer. */
		if (fabs(x0 - a.x) == 0.5 && a.y < 0.0) {
		x0 -= 1.0;
		}
	}

	renorm(x0, x1, x2, x3);
	return make_qd(x0, x1, x2, x3);
}

__device__
gqd_real fabs(const gqd_real &a) {
	return abs(a);
}


#endif


