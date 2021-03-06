#ifndef __GDD_REAL_H__
#define __GDD_REAL_H__


#include <iostream>
#include <cmath>
#include <limits.h>
#include <float.h>
#include <vector_types.h>

//#include "cuda_header.h"
#include "inline.cu"

#define USE_FMA 1


/* constants */

extern __device__ __constant__ double _dd_eps;	// 2^-104

extern __device__ __constant__ double __dd_zero;
extern __device__ __constant__ double __dd_one;
extern __device__ __constant__ double  __dd_inf[2];	// h_inf, CUDART_INF;
extern __device__ __constant__ double __dd_qnan[2];	// h_qnan, CUDART_NAN;

extern __device__ __constant__ double     __dd_e[2];
extern __device__ __constant__ double  __dd_log2[2];
extern __device__ __constant__ double __dd_log10[2];
extern __device__ __constant__ double   __dd_2pi[2];
extern __device__ __constant__ double    __dd_pi[2];
extern __device__ __constant__ double   __dd_pi2[2];
extern __device__ __constant__ double  __dd_pi16[2];
extern __device__ __constant__ double   __dd_pi4[2];
extern __device__ __constant__ double  __dd_3pi4[2];

//#define _dd_zero	const gdd_real(__dd_zero, __dd_zero)	// 0.0, 0.0
//#define _dd_one		const gdd_real(__dd_one,  __dd_zero)	// 1.0, 0.0
//
//#define _dh_inf		const gdd_real( __dd_inf[0],  __dd_inf[0])	// __h_inf,  __h_inf
//#define _dh_qnan	const gdd_real(__dd_qnan[0], __dd_qnan[0])	// __h_qnan, __h_qnan
//#define _dd_inf		const gdd_real( __dd_inf[1],  __dd_inf[1])	// dd_inf,  dd_inf
//#define _dd_qnan	const gdd_real(__dd_qnan[1], __dd_qnan[1])	// dd_qnan, dd_qnan
//
//#define _dd_e		const gdd_real(    __dd_e[0],     __dd_e[1])
//#define _dd_log2	const gdd_real( __dd_log2[0],  __dd_log2[1])
//#define _dd_log10	const gdd_real(__dd_log10[0], __dd_log10[1])
//#define _dd_2pi		const gdd_real(  __dd_2pi[0],   __dd_2pi[1])
//#define _dd_pi		const gdd_real(   __dd_pi[0],    __dd_pi[1])
//#define _dd_pi2		const gdd_real(  __dd_pi2[0],   __dd_pi2[1])
//#define _dd_pi16	const gdd_real( __dd_pi16[0],  __dd_pi16[1])
//#define _dd_pi4		const gdd_real(  __dd_pi4[0],   __dd_pi4[1])
//#define _dd_3pi4	const gdd_real( __dd_3pi4[0],  __dd_3pi4[1])
//
//#define _dd_max         const gdd_real(1.79769313486231570815e+308, 9.97920154767359795037e+291)
//#define _dd_safe_max    const gdd_real(1.7976931080746007281e+308,  9.97920154767359795037e+291)
//#define _dd_min_normed  (2.0041683600089728e-292)  // = 2^(-1022 + 53)
//#define _dd_digits      (31)

#define _dd_zero	 gdd_real(__dd_zero, __dd_zero)	// 0.0, 0.0
#define _dd_one		 gdd_real(__dd_one,  __dd_zero)	// 1.0, 0.0

#define _dh_inf		 gdd_real( __dd_inf[0],  __dd_inf[0])	// __h_inf,  __h_inf
#define _dh_qnan	 gdd_real(__dd_qnan[0], __dd_qnan[0])	// __h_qnan, __h_qnan
#define _dd_inf		 gdd_real( __dd_inf[1],  __dd_inf[1])	// dd_inf,  dd_inf
#define _dd_qnan	 gdd_real(__dd_qnan[1], __dd_qnan[1])	// dd_qnan, dd_qnan

#define _dd_e		 gdd_real(    __dd_e[0],     __dd_e[1])
#define _dd_log2	 gdd_real( __dd_log2[0],  __dd_log2[1])
#define _dd_log10	 gdd_real(__dd_log10[0], __dd_log10[1])
#define _dd_2pi		 gdd_real(  __dd_2pi[0],   __dd_2pi[1])
#define _dd_pi		 gdd_real(   __dd_pi[0],    __dd_pi[1])
#define _dd_pi2		 gdd_real(  __dd_pi2[0],   __dd_pi2[1])
#define _dd_pi16	 gdd_real( __dd_pi16[0],  __dd_pi16[1])
#define _dd_pi4		 gdd_real(  __dd_pi4[0],   __dd_pi4[1])
#define _dd_3pi4	 gdd_real( __dd_3pi4[0],  __dd_3pi4[1])

#define _dd_max         gdd_real(1.79769313486231570815e+308, 9.97920154767359795037e+291)
#define _dd_safe_max    gdd_real(1.7976931080746007281e+308,  9.97920154767359795037e+291)
#define _dd_min_normed  (2.0041683600089728e-292)  // = 2^(-1022 + 53)
#define _dd_digits      (31)


/* data in the constant memory */
#define n_dd_inv_fact (15)
//__device__ __constant__ double dd_inv_fact[n_dd_inv_fact][2] = {
//	{ 1.66666666666666657e-01,  9.25185853854297066e-18 },
//	{ 4.16666666666666644e-02,  2.31296463463574266e-18 },
//	{ 8.33333333333333322e-03,  1.15648231731787138e-19 },
//	{ 1.38888888888888894e-03, -5.30054395437357706e-20 },
//	{ 1.98412698412698413e-04,  1.72095582934207053e-22 },
//	{ 2.48015873015873016e-05,  2.15119478667758816e-23 },
//	{ 2.75573192239858925e-06, -1.85839327404647208e-22 },
//	{ 2.75573192239858883e-07,  2.37677146222502973e-23 },
//	{ 2.50521083854417202e-08, -1.44881407093591197e-24 },
//	{ 2.08767569878681002e-09, -1.20734505911325997e-25 },
//	{ 1.60590438368216133e-10,  1.25852945887520981e-26 },
//	{ 1.14707455977297245e-11,  2.06555127528307454e-28 },
//	{ 7.64716373181981641e-13,  7.03872877733453001e-30 },
//	{ 4.77947733238738525e-14,  4.39920548583408126e-31 },
//	{ 2.81145725434552060e-15,  1.65088427308614326e-31 }
//};
//
//__device__ __constant__ double dd_sin_table[4][2] = {
//	{ 1.950903220161282758e-01, -7.991079068461731263e-18 },
//	{ 3.826834323650897818e-01, -1.005077269646158761e-17 },
//	{ 5.555702330196021776e-01,  4.709410940561676821e-17 },
//	{ 7.071067811865475727e-01, -4.833646656726456726e-17 }
//};
//
//__device__ __constant__ double dd_cos_table[4][2] = {
//	{ 9.807852804032304306e-01,  1.854693999782500573e-17 },
//	{ 9.238795325112867385e-01,  1.764504708433667706e-17 },
//	{ 8.314696123025452357e-01,  1.407385698472802389e-18 },
//	{ 7.071067811865475727e-01, -4.833646656726456726e-17 }
//};

class gqd_real;	// for friend declaration


class gdd_real {
private:
	double2 dd;
	__device__ gdd_real operator^(double n);
public:
	//double2 dd;

	//default constructor
	__device__ __host__	gdd_real();
	__device__ __host__	gdd_real(double hi, double lo);
	__device__ __host__	gdd_real(double d);
	__device__ __host__	explicit gdd_real(int i);
	__device__ __host__	explicit gdd_real(const double *d);

	// copy constructor
	__device__ __host__	gdd_real(const gdd_real &a);
	// destructor
	__device__ __host__	~gdd_real();

	static void error(const char *msg);

	__device__ __host__ gdd_real operator-() const;

	__device__ __host__ gdd_real &operator=(const gdd_real &a);
	__device__ __host__ gdd_real &operator=(double a);
	__device__ gdd_real &operator+=(double a);
	__device__ gdd_real &operator+=(const gdd_real &a);
	__device__ gdd_real &operator-=(const gdd_real &a);
	__device__ gdd_real &operator-=(double b);
	__device__ gdd_real &operator*=(double a);
	__device__ gdd_real &operator*=(const gdd_real &a);
	__device__ gdd_real &operator/=(double a);
	__device__ gdd_real &operator/=(const gdd_real &a);

	__device__ gdd_real operator^(int n);
	

	__host__ void to_digits(char *s, int &expn, int precision = _dd_digits) const;
	__host__ void write(char *s, int len, int precision = _dd_digits,
									bool showpos = false, bool uppercase = false) const;
	//__host__ std::string to_string(int precision = _dd_digits, int width = 0,
	//								std::ios_base::fmtflags fmt = static_cast<std::ios_base::fmtflags>(0),
	//								bool showpos = false, bool uppercase = false, char fill = ' ') const;
	__host__ int read(const char *s, gdd_real &a);

	/* Debugging Methods */
	__host__ void dump(const std::string &name = "", std::ostream &os = std::cerr) const;
	__host__ void dump_bits(const std::string &name = "",	std::ostream &os = std::cerr) const;

	__device__ __host__ static gdd_real rand(void);
	__device__ __host__ static gdd_real debug_rand(void);

	friend __device__ __host__ gdd_real negative(const gdd_real &a);
	friend __device__ gdd_real operator+(const gdd_real &a, double b);
	friend __forceinline__ __device__ gdd_real ieee_add(const gdd_real &a, const gdd_real &b);
	friend __forceinline__ __device__ gdd_real sloppy_add(const gdd_real &a, const gdd_real &b);
	friend __device__ gdd_real operator-(const gdd_real &a, const gdd_real &b);
	friend __device__ gdd_real operator-(const gdd_real &a, double b);
	friend __device__ gdd_real operator-(double a, const gdd_real &b);
	friend __device__ gdd_real sqr(const gdd_real &a);
	friend __device__ gdd_real ldexp(const gdd_real &a, int exp);
	friend __device__ gdd_real mul_pwr2(const gdd_real &a, double b);
	friend __device__ gdd_real operator*(const gdd_real &a, const gdd_real &b);
	friend __device__ gdd_real operator*(const gdd_real &a, double b);
	friend __forceinline__ __device__ gdd_real accurate_div(const gdd_real &a, const gdd_real &b);
	friend __forceinline__ __device__ gdd_real sloppy_div(const gdd_real &a, const gdd_real &b);
	friend __device__ gdd_real operator/(const gdd_real &a, double b);
	friend __device__ bool is_zero(const gdd_real &a);
	friend __device__ bool is_pzero(const gdd_real &a);
	friend __device__ bool is_nzero(const gdd_real &a);
	friend __device__ bool is_one(const gdd_real &a);
	friend __device__ bool is_positive(const gdd_real &a);
	friend __device__ bool is_negative(const gdd_real &a);
	friend __device__ bool isnan(const gdd_real &a);
	friend __device__ bool isfinite(const gdd_real &a);
	friend __device__ bool isinf(const gdd_real &a);
	friend __device__ bool is_pinf(const gdd_real &a);
	friend __device__ bool is_ninf(const gdd_real &a);
	friend __device__ __host__ double to_double(const gdd_real &a);
	friend __device__ __host__ int to_int(const gdd_real &a);
	friend __device__ bool operator==(const gdd_real &a, double b);
	friend __device__ bool operator==(double a, const gdd_real &b);
	friend __device__ bool operator==(const gdd_real &a, const gdd_real &b);
	friend __device__ bool operator<=(const gdd_real &a, double b);
	friend __device__ bool operator<=(const gdd_real &a, const gdd_real &b);
	friend __device__ bool operator>=(const gdd_real &a, double b);
	friend __device__ bool operator>=(const gdd_real &a, const gdd_real &b);
	friend __device__ bool operator<(const gdd_real &a, double b);
	friend __device__ bool operator<(double a, const gdd_real &b);
	friend __device__ bool operator<(const gdd_real &a, const gdd_real &b);
	friend __device__ bool operator>(const gdd_real &a, double b);
	friend __device__ bool operator>(double a, const gdd_real &b);
	friend __device__ bool operator>(const gdd_real &a, const gdd_real &b);
	friend __device__ bool operator!=(const gdd_real &a, double b);
	friend __device__ bool operator!=(double a, const gdd_real &b);
	friend __device__ bool operator!=(const gdd_real &a, const gdd_real &b);
	friend __device__ gdd_real aint(const gdd_real &a);
	friend __device__ gdd_real nint(const gdd_real &a);
	friend __device__ gdd_real floor(const gdd_real &a);
	friend __device__ gdd_real ceil(const gdd_real &a);
	friend __device__ gdd_real abs(const gdd_real &a);
	friend __device__ gdd_real sqrt(const gdd_real &a);
	friend __device__ gdd_real nroot(const gdd_real &a, int n);
	friend __device__ gdd_real exp(const gdd_real &a);
	friend __device__ gdd_real log(const gdd_real &a);
	friend __device__ void sincos_taylor(const gdd_real &a, gdd_real &sin_a, gdd_real &cos_a);
	friend __device__ gdd_real sin(const gdd_real &a);
	friend __device__ gdd_real cos(const gdd_real &a);
	friend __device__ void sincos(const gdd_real &a, gdd_real &sin_a, gdd_real &cos_a);
	friend __device__ gdd_real atan2(const gdd_real &y, const gdd_real &x);

	friend gqd_real;
	friend __device__ __forceinline__	static gqd_real accurate_div(const gqd_real &a, const gdd_real &b);
	friend __device__ __forceinline__	static gqd_real sloppy_div(const gqd_real &a, const gdd_real &b);

	friend __device__ gqd_real operator+(const gqd_real &a, const gdd_real &b);
	friend __device__ gqd_real operator*(const gqd_real &a, const gdd_real &b);
	friend __device__ gqd_real operator/(const gqd_real &a, const gdd_real &b);

	friend __device__ bool operator==(const gqd_real &a, const gdd_real &b);
	friend __device__ bool operator<(const gqd_real &a, const gdd_real &b);
	friend __device__ bool operator<=(const gqd_real &a, const gdd_real &b);
	friend __device__ bool operator>(const gqd_real &a, const gdd_real &b);
	friend __device__ bool operator>=(const gqd_real &a, const gdd_real &b);

};


namespace std {
	template <>
	class numeric_limits<gdd_real> : public numeric_limits<double>{
	public:
		__device__ static double epsilon() { return _dd_eps; }
		__device__ static double min() { return _dd_min_normed; }
		__device__ static gdd_real max() { return _dd_max; }
		__device__ static gdd_real safe_max() { return _dd_safe_max; }
		__device__ static int digits(){ return 104; }
		__device__ static int digits10(){ return 31; }
		//#define digits    104
		//#define digits10  31
	};
}


__device__ __host__ gdd_real negative(const gdd_real &a);
__device__ __host__ gdd_real ddrand(void);


//__device__ __host__ gdd_real polyeval(const gdd_real *c, int n, const gdd_real &x);
//__device__ __host__ gdd_real polyroot(const gdd_real *c, int n, const gdd_real &x0, int max_iter = 32, double thresh = 0.0);


/* Computes  dd * d  where d is known to be a power of 2. */
__device__ gdd_real mul_pwr2(const gdd_real &dd, double d);
__device__ gdd_real ldexp(const gdd_real &a, int exp);
__device__ gdd_real sqr(const gdd_real &a);

__device__ gdd_real operator+(const gdd_real &a, double b);
__device__ gdd_real operator+(double a, const gdd_real &b);
__device__ gdd_real operator+(const gdd_real &a, const gdd_real &b);

__device__ gdd_real operator-(const gdd_real &a, double b);
__device__ gdd_real operator-(double a, const gdd_real &b);
__device__ gdd_real operator-(const gdd_real &a, const gdd_real &b);

__device__ gdd_real operator*(const gdd_real &a, double b);
__device__ gdd_real operator*(double a, const gdd_real &b);
__device__ gdd_real operator*(const gdd_real &a, const gdd_real &b);

__device__ gdd_real operator/(const gdd_real &a, double b);
__device__ gdd_real operator/(double a, const gdd_real &b);
__device__ gdd_real operator/(const gdd_real &a, const gdd_real &b);

__device__ bool operator==(const gdd_real &a, double b);
__device__ bool operator==(double a, const gdd_real &b);
__device__ bool operator==(const gdd_real &a, const gdd_real &b);

__device__ bool operator!=(const gdd_real &a, double b);
__device__ bool operator!=(double a, const gdd_real &b);
__device__ bool operator!=(const gdd_real &a, const gdd_real &b);

__device__ bool operator>(const gdd_real &a, double b);
__device__ bool operator>(double a, const gdd_real &b);
__device__ bool operator>(const gdd_real &a, const gdd_real &b);

__device__ bool operator<(const gdd_real &a, double b);
__device__ bool operator<(double a, const gdd_real &b);
__device__ bool operator<(const gdd_real &a, const gdd_real &b);

__device__ bool operator>=(const gdd_real &a, double b);
__device__ bool operator>=(double a, const gdd_real &b);
__device__ bool operator>=(const gdd_real &a, const gdd_real &b);

__device__ bool operator<=(const gdd_real &a, double b);
__device__ bool operator<=(double a, const gdd_real &b);
__device__ bool operator<=(const gdd_real &a, const gdd_real &b);

__device__ bool is_zero(const gdd_real &a);
__device__ bool is_pzero(const gdd_real &a);
__device__ bool is_nzero(const gdd_real &a);
__device__ bool is_one(const gdd_real &a);
__device__ bool is_positive(const gdd_real &a);
__device__ bool is_negative(const gdd_real &a);

__device__ bool isnan(const gdd_real &a);
__device__ bool isfinite(const gdd_real &a);
__device__ bool isinf(const gdd_real &a);
__device__ bool is_pinf(const gdd_real &a);
__device__ bool is_ninf(const gdd_real &a);

__device__ __host__ double to_double(const gdd_real &a);
__device__ __host__ int    to_int(const gdd_real &a);
__device__ __host__ int    to_int(double a);

//__device__ __host__ gdd_real drem(const gdd_real &a, const gdd_real &b);
//__device__ __host__ gdd_real divrem(const gdd_real &a, const gdd_real &b, gdd_real &r);


__device__ gdd_real nint(const gdd_real &a);
__device__ gdd_real floor(const gdd_real &a);
__device__ gdd_real ceil(const gdd_real &a);
__device__ gdd_real aint(const gdd_real &a);

__device__ gdd_real fabs(const gdd_real &a);
__device__ gdd_real abs(const gdd_real &a);   /* same as fabs */
__device__ gdd_real inv(const gdd_real &a);
__device__ gdd_real fmod(const gdd_real &a, const gdd_real &b);

__device__ gdd_real pow(const gdd_real &a, int n);
__device__ gdd_real pow(const gdd_real &a, const gdd_real &b);
__device__ gdd_real npwr(const gdd_real &a, int n);
__device__ gdd_real operator^(const gdd_real &a, int n);

__device__ gdd_real max(const gdd_real &a, const gdd_real &b);
__device__ gdd_real min(const gdd_real &a, const gdd_real &b);

__device__ gdd_real sqrt(const gdd_real &a);
__device__ gdd_real nroot(const gdd_real &a, int n);

__device__ gdd_real exp(const gdd_real &a);

__device__ gdd_real log(const gdd_real &a);
__device__ gdd_real log2(const gdd_real &a);
__device__ gdd_real log10(const gdd_real &a);

__device__ gdd_real sin(const gdd_real &a);
__device__ gdd_real cos(const gdd_real &a);
__device__ gdd_real tan(const gdd_real &a);
__device__ void sincos(const gdd_real &a, gdd_real &sin_a, gdd_real &cos_a);

#ifdef ALL_MATH
__device__ gdd_real asin(const gdd_real &a);
__device__ gdd_real acos(const gdd_real &a);
__device__ gdd_real atan(const gdd_real &a);
__device__ gdd_real atan2(const gdd_real &y, const gdd_real &x);

__device__ gdd_real sinh(const gdd_real &a);
__device__ gdd_real cosh(const gdd_real &a);
__device__ gdd_real tanh(const gdd_real &a);
__device__ void sincosh(const gdd_real &a, gdd_real &sinh_a, gdd_real &cosh_a);

__device__ gdd_real asinh(const gdd_real &a);
__device__ gdd_real acosh(const gdd_real &a);
__device__ gdd_real atanh(const gdd_real &a);
#endif

// generate gdd_real from double operation
__device__ gdd_real dd_add(double a, double b);
__device__ gdd_real dd_sub(double a, double b);
__device__ gdd_real dd_mul(double a, double b);
__device__ gdd_real dd_div(double a, double b);
__device__ gdd_real dd_sqr(double a, double b);

//std::ostream& operator<<(std::ostream &s, const gdd_real &a);
//std::istream& operator>>(std::istream &s, gdd_real &a);




#endif	//__GDD_REAL_H__