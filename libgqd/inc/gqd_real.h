
#ifndef __GQD_REAL_H__
#define __GQD_REAL_H__

#define USE_FMA 1

#include <iostream>
#include <string>
#include <limits>
#include <cmath>

#include "cuda_header.h"



#define n_qd_inv_fact (15)


extern __device__ __constant__ double _qd_eps; // = 2^-209
extern __device__ __constant__ double __qd_zero;
extern __device__ __constant__ double __qd_one;
extern __device__ __constant__ double  __qd_inf[2];	// h_inf, CUDART_INF;
extern __device__ __constant__ double __qd_qnan[2];	// h_qnan, CUDART_NAN;

extern __device__ __constant__ double      __qd_e[4];
extern __device__ __constant__ double   __qd_log2[4];
extern __device__ __constant__ double  __qd_log10[4];
extern __device__ __constant__ double    __qd_2pi[4];
extern __device__ __constant__ double     __qd_pi[4];
extern __device__ __constant__ double    __qd_pi2[4];
extern __device__ __constant__ double __qd_pi1024[4];
extern __device__ __constant__ double    __qd_pi4[4];
extern __device__ __constant__ double   __qd_3pi4[4];


#define _qd_zero	const gqd_real(__qd_zero, __qd_zero, __qd_zero, __qd_zero)	// 0.0, 0.0, 0.0, 0.0
#define _qd_one		const gqd_real(__qd_one,  __qd_zero, __qd_zero, __qd_zero)	// 1.0, 0.0, 0.0, 0.0

#define _qh_inf		const gqd_real( __qd_inf[0],  __qd_inf[0],  __qd_inf[0],  __qd_inf[0])	// __h_inf,  __h_inf,  __h_inf,  __h_inf
#define _qh_qnan	const gqd_real(__qd_qnan[0], __qd_qnan[0], __qd_qnan[0], __qd_qnan[0])	// __h_qnan, __h_qnan, __h_qnan, __h_qnan
#define _qd_inf		const gqd_real( __qd_inf[1],  __qd_inf[1],  __qd_inf[1],  __qd_inf[1])	// qd_inf,  qd_inf,  qd_inf,  qd_inf
#define _qd_qnan	const gqd_real(__qd_qnan[1], __qd_qnan[1], __qd_qnan[1], __qd_qnan[1])	// qd_qnan, qd_qnan, qd_qnan, qd_qnan

#define _qd_e		const gqd_real(      __qd_e[0],      __qd_e[1],      __qd_e[2],      __qd_e[3] )
#define _qd_log2	const gqd_real(   __qd_log2[0],   __qd_log2[1],   __qd_log2[2],   __qd_log2[3] )
#define _qd_log10	const gqd_real(  __qd_log10[0],  __qd_log10[1],  __qd_log10[2],  __qd_log10[3] )
#define _qd_2pi		const gqd_real(    __qd_2pi[0],    __qd_2pi[1],    __qd_2pi[2],    __qd_2pi[3] )
#define _qd_pi		const gqd_real(     __qd_pi[0],     __qd_pi[1],     __qd_pi[2],     __qd_pi[3] )
#define _qd_pi2		const gqd_real(    __qd_pi2[0],    __qd_pi2[1],    __qd_pi2[2],    __qd_pi2[3] )
#define _qd_pi1024	const gqd_real( __qd_pi1024[0], __qd_pi1024[1], __qd_pi1024[2], __qd_pi1024[3] )
#define _qd_pi4		const gqd_real(    __qd_pi4[0],    __qd_pi4[1],    __qd_pi4[2],    __qd_pi4[3] )
#define _qd_3pi4	const gqd_real(   __qd_3pi4[0],   __qd_3pi4[1],   __qd_3pi4[2],   __qd_3pi4[3] )


#define	_qd_max			const gqd_real(	1.79769313486231570815e+308, 9.97920154767359795037e+291, 5.53956966280111259858e+275, 3.07507889307840487279e+259)
#define	_qd_safe_max	const gqd_real( 1.7976931080746007281e+308,  9.97920154767359795037e+291, 5.53956966280111259858e+275, 3.07507889307840487279e+259)
#define	_qd_min_normed	(1.6259745436952323e-260) // = 2^(-1022 + 3*53)
#define	_qd_digits		(62)

//class gdd_real;

class gqd_real {
private:
	// The Components.
	double4 qd;
	__device__ gqd_real operator^(double n);

public:
	//double4 qd;

	__device__ __host__ gqd_real();
	__device__ __host__ ~gqd_real();
	__device__ __host__ gqd_real(double x0, double x1, double x2, double x3);
	__device__ __host__ explicit gqd_real(double d);
	__device__ __host__ explicit gqd_real(const double *d4);	// assume double[4]
	__device__ __host__ gqd_real(const gqd_real &qd);
	__device__ __host__ gqd_real(const gdd_real &dd);
	__device__ __host__ gqd_real(int i);


	__device__ __host__ double operator[](int i) const;
	__device__ __host__ double &operator[](int i);

	static void error(const char *msg);

	/* Eliminates any zeros in the middle component(s). */
	//__device__ __host__ void zero_elim();
	//__device__ __host__ void zero_elim(double &e);

	__device__ __host__ void renorm();
	__device__ __host__ void renorm(double &e);


	__device__ __host__ gqd_real &operator=(double a);
	__device__ __host__ gqd_real &operator=(const gdd_real &a);
	__device__ __host__ gqd_real &operator=(const gqd_real &a);

	__device__ __host__ gqd_real operator-() const;

	__device__ __host__ gqd_real &operator+=(double a);
	__device__ __host__ gqd_real &operator+=(const gdd_real &a);
	__device__ __host__ gqd_real &operator+=(const gqd_real &a);

	__device__ __host__ gqd_real &operator-=(double a);
	__device__ __host__ gqd_real &operator-=(const gdd_real &a);
	__device__ __host__ gqd_real &operator-=(const gqd_real &a);

	__device__ __host__ gqd_real &operator*=(double a);
	__device__ __host__ gqd_real &operator*=(const gdd_real &a);
	__device__ __host__ gqd_real &operator*=(const gqd_real &a);

	__device__ __host__ gqd_real &operator/=(double a);
	__device__ __host__ gqd_real &operator/=(const gdd_real &a);
	__device__ __host__ gqd_real &operator/=(const gqd_real &a);





	//__device__ __host__ bool is_zero() const;
	//__device__ __host__ bool is_one() const;
	//__device__ __host__ bool is_positive() const;
	//__device__ __host__ bool is_negative() const;
	//__device__ __host__ bool operator==(const gdd_real &a);
	//__device__ __host__ bool operator<(const gdd_real &a);
	//__device__ __host__ bool operator<=(const gdd_real &a);
	//__device__ __host__ bool operator>(const gdd_real &a);
	//__device__ __host__ bool operator>=(const gdd_real &a);


	__device__ __host__ static gqd_real rand(void);

	void to_digits(char *s, int &expn, int precision = _qd_digits) const;
	void write(char *s, int len, int precision = _qd_digits,
		bool showpos = false, bool uppercase = false) const;
	std::string to_string(int precision = _qd_digits, int width = 0,
		std::ios_base::fmtflags fmt = static_cast<std::ios_base::fmtflags>(0), 
		bool showpos = false, bool uppercase = false, char fill = ' ') const;
	static int read(const char *s, gqd_real &a);

	/* Debugging methods */
	void dump(const std::string &name = "", std::ostream &os = std::cerr) const;
	void dump_bits(const std::string &name = "", 
					std::ostream &os = std::cerr) const;

	static gqd_real debug_rand();

	friend __device__ __host__ gqd_real negative(const gqd_real &a);
	//friend __device__ __host__ gqd_real operator+(const gqd_real &a, double b);
	//friend __device__ __host__ gqd_real operator*(const gqd_real &a, double b);
	//friend __device__ __host__ __forceinline__ static gqd_real sloppy_mul(const gqd_real &a, const gqd_real &b);
	//friend __device__ __host__ __forceinline__ static gqd_real accurate_mul(const gqd_real &a, const gqd_real &b);
	//friend __device__ __host__ __forceinline__ static gqd_real accurate_div(const gqd_real &a, const gqd_real &b);
	//friend __device__ __host__ __forceinline__ static gqd_real sloppy_div(const gqd_real &a, const gqd_real &b);

	friend __device__ __host__ gqd_real sqr(const gqd_real &a);
	friend __device__ __host__ gqd_real mul_pwr2(const gqd_real &a, double b);
	friend __device__ __host__ gqd_real ldexp(const gqd_real &a, int n);
	friend __device__ gqd_real abs(const gqd_real &a);

	friend __device__ __host__ double to_double(const gqd_real &a);
	friend __device__ __host__ int to_int(const gqd_real &a);
	friend __device__ bool is_zero(const gqd_real &a);
	friend __device__ bool is_one(const gqd_real &a);
	friend __device__ bool is_positive(const gqd_real &a);
	friend __device__ bool is_negative(const gqd_real &a);

	friend __device__ __host__ bool isnan(const gqd_real &a);
	friend __device__ __host__ bool isfinite(const gqd_real &a);
	friend __device__ __host__ bool isinf(const gqd_real &a);
	friend __device__ bool is_pinf(const gqd_real &a);
	friend __device__ bool is_ninf(const gqd_real &a);

};

namespace std {
	template <>
	class numeric_limits<gqd_real> : public numeric_limits<double>{
	public:
		__device__ __inline__ static double epsilon() { return _qd_eps; }
		__device__ __inline__ static double min() { return _qd_min_normed; }
		__device__ __inline__ static gqd_real max() { return _qd_max; }
		__device__ __inline__ static gqd_real safe_max() { return _qd_safe_max; }
		__device__ __inline__ static int digits() { return 209; }
		__device__ __inline__ static int digits10() { return 62; }
	};
}

__device__ __host__ void quick_renorm(double &c0, double &c1, double &c2, double &c3, double &c4);
__device__ __host__ void renorm(double &c0, double &c1, double &c2, double &c3);
__device__ __host__ void renorm(double &c0, double &c1, double &c2, double &c3, double &c4);

//__device__ __host__ gqd_real polyeval(const gqd_real *c, int n, const gqd_real &x);
//__device__ __host__ gqd_real polyroot(const gqd_real *c, int n, const gqd_real &x0, int max_iter = 64, double thresh = 0.0);

__device__ __host__ gqd_real negative(const gqd_real &a);
__device__ __host__ gqd_real qdrand(void);


/* Computes  qd * d  where d is known to be a power of 2.
   This can be done component wise.                      */
__device__ __host__ gqd_real mul_pwr2(const gqd_real &qd, double d);
__device__ __host__ gqd_real ldexp(const gqd_real &a, int n);
__device__ __host__ gqd_real sqr(const gqd_real &a);

__device__ __host__ gqd_real operator+(const gqd_real &a, double b);
__device__ __host__ gqd_real operator+(double a, const gqd_real &b);
__device__ __host__ gqd_real operator+(const gqd_real &a, const gqd_real &b);
__device__ __host__ gqd_real operator+(const gqd_real &a, const gdd_real &b);
__device__ __host__ gqd_real operator+(const gdd_real &a, const gqd_real &b);

__device__ __host__ gqd_real operator-(const gqd_real &a, double b);
__device__ __host__ gqd_real operator-(double a, const gqd_real &b);
__device__ __host__ gqd_real operator-(const gqd_real &a, const gqd_real &b);
__device__ __host__ gqd_real operator-(const gqd_real &a, const gdd_real &b);
__device__ __host__ gqd_real operator-(const gdd_real &a, const gqd_real &b);

__device__ __host__ gqd_real operator*(const gqd_real &a, double b);
__device__ __host__ gqd_real operator*(double a, const gqd_real &b);
__device__ __host__ gqd_real operator*(const gqd_real &a, const gqd_real &b);
__device__ __host__ gqd_real operator*(const gqd_real &a, const gdd_real &b);
__device__ __host__ gqd_real operator*(const gdd_real &a, const gqd_real &b);

__device__ __host__ gqd_real operator/(const gqd_real &a, double b);
__device__ __host__ gqd_real operator/(double a, const gqd_real &b);
__device__ __host__ gqd_real operator/(const gqd_real &a, const gqd_real &b);
__device__ __host__ gqd_real operator/(const gqd_real &a, const gdd_real &b);
__device__ __host__ gqd_real operator/(const gdd_real &a, const gqd_real &b);




__device__ __host__ gqd_real rem(const gqd_real &a, const gqd_real &b);
__device__ __host__ gqd_real drem(const gqd_real &a, const gqd_real &b);
__device__ __host__ gqd_real divrem(const gqd_real &a, const gqd_real &b, gqd_real &r);


__device__ __host__ bool operator==(const gqd_real &a, const gqd_real &b);
__device__ __host__ bool operator==(const gqd_real &a, const gdd_real &b);
__device__ __host__ bool operator==(const gdd_real &a, const gqd_real &b);
__device__ __host__ bool operator==(const gqd_real &a, double b);
__device__ __host__ bool operator==(double a, const gqd_real &b);

__device__ __host__ bool operator!=(const gqd_real &a, const gqd_real &b);
__device__ __host__ bool operator!=(const gqd_real &a, const gdd_real &b);
__device__ __host__ bool operator!=(const gdd_real &a, const gqd_real &b);
__device__ __host__ bool operator!=(const gqd_real &a, double b);
__device__ __host__ bool operator!=(double a, const gqd_real &b);

__device__ __host__ bool operator>(const gqd_real &a, const gqd_real &b);
__device__ __host__ bool operator>(const gqd_real &a, const gdd_real &b);
__device__ __host__ bool operator>(const gdd_real &a, const gqd_real &b);
__device__ __host__ bool operator>(const gqd_real &a, double b);
__device__ __host__ bool operator>(double a, const gqd_real &b);

__device__ __host__ bool operator>=(const gqd_real &a, const gqd_real &b);
__device__ __host__ bool operator>=(const gqd_real &a, const gdd_real &b);
__device__ __host__ bool operator>=(const gdd_real &a, const gqd_real &b);
__device__ __host__ bool operator>=(const gqd_real &a, double b);
__device__ __host__ bool operator>=(double a, const gqd_real &b);

__device__ __host__ bool operator<(const gqd_real &a, const gqd_real &b);
__device__ __host__ bool operator<(const gqd_real &a, const gdd_real &b);
__device__ __host__ bool operator<(const gdd_real &a, const gqd_real &b);
__device__ __host__ bool operator<(const gqd_real &a, double b);
__device__ __host__ bool operator<(double a, const gqd_real &b);

__device__ __host__ bool operator<=(const gqd_real &a, const gqd_real &b);
__device__ __host__ bool operator<=(const gqd_real &a, const gdd_real &b);
__device__ __host__ bool operator<=(const gdd_real &a, const gqd_real &b);
__device__ __host__ bool operator<=(const gqd_real &a, double b);
__device__ __host__ bool operator<=(double a, const gqd_real &b);

__device__ bool is_zero(const gqd_real &a);
__device__ bool is_one(const gqd_real &a);
__device__ bool is_positive(const gqd_real &a);
__device__ bool is_negative(const gqd_real &a);

__device__ __host__ bool isnan(const gqd_real &a);
__device__ __host__ bool isfinite(const gqd_real &a);
__device__ __host__ bool isinf(const gqd_real &a);
__device__ bool is_pinf(const gqd_real &a);
__device__ bool is_ninf(const gqd_real &a);

__device__ __host__ gdd_real to_gdd_real(const gqd_real &a);
__device__ __host__ double   to_double(const gqd_real &a);
__device__ __host__ int      to_int(const gqd_real &a);
//__device__ __host__ int to_int(const double a);


__device__ __host__ gqd_real nint(const gqd_real &a);
__device__ __host__ gqd_real quick_nint(const gqd_real &a);
__device__ __host__ gqd_real floor(const gqd_real &a);
__device__ __host__ gqd_real ceil(const gqd_real &a);
__device__ __host__ gqd_real aint(const gqd_real &a);

__device__ gqd_real fabs(const gqd_real &a);
__device__ gqd_real abs(const gqd_real &a);    // same as fabs
__device__ gqd_real inv(const gqd_real &a);
__device__ __host__ gqd_real fmod(const gqd_real &a, const gqd_real &b);

__device__ gqd_real pow(const gqd_real &a, int n);
__device__ gqd_real pow(const gqd_real &a, const gqd_real &b);
__device__ gqd_real npwr(const gqd_real &a, int n);
__device__ gqd_real operator^(const gqd_real &a, int n);


__device__ gqd_real sqrt(const gqd_real &a);
__device__ gqd_real nroot(const gqd_real &a, int n);

__device__  gqd_real exp(const gqd_real &a);

__device__  gqd_real log(const gqd_real &a);
__device__  gqd_real log2(const gqd_real &a);
__device__  gqd_real log10(const gqd_real &a);

__device__  gqd_real sin(const gqd_real &a);
__device__  gqd_real cos(const gqd_real &a);
__device__  gqd_real tan(const gqd_real &a);
__device__  void sincos(const gqd_real &a, gqd_real &s, gqd_real &c);

#ifdef ALL_MATH
__device__  gqd_real asin(const gqd_real &a);
__device__  gqd_real acos(const gqd_real &a);
__device__  gqd_real atan(const gqd_real &a);
__device__  gqd_real atan2(const gqd_real &y, const gqd_real &x);

__device__  gqd_real sinh(const gqd_real &a);
__device__  gqd_real cosh(const gqd_real &a);
__device__  gqd_real tanh(const gqd_real &a);
__device__  void sincosh(const gqd_real &a, gqd_real &s, gqd_real &c);

__device__  gqd_real asinh(const gqd_real &a);
__device__  gqd_real acosh(const gqd_real &a);
__device__  gqd_real atanh(const gqd_real &a);
#endif

__device__ __host__ gqd_real qdrand(void);

__device__ __host__ gqd_real max(const gqd_real &a, const gqd_real &b);
__device__ __host__ gqd_real max(const gqd_real &a, const gqd_real &b, const gqd_real &c);
__device__ __host__ gqd_real min(const gqd_real &a, const gqd_real &b);
__device__ __host__ gqd_real min(const gqd_real &a, const gqd_real &b, const gqd_real &c);


//__device__ __host__ std::ostream &operator<<(std::ostream &s, const gqd_real &a);
//__device__ __host__ std::istream &operator>>(std::istream &s, gqd_real &a);



#endif /* __GQD_REAL_H__ */
