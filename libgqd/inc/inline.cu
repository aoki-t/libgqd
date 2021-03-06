#ifndef __GDD_GQD_INLINE_CU__
#define __GDD_GQD_INLINE_CU__


#define _QD_SPLITTER            (134217729.0)                   // = 2^27 + 1
#define _QD_SPLIT_THRESH        (6.6969287949141707e+299)       // = 2^996


// For translate
union trans {
	unsigned __int64 asInt64;
	double  asDouble;
};


// Basic functions =============================================================

// computes fl( a + b ) and err( a + b ), assumes |a| > |b|
__forceinline__ __device__
double quick_two_sum( double a, double b, double &err ) {
	//double abs_a = fabs(a);
	//double abs_b = fabs(b);
	//if (!(abs_a > abs_b)) {
	//	double t = a;
	//	a = b;
	//	b = t;
	//}
	//assert(fabs(a) >= fabs(b));

	//if(b == 0.0) {
	//	err = 0.0;
	//	return a;
	//}

	double s = a + b;
	err = b - (s - a);
	return s;
}


// Computes fl(a+b) and err(a+b).
__forceinline__ __device__
double two_sum( double a, double b, double &err ) {

	//if( (a == 0.0) || (b == 0.0) ) {
	//	err = 0.0;
	//	return (a + b);
	//}

	double s = a + b;
	double bb = s - a;
	err = (a - (s - bb)) + (b - bb);
	
	return s;
}


// Computes fl( a - b ) and err( a - b ), assumes |a| >= |b|
__forceinline__ __device__
double quick_two_diff( double a, double b, double &err ) {

	//if (!(fabs(a) >= fabs(b))) {
	//	double t = a;
	//	a = b;
	//	b = t;
	//}
	//assert(fabs(a) >= fabs(b));

	//if (a == b) {
	//	err = 0.0;
	//	return 0.0;
	//}

	double s = a - b;
	err = (a - s) - b;
	return s;
}


// Computes fl( a - b ) and err( a - b )
__forceinline__ __device__
double two_diff( double a, double b, double &err ) {

	//if(a == b) {
	//	err = 0.0;
	//	return 0.0;
	//}

	double s = a - b;
	double bb = s - a;
	err = (a - (s - bb)) - (b + bb);
	return s;
}


// Computes high word and lo word of a 
#ifndef USE_FMA
__forceinline__ __device__
void split(double a, double &hi, double &lo) {
	double temp;
	if (a > _QD_SPLIT_THRESH || a < -_QD_SPLIT_THRESH)
	{
		a *= 3.7252902984619140625e-09;  // 2^-28
		temp = _QD_SPLITTER * a;
		hi = temp - (temp - a);
		lo = a - hi;
		hi *= 268435456.0;          // 2^28
		lo *= 268435456.0;          // 2^28
	} else 	{
		temp = _QD_SPLITTER * a;
		hi = temp - (temp - a);
		lo = a - hi;
	}

}
#endif


// Computes fl(a*b) and err(a*b).
__forceinline__  __device__
double two_prod(double a, double b, double &err) {
#ifdef USE_FMA
	double p = a * b;
	err = fma(a, b, -p);
	return p;

#else
	double a_hi, a_lo, b_hi, b_lo;
	double p = a * b;
	split(a, a_hi, a_lo);
	split(b, b_hi, b_lo);
	
	//err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
	err = (a_hi*b_hi) - p + (a_hi*b_lo) + (a_lo*b_hi) + (a_lo*b_lo); 

	return p;
#endif
}


// Computes fl(a*a) and err(a*a).  Faster than calling two_prod(a, a, err).
__forceinline__ __device__
double two_sqr(double a, double &err) {
#ifdef USE_FMA
	double p = a * a;
	err = fma(a, a, -p);
	return p;

#else
	double hi, lo;
	double q = a * a;
	split(a, hi, lo);
	err = ((hi * hi - q) + 2.0 * hi * lo) + lo * lo;
	return q;
#endif
}


// Computes the nearest integer to d.
__forceinline__ __device__
double nint(double d) {
	if (d == std::floor(d)){
		return d;
	}
	return std::floor(d + 0.5);
}


__device__
bool is_positive(double a) {
	const unsigned __int64 cons = 0x8000000000000000ULL;
	trans t;
	t.asDouble = a;

	if (t.asInt64 == 0x7ff8000000000000ULL) return false;
	bool result = ((t.asInt64 & cons) == 0);
	return result;
}


__device__
bool is_negative(double a) {
	const unsigned __int64 cons = 0x8000000000000000ULL;
	trans t;
	t.asDouble = a;

	if (t.asInt64 == 0xfff8000000000000ULL) return false;
	bool result = ((t.asInt64 & cons) == cons);
	return result;
}




#endif /* __GDD_GQD_INLINE_CU__ */
