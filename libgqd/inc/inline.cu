#ifndef __GDD_GQD_INLINE_CU__
#define __GDD_GQD_INLINE_CU__


#define _QD_SPLITTER            (134217729.0)                   // = 2^27 + 1
#define _QD_SPLIT_THRESH        (6.69692879491417e+299)         // = 2^996


/****************Basic Funcitons *********************/

//computs fl( a + b ) and err( a + b ), assumes |a| > |b|
__forceinline__ __device__ __host__
double quick_two_sum( double a, double b, double &err ) {

	if(b == 0.0) {
		err = 0.0;
		return (a + b);
	}

	double s = a + b;
	err = b - (s - a);

	return s;
}

__forceinline__ __device__ __host__
double two_sum( double a, double b, double &err ) {

	if( (a == 0.0) || (b == 0.0) ) {
		err = 0.0;
		return (a + b);
	}

	double s = a + b;
	double bb = s - a;
	err = (a - (s - bb)) + (b - bb);
	
	return s;
}


//computes fl( a - b ) and err( a - b ), assumes |a| >= |b|
__forceinline__ __device__ __host__
double quick_two_diff( double a, double b, double &err ) {
	if(a == b) {
		err = 0.0;
		return 0.0;
	}

	double s = a - b;
	err = (a - s) - b;
	return s;
}

//computes fl( a - b ) and err( a - b )
__forceinline__ __device__ __host__
double two_diff( double a, double b, double &err ) {
	if(a == b) {
		err = 0.0;
		return 0.0;
	}

	double s = a - b;
	double bb = s - a;
	err = (a - (s - bb)) - (b + bb);
	return s;
}

// Computes high word and lo word of a 
__forceinline__ __device__ __host__
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

/* Computes fl(a*b) and err(a*b). */
__forceinline__  __device__ __host__
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
	
	//err = (a_hi*b_hi) - p + (a_hi*b_lo) + (a_lo*b_hi) + (a_lo*b_lo); 
	err = (a_hi*b_hi) - p + (a_hi*b_lo) + (a_lo*b_hi) + (a_lo*b_lo); 

	return p;
#endif
}

/* Computes fl(a*a) and err(a*a).  Faster than the above method. */
__forceinline__ __device__ __host__
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

/* Computes the nearest integer to d. */
__forceinline__ __device__ __host__
double nint(double d) {
	if (d == std::floor(d)){
		return d;
	}
	return std::floor(d + 0.5);
}



#endif /* __GDD_GQD_INLINE_CU__ */
