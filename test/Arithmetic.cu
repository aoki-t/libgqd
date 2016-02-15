
#include "../libgqd/inc/cuda_header.h"
#include "../libgqd/inc/gdd_real.h"
#include "../libgqd/inc/gqd_real.h"

#include "kernel.h"
#include "Test.h"




// debug function
__device__
void newline() {
	printf("\n");
}

__device__
void cout(const char *msg) {
	printf(msg);
}




// ==============================================================================

__global__
void eps_test() {

	EXPECT_EQ(0x3970000000000000ULL, _dd_eps);	// 2^-104 = 1.0 * 2^(-104 + 1023)
	EXPECT_EQ(0x32e0000000000000ULL, _qd_eps);	// 2^-209 = 1.0 * 2^(-209 + 1023)
	newline();
}


__global__
void posinega_test() {
	unsigned __int64 data[20] = {
		0xfff8000000000000ULL,	// qnan 
		0x7ff8000000000000ULL,	// -qnan (host qnan)
		0x7ff0000000000000ULL,	// inf   (host inf)
		0xfff0000000000000ULL,	// -inf
		0x7fefffffffffffffULL,	// max
		0xffefffffffffffffULL,	// -max
		0x0010000000000000ULL,	// min_norm
		0x8010000000000000ULL,	// -min_norm
		0x000fffffffffffffULL,	// subnormal_max
		0x800fffffffffffffULL,	// -subnormal_max
		0x0000000000000001ULL,	// subnormal_min
		0x8000000000000001ULL,	// -subnormal_min
		0x4011000000000000ULL,	// 3.0
		0x4000000000000000ULL,	// 2.0
		0x3ff0000000000000ULL,	// 1.0
		0x0000000000000000ULL,	// 0.0
		0x8000000000000000ULL,	// -0.0
		0xbff0000000000000ULL,	// -1.0
		0xc000000000000000ULL,	// -2.0
		0xc011000000000000ULL,	// -3.0
	};
	trans t[20];
	for (int i = 0; i < 20; i++) {
		t[i].asInt64 = data[i];
	}
	double t_pqnan = t[0].asDouble;
	double t_nqnan = t[1].asDouble;
	double t_pinf = t[2].asDouble;
	double t_ninf = t[3].asDouble;
	double t_pmax = t[4].asDouble;
	double t_nmax = t[5].asDouble;
	double t_pmin_norm = t[6].asDouble;
	double t_nmin_norm = t[7].asDouble;
	double t_psubnorm_max = t[8].asDouble;
	double t_nsubnorm_max = t[9].asDouble;
	double t_psubnorm_min = t[10].asDouble;
	double t_nsubnorm_min = t[11].asDouble;
	double t_three = t[12].asDouble;
	double t_two = t[13].asDouble;
	double t_one = t[14].asDouble;
	double t_zero = t[15].asDouble;
	double t_nzero = t[16].asDouble;
	double t_none = t[17].asDouble;
	double t_ntwo = t[18].asDouble;
	double t_nthree = t[19].asDouble;

	printf("check is_positive ---------------------------------------------\n");

	EXPECT_EQ(false, is_positive(t_pqnan));
	EXPECT_EQ(false, is_positive(t_nqnan));
	EXPECT_EQ(true, is_positive(t_pinf));
	EXPECT_EQ(false, is_positive(t_ninf));
	EXPECT_EQ(true, is_positive(t_pmax));
	EXPECT_EQ(false, is_positive(t_nmax));
	EXPECT_EQ(true, is_positive(t_pmin_norm));
	EXPECT_EQ(false, is_positive(t_nmin_norm));
	EXPECT_EQ(true, is_positive(t_psubnorm_max));
	EXPECT_EQ(false, is_positive(t_nsubnorm_max));
	EXPECT_EQ(true, is_positive(t_psubnorm_min));
	EXPECT_EQ(false, is_positive(t_nsubnorm_min));
	EXPECT_EQ(true, is_positive(t_three));
	EXPECT_EQ(true, is_positive(t_two));
	EXPECT_EQ(true, is_positive(t_one));
	EXPECT_EQ(true, is_positive(t_zero));
	EXPECT_EQ(false, is_positive(t_nzero));
	EXPECT_EQ(false, is_positive(t_none));
	EXPECT_EQ(false, is_positive(t_ntwo));
	EXPECT_EQ(false, is_positive(t_nthree));
	newline();

	printf("check is_negative ---------------------------------------------\n");
	EXPECT_EQ(false, is_negative(t_pqnan));
	EXPECT_EQ(false, is_negative(t_nqnan));
	EXPECT_EQ(false, is_negative(t_pinf));
	EXPECT_EQ(true, is_negative(t_ninf));
	EXPECT_EQ(false, is_negative(t_pmax));
	EXPECT_EQ(true, is_negative(t_nmax));
	EXPECT_EQ(false, is_negative(t_pmin_norm));
	EXPECT_EQ(true, is_negative(t_nmin_norm));
	EXPECT_EQ(false, is_negative(t_psubnorm_max));
	EXPECT_EQ(true, is_negative(t_nsubnorm_max));
	EXPECT_EQ(false, is_negative(t_psubnorm_min));
	EXPECT_EQ(true, is_negative(t_nsubnorm_min));
	EXPECT_EQ(false, is_negative(t_three));
	EXPECT_EQ(false, is_negative(t_two));
	EXPECT_EQ(false, is_negative(t_one));
	EXPECT_EQ(false, is_negative(t_zero));
	EXPECT_EQ(true, is_negative(t_nzero));
	EXPECT_EQ(true, is_negative(t_none));
	EXPECT_EQ(true, is_negative(t_ntwo));
	EXPECT_EQ(true, is_negative(t_nthree));
	newline();

}

__global__
void two_sum_test() {

	unsigned __int64 list[][2] = {
		// a, b, expected
		{ 0x3ff0000000000001ULL, 0x0030000000000003ULL },	//	|a| > |b| norm vs norm
		{ 0x0009000000000000ULL, 0x0007000000000000ULL },	//	|a| > |b| subnorm vs subnorm
		{ 0x3ff0000000000001ULL, 0x0007000000000000ULL },	//	|a| > |b| norm vs subnorm
		{ 0x3ff0000000000001ULL, 0xbff0000000000001ULL },	//	|a| = |-a| norm
		{ 0x0007000000000000ULL, 0x8007000000000000ULL },	//	|a| = |-a| subnorm
		{ 0x3ff0000000000001ULL, 0x3ff0000000000003ULL },	//  |a| < |b| near norm
		{ 0x3ff0000000000001ULL, 0xbff0000000000002ULL },	//	|a| < |b| 
	};

	unsigned __int64 rlist[][2] = {
		{ 0x3ff0000000000001ULL, 0x0030000000000003ULL },
		{ 0x0010000000000000ULL, 0x0000000000000000ULL },
		{ 0x3ff0000000000001ULL, 0x0007000000000000ULL },
		{ 0x0000000000000000ULL, 0x0000000000000000ULL },
		{ 0x0000000000000000ULL, 0x0000000000000000ULL },
		{ 0x4000000000000002ULL, 0x0000000000000000ULL },
		{ 0xbcb0000000000000ULL, 0x0000000000000000ULL },
	};
	const int n_lists = 7;
	double dlist[n_lists][2];
	double drlist[n_lists][2];
	
	double s1, e1, s2, e2, a, b, r1, r2;

	// data prepare
	trans t1, t2;
	int lsize = sizeof(list) / sizeof(list[0]);
	for (int i = 0; i < lsize; i++) {
		t1.asInt64 = list[i][0];
		t2.asInt64 = list[i][1];
		dlist[i][0] = t1.asDouble;
		dlist[i][1] = t2.asDouble;

		t1.asInt64 = rlist[i][0];
		t2.asInt64 = rlist[i][1];
		drlist[i][0] = t1.asDouble;
		drlist[i][1] = t2.asDouble;

	}

	printf("check two_sum vs quick_two_sum ----------------------------\n");
	for (int i = 0; i < lsize; i++) {
		a = dlist[i][0];
		b = dlist[i][1];
		r1 = drlist[i][0];
		r2 = drlist[i][1];

		s1 = two_sum(a, b, e1);
		s2 = quick_two_sum(a, b, e2);
		EXPECT_EQ2(s1, e1, s2, e2);
		EXPECT_EQ2(r1, r2, s1, e1);
		newline();
	}
	newline();

	printf("check two_sum ---------------------------------------------\n");
	for (int i = 0; i < lsize; i++) {
		a = dlist[i][0];
		b = dlist[i][1];
		r1 = drlist[i][0];
		r2 = drlist[i][1];

		s1 = two_sum(a, b, e1);
		s2 = two_sum(b, a, e2);
		EXPECT_EQ2(s1, e1, s2, e2);
		EXPECT_EQ2(r1, r2, s1, e1);
		newline();
	}
	newline();

	printf("check quick_two_sum ---------------------------------------\n");
	int i = 0;
	a = dlist[i][0],
	b = dlist[i][1];
	r1 = drlist[i][0];
	r2 = drlist[i][1];
	s1 = quick_two_sum(a, b, e1);
	s2 = quick_two_sum(b, a, e2);
	EXPECT_NE2(s1, e1, s2, e2);
	EXPECT_EQ2(r1, r2, s1, e1);
	EXPECT_NE2(r1, r2, s2, e2);
	i++;
	newline();

	a = dlist[i][0],
	b = dlist[i][1];
	r1 = drlist[i][0];
	r2 = drlist[i][1];
	s1 = quick_two_sum(a, b, e1);
	s2 = quick_two_sum(b, a, e2);
	EXPECT_EQ2(s1, e1, s2, e2);
	EXPECT_EQ2(r1, r2, s1, e1);
	EXPECT_EQ2(r1, r2, s2, e2);
	i++;
	newline();

	a = dlist[i][0],
	b = dlist[i][1];
	r1 = drlist[i][0];
	r2 = drlist[i][1];
	s1 = quick_two_sum(a, b, e1);
	s2 = quick_two_sum(b, a, e2);
	EXPECT_NE2(s1, e1, s2, e2);
	EXPECT_EQ2(r1, r2, s1, e1);
	EXPECT_NE2(r1, r2, s2, e2);
	i++;
	newline();

	a = dlist[i][0],
	b = dlist[i][1];
	r1 = drlist[i][0];
	r2 = drlist[i][1];
	s1 = quick_two_sum(a, b, e1);
	s2 = quick_two_sum(b, a, e2);
	EXPECT_EQ2(s1, e1, s2, e2);
	EXPECT_EQ2(r1, r2, s1, e1);
	EXPECT_EQ2(r1, r2, s2, e2);
	i++;
	newline();

	a = dlist[i][0],
	b = dlist[i][1];
	r1 = drlist[i][0];
	r2 = drlist[i][1];
	s1 = quick_two_sum(a, b, e1);
	s2 = quick_two_sum(b, a, e2);
	EXPECT_EQ2(s1, e1, s2, e2);
	EXPECT_EQ2(r1, r2, s1, e1);
	EXPECT_EQ2(r1, r2, s2, e2);
	i++;
	newline();

	a = dlist[i][0],
	b = dlist[i][1];
	r1 = drlist[i][0];
	r2 = drlist[i][1];
	s1 = quick_two_sum(a, b, e1);
	s2 = quick_two_sum(b, a, e2);
	EXPECT_EQ2(s1, e1, s2, e2);
	EXPECT_EQ2(r1, r2, s1, e1);
	EXPECT_EQ2(r1, r2, s2, e2);
	i++;
	newline();

	a = dlist[i][0],
	b = dlist[i][1];
	r1 = drlist[i][0];
	r2 = drlist[i][1];
	s1 = quick_two_sum(a, b, e1);
	s2 = quick_two_sum(b, a, e2);
	EXPECT_EQ2(s1, e1, s2, e2);
	EXPECT_EQ2(r1, r2, s1, e1);
	EXPECT_EQ2(r1, r2, s2, e2);
	i++;
	newline();

}

__global__
void two_diff_test() {

	unsigned __int64 list[][2] = {
		// a, b, expected
		{ 0x3ff0000000000001ULL, 0x0030000000000003ULL },	//	|a| > |b| norm vs norm
		{ 0x0009000000000000ULL, 0x0007000000000000ULL },	//	|a| > |b| subnorm vs subnorm
		{ 0x3ff0000000000001ULL, 0x0007000000000000ULL },	//	|a| > |b| norm vs subnorm
		{ 0x3ff0000000000001ULL, 0xbff0000000000001ULL },	//	|a| = |-a| norm
		{ 0x0007000000000000ULL, 0x8007000000000000ULL },	//	|a| = |-a| subnorm
		{ 0x3ff0000000000001ULL, 0x3ff0000000000003ULL },	//  |a| < |b| near norm
		{ 0x3ff0000000000001ULL, 0xbff0000000000002ULL },	//	|a| < |b| 
	};

	unsigned __int64 rlist[][2] = {
		{ 0x3ff0000000000001ULL, 0x8030000000000003ULL },
		{ 0x0002000000000000ULL, 0x0000000000000000ULL },
		{ 0x3ff0000000000001ULL, 0x8007000000000000ULL },
		{ 0x4000000000000001ULL, 0x0000000000000000ULL },
		{ 0x000e000000000000ULL, 0x0000000000000000ULL },
		{ 0xbcc0000000000000ULL, 0x0000000000000000ULL },
		{ 0x4000000000000002ULL, 0xbcb0000000000000ULL },
	};
	const int n_lists = 7;
	double dlist[n_lists][2];
	double drlist[n_lists][2];

	double s1, e1, s2, e2, a, b, r1, r2;

	// data prepare
	trans t1, t2;
	int lsize = sizeof(list) / sizeof(list[0]);
	for (int i = 0; i < lsize; i++) {
		t1.asInt64 = list[i][0];
		t2.asInt64 = list[i][1];
		dlist[i][0] = t1.asDouble;
		dlist[i][1] = t2.asDouble;

		t1.asInt64 = rlist[i][0];
		t2.asInt64 = rlist[i][1];
		drlist[i][0] = t1.asDouble;
		drlist[i][1] = t2.asDouble;

	}

	printf("check two_diff vs quick_two_diff ----------------------------\n");
	for (int i = 0; i < lsize; i++) {
		a = dlist[i][0];
		b = dlist[i][1];
		r1 = drlist[i][0];
		r2 = drlist[i][1];

		s1 = two_diff(a, b, e1);
		s2 = quick_two_diff(a, b, e2);
		EXPECT_EQ2(s1, e1, s2, e2);
		EXPECT_EQ2(r1, r2, s1, e1);
		newline();
	}
	newline();


}


__global__
void split_test() {
	unsigned __int64 list[][1] = {
		{ 0x3feffffffe000000ULL },
		{ 0x3fe0000001ffffffULL },
		{ 0x3fefffffffffffffULL },
		{ 0x3fe0000003000000ULL },
		{ 0x7e3ffffffe000000ULL },
		{ 0x7e30000001ffffffULL },
		{ 0x7e3fffffffffffffULL },
		{ 0x7e30000003000000ULL },
		{ 0x7e31234567898765ULL },	//  1234567898765 * 2^996
		{ 0xfe31234567898765ULL },	// -1234567898765 * 2^996
		{ 0x7e3fffffffffffffULL },	//  ff... * 2^996
		{ 0xfe3fffffffffffffULL },	// -ff... * 2^996
		{ 0x7e30000000000001ULL },	//  2^996 + ulp(1)
		{ 0xfe30000000000001ULL },	// -2^996 - ulp(1)
		{ 0xfff8000000000000ULL },	// qnan 
		{ 0x7ff8000000000000ULL },	// -qnan (host qnan)
		{ 0x7ff0000000000000ULL },	// inf   (host inf)
		{ 0xfff0000000000000ULL },	// -inf
		{ 0x7fefffffffffffffULL },	// max
		{ 0xffefffffffffffffULL },	// -max
		{ 0x0010000000000000ULL },	// min_norm
		{ 0x8010000000000000ULL },	// -min_norm
		{ 0x000fffffffffffffULL },	// subnormal_max
		{ 0x800fffffffffffffULL },	// -subnormal_max
		{ 0x0000000000000001ULL },	// subnormal_min
		{ 0x8000000000000001ULL },	// -subnormal_min
	};

	unsigned __int64 rlist[][2] = {
		{ 0x3ff0000000000000ULL, 0xbe30000000000000ULL },
		{ 0x3fe0000000000000ULL, 0x3e2ffffff0000000ULL },
		{ 0x3ff0000000000000ULL, 0xbca0000000000000ULL },
		{ 0x3fe0000000000000ULL, 0x3e38000000000000ULL },
		{ 0x7e40000000000000ULL, 0xfc80000000000000ULL },
		{ 0x7e30000000000000ULL, 0x7c7ffffff0000000ULL },
		{ 0x7e40000000000000ULL, 0xfaf0000000000000ULL },
		{ 0x7e30000000000000ULL, 0x7c88000000000000ULL },
		{ 0x7e31234568000000ULL, 0xfc5d9e26c0000000ULL },
		{ 0xfe31234568000000ULL, 0x7c5d9e26c0000000ULL },
		{ 0x7e40000000000000ULL, 0xfaf0000000000000ULL },
		{ 0xfe40000000000000ULL, 0x7af0000000000000ULL },
		{ 0x7e30000000000000ULL, 0x7af0000000000000ULL },
		{ 0xfe30000000000000ULL, 0xfaf0000000000000ULL },
		{ 0xfff8000000000000ULL, 0xfff8000000000000ULL },	// qnan 
		{ 0x7ff8000000000000ULL, 0x7ff8000000000000ULL },	// -qnan (host qnan)
		{ 0x7ff0000000000000ULL, 0x7ff0000000000000ULL },	// inf   (host inf)
		{ 0xfff0000000000000ULL, 0xfff0000000000000ULL },	// -inf
		{ 0x7ff0000000000000ULL, 0xfca0000000000000ULL },	// max
		{ 0xfff0000000000000ULL, 0x7ca0000000000000ULL },	// -max
		{ 0x0010000000000000ULL, 0x0000000000000000ULL },	// min_norm
		{ 0x8010000000000000ULL, 0x0000000000000000ULL },	// -min_norm
		{ 0x0010000000000000ULL, 0x8000000000000001ULL },	// subnormal_max
		{ 0x8010000000000000ULL, 0x0000000000000001ULL },	// -subnormal_max
		{ 0x0000000000000001ULL, 0x0000000000000000ULL },	// subnormal_min
		{ 0x8000000000000001ULL, 0x0000000000000000ULL },	// -subnormal_min

	};
	const int n_lists = 26;
	double dlist[n_lists][1];
	double drlist[n_lists][2];

	double s1, e1, s2, e2, a, b, r1, r2;

	// data prepare
	trans t1, t2;
	int lsize = sizeof(list) / sizeof(list[0]);
	for (int i = 0; i < lsize; i++) {
		t1.asInt64 = list[i][0];
		dlist[i][0] = t1.asDouble;

		t1.asInt64 = rlist[i][0];
		t2.asInt64 = rlist[i][1];
		drlist[i][0] = t1.asDouble;
		drlist[i][1] = t2.asDouble;

	}
	
	printf("check Split -------------------------------------------------\n");

	double hi, lo, hi_e, lo_e;
	for (int i = 0; i < lsize; i++){
		a = dlist[i][0];
		split(a, hi, lo);
		hi_e = drlist[i][0];
		lo_e = drlist[i][1];
		printf("[0x%016llx] {0x%016llx ,0x%016llx}\n", a, hi, lo);

		EXPECT_EQ2(hi_e, lo_e, hi, lo);
		newline();
	}
	newline();

}

cudaError_t k_checkInline() {
	printf("checkInline ---------------------------------------------------\n");

	eps_test<<<1, 1 >>>();
	posinega_test<<<1, 1>>>();
	two_sum_test<<<1, 1>>>();
	two_diff_test<<<1, 1>>>();

#ifndef USE_FMA
	split_test<<<1,1>>>();
#endif

	cudaError_t cudaStatus = cudaGetLastError();
	return cudaStatus;
}

//namespace std{
//
//class string{
//private:
//	static short n_threads;
//	char *m_str;
//	short size = 0;
//	short alloc_size = 0;
//protected:
//	__device__
//	short get_alloc_size(const char* s) {
//		unsigned int t = sizeof(s);
//		if (t > SHRT_MAX) return -1;
//		
//		int denominator = 64;
//		int quotient = t / denominator;
//		while (quotient != 0 || denominator < SHRT_MAX){
//			quotient = t / denominator;
//			denominator *= 2;
//		}
//		return (short)denominator;
//	}
//
//public:
//	__device__
//	static void initialize(cudaDeviceProp *info) {
//		n_threads = (short)(info->maxThreadsPerBlock);
//		//new 
//	}
//	__device__
//	static void finalize() {
//		n_threads = 0;
//		//delete s;
//	}
//
//	__device__
//	string()
//	:m_str(""), size(0){
//
//	}
//	__device__
//	~string() {
//		delete m_str;
//		m_str = NULL;
//		size = 0;
//	}
//	__device__
//	string(char *str) {
//		int bytes = sizeof(str);
//		m_str = new char(bytes);
//	}
//
//	__device__
//	string(const string &str) {
//		if (*this == str) return;
//
//		delete m_str;
//		size = sizeof(str);
//		m_str = new char(size);
//	}
//	__device__
//	string &operator=(const string &s) {
//		if (*this == s) return;
//		if (m_str != NULL) { 
//			delete m_str;
//			size = 0;
//		}
//
//		int bytes = sizeof(s.m_str);
//		m_str = new char(bytes);
//
//	}
//	__device__
//	bool operator==(const string &s) {
//		if (this->m_str != NULL && this->m_str == s.m_str) {
//			return true;
//		}
//		return false;
//	}
//	string operator+(const string &s) {
//
//	}
//	string operator+(const char *c) {
//
//	}
//
//};
//
//};	// namespace std
//


