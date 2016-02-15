#ifndef __TEST_H__
#define __TEST_H__

#include "../libgqd/inc/cuda_header.h"



class Test{
public:
	__device__ Test();
	__device__ ~Test();
	__device__ static void expect_eq(double e, double a, char *ex, char *ac);
	__device__ static void expect_eq(unsigned __int64 e, double a, char *ex, char *ac);
	__device__ static void expect_eq(bool e, bool a, char *ex, char *ac);

	__device__ static void expect_eq2(double e1, double e2, double a1, double a2, char *ex1, char *ex2, char *ac1, char *ac2);

	__device__ static void expect_ne(double e, double a, char *ex, char *ac);
	__device__ static void expect_ne(unsigned __int64 e, double a, char *ex, char *ac);
	__device__ static void expect_ne(bool e, bool a, char *ex, char *ac);

	__device__ static void expect_ne2(double e1, double e2, double a1, double a2, char *ex1, char *ex2, char *ac1, char *ac2);

};


#define EXPECT_EQ(e, a) Test::expect_eq(e, a, #e, #a)
#define EXPECT_NE(e, a) Test::expect_ne(e, a, #e, #a)

#define EXPECT_EQ2(e1, e2, a1, a2) Test::expect_eq2(e1, e2, a1, a2, #e1, #e2, #a1, #a2)
#define EXPECT_NE2(e1, e2, a1, a2) Test::expect_ne2(e1, e2, a1, a2, #e1, #e2, #a1, #a2)


#endif	// __TEST_H__	
