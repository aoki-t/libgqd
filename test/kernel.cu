#include <float.h>
#include <stdio.h>
#include <iostream>
//#include <math.h>


#include "../libgqd/inc/cuda_header.h"
#include "../libgqd/inc/common.cu"
#include "../libgqd/inc/gdd_real.h"
#include "../libgqd/inc/gqd_type.h"
#include "../libgqd/inc/gqd.cu"

#include "kernel.h"
#include "Arithmetic.cu"

//#include "Memory.cu"
//cudaError_t addWithCuda(gdd_real c[], const gdd_real a[], const gdd_real b[], unsigned int size);

//// For translate
//static union trans {
//	__int64 asInt64;
//	double  asDouble;
//};

//cudaError_t addWithCuda(gdd_real *c, const gdd_real *a, const gdd_real *b, unsigned int size);
//cudaError_t addWithCudaq(gqd_real *c, const gqd_real *a, const gqd_real *b, unsigned int size);
//cudaError_t IEEE_check();

__global__ void addKernel(gdd_real c[],  gdd_real a[],  gdd_real b[])
{
	/*
	operator+(gdd_real const&, gdd_real const&)
	gdd_real::operator=(gdd_real const&)

	gdd_real::operator*(gdd_real const&)
	gdd_real::operator-(gdd_real const&)
	gdd_real::operator/(gdd_real const&)
	*/

    int i = threadIdx.x;
	//c[i] = a[i];
    c[i] = a[i] + b[i];
	c[i] = c[i] * b[i];
	c[i] = c[i] - b[i];
	c[i] = c[i] / b[i];
	c[i] += b[i];
	c[i] *= b[i];
	c[i] -= b[i];
	c[i] /= b[i];
	gdd_real xxx = sin(a[i]);
	c[i] = xxx;

	//printf("[%d, %d]:\t\tValue is:%llf\n", \
	//	blockIdx.y*gridDim.x + blockIdx.x, \
	//	threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x, \
	//	c[i].dd.x);

	gdd_real ze = gdd_real(0.0, 0.0);
	gdd_real neg_ze = gdd_real(CUDART_NEG_ZERO, CUDART_NEG_ZERO);
	gdd_real na = gdd_real(CUDART_NAN, CUDART_NAN);
	gdd_real ia = gdd_real(CUDART_INF, CUDART_INF);
	trans t;
	t.asInt64 = 0x0000000000000003ULL;	//subnormal
	trans t4, t5;
	t4.asDouble = CUDART_INF;	//inf
	t5.asDouble = CUDART_NAN;	//nan

	bool ip = is_positive(ze);
	bool in = is_negative(ze);
	bool iz = is_zero(ze);
	bool ip2 = is_positive(neg_ze);
	bool in2 = is_negative(neg_ze);
	bool iz2 = is_zero(neg_ze);

	bool io = is_one(negative(gdd_real(-1.0)));
	bool iq = isnan(-na);
	bool ipi = isinf(ia);
	bool ini = isinf(-ia);
	bool ipf = isfinite(ia);
	bool inf = isfinite(-ia);
	bool isn = isfinite(t.asDouble);	// subnormal
	bool isn2 = isfinite(-t.asDouble);	// subnormal


	gqd_real infini = gqd_real(t4.asDouble);
	printf("gqd_real( infinity)        is:% f\n", infini[0]);
	gqd_real n_infini = gqd_real(-t4.asDouble);
	printf("gqd_real(-infinity)        is:% f\n", n_infini[0]);

	gqd_real q_nan = gqd_real(t5.asDouble);
	printf("gqd_real( q_nan)           is:% f\n", q_nan[0]);
	gqd_real n_q_nan = gqd_real(-t5.asDouble);
	printf("gqd_real(-q_nan)           is:% f\n", n_q_nan[0]);

	gqd_real subn = gqd_real(t.asDouble);
	printf("gqd_real( subnormal)       is:% f\n", subn[0]);
	gqd_real n_subn = gqd_real(-t.asDouble);
	printf("gqd_real(-subnormal)       is:% f\n", n_subn[0]);



	gdd_real bb(0.5), cc;
	printf("is_positive(zero)        is:%d\n", ip);
	printf("is_negative(zero)        is:%d\n", in);
	printf("is_zero(zero)            is:%d\n", iz);
	printf("is_positive(neg_zero)    is:%d\n", ip2);
	printf("is_negative(neg_zero)    is:%d\n", in2);
	printf("is_zero(neg_zero)        is:%d\n", iz2);

	printf("is_one(-one)    is:%d\n", io);
	printf("is_nan          is:%d\n", iq);
	printf("is_inf(pinf)	is:%d\n", ipi);
	printf("is_inf(ninf)	is:%d\n", ini);
	printf("is_finite(pinf)	is:%d\n", ipf);
	printf("is_finite(ninf)	is:%d\n", inf);
	printf("is_finite( subnorm)is:%d\n", isn);
	printf("is_finite(-subnorm)is:%d\n", isn2);


	gqd_real q_none = gqd_real(-1.0);
	printf("-one	  is:%08llx %08llx %08llx %08llx \n", q_none[0], q_none[1], q_none[2], q_none[3]);
	printf("-one	  is:%0llf %0llf %0llf %0llf \n", q_none[0], q_none[1], q_none[2], q_none[3]);

	q_none = negative(q_none);
	printf("neg(-one) is:%08llx %08llx %08llx %08llx \n", q_none[0], q_none[1], q_none[2], q_none[3]);
	printf("neg(-one) is:%0llf %0llf %0llf %0llf \n", q_none[0], q_none[1], q_none[2], q_none[3]);
	
	io = is_one(q_none);
	printf("is_one(neg(-one))	is:%d\n", io);

	if (0.0 == CUDART_NEG_ZERO) {
		printf("0.0 == CUDART_NEG_ZERO	is: True\n");
	} else {
		printf("0.0 == CUDART_NEG_ZERO	is: False\n");
	}
	if (0.0 > CUDART_NEG_ZERO) {
		printf("0.0 > CUDART_NEG_ZERO	is: True\n");
	} else {
		printf("0.0 > CUDART_NEG_ZERO	is: False\n");
	}

	printf("-----------------------------------------------------------------\n");
	trans t1, t2;
	unsigned __int64 list[][2] = {
		{ 0x3ff0000000000001ULL, 0x3ff0000000000003ULL },	//  |a| < |b|
		{ 0x3ff0000000000001ULL, 0x0030000000000003ULL },	//	|a| > |b|
		{ 0x3ff0000000000001ULL, 0xbff0000000000001ULL },	//	|a| = |-a|
		{ 0x3ff0000000000001ULL, 0xbff0000000000002ULL },	//	|a| < |b|
	};
	int max = sizeof(list) / sizeof(list[0]);
	for (int i = 0; i < max; i++) {
		t1.asInt64 = list[i][0];
		t2.asInt64 = list[i][1];
		double e1, e2, e3;
		
		printf("in     [% e, % e] [%016llx, %016llx]\n", t1.asDouble, t2.asDouble, t1.asDouble, t2.asDouble);
		double s1 = quick_two_sum(t1.asDouble, t2.asDouble, e1);
		printf("qt-out [% e, % e] [%016llx, %016llx]\n", s1, e1, s1, e1);
		double s2 = quick_two_sum(t2.asDouble, t1.asDouble, e2);
		printf("qt-out [% e, % e] [%016llx, %016llx]\n", s2, e2, s2, e2);

		double s3 = two_sum(t2.asDouble, t1.asDouble, e3);
		printf(" t-out [% e, % e] [%016llx, %016llx]\n", s3, e3, s3, e3);

		if (s1 == s2 && e1 == e2) {
			printf("same\n\n");
		} else {
			printf("different \n\n");
		}
	}
	printf("-----------------------------------------------------------------\n");
	//cc = npwr(bb, 2);
	//printf("0.5^2 = %f\n", cc.dd.x);
	//cc = npwr(bb, 1);
	//printf("0.5^1 = %f\n", cc.dd.x);

	//cc = npwr(bb, -1);
	//printf("0.5^-1 = %f\n", cc.dd.x);
	//cc = npwr(bb, -2);
	//printf("0.5^-2 = %f\n", cc.dd.x);

	//cc = pow(bb, bb);
	//printf("0.5^0.5 = %f, %f\n", cc.dd.x, cc.dd.y);


}
__global__ void addKernelq(gqd_real *c, const gqd_real *a, const gqd_real *b)
{
    //int i = threadIdx.x;
    //c[i] = a[i] + b[i];
	//c[i] = c[i] * b[i];
	//c[i] = c[i] - b[i];
	//c[i] = c[i] / b[i];

}

__global__ void NanOperation(double *c, int N)
{
	extern __shared__ double res[];
	const double inf = CUDART_INF;
	const double cunan = CUDART_NAN;
	const double nzero = CUDART_NEG_ZERO;
	const double zero = 0.0;
	const double one = 1.0;
	//const int N = 40;
	

	//int i = threadIdx.x;

	res[0] = pow(zero, zero);	// 0^0
	res[1] = pow(one, zero);	// 1^0
	res[2] = pow(cunan, zero);	// NaN^0	(2008では1を返す)
	res[3] = pow(inf, zero);	// Inf^0

	res[4] = pow(zero, one);	// 0^1
	res[5] = pow(one, one);		// 1^1
	res[6] = pow(cunan, one);	// NaN^1
	res[7] = pow(inf, one);		// Inf^1

	res[8] = pow(zero, inf);	// 0^inf
	res[9] = pow(one, inf);		// 1^inf
	res[10] = pow(2.0, inf);	// 2^inf
	res[11] = pow(cunan, inf);	// NaN^inf
	res[12] = pow(inf, inf);	// Inf^inf

	res[13] = pow(zero, cunan);	// 0^NaN
	res[14] = pow(one, cunan);	// 1^NaN	(2008では1を返す)
	res[15] = pow(inf, cunan);	// inf^NaN
	res[16] = pow(cunan, cunan);	// NaN^NaN

	res[17] = exp(cunan);	// exp(NaN)
	res[18] = exp(inf);		// exp(inf)
	res[19] = exp(-inf);	// exp(-inf)

	// log関数仕様確認
	res[20] = log(zero);
	res[21] = log(nzero);
	res[22] = log(one);
	res[23] = log(-one);
	res[24] = log( inf);
	res[25] = log(-inf);
	res[26] = log(cunan);

	// sqrt関数仕様確認
	res[27] = sqrt(zero);
	res[28] = sqrt(nzero);
	res[29] = sqrt( inf);
	res[30] = sqrt(-one);
	res[31] = sqrt(-inf);
	res[32] = sqrt(cunan);

	// comparition
	res[33] = (cunan == cunan);
	res[34] = (cunan != cunan);
	res[35] = (cunan > cunan);
	res[36] = (cunan >= cunan);
	res[37] = (cunan < cunan);
	res[38] = (cunan <= cunan);

	res[39] = (cunan == inf);
	res[40] = (cunan != inf);
	res[41] = (cunan >  inf);
	res[42] = (cunan >= inf);
	res[43] = (cunan <  inf);
	res[44] = (cunan <= inf);

	res[45] = (cunan == -inf);
	res[46] = (cunan != -inf);
	res[47] = (cunan >  -inf);
	res[48] = (cunan >= -inf);
	res[49] = (cunan <  -inf);
	res[50] = (cunan <= -inf);

	res[51] = (cunan == 1.0);
	res[52] = (cunan != 1.0);
	res[53] = (cunan >  1.0);
	res[54] = (cunan >= 1.0);
	res[55] = (cunan <  1.0);
	res[56] = (cunan <= 1.0);

	// nan ops
	res[57] = (cunan / 1.0);
	res[58] = (1.0 / cunan);
	res[59] = (0.0 / 0.0  );
	res[60] = (0.0 / nzero);
	res[61] = (nzero / 0.0);
	res[62] = (nzero/nzero);
	res[63] = ( inf /  inf);
	res[64] = ( inf / -inf);
	res[65] = (-inf /  inf);
	res[66] = (-inf / -inf);
	res[67] = ( 0.0 /  1.0);
	res[68] = ( 0.0 / -1.0);
	res[69] = ( 0.0 /  inf);
	res[70] = ( 0.0 / -inf);
	res[71] = (nzero/  1.0);
	res[72] = (nzero/ -1.0);
	res[73] = (nzero/  inf);
	res[74] = (nzero/ -inf);

	res[75] = ( 1.0 /  0.0);
	res[76] = (-1.0 /  0.0);
	res[77] = ( inf /  0.0);
	res[78] = (-inf /  0.0);
	res[79] = ( 1.0 /nzero);
	res[80] = (-1.0 /nzero);
	res[81] = ( inf /nzero);
	res[82] = (-inf /nzero);


	for (int i = 0; i < N; ++i) {
		c[i] = res[i];
	}
}

/*
int main2()
{
	GDDStart(0);
	GQDStart(0);
	std::cout << "cuda GQD sample" << std::endl;

    const int arraySize = 5;
	const gdd_real a[arraySize] = { gdd_real(1.0, 0.0), gdd_real(2.0, 0.0), gdd_real(3.0, 0.0), gdd_real(4.0, 0.0), gdd_real(5.0, 0.0) };
	const gdd_real b[arraySize] = { gdd_real(10.0, 0.0), gdd_real(20.0, 0.0), gdd_real(30.0, 0.0), gdd_real(40.0, 0.0), gdd_real(50.0, 0.0) };
	gdd_real c[arraySize] = { gdd_real(0.0, 0.0), gdd_real(0.0, 0.0), gdd_real(0.0, 0.0), gdd_real(0.0, 0.0), gdd_real(0.0, 0.0), };

	//const gqd_real q_a[arraySize] = { make_qd(1), make_qd(2), make_qd(3), make_qd(4), make_qd(5) };
	//const gqd_real q_b[arraySize] = { make_qd(10), make_qd(20), make_qd(30), make_qd(40), make_qd(50) };
	//gqd_real q_c[arraySize] = { make_qd(0.0, 0.0,0.0, 0.0) };

	gdd_real temp = gdd_real(3.14, 0.0);

	//gqd_real temp2 = make_qd(3.14);
	//gqd_real temp2;

	//================================================================
	
	union aa{
		float f;
		unsigned int i;
	};
	union dd{
		double d;
		__int64 L64;
	};
	double inn = std::numeric_limits<double>::infinity();
	std::cout << " inf_host double = ";
	printf("%0I64x\n", inn);


	double nn = std::numeric_limits<double>::quiet_NaN();	//0x7ff8000 00000000
	std::cout << "qnan_host double = ";
	printf("%0I64x\n", nn);

	double ss = std::numeric_limits<double>::signaling_NaN();	//0x7ff0000 00000001
	std::cout << "snan_host double = ";
	printf("%0I64x\n\n", ss);

	dd inf_d;
	inf_d.d = std::numeric_limits<double>::max() * 10.0;
	std::cout << " inf_host double = ";
	printf("%0I64x\n", inf_d);

	dd qnan_d;
	//qnan_d.d = -(inf_d.d) * 0.0;// inf_d.d;
	qnan_d.d = (inf_d.d) - inf_d.d;
	std::cout << "qnan_d (inf - inf) = ";
	printf("%0I64x\n", qnan_d);

	qnan_d.d = (inf_d.d) / inf_d.d;
	std::cout << "qnan_d (inf / inf) = ";
	printf("%0I64x\n", qnan_d);

	qnan_d.d = (inf_d.d) * 0.0;
	std::cout << "qnan_d (inf * 0.0) = ";
	printf("%0I64x\n", qnan_d);


	aa nnf;
	nnf.f = std::numeric_limits<float>::quiet_NaN();	//0x7fc00000
	//std::cout << nnf << std::endl;
	std::cout << "qnan_f = ";
	printf("%0I32x\n", nnf.i);

	aa ssf;
	ssf.f = std::numeric_limits<float>::signaling_NaN();//0x7f800001
	std::cout << "snan_f = ";
	printf("%0I32x\n", ssf.i);

	//================================================================

	std::cout << "size of gdd = " << sizeof(gdd_real) << "bytes" << std::endl;
	std::cout << "size of gqd = " << sizeof(gqd_real) << "bytes" << std::endl;
	cudaError_t cudaStatus;

    // Add vectors in parallel.
    cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	double *f = (double*)c;
	
	for (int i = 0; i < arraySize / 2; ++i){
		std::cout << *f<<", ";
		f++;
		std::cout << *f;
		f++;
		std::cout << std::endl;
	}
	std::cout << "-----------------------------" << std::endl;

	for (int i = 0; i < arraySize ; ++i){
		//std::cout << c[i].dd.x << ", ";
		//std::cout << c[i].dd.y;
		std::cout << std::endl;
	}

	

	//cudaStatus = MemoryBandwidthCheck();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, " MemoryBandwidthCheck() failed!");
	//	return 1;
	//}

	// gdd_real^N check
	//gdd_real aa(2.0),bb(0.5), cc;
	//aa = aa^100;
	//std::cout << "2^100 = 1267650600228229401496703205376" << std::endl;
	//std::cout << "2^100 = "<< aa.dd.x << std::endl;

	//dd ref, cmp;
	//ref.L64 = 0x7ff0000000000000ULL;	// CUDART_INF;
	//cmp.d = inf();
	//std::cout << "Inf cudart = " << ref.L64 << std::endl;
	//std::cout << "Inf func   = " << cmp.L64 << std::endl;

	//ref.L64 = 0xfff8000000000000ULL;	// CUDART_NAN;
	//cmp.d = qnan();
	//std::cout << "NaN cudart = " << ref.L64 << std::endl;
	//std::cout << "NaN func   = " << cmp.L64 << std::endl;


	//cudaStatus = addWithCudaq(q_c, q_a, q_b, arraySize);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "addWithCuda failed!");
	//	return 1;
	//}

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        //c[0], c[1], c[2], c[3], c[4]);

//	std::cout << " c0=" << c[0] << " c1=" << c[1] << " c2=" << c[2] << " c3=" << c[3] << " c4=" << c[4] << std::endl;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	GDDEnd();
	GQDEnd();
    return 0;
}
*/

// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(gdd_real *c, const gdd_real *a, const gdd_real *b, unsigned int size)
cudaError_t k_addWithCuda()
{
	gdd_real *dev_a = 0;
	gdd_real *dev_b = 0;
	gdd_real *dev_c = 0;
    cudaError_t cudaStatus;
	
	const int arraySize = 5;
	const gdd_real a[arraySize] = { gdd_real(1.0, 0.0), gdd_real(2.0, 0.0), gdd_real(3.0, 0.0), gdd_real(4.0, 0.0), gdd_real(5.0, 0.0) };
	const gdd_real b[arraySize] = { gdd_real(10.0, 0.0), gdd_real(20.0, 0.0), gdd_real(30.0, 0.0), gdd_real(40.0, 0.0), gdd_real(50.0, 0.0) };
	gdd_real c[arraySize] = { gdd_real(0.0, 0.0), gdd_real(0.0, 0.0), gdd_real(0.0, 0.0), gdd_real(0.0, 0.0), gdd_real(0.0, 0.0), };

	int size = arraySize;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(gdd_real));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(gdd_real));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(gdd_real));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(gdd_real), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, 1>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(gdd_real), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	double *f = (double*)c;

	for (int i = 0; i < arraySize / 2; ++i) {
		std::cout << *f << ", ";
		f++;
		std::cout << *f;
		f++;
		std::cout << std::endl;
	}
	std::cout << "-----------------------------" << std::endl;

	for (int i = 0; i < arraySize; ++i) {
		//std::cout << c[i].dd.x << ", ";
		//std::cout << c[i].dd.y;
		std::cout << std::endl;
	}

	cudaStatus = cudaSuccess;

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}



union uni_cuda{
	double d;
	__int64 L64;
};

std::string checkCudaValue(uni_cuda a){
#define CUDA_INF		(0x7ff0000000000000ULL)
#define CUDA_NAN		(0xfff8000000000000ULL)
#define CUDA_PNAN		(0x7ff8000000000000ULL)
#define CUDA_NEG_ZERO	(0x8000000000000000ULL)

	uni_cuda cuda_nan, cuda_pnan, cuda_pinf, cuda_ninf, cuda_nzero;
	cuda_nan.L64 = CUDA_NAN;
	cuda_pnan.L64 = CUDA_PNAN;
	cuda_pinf.L64 = CUDA_INF;
	cuda_ninf.d = -(cuda_pinf.d);
	cuda_nzero.L64 = CUDA_NEG_ZERO;

	if (a.L64 == cuda_nan.L64){
		if (isnan(a.d)){
			return std::string(" ( CUDART_NAN) isNaN");
		}else{
			return std::string(" ( CUDART_NAN)");
		}
	}
	if (a.L64 == cuda_pnan.L64){
		if (isnan(a.d)){
			return std::string(" (+CUDART_NAN) isNaN");
		}else{
			return std::string(" (+CUDART_NAN)");
		}
	}
	if (a.L64 == 0x3ff0000000000000) { return " (1.0)"; }
	if (a.L64 == cuda_pinf.L64){ return std::string(" (CUDART_Pinf)"); }
	if (a.L64 == cuda_ninf.L64){ return std::string(" (CUDART_Ninf)"); }
	if (a.L64 == cuda_nzero.L64){ return std::string(" (CUDART_NZERO)"); }
	return std::string("");
}



//cudaError_t addWithCudaq(gqd_real *c, const gqd_real *a, const gqd_real *b, unsigned int size)
cudaError_t k_addWithCudaq()
{
	gqd_real *dev_a = 0;
	gqd_real *dev_b = 0;
	gqd_real *dev_c = 0;
	cudaError_t cudaStatus;
	
	const int arraySize = 5;
	const gdd_real a[arraySize] = { gdd_real(1.0, 0.0), gdd_real(2.0, 0.0), gdd_real(3.0, 0.0), gdd_real(4.0, 0.0), gdd_real(5.0, 0.0) };
	const gdd_real b[arraySize] = { gdd_real(10.0, 0.0), gdd_real(20.0, 0.0), gdd_real(30.0, 0.0), gdd_real(40.0, 0.0), gdd_real(50.0, 0.0) };
	gdd_real c[arraySize] = { gdd_real(0.0, 0.0), gdd_real(0.0, 0.0), gdd_real(0.0, 0.0), gdd_real(0.0, 0.0), gdd_real(0.0, 0.0), };

	int size = arraySize;


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(gqd_real));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(gqd_real));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(gqd_real));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(gqd_real), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernelq<<<1, size >>>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(gqd_real), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaSuccess;

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t k_IEEE_check()
{
	const int elements = 100;
	uni_cuda nan_res[elements];
	double *dev_nan_res;
	int dev_mem_size = sizeof(double)*elements;

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	checkCudaErrors(cudaMalloc((void**)&dev_nan_res, dev_mem_size));

	checkCudaErrors(cudaMemset(dev_nan_res, '\0', dev_mem_size));

	// Launch a kernel on the GPU with one thread for each element.
	NanOperation<<<1, 1, dev_mem_size >>>(dev_nan_res, elements);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "NanOperation launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	checkCudaErrors(cudaDeviceSynchronize());

	// Copy output vector from GPU buffer to host memory.
	checkCudaErrors(cudaMemcpy(nan_res, dev_nan_res, dev_mem_size, cudaMemcpyDeviceToHost));
	
	std::cout << std::hex;
	std::cout << "|expres|expect|result  ----------------------------------| " << std::endl;
	std::cout << "0^0   = [ 1.0] " << nan_res[0].d << checkCudaValue(nan_res[0]) << std::endl;
	std::cout << "1^0   = [ 1.0] " << nan_res[1].d << checkCudaValue(nan_res[1]) << std::endl;
	std::cout << "NaN^0 = [ 1.0] " << nan_res[2].d << checkCudaValue(nan_res[2]) << " (but should be NaN)" << std::endl;
	std::cout << "Inf^0 = [ 1.0] " << nan_res[3].d << checkCudaValue(nan_res[3]) << std::endl;
	std::cout << std::endl;
	std::cout << "0^1   = [ 0.0] " << nan_res[4].d << checkCudaValue(nan_res[4]) << std::endl;
	std::cout << "1^1   = [ 1.0] " << nan_res[5].d << checkCudaValue(nan_res[5]) << std::endl;
	std::cout << "NaN^1 = [ NaN] " << nan_res[6].L64 << checkCudaValue(nan_res[6]) << std::endl;
	std::cout << "Inf^1 = [ inf] " << nan_res[7].L64 << checkCudaValue(nan_res[7]) << std::endl;
	std::cout << std::endl;
	std::cout << "0^Inf   = [ 0.0] " << nan_res[8].d << checkCudaValue(nan_res[8]) << std::endl;
	std::cout << "1^Inf   = [ 1.0] " << nan_res[9].d << checkCudaValue(nan_res[9]) << std::endl;
	std::cout << "2^Inf   = [ inf] " << nan_res[10].L64 << checkCudaValue(nan_res[10]) << std::endl;
	std::cout << "NaN^Inf = [ NaN] " << nan_res[11].L64 << checkCudaValue(nan_res[11]) << std::endl;
	std::cout << "Inf^Inf = [ inf] " << nan_res[12].L64 << checkCudaValue(nan_res[12]) << std::endl;
	std::cout << std::endl;

	std::cout << "0^NaN     = [ NaN] " << nan_res[13].L64 << checkCudaValue(nan_res[13]) << std::endl;
	std::cout << "1^NaN     = [ 1.0] " << nan_res[14].d << checkCudaValue(nan_res[14]) << std::endl;
	std::cout << "Inf^NaN   = [ NaN] " << nan_res[15].L64 << checkCudaValue(nan_res[15]) << std::endl;
	std::cout << "NaN^NaN   = [ NaN] " << nan_res[16].L64 << checkCudaValue(nan_res[16]) << std::endl;
	std::cout << std::endl;

	std::cout << "exp( nan) = [ NaN] " << nan_res[17].L64 << checkCudaValue(nan_res[17]) << std::endl;
	std::cout << "exp( inf) = [ inf] " << nan_res[18].L64 << checkCudaValue(nan_res[18]) << std::endl;
	std::cout << "exp(-inf) = [ 0.0] " << nan_res[19].L64 << checkCudaValue(nan_res[19]) << std::endl;
	std::cout << std::endl;


	std::cout << "Log spec ---------------------" << std::endl;
	std::cout << "log(+0.0)  = [-inf] " << nan_res[20].L64 << checkCudaValue(nan_res[20]) << std::endl;
	std::cout << "log(-0.0)  = [-inf] " << nan_res[21].L64 << checkCudaValue(nan_res[21]) << std::endl;
	std::cout << "log( 1.0)  = [+0.0] " << nan_res[22].L64 << checkCudaValue(nan_res[22]) << std::endl;
	std::cout << "log(-1.0)  = [ NaN] " << nan_res[23].L64 << checkCudaValue(nan_res[23]) << std::endl;
	std::cout << "log( inf)  = [ inf] " << nan_res[24].L64 << checkCudaValue(nan_res[24]) << std::endl;
	std::cout << "log(-inf)  = [ NaN] " << nan_res[25].L64 << checkCudaValue(nan_res[25]) << std::endl;
	std::cout << "log( NaN)  = [ NaN] " << nan_res[26].L64 << checkCudaValue(nan_res[26]) << std::endl;
	std::cout << std::endl;

	std::cout << "SQRT spec ---------------------" << std::endl;
	std::cout << "sqrt(+0.0) = [+0.0] " << nan_res[27].L64 << checkCudaValue(nan_res[27]) << std::endl;
	std::cout << "sqrt(-0.0) = [-0.0] " << nan_res[28].L64 << checkCudaValue(nan_res[28]) << std::endl;
	std::cout << "sqrt( inf) = [ inf] " << nan_res[29].L64 << checkCudaValue(nan_res[29]) << std::endl;
	std::cout << "sqrt(-1.0) = [ NaN] " << nan_res[30].L64 << checkCudaValue(nan_res[30]) << std::endl;
	std::cout << "sqrt(-inf) = [ NaN] " << nan_res[31].L64 << checkCudaValue(nan_res[31]) << std::endl;
	std::cout << "sqrt( NaN) = [ NaN] " << nan_res[32].L64 << checkCudaValue(nan_res[32]) << std::endl;
	std::cout << std::endl;

	std::cout << "Nan comparition ---------------" << std::endl;
	std::cout << "nan == nan = [   0] " << nan_res[33].L64 << checkCudaValue(nan_res[33]) << std::endl;
	std::cout << "nan != nan = [   1] " << nan_res[34].L64 << checkCudaValue(nan_res[34]) << std::endl;
	std::cout << "nan >  nan = [   0] " << nan_res[35].L64 << checkCudaValue(nan_res[35]) << std::endl;
	std::cout << "nan >= nan = [   0] " << nan_res[36].L64 << checkCudaValue(nan_res[36]) << std::endl;
	std::cout << "nan <  nan = [   0] " << nan_res[37].L64 << checkCudaValue(nan_res[37]) << std::endl;
	std::cout << "nan <= nan = [   0] " << nan_res[38].L64 << checkCudaValue(nan_res[38]) << std::endl;
	std::cout << std::endl;

	std::cout << "nan == inf = [   0] " << nan_res[39].L64 << checkCudaValue(nan_res[39]) << std::endl;
	std::cout << "nan != inf = [   1] " << nan_res[40].L64 << checkCudaValue(nan_res[40]) << std::endl;
	std::cout << "nan >  inf = [   0] " << nan_res[41].L64 << checkCudaValue(nan_res[41]) << std::endl;
	std::cout << "nan >= inf = [   0] " << nan_res[42].L64 << checkCudaValue(nan_res[42]) << std::endl;
	std::cout << "nan <  inf = [   0] " << nan_res[43].L64 << checkCudaValue(nan_res[43]) << std::endl;
	std::cout << "nan <= inf = [   0] " << nan_res[44].L64 << checkCudaValue(nan_res[44]) << std::endl;
	std::cout << std::endl;

	std::cout << "nan == -inf = [   0] " << nan_res[45].L64 << checkCudaValue(nan_res[45]) << std::endl;
	std::cout << "nan != -inf = [   1] " << nan_res[46].L64 << checkCudaValue(nan_res[46]) << std::endl;
	std::cout << "nan >  -inf = [   0] " << nan_res[47].L64 << checkCudaValue(nan_res[47]) << std::endl;
	std::cout << "nan >= -inf = [   0] " << nan_res[48].L64 << checkCudaValue(nan_res[48]) << std::endl;
	std::cout << "nan <  -inf = [   0] " << nan_res[49].L64 << checkCudaValue(nan_res[49]) << std::endl;
	std::cout << "nan <= -inf = [   0] " << nan_res[50].L64 << checkCudaValue(nan_res[50]) << std::endl;
	std::cout << std::endl;

	std::cout << "nan == 1.0 = [   0] " << nan_res[51].L64 << checkCudaValue(nan_res[51]) << std::endl;
	std::cout << "nan != 1.0 = [   1] " << nan_res[52].L64 << checkCudaValue(nan_res[52]) << std::endl;
	std::cout << "nan >  1.0 = [   0] " << nan_res[53].L64 << checkCudaValue(nan_res[53]) << std::endl;
	std::cout << "nan >= 1.0 = [   0] " << nan_res[54].L64 << checkCudaValue(nan_res[54]) << std::endl;
	std::cout << "nan <  1.0 = [   0] " << nan_res[55].L64 << checkCudaValue(nan_res[55]) << std::endl;
	std::cout << "nan <= 1.0 = [   0] " << nan_res[56].L64 << checkCudaValue(nan_res[56]) << std::endl;
	std::cout << std::endl;

	std::cout << "Nan ops ---------------" << std::endl;
	std::cout << "cunan / 1.0 = [ NaN] " << nan_res[57].L64 << checkCudaValue(nan_res[57]) << std::endl;
	std::cout << "1.0 / cunan = [ NaN] " << nan_res[58].L64 << checkCudaValue(nan_res[58]) << std::endl;
	std::cout << "0.0 / 0.0   = [ NaN] " << nan_res[59].L64 << checkCudaValue(nan_res[59]) << std::endl;
	std::cout << "0.0 / nzero = [?NaN] " << nan_res[60].L64 << checkCudaValue(nan_res[60]) << std::endl;
	std::cout << "nzero / 0.0 = [?NaN] " << nan_res[61].L64 << checkCudaValue(nan_res[61]) << std::endl;
	std::cout << "nzero/nzero = [ NaN] " << nan_res[62].L64 << checkCudaValue(nan_res[62]) << std::endl;
	std::cout << " inf /  inf = [ NaN] " << nan_res[63].L64 << checkCudaValue(nan_res[63]) << std::endl;
	std::cout << " inf / -inf = [ NaN] " << nan_res[64].L64 << checkCudaValue(nan_res[64]) << std::endl;
	std::cout << "-inf /  inf = [ NaN] " << nan_res[65].L64 << checkCudaValue(nan_res[65]) << std::endl;
	std::cout << "-inf / -inf = [ NaN] " << nan_res[66].L64 << checkCudaValue(nan_res[66]) << std::endl;
	std::cout << " 0.0 /  1.0 = [ 0.0] " << nan_res[67].L64 << checkCudaValue(nan_res[67]) << std::endl;
	std::cout << " 0.0 / -1.0 = [-0.0] " << nan_res[68].L64 << checkCudaValue(nan_res[68]) << std::endl;
	std::cout << " 0.0 /  inf = [ 0.0] " << nan_res[69].L64 << checkCudaValue(nan_res[69]) << std::endl;
	std::cout << " 0.0 / -inf = [-0.0] " << nan_res[70].L64 << checkCudaValue(nan_res[70]) << std::endl;
	std::cout << "nzero/  1.0 = [-0.0] " << nan_res[71].L64 << checkCudaValue(nan_res[71]) << std::endl;
	std::cout << "nzero/ -1.0 = [ 0.0] " << nan_res[72].L64 << checkCudaValue(nan_res[72]) << std::endl;
	std::cout << "nzero/  inf = [-0.0] " << nan_res[73].L64 << checkCudaValue(nan_res[73]) << std::endl;
	std::cout << "nzero/ -inf = [ 0.0] " << nan_res[74].L64 << checkCudaValue(nan_res[74]) << std::endl;

	std::cout << " 1.0 /  0.0 = [ inf] " << nan_res[75].L64 << checkCudaValue(nan_res[75]) << std::endl;
	std::cout << "-1.0 /  0.0 = [-inf] " << nan_res[76].L64 << checkCudaValue(nan_res[76]) << std::endl;
	std::cout << " inf /  0.0 = [ inf] " << nan_res[77].L64 << checkCudaValue(nan_res[77]) << std::endl;
	std::cout << "-inf /  0.0 = [-inf] " << nan_res[78].L64 << checkCudaValue(nan_res[78]) << std::endl;
	std::cout << " 1.0 /nzero = [-inf] " << nan_res[79].L64 << checkCudaValue(nan_res[79]) << std::endl;
	std::cout << "-1.0 /nzero = [ inf] " << nan_res[80].L64 << checkCudaValue(nan_res[80]) << std::endl;
	std::cout << " inf /nzero = [-inf] " << nan_res[81].L64 << checkCudaValue(nan_res[81]) << std::endl;
	std::cout << "-inf /nzero = [ inf] " << nan_res[82].L64 << checkCudaValue(nan_res[82]) << std::endl;

	std::cout << std::endl;

	cudaStatus = cudaSuccess;

Error:
	cudaFree(dev_nan_res);

	return cudaStatus;
}


//cudaError_t addWithCuda(gdd_real *c, const gdd_real *a, const gdd_real *b, unsigned int size);
//cudaError_t addWithCudaq(gqd_real *c, const gqd_real *a, const gqd_real *b, unsigned int size);
//cudaError_t IEEE_check();

void addWithCudaq() {
	k_addWithCudaq();
}

void addWithCuda() {
	k_addWithCuda();
}

void IEEE_check() {
	k_IEEE_check();
}


void checkInline() {
	k_checkInline();
}


void Initialize(int n) {
	GDDStart(n);
	GQDStart(n);
}

void Finalize(int n) {
	int current;
	cudaGetDevice(&current);
	if (current != n) {
		cudaSetDevice(n);
	}
	
	GDDEnd();
	GQDEnd();
}


double testdata[20] = {
	
	0xfff8000000000000ULL,	// qnan 
	0x7ff8000000000000ULL,	// -qnan
	0x7ff0000000000000ULL,	// inf
	0xfff0000000000000ULL,	// -inf
	0x3fefffffffffffffULL,	// max
	0xbfefffffffffffffULL,	// -max
	0x0010000000000000ULL,	// min_norm
	0x8010000000000000ULL,	// -min_norm
	0x0001111111111111ULL,	// subnormal_max
	0x0000000000000001ULL,	// subnormal_min
	0x8001111111111111ULL,	// -subnormal_max
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

