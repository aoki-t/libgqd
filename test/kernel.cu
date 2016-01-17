//#include <cuda.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

////#include "cuda_device_runtime_api.h"
////#include "device_double_functions.h"
////#include "device_functions.h"

//#pragma comment(lib,"D:\\libgqd_class2\\bin\\x64_Release\\libgqd.lib")
#include <float.h>
#include <stdio.h>
#include <iostream>
//#include <math.h>

//#include <immintrin.h>
//#include <amp_math.h>

//#include <gqd_type.h>
//#include <gdd_basic.h>
//#include <gqd_basic.h>
//#include <gqd_function.h>

//#ifdef __cplusplus
//extern "C" {
//#endif

#include "../libgqd/inc/cuda_header.h"
#include "../libgqd/inc/gqd_type.h"
#include "../libgqd/inc/common.cu"
#include "../libgqd/inc/gdd_real.h"
#include "../libgqd/inc/gqd.cu"
#include "../libgqd/inc/gdd_log.cu"
#include "../libgqd/inc/gdd_exp.cu"
#include "../libgqd/inc/gdd_basic.cu"
#include "../libgqd/inc/gdd_sincos.cu"
//#include "../libgqd/inc/gqd_basic.h"
//#include "../libgqd/inc/gqd_function.h"

//#include "../libgqd/libgqd/inc/gdd_common.cu"
//#include "../libgqd/libgqd/inc/gqd_common.cu"

//#ifdef __cplusplus
//}
//#endif

/*
リンクエラー
operator+(double2 const&, double2 const&)
operator*(double2 const&, double2 const&)
operator-(double2 const&, double2 const&)
operator/(double2 const&, double2 const&)

*/

//cudaError_t addWithCuda(gdd_real c[], const gdd_real a[], const gdd_real b[], unsigned int size);

cudaError_t addWithCuda(gdd_real *c, const gdd_real *a, const gdd_real *b, unsigned int size);
cudaError_t addWithCudaq(gqd_real *c, const gqd_real *a, const gqd_real *b, unsigned int size);
cudaError_t IEEE_check();

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

	printf("[%d, %d]:\t\tValue is:%llf\n", \
		blockIdx.y*gridDim.x + blockIdx.x, \
		threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x, \
		c[i].dd.x);

	gdd_real ze = gdd_real(CUDART_NEG_ZERO, CUDART_NEG_ZERO);
	gdd_real na = gdd_real(CUDART_NAN, CUDART_NAN);
	bool ip = is_positive(ze);
	bool in = is_negative(ze);
	bool iz = is_zero(ze);
	bool iq = isnan(na);

	gdd_real bb(0.5), cc;
	printf("is_positive is:%d\n", ip);
	printf("is_negative is:%d\n", in);
	printf("is_zero     is:%d\n", iz);
	printf("is_nan      is:%d\n", iq);

	cc = npwr(bb, 2);
	printf("0.5^2 = %f\n", cc.dd.x);
	cc = npwr(bb, 1);
	printf("0.5^1 = %f\n", cc.dd.x);

	cc = npwr(bb, -1);
	printf("0.5^-1 = %f\n", cc.dd.x);
	cc = npwr(bb, -2);
	printf("0.5^-2 = %f\n", cc.dd.x);

	cc = pow(bb, bb);
	printf("0.5^0.5 = %f, %f\n", cc.dd.x, cc.dd.y);


}
__global__ void addKernelq(gqd_real *c, const gqd_real *a, const gqd_real *b)
{
    //int i = threadIdx.x;
    //c[i] = a[i] + b[i];
	//c[i] = c[i] * b[i];
	//c[i] = c[i] - b[i];
	//c[i] = c[i] / b[i];

}

__global__ void NanOperation(double *c)
{
	const double inf = CUDART_INF;
	const double cunan = CUDART_NAN;
	const double nzero = CUDART_NEG_ZERO;
	const double zero = 0.0;
	const double one = 1.0;
	const int N = 40;
	__shared__ double res[N];


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

	for (int i = 0; i < N; ++i){
		c[i] = res[i];
	}
}


int main()
{
	GDDStart(0);

	std::cout << "cuda GQD sample" << std::endl;

    const int arraySize = 5;
	const gdd_real a[arraySize] = { gdd_real(1.0, 0.0), gdd_real(2.0, 0.0), gdd_real(3.0, 0.0), gdd_real(4.0, 0.0), gdd_real(5.0, 0.0) };
	const gdd_real b[arraySize] = { gdd_real(10.0, 0.0), gdd_real(20.0, 0.0), gdd_real(30.0, 0.0), gdd_real(40.0, 0.0), gdd_real(50.0, 0.0) };
	gdd_real c[arraySize] = { gdd_real(0.0, 0.0) ,0, 0, 0, 0,};

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
	std::cout << " inf_d = ";
	printf("%0I64x\n", inn);


	double nn = std::numeric_limits<double>::quiet_NaN();	//0x7ff8000 00000000
	std::cout << "qnan_d = ";
	printf("%0I64x\n", nn);

	double ss = std::numeric_limits<double>::signaling_NaN();	//0x7ff0000 00000001
	std::cout << "snan_d = ";
	printf("%0I64x\n\n", ss);

	dd inf_d;
	inf_d.d = std::numeric_limits<double>::max() * 10.0;
	std::cout << " inf_d = ";
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
		std::cout << c[i].dd.x << ", ";
		std::cout << c[i].dd.y;
		std::cout << std::endl;
	}

	
	cudaStatus = IEEE_check();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "IEEE_check failed!");
		return 1;
	}

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

    return 0;
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(gdd_real *c, const gdd_real *a, const gdd_real *b, unsigned int size)
{
	gdd_real *dev_a = 0;
	gdd_real *dev_b = 0;
	gdd_real *dev_c = 0;
    cudaError_t cudaStatus;

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
	if (a.L64 == cuda_pinf.L64){ return std::string(" (CUDART_Pinf)"); }
	if (a.L64 == cuda_ninf.L64){ return std::string(" (CUDART_Ninf)"); }
	if (a.L64 == cuda_nzero.L64){ return std::string(" (CUDART_NZERO)"); }
	return std::string("");
}

cudaError_t addWithCudaq(gqd_real *c, const gqd_real *a, const gqd_real *b, unsigned int size)
{
	gqd_real *dev_a = 0;
	gqd_real *dev_b = 0;
	gqd_real *dev_c = 0;
	cudaError_t cudaStatus;

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

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t IEEE_check()
{
	const int n_nan_elements = 40;
	uni_cuda nan_res[n_nan_elements];
	double *dev_nan_res;
	int dev_mem_size = sizeof(double)*n_nan_elements;

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_nan_res, dev_mem_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaMemset(dev_nan_res, '\0', dev_mem_size);

	// Launch a kernel on the GPU with one thread for each element.
	NanOperation<<<1, 1 >>>(dev_nan_res);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "NanOperation launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
	cudaStatus = cudaMemcpy(nan_res, dev_nan_res, dev_mem_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	std::cout << std::hex;
	std::cout << "0^0   = [ 1.0] " << nan_res[0].d << checkCudaValue(nan_res[0]) << std::endl;
	std::cout << "1^0   = [ 1.0] " << nan_res[1].d << checkCudaValue(nan_res[1]) << std::endl;
	std::cout << "NaN^0 = [ 1.0] " << nan_res[2].d << checkCudaValue(nan_res[2]) << std::endl;
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


	cudaFree(dev_nan_res);
	return cudaStatus;

Error:
	cudaFree(dev_nan_res);

	return cudaStatus;
}


/*

__forceinline__ __host__ __device__
double two_sqr_fma(double a, double &err) {
	double p = a * a;
	err = fma(a, a, -p);
	return p;

}

__host__
double h_two_sqr_fma(double a, double &err) {
	double p = a * a;
	err = Concurrency::precise_math::fma(a, a, -p);
	return p;

}


__forceinline__ __host__ __device__
double two_sqr_nofma(double a, double &err) {
	double hi, lo;
	double q = a * a;
	split(a, hi, lo);
	err = ((hi * hi - q) + 2.0 * hi * lo) + lo * lo;
	return q;
}

double h_two_sqr_nofma(double a, double &err) {
	double hi, lo;
	double q = a * a;
	split(a, hi, lo);
	err = ((hi * hi - q) + 2.0 * hi * lo) + lo * lo;
	return q;
}


__global__
void test_fma_diff_kernel(double *in_val, double *fma_res, double *fma_err, double *no_fma_res, double *no_fma_err){
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	fma_res[i] = two_sqr_fma(in_val[i], fma_err[i]);
	no_fma_res[i] = two_sqr_nofma(in_val[i], no_fma_err[i]);

}

void test_fma_diff(){
#define elements 100
	double in[elements];
	double fma_res[elements];
	double fma_err[elements];
	double no_fma_res[elements];
	double no_fma_err[elements];
	double host_fma_res[elements];
	double host_fma_err[elements];
	double host_no_fma_res[elements];
	double host_no_fma_err[elements];

	double *dev_in;
	double *dev_fma_res;
	double *dev_fma_err;
	double *dev_no_fma_res;
	double *dev_no_fma_err;

	cudaMalloc(&dev_in, sizeof(double)*elements);
	cudaMalloc(&dev_fma_res, sizeof(double)*elements);
	cudaMalloc(&dev_fma_err, sizeof(double)*elements);
	cudaMalloc(&dev_no_fma_res, sizeof(double)*elements);
	cudaMalloc(&dev_no_fma_err, sizeof(double)*elements);
	dim3 gridsize = 1;
	dim3 blocksize = elements;

	for (int i = 0; i < elements; i++){
		in[i] = 1.0 - (1 / std::pow(10, i));
	}

	cudaMemcpy(&dev_in, in, sizeof(double)*elements, cudaMemcpyHostToDevice);
	cudaMemset(&dev_fma_res, 0, sizeof(double)*elements);
	cudaMemset(&dev_fma_err, 0, sizeof(double)*elements);
	cudaMemset(&dev_no_fma_res, 0, sizeof(double)*elements);
	cudaMemset(&dev_no_fma_err, 0, sizeof(double)*elements);

	test_fma_diff_kernel <<<gridsize, blocksize>>>(in, fma_res, fma_err, no_fma_res, no_fma_err);

	cudaMemcpy(&fma_res, dev_fma_res, sizeof(double)*elements, cudaMemcpyDeviceToHost);
	cudaMemcpy(&fma_err, dev_fma_err, sizeof(double)*elements, cudaMemcpyDeviceToHost);
	cudaMemcpy(&no_fma_res, dev_no_fma_res, sizeof(double)*elements, cudaMemcpyDeviceToHost);
	cudaMemcpy(&no_fma_err, dev_no_fma_err, sizeof(double)*elements, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();


	for (int i = 0; i < elements; i++){
		host_fma_res[i] = h_two_sqr_fma(in[i], host_fma_err[i]);
		host_no_fma_res[i] = h_two_sqr_nofma(in[i], host_no_fma_err[i]);

		gdd_real h_fma, h_nofma, d_fma, d_nofma;
		//host fma<->nofma
		//double diff_h2h_res = host_fma_res - host_no_fma_res;
		//double diff_h2h_err = host_fma_err - host_no_fma_err;

		h_fma = gdd_real(host_fma_res[i], host_fma_err[i]);
		h_nofma = gdd_real(host_no_fma_res[i], host_no_fma_err[i]);


		//fma host<->dev
		//double diff_d2h_res = dev_fma_res - host_fma_res;
		//double diff_d2h_err = dev_fma_err - host_fma_err;
		d_fma = gdd_real(dev_fma_res[i], dev_fma_err[i]);

		//nofma host<->dev
		//double diff_d2h_no_res = dev_no_fma_res - host_no_fma_res;
		//double diff_d2h_no_err = dev_no_fma_err - host_no_fma_err;
		d_nofma = gdd_real(dev_no_fma_res[i], dev_no_fma_err[i]);

		if (d_fma != d_nofma){
			std::cout << "in val            " << in[i] << std::endl;
			std::cout << "host fma<->nofma  " << (h_fma - h_nofma) << std::endl;
			std::cout << "  fma  dev<->host " << (d_fma - h_fma) << std::endl;
			std::cout << "nofma  dev<->host " << (d_nofma - h_nofma) << std::endl;
			std::cout << std::endl;
		}
	}

	cudaFree(dev_in);
	cudaFree(dev_fma_res);
	cudaFree(dev_fma_err);
	cudaFree(dev_no_fma_res);
	cudaFree(dev_no_fma_err);

}
*/

