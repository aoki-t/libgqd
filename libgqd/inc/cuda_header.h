
#ifndef _CUDA_HEADER_CU_
#define _CUDA_HEADER_CU_

#include <stdio.h>
#include <stdlib.h>
//#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <device_functions.h>
#include <math_constants.h>
#include <device_double_functions.h>	//CUDART_INF


#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>

//#if __CUDA_ARCH__ >= 200
////#include <sm_20_atomic_functions.h>
////#include <sm_20_intrinsics.h>
//#endif
//#if __CUDA_ARCH__ >= 300
////#include <sm_30_intrinsics.h>
//#endif
//#if __CUDA_ARCH__ >= 320
////#include <sm_32_atomic_functions.h>
////#include <sm_32_intrinsics.h>
//#endif
//#if __CUDA_ARCH__ >= 350
////#include <sm_35_atomic_functions.h>
////#include <sm_35_intrinsics.h>
//#endif
#if (defined(CUDART_VERSION)) &&  CUDART_VERSION < 5000
#error CUDA Tookkit 5.0 or later needed.
#endif
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 200)
#error Conpute Capability 2.0 or later needed
#endif


/** for CUDA 2.0 */
#ifdef CUDA_2
#define cutilCheckMsg CUT_CHECK_ERROR
#define cutilSafeCall CUDA_SAFE_CALL
#endif


/* kernel macros */
#define NUM_TOTAL_THREAD (gridDim.x*blockDim.x)
#define GLOBAL_THREAD_OFFSET (blockDim.x*blockIdx.x + threadIdx.x)

/** macro utility */
#define GPUMALLOC(D_DATA, MEM_SIZE)         checkCudaErrors( cudaMalloc(D_DATA, MEM_SIZE) )
#define TOGPU(D_DATA, H_DATA, MEM_SIZE)     checkCudaErrors( cudaMemcpy(D_DATA, H_DATA, MEM_SIZE, cudaMemcpyHostToDevice) )
#define FROMGPU( H_DATA, D_DATA, MEM_SIZE ) checkCudaErrors( cudaMemcpy( H_DATA, D_DATA, MEM_SIZE, cudaMemcpyDeviceToHost) )
#define GPUTOGPU( DEST, SRC, MEM_SIZE )     checkCudaErrors( cudaMemcpy( DEST, SRC, MEM_SIZE, cudaMemcpyDeviceToDevice ) )
#define GPUFREE( MEM )                      checkCudaErrors( cudaFree(MEM) );


/*timing functions*/
//return ms, rather than second!

inline void startTimer(cudaEvent_t& start, cudaEvent_t& stop) {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

inline float endTimer(cudaEvent_t& start, cudaEvent_t& stop) {
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime = 0;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsedTime;
}

class Timer {
protected:
	float _t; //ms
public:

	Timer() :
	_t(0.0f) {
	};

	virtual ~Timer() {
	};

	virtual void go() = 0;
	virtual void stop() = 0;

	void reset() {
		_t = 0;
	}

	float report() const {
		return _t;
	}
	
	float reportAndPrint(const char* name) const {
		float sec = _t / 1000.f;
		printf("Processing time for %s: %f sec\n", name, sec);
		return sec;
	}
};

class CPUTimer : public Timer {
private:
//	struct timeval _start, _end;
public:

//	CPUTimer() :
//	_start(), _end() {
//	}

	CPUTimer(){}

	~CPUTimer() {
	}

	void go() {
//		gettimeofday(&_start, NULL);
	}

	void stop() {
//		gettimeofday(&_end, NULL);
//		_t += ((_end.tv_sec - _start.tv_sec)*1000.0f + (_end.tv_usec - _start.tv_usec) / 1000.0f);
	}
};

class CUDATimer : public Timer {
private:
	cudaEvent_t _start, _stop;
public:

	CUDATimer() : _start(), _stop() {
	};

	inline void go() {
		startTimer(_start, _stop);
	};

	inline void stop() {
		_t += endTimer(_start, _stop);
	};
};



/*CUDA helper functions*/

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
//#define checkCudaErrors(err)		   __checkCudaErrors (err, __FILE__, __LINE__)
//
//inline void __checkCudaErrors(cudaError err, const char *file, const int line) {
//	if (cudaSuccess != err) {
//		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
//				file, line, (int) err, cudaGetErrorString(err));
//		exit(-1);
//	}
//}

// This will output the proper error string when calling cudaGetLastError
//#define getLastCudaError(msg)	  __getLastCudaError (msg, __FILE__, __LINE__)
//
//inline void __getLastCudaError(const char *errorMessage, const char *file, const int line) {
//	cudaError_t err = cudaGetLastError();
//	if (cudaSuccess != err) {
//		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
//				file, line, errorMessage, (int) err, cudaGetErrorString(err));
//		exit(-1);
//	}
//}






#endif

