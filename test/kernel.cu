//#include <cuda.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

////#include "cuda_device_runtime_api.h"
////#include "device_double_functions.h"
////#include "device_functions.h"

#include <stdio.h>
#include <iostream>

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
//#include "../libgqd/inc/gdd_basic.h"
//#include "../libgqd/inc/gqd_basic.h"
//#include "../libgqd/inc/gqd_function.h"

//#include "../libgqd/libgqd/inc/gdd_common.cu"
//#include "../libgqd/libgqd/inc/gqd_common.cu"

//#ifdef __cplusplus
//}
//#endif

/*
ÉäÉìÉNÉGÉâÅ[
operator+(double2 const&, double2 const&)
operator*(double2 const&, double2 const&)
operator-(double2 const&, double2 const&)
operator/(double2 const&, double2 const&)

*/

//cudaError_t addWithCuda(gdd_real c[], const gdd_real a[], const gdd_real b[], unsigned int size);

cudaError_t addWithCuda(gdd_real *c, const gdd_real *a, const gdd_real *b, unsigned int size);
cudaError_t addWithCudaq(gqd_real *c, const gqd_real *a, const gqd_real *b, unsigned int size);

__global__ void addKernel(gdd_real c[], const gdd_real a[], const gdd_real b[])
{
    int i = threadIdx.x;
 //   c[i] = a[i] + b[i];
	//c[i] = c[i] * b[i];
	//c[i] = c[i] - b[i];
	//c[i] = c[i] / b[i];
}
__global__ void addKernelq(gqd_real *c, const gqd_real *a, const gqd_real *b)
{
    //int i = threadIdx.x;
 //   c[i] = a[i] + b[i];
	//c[i] = c[i] * b[i];
	//c[i] = c[i] - b[i];
	//c[i] = c[i] / b[i];
}


int main()
{
	std::cout << "cuda GQD sample" << std::endl;
	
    const int arraySize = 5;
	const gdd_real a[arraySize] = { make_dd(1.0, 0.0), make_dd(2.0, 0.0), make_dd(3.0, 0.0), make_dd(4.0, 0.0), make_dd(5.0, 0.0) };
	const gdd_real b[arraySize] = { make_dd(10.0, 0.0), make_dd(20.0, 0.0), make_dd(30.0, 0.0), make_dd(40.0, 0.0), make_dd(50.0, 0.0) };
	gdd_real c[arraySize] = { make_dd(0.0, 0.0) };

	//const gqd_real q_a[arraySize] = { make_qd(1), make_qd(2), make_qd(3), make_qd(4), make_qd(5) };
	//const gqd_real q_b[arraySize] = { make_qd(10), make_qd(20), make_qd(30), make_qd(40), make_qd(50) };
	//gqd_real q_c[arraySize] = { make_qd(0.0, 0.0,0.0, 0.0) };

	gdd_real temp = make_dd(3.14, 0.0);
	//gqd_real temp2 = make_qd(3.14);
	//gqd_real temp2;

	
	std::cout << "size of gdd = " << sizeof(gdd_real) << "bytes" << std::endl;
	std::cout << "size of gqd = " << sizeof(gqd_real) << "bytes" << std::endl;
	cudaError_t cudaStatus;

    // Add vectors in parallel.
    cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

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
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

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
	addKernelq << <1, size >> >(dev_c, dev_a, dev_b);

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

