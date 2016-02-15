#ifndef __MEMORY_CU__
#define __MEMORY_CU__

#include <Windows.h>
#include <iostream>
#include <stdio.h>


#include <mmsystem.h>	// timeGetTime()
#pragma comment( lib, "winmm.lib" )

#include "../libgqd/inc/cuda_header.h"


cudaError MemoryBandwidthCheck(){

	int elem_size = 4;
	int tortal_size = (512 * 1024 * 1024);

	// host memory set up
	int *p_hmem = (int*)malloc(tortal_size);
	for (int i = 0; i < tortal_size / sizeof(int); ++i) {
		*p_hmem++ = i;
	}

	// device memory set up
	cudaError_t cudaStatus;
	size_t avai, total;
	cudaStatus = cudaMemGetInfo(&avai, &total);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemGetInfo failed!");
		goto cuError;
	}

	if (avai + (1024*1024) < tortal_size ) {
		fprintf(stderr, "avairable memory not enough!");
		goto cuError;
	}


	int *p_dmem;
	cudaStatus = cudaMalloc((void**)&p_dmem, tortal_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto cuError;
	}


	printf("QueryPerformanceCounter()\n");
	LARGE_INTEGER	liFreq;
	BOOL QPCsupported = QueryPerformanceFrequency(&liFreq);
	if (!QPCsupported) {
		printf("QueryPerformanceCounter not supported.\n");
		exit(0);
	}
	printf("Freq = %9.4ld[counts/sec]\n", liFreq.QuadPart);
	LARGE_INTEGER	cuStart, cuEnd;
	QueryPerformanceCounter(&cuStart);

	cudaStatus = cudaMemcpy(p_dmem, p_hmem, elem_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto cuError;
	}


	QueryPerformanceCounter(&cuEnd);
	printf("%9.4lf[ms]\n", 1000.0 * (double)(cuEnd.QuadPart - cuStart.QuadPart) / liFreq.QuadPart);


	free(p_hmem);
	cudaFree(p_dmem);
	return cudaSuccess;

cuError:
	free(p_hmem);
	cudaFree(p_dmem);
	return cudaError::cudaErrorInvalidValue;
}

#endif
