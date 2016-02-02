#ifndef __GQD_COMMON_CU__
#define __GQD_COMMON_CU__

#include <stdio.h>
//#include <stdlib.h>


//#include "gqd_type.h"		//type definitions for gdd_real and gqd_real
#include "cuda_header.h"
#include "gdd_real.h"
//#include "inline.cu"		//basic functions used by both gdd_real and gqd_real

#include "gqd_real.h"
//#define SLOPPY_ADD 1
//#define SLOPPY_MUL 1
//#define SLOPPY_DIV 1
//#define USE_FMA 1

/* type definitions, defined in the type.h */
union type_trans_dbl{
	__int64 i64_value;
	double	dbl_value;
};

/** initialization function */
void GDDStart(int device) {
	printf("GDD turns on ...");
	cudaSetDevice(device);


	cudaStream_t st_spcl_val;
	cudaStreamCreate(&st_spcl_val);

	double h_special_tbl[2];
	type_trans_dbl trans;
	
	h_special_tbl[0] = std::numeric_limits<double>::infinity();
	trans.i64_value = (0x7ff0000000000000ULL);	//CUDART_INF
	h_special_tbl[1] = trans.dbl_value;
	checkCudaErrors(cudaMemcpyToSymbolAsync(__qd_inf, &h_special_tbl, sizeof(h_special_tbl), 0, cudaMemcpyHostToDevice, st_spcl_val));
	cudaStreamSynchronize(st_spcl_val);

	h_special_tbl[0] = std::numeric_limits<double>::quiet_NaN();
	trans.i64_value = (0xfff8000000000000ULL);	//CUDART_NAN
	h_special_tbl[1] = trans.dbl_value;
	checkCudaErrors(cudaMemcpyToSymbolAsync(__qd_qnan, &h_special_tbl, sizeof(h_special_tbl), 0, cudaMemcpyHostToDevice, st_spcl_val));
	cudaStreamSynchronize(st_spcl_val);

	cudaStreamDestroy(st_spcl_val);

	printf("\tdone.\n");
}

void GDDEnd() {
	printf("GQD turns off...");
	cudaDeviceReset();
	printf("\tdone.\n");
}


void GQDEnd() {
	printf("GQD turns off...");
	cudaDeviceReset();
	printf("\tdone.\n");
}

void GQDStart(int device) {
	printf("GQD turns on ...");
	
	cudaSetDevice(device);

	cudaStream_t st_spcl_val;
	cudaStreamCreate(&st_spcl_val);

	double h_special_tbl[2];
	type_trans_dbl trans;

	h_special_tbl[0] = std::numeric_limits<double>::infinity();
	trans.i64_value = (0x7ff0000000000000ULL);	//CUDART_INF
	h_special_tbl[1] = trans.dbl_value;
	checkCudaErrors(cudaMemcpyToSymbolAsync(__dd_inf, &h_special_tbl, sizeof(h_special_tbl), 0, cudaMemcpyHostToDevice, st_spcl_val));
	cudaStreamSynchronize(st_spcl_val);

	h_special_tbl[0] = std::numeric_limits<double>::quiet_NaN();
	trans.i64_value = (0xfff8000000000000ULL);	//CUDART_NAN
	h_special_tbl[1] = trans.dbl_value;
	checkCudaErrors(cudaMemcpyToSymbolAsync(__dd_qnan, &h_special_tbl, sizeof(h_special_tbl), 0, cudaMemcpyHostToDevice, st_spcl_val));
	cudaStreamSynchronize(st_spcl_val);

	cudaStreamDestroy(st_spcl_val);


	printf("\tdone.\n");
}

//__host__
//int convSPValForHost(gdd_real *buff, unsigned int elements){
//	double h_qnan = std::numeric_limits<double>::quiet_NaN();
//	double h_inf = std::numeric_limits<double>::infinity();
//
//	for (unsigned int i = 0; i < elements; i++){
//		// transform to hosts qnan if element is gpu_nan
//		gdd_real t = buff[i];
//		if (isnan(t)){
//			buff[i].dd.x = h_qnan;
//			buff[i].dd.y = h_qnan;
//		} else if (isinf(t)) {
//			buff[i].dd.x = h_inf;
//			buff[i].dd.y = h_inf;
//		} else {
//			// nothing to do
//		}
//		i++;
//	}
//	return 0;
//}
//__device__
//int convSPValFordevice(gdd_real *buff, unsigned int elements){
//	for (unsigned int i = 0; i < elements; i++) {
//		// transform to hosts qnan if element is gpu_nan
//		gdd_real t = buff[i];
//		if (isnan(t)) {
//			buff[i] = _dd_qnan;
//		} else if (isinf(t)) {
//			buff[i] = _dd_inf;
//		} else {
//			// nothing to do
//		}
//		i++;
//	}
//	return 0;
//}

#endif /* __GQD_COMMON_CU__ */


