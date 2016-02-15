#ifndef __KERNEL_H__
#define __KERNEL_H__

//#ifdef __cplusplus
//extern "C" {
//#endif


//union trans {
//	unsigned __int64 asInt64;
//	double asDouble;
//};

void Initialize(int n = 0);
void Finalize(int n = 0);

//cudaError_t addWithCuda(gdd_real *c, const gdd_real *a, const gdd_real *b, unsigned int size);
//cudaError_t addWithCudaq(gqd_real *c, const gqd_real *a, const gqd_real *b, unsigned int size);

void checkInline();


void addWithCudaq();

void addWithCuda();


void IEEE_check();




//#ifdef __cplusplus
//}
//#endif


#endif // __KERNEL_H__