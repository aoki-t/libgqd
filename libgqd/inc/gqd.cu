#ifndef __GQD_CU__
#define __GQD_CU__

/**
* the API file
* includes every thing for this library
*/

#include "cuda_header.h"

#include "gdd_real.h"
//#include "inline.cu"		//basic functions used by both gdd_real and gqd_real

/* gdd_library */
#include "gdd_basic.cu"
#include "gdd_sqrt.cu"
#include "gdd_exp.cu"
#include "gdd_log.cu"
#include "gdd_sincos.cu"


/* gqd_libraray */
#include "gqd_real.h"
#include "gqd_basic.cu"
#include "gqd_sqrt.cu"
#include "gqd_exp.cu"
#include "gqd_log.cu"
#include "gqd_sincos.cu"

#endif // __GQD_CU__
