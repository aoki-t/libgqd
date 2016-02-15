#ifndef __GQD_TYPE_H__
#define __GQD_TYPE_H__

//#include <vector_types.h>


/* compiler switch */
/**
 * ALL_MATH will include advanced math functions, including
 * atan, acos, asin, sinh, cosh, tanh, asinh, acosh, atanh
 * WARNING: these functions take long time to compile, 
 * e.g., several hours
 * */
#define ALL_MATH 1

/* type definition */
//typedef double2 gdd_real;

//typedef double4 gqd_real;


/* initialization functions, these can be called by hosts */
void GDDStart(int device = 0);
void GDDEnd();
void GQDStart(int device = 0);
void GQDEnd();

#endif /*__GQD_TYPE_H__*/
