#ifndef CAKERNEL__GPU_CUDA__VARS_H
#define CAKERNEL__GPU_CUDA__VARS_H

/* definition of CCTK_REAL */
#include "cctk.h"

#ifdef __CUDACC__

/* device pointers */
/* All the external variables will be declared here.
 * We will generate the this file from the parser directly but we
 * need to generate the memory allocation routines as well.
 */


extern CCTK_REAL * d_vx;

extern CCTK_REAL * d_vy;

extern CCTK_REAL * d_vz;

extern CCTK_REAL * d_p;



extern CCTK_REAL * d_vx_out;

extern CCTK_REAL * d_vy_out;

extern CCTK_REAL * d_vz_out;


#endif

#endif /* CAKERNEL__GPU_CUDA__VARS_H */
