/*@@
 * @file    CaKernel__gpu_cuda__boundary__Update_Boundaries.h
 * @date    Tue Sep 20 15:47:56 CDT 2011
 * @author  Marek Blazewicz and Steven R. Brandt
 * @desc
 * WARNING: Automatically Generated. Do not modify.
 * The prototype of the CaCUDA computational schema. It contains macros
 * which enable to declare, define and launch kernels as well as to copy
 * the data required for proper computations. The macros presented
 * in this file in the future will be automatically generated by the
 * Cactus parser depending on the input in interfaces.ccl file.
 * @enddesc
 * @version  $Header$
 *
 @@*/

#ifndef CAKERNEL__GPU_CUDA__BOUNDARY__UPDATE_BOUNDARIES_H
#define CAKERNEL__GPU_CUDA__BOUNDARY__UPDATE_BOUNDARIES_H

/* definition of CCTK_REAL */
#include "cctk.h"

/* CaCUDAUtil.h shall be visible to all CaCUDA developers at some point */
#include "CaCUDA/CaCUDALib/src/CaCUDAUtil.h"

#include "cctk_Parameters.h"
#include "cctk_Arguments.h"


#ifdef __CUDACC__

/// !!!!!!!!!!!!! BEGIN of global definitions (not auto generated) !!!!!!!!!!!!!!!!


#define CAKERNEL_Threadsx 16
#define CAKERNEL_Threadsy 16
#define CAKERNEL_Threadsz 1

/* JT: 16x16x16 failed to compile on spider. 8x8x8 is ok to compile.
 * Let's make 8x8x8 as the default setting temporarily.
 * We will need to estimate the best configuration based on
 * the number of variables and the memory available.
 * */
#define CAKERNEL_Tilex 16
#define CAKERNEL_Tiley 16
#define CAKERNEL_Tilez 16

/// !!!!!!!!!!!!! END of global definitions (not auto generated) !!!!!!!!!!!!!!!!

/// !!!!!!!!!!!!!!!!!!!!!!!!! BEGIN Update Pressure Kernel macors !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef stncl_xn
#error "You can't include two header file in one execution file"
#endif

#define stncl_xn 1
#define stncl_xp 1
#define stncl_yn 1
#define stncl_yp 1
#define stncl_zn 1
#define stncl_zp 1

/** Element  */
#define CAKERNEL_GFINDEX3D_Update_Boundaries(ptr, i, j, k)                     \
 ptr[(i + gi) + params.cagh_ni * ((j + gj) + params.cagh_nj * (k + gk))]

#define I3D CAKERNEL_GFINDEX3D_Update_Boundaries          

/** The '_s' postfix means that the macro is more 'specific'. The user gets
more control to the compilation process, by for example skipping some of the
macros. But it requires a greater knowledge of what exactly is done inside
these macros (advanced programmers only - like Jian) */
#define CAKERNEL_Update_Boundaries_Declare_Begin_s                             \
template<short x, short y, short z>                                            \
__global__ void CAKERNEL_Update_Boundaries(                                    \
/** Variables automatically added by the Cactus parser:  */                    \
                                                                               \
/*inout*/CCTK_REAL *vx,/*inout*/CCTK_REAL *vy,/*inout*/CCTK_REAL *vz,          \
                                                                               \
                                                                               \
/** Statically added variables to each kernel: */                              \
const CaCUDA_Kernel_Launch_Parameters params)                                  \
{                                                                           
# define CAKERNEL_Update_Boundaries_Declare_Cached_Variables_s         
# define CAKERNEL_Update_Boundaries_Declare_Flow_Variables_s                   \
/** Kernel specific variables declaration. */                                  \
/** Common variables declaration; values are kernel specific. */               \
  int gi, gj, gk;                                                              \
  if(x){                                                                       \
    gj = blockIdx.x * blockDim.x + threadIdx.x;                                \
    gk = blockIdx.y * blockDim.y + threadIdx.y;                                \
    if(x > 0) gi = params.cagh_ni - 1;                                         \
    else      gi = 0;                                                          \
  }                                                                            \
  if(y){                                                                       \
    gi = blockIdx.x * blockDim.x + threadIdx.x;                                \
    gk = blockIdx.y * blockDim.y + threadIdx.y;                                \
    if(y > 0) gj = params.cagh_nj - 1;                                         \
    else      gj = 0;                                                          \
  }                                                                            \
  if(z){                                                                       \
    gi = blockIdx.x * blockDim.x + threadIdx.x;                                \
    gj = blockIdx.y * blockDim.y + threadIdx.y;                                \
    if(z > 0) gk = params.cagh_nk - 1;                                         \
    else      gk = 0;                                                          \
  }                                                                            \
  bool fetch_data = gi < params.cagh_ni && gj < params.cagh_nj                 \
                    && gk < params.cagh_nk;                                 
# define CAKERNEL_Update_Boundaries_Limit_Threads_To_LSH_Begin_s               \
  if(fetch_data)                                                               \
  {                                                                         
#   define CAKERNEL_Update_Boundaries_Computations_Begin_s                     \
    
# define CAKERNEL_Update_Boundaries_Limit_Threads_To_LSH_End_s                 \
  }                                                                         
#define CAKERNEL_Update_Boundaries_Declare_End_s                               \
}


#define CAKERNEL_Update_Boundaries_Begin                                       \
CAKERNEL_Update_Boundaries_Declare_Begin_s                                     \
  CAKERNEL_Update_Boundaries_Declare_Cached_Variables_s                        \
  CAKERNEL_Update_Boundaries_Declare_Flow_Variables_s                          \
  CAKERNEL_Update_Boundaries_Limit_Threads_To_LSH_Begin_s                      \

#define CAKERNEL_Update_Boundaries_End                                         \
  CAKERNEL_Update_Boundaries_Limit_Threads_To_LSH_End_s                        \
CAKERNEL_Update_Boundaries_Declare_End_s

//template<short x, short y, short z> 
//__global__ void CAKERNEL_Update_Boundaries(                            
//
///*inout*/CCTK_REAL *vx,/*inout*/CCTK_REAL *vy,/*inout*/CCTK_REAL *vz,
//
//
//const CaCUDA_Kernel_Launch_Parameters params);                               
//
//
//extern CCTK_REAL *d_vx;
extern CCTK_REAL *d_vy;
extern CCTK_REAL *d_vz;

//
//


/// !!!!!!!!!!!!!!!!!!!!!!!!! END Update Pressure Kernel macors !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#endif

#endif /* CAKERNEL__GPU_CUDA__BOUNDARY__UPDATE_BOUNDARIES_H */