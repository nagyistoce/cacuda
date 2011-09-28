#ifndef _CACUDALIB_H_
#define _CACUDALIB_H_

#ifndef CACUDADEBUG
#define CACUDADEBUG 1
#endif

#include "cctk.h"
#include "CaCUDAUtil.h"

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>

extern "C"
{
#endif

/* CUDA Programming Guid 3.2 - B 1.4.1
 * __global__ function parameters are passed to the device:
 * via shared memory and are limited to 256 bytes on devices of compute capability 1.x,
 * via constant memory and are limited to 4 KB on devices of compute capability 2.x.
 */

/* CaCUDA arguments declaration  */
#define DECLARE_CCTK_CUDA_ARGUMENTS                                 \
  CCTK_REAL * cagh_bufferIn = d_bufferIn;                           \
  CCTK_REAL * cagh_bufferOut = d_bufferOut;                         \
  int const cagh_it = cctk_iteration;                               \
  int const cagh_ni = cctk_lsh[0];                                  \
  int const cagh_nj = cctk_lsh[1];                                  \
  int const cagh_nk = cctk_lsh[2];                                  \
  int const cagh_nghostsi = cctk_nghostzones[0];                    \
  int const cagh_nghostsj = cctk_nghostzones[1];                    \
  int const cagh_nghostsk = cctk_nghostzones[2];                    \
  CCTK_REAL const cagh_dx = cctk_delta_space[0];                    \
  CCTK_REAL const cagh_dy = cctk_delta_space[1];                    \
  CCTK_REAL const cagh_dz = cctk_delta_space[2];                    \
  CCTK_REAL const cagh_dt = cctk_delta_time;                        \
  CCTK_REAL const cagh_xmin = cctk_origin_space[0];                 \
  CCTK_REAL const cagh_ymin = cctk_origin_space[1];                 \
  CCTK_REAL const cagh_zmin = cctk_origin_space[2];                 \
  CCTK_REAL const cagh_time = cctk_time;

/* CaCUDA arguments */
#define CCTK_CUDA_ARGUMENTS                                         \
  CCTK_REAL * cagh_bufferIn,                                        \
  CCTK_REAL * cagh_bufferOut,                                       \
  int const cagh_it,                                                \
  int const cagh_ni,                                                \
  int const cagh_nj,                                                \
  int const cagh_nk,                                                \
  int const cagh_nghostsi,                                          \
  int const cagh_nghostsj,                                          \
  int const cagh_nghostsk,                                          \
  CCTK_REAL const cagh_dx,                                          \
  CCTK_REAL const cagh_dy,                                          \
  CCTK_REAL const cagh_dz,                                          \
  CCTK_REAL const cagh_dt,                                          \
  CCTK_REAL const cagh_xmin,                                        \
  CCTK_REAL const cagh_ymin,                                        \
  CCTK_REAL const cagh_zmin,                                        \
  CCTK_REAL const cagh_time

/* CaCUDA Arguments List */
#define CCTK_CUDA_ARGUMENT_LIST                                     \
  (cagh_bufferIn, cagh_bufferOut,                                    \
  cagh_it, cagh_ni, cagh_nj, cagh_nk,                               \
  cagh_nghostsi, cagh_nghostsj, cagh_nghostsk,                      \
  cagh_dx, cagh_dy, cagh_dz, cagh_dt,                               \
  cagh_xmin, cagh_ymin, cagh_zmin, cagh_time)

/* Cactus CUDA call wrapper */
/* we will use constants defined in CaCUDAConst.h to define a shared mem block
 * necessary parameter checks will be done to give either suggestions or
 * warnings
 * */
#ifdef __CUDACC__

#  define CCTK_CUDA_CALL __global__ void
#  define CCTK_CUDA_EXECUTE(func, gridDim, blockDim, args)          \
do {                                                                \
  func <<<gridDim, blockDim, 0, 0>>> args;                          \
  CUDA_SAFE_CALL(cudaThreadSynchronize());                          \
} while(0)

/* Declare a shared memory block
 * won't use this untill we know how to allocate a 2D/3D tile
 #  define DECLARE_CCTK_CUDA_SHARED(type,var) extern __shared__ type var[];
 */

#else

#define CCTK_CUDACALL

#endif


/* Processing Unit Info (only GPU's)
 *
 * We are not doing load balance at the very beginning. We will start with
 * homogeneous GPU cluster where only a certain number of GPUs are available
 * on each node. When there are no GPU's available on any node, we will terminate
 * the code.
 * */
  typedef struct _cacuda_pui
  {
    /* dev number for each mpi process mydev and myproc default to -1 */
    CCTK_INT mydev;
    CCTK_INT myproc;
    struct cudaDeviceProp devprop;
  } CACUDA_PUI;

/*
 * CaCUDA grid hierarchy is a simplified version of Cactus cGH
  typedef struct _cacuda_gh
  {
	    int cagh_iteration;
	    int cagh_lsh[3];
	    CCTK_REAL cagh_delta_time;
	    CCTK_REAL cagh_delta_space[3];
	    int cagh_nghostzones[3];
	    CCTK_REAL cagh_time;
   } CACUDA_GH;
*/


/**
 * CCTK_DevBanner
 * Output the CaCUDA banner.
 */
  CCTK_INT CaCUDALib_Banner (void);

/**
 * CCTK_GetDevInfo
 * Output device information.
 */
  void CaCUDALib_GetDevInfo (CCTK_INT numdev);

/**
 * CCTK_OutputDevInfo
 * Output device information.
 */
  CCTK_INT CaCUDALib_OutputDevInfo (void);

/**
 * CCTK_RegDevVar
 * Register device variables with Cactus variable names
 * CCTK_INT CaCUDALib_RegVar(const char * varname);
 */

/**
 * CCTK_RegDevGrp
 * Register device groups with Cactus group names
 * CCTK_INT CaCUDALib_RegGrp(const char * grpname);
 */

/**
 * CCTK_InitDev
 * Initialize GPU devices.
 *
 * TODO: Here I refer to Marek's init_s() where we shall have a 1D array holding the
 * distribution of GPU's among all the nodes in the future. The total number of GPU's is
 * not good enough for heterogeneous clusters. E.g., what if we have 4 mpi processes on
 * one node where there are only 2 GPU's ?
 */
  CCTK_INT CaCUDALib_InitDev (void);

/**
 * CCTK_ShutDownDev
 * Shutdown GPU devices.
 */
  CCTK_INT CaCUDALib_ShutDownDev (void);
/**
 * CCTK_AllocDevMem
 * Allocates memory of grid variables on GPU devices.
 * We will keep a list of registered variables that users request to
 * allocate memory on GPU.
 */

/**
 * CCTK_FreeDevMem
 * Free the memory of grid variables on GPU devices.
 */
//  CCTK_INT CaCUDALib_FreeDevMem (void);

/**
 * CCTK_CopyToDev
 * Copying data from CPU to GPU.
 */
//  CCTK_INT CaCUDALib_CopyToDev (void **data);

/**
 * CCTK_CopyFromDev
 * Copies the data responsible for the copying from the GPU to CPU memory space.
 */
//  CCTK_INT CaCUDALib_CopyFromDev (void **data);

/**
 *  variables shared among routines in CaCUDALib
 * */

  /* array of processing units */
  extern CACUDA_PUI *cacuda_pui;

  /* CaCUDA grid hierarchy will be in the contant memory
   * notice that constant memory block can only be claimed once
   * for each variable
   *
   * Constant memory has to be initialized within the same file which makes
   * it hard to use it for transfer data. We may use it implicitly by declare
   * const input arguments and let CUDA compiler to optimize them. Those
   * input arguments will usually be copied to the const memory but we need to
   * verify this later.
   * */
  //__constant__ CACUDA_GH d_cagh;

  /* number of devices */
  extern CCTK_INT num_dev;

#ifdef __CUDACC__
}
#endif

#endif                          /* _CACUDALIB_H_ */
