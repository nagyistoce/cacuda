#include <stdio.h>
#include <math.h>
#include "cctk.h"
#include "cctk_Parameters.h"
#include "cctk_Arguments.h"

#include "CaCUDALib.h"
#include "CaCUDAUtil.h"

#ifdef __CUDACC__
#  include <cuda.h>
#  include <cuda_runtime.h>
#endif

/* input buffer on devices */
CCTK_REAL **d_bufferIn;

/* output buffer on devices */
CCTK_REAL **d_bufferOut;

void CaCUDALib_AllocDevMem (CCTK_ARGUMENTS)
{
	DECLARE_CCTK_ARGUMENTS;
	DECLARE_CCTK_PARAMETERS;

    const size_t volumeSize = cctk_lsh[0] * cctk_lsh[1] * cctk_lsh[2];

#ifdef CACUDADEBUG
	CCTK_INFO("allocate memory for registered CaCUDA variables");
#endif

    CUDA_SAFE_CALL(cudaMalloc((void **)&d_bufferOut, volumeSize * sizeof(CCTK_REAL)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_bufferIn,  volumeSize * sizeof(CCTK_REAL)));

  return ;
}

/**
 * Free the memory of grid variables on GPU devices.
 */

CCTK_INT CaCUDALib_FreeDevMem (void)
{
  CUDA_SAFE_CALL(cudaFree(d_bufferIn));
  CUDA_SAFE_CALL(cudaFree(d_bufferOut));

  return 0;
}

/**
 * Copying data from CPU to GPU.
 */

CCTK_INT CaCUDALib_CopyToDev(void **data)
{
  return 0;
}

/**
 * Copies the data responsible for the copying from the GPU to CPU memory space.
 */
CCTK_INT CaCUDALib_CopyFromDev(void **data)
{
  return 0;
}
