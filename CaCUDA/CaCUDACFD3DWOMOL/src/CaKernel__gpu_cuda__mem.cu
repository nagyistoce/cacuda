/* Assume Piraha will generate this file and this file will be pushed here as well */

#include "cctk.h"
#include "cctk_Parameters.h"
#include "cctk_Arguments.h"

#include "CaCUDA/CaCUDALib/src/CaCUDAUtil.h"

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif



CCTK_REAL * d_vx;
CCTK_REAL * d_vy;
CCTK_REAL * d_vz;
CCTK_REAL * d_p;


CCTK_REAL * d_vx_out; 
CCTK_REAL * d_vy_out; 
CCTK_REAL * d_vz_out; 


void CaCUDA_AllocDevMem (CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  const size_t dataSize = cctk_lsh[0] * cctk_lsh[1] * cctk_lsh[2]
    * sizeof (CCTK_REAL);

  
  CUDA_SAFE_CALL (cudaMalloc ((void **) &(d_vx), dataSize));
  CUDA_SAFE_CALL (cudaMalloc ((void **) &(d_vy), dataSize));
  CUDA_SAFE_CALL (cudaMalloc ((void **) &(d_vz), dataSize));
  CUDA_SAFE_CALL (cudaMalloc ((void **) &(d_p), dataSize));

  
  CUDA_SAFE_CALL (cudaMalloc ((void **) &(d_vx_out), dataSize));
  CUDA_SAFE_CALL (cudaMalloc ((void **) &(d_vy_out), dataSize));
  CUDA_SAFE_CALL (cudaMalloc ((void **) &(d_vz_out), dataSize));

  CUDA_CHECK_LAST_CALL("memalloc failed");
  CCTK_INFO ("device memory for CaCUDA variables successfully allocated");
}

/**
 * Free the memory of grid variables on GPU devices.
 */

void CaCUDA_FreeDevMem (CCTK_ARGUMENTS)
{
  
  CUDA_SAFE_CALL (cudaFree (d_vx));
  CUDA_SAFE_CALL (cudaFree (d_vy));
  CUDA_SAFE_CALL (cudaFree (d_vz));
  CUDA_SAFE_CALL (cudaFree (d_p));

  
  CUDA_SAFE_CALL (cudaFree (d_vx_out));
  CUDA_SAFE_CALL (cudaFree (d_vy_out));
  CUDA_SAFE_CALL (cudaFree (d_vz_out));

  CUDA_CHECK_LAST_CALL("memalloc failed");
  CCTK_INFO ("device memory for CaCUDA variables successfully freed");
}
