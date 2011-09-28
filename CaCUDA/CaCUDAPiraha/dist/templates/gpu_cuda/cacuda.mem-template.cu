/* Assume Piraha will generate this file and this file will be pushed here as well */

#include "cctk.h"
#include "cctk_Parameters.h"
#include "cctk_Arguments.h"

#include "CaCUDA/CaCUDALib/src/CaCUDAUtil.h"

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif


%var_loop() %[
CCTK_REAL * d_%{vname};]%

%var_loop("intent=separateinout") %[
CCTK_REAL * d_%{vname}_out; ]%


void CaCUDA_AllocDevMem (CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  const size_t dataSize = cctk_lsh[0] * cctk_lsh[1] * cctk_lsh[2]
    * sizeof (CCTK_REAL);

  %var_loop() %[
  CUDA_SAFE_CALL (cudaMalloc ((void **) &(d_%{vname}), dataSize));]%

  %var_loop("intent=separateinout") %[
  CUDA_SAFE_CALL (cudaMalloc ((void **) &(d_%{vname}_out), dataSize));]%

  CUDA_CHECK_LAST_CALL("memalloc failed");
  CCTK_INFO ("device memory for CaCUDA variables successfully allocated");
}

/**
 * Free the memory of grid variables on GPU devices.
 */

void CaCUDA_FreeDevMem (CCTK_ARGUMENTS)
{
  %var_loop() %[
  CUDA_SAFE_CALL (cudaFree (d_%{vname}));]%

  %var_loop("intent=separateinout") %[
  CUDA_SAFE_CALL (cudaFree (d_%{vname}_out));]%

  CUDA_CHECK_LAST_CALL("memalloc failed");
  CCTK_INFO ("device memory for CaCUDA variables successfully freed");
}
