/* Assume Piraha will generate this file and this file will be pushed here as well */

#include "cctk.h"
#include "cctk_Parameters.h"
#include "cctk_Arguments.h"

#include "CaCUDA/CaCUDALib/src/CaCUDAUtil.h"
#include "CaCUDACFD3D_Kernels_Vars.h"
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

/* now we just copy everything back and forward for each time step */
/* we may schedule one communication for each kernel */
void CaCUDA_CopyToDev( CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;
  const size_t dataSize = sizeof(CCTK_REAL) * cctkGH->cctk_lsh[0]
      * cctkGH->cctk_lsh[1] * cctkGH->cctk_lsh[2];
%var_loop() %[
  CUDA_SAFE_CALL (cudaMemcpy((void *)d_%{vname}, %{vname},
          dataSize, cudaMemcpyHostToDevice));
]%
  CCTK_INFO("CaCUDA variables %var_loop('delimit=,','%vname') have been successfully copied to device");
}

void CaCUDA_CopyFromDev( CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;

  const size_t dataSize = sizeof(CCTK_REAL) * cctkGH->cctk_lsh[0]
      * cctkGH->cctk_lsh[1] * cctkGH->cctk_lsh[2];
%var_loop() %[
  CUDA_SAFE_CALL (cudaMemcpy(%{vname}_rhs, d_%{vname},
          dataSize, cudaMemcpyDeviceToHost));
]%
  CCTK_INFO("CaCUDA variables %var_loop('delimit=,','%vname') have been successfully copied from device");
}
