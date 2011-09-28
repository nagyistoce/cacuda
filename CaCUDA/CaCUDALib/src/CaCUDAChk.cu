#include "cctk.h"
#include "cctk_Parameters.h"
#include "cctk_Arguments.h"

#include "CaCUDALib.h"
#include "CaCUDAUtil.h"

#ifdef __CUDACC__
#  include <cuda.h>
#  include <cuda_runtime.h>
#endif

/* Check that the problem size will give the best performance
 *
 * This will need the device properties and problem size to give
 * the best guess.
 * */

void CaCUDALib_ParamCheck (CCTK_ARGUMENTS)
{
	DECLARE_CCTK_ARGUMENTS;
	DECLARE_CCTK_PARAMETERS;

/*get attributes of the registered Kernel function */
	  struct cudaFuncAttributes funcAttrib;
	  CUDA_SAFE_CALL(cudaFuncGetAttributes(&funcAttrib, globalfunc));

}
