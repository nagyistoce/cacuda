#include <stdio.h>
#include <math.h>

#include "cctk.h"
#include "cctk_Parameters.h"
#include "cctk_Arguments.h"
#include "CaCUDALib.h"
#include "CaCUDAUtil.h"

/* definition of global variables */

CACUDA_PUI *cacuda_pui;
CCTK_INT num_dev;

/* initialization */
CCTK_INT CaCUDALib_InitDev (void)
{
#ifdef __CUDACC__

  DECLARE_CCTK_PARAMETERS;

/* get number of devices */
  CUDA_SAFE_CALL (cudaGetDeviceCount (&num_dev));

  if (num_dev == 0)
  {
    CCTK_WARN
      (0, "There are no CUDA devices available. (They may be busy, or the driver may not be installed.)");
  }

  MALLOC_SAFE_CALL(cacuda_pui = (CACUDA_PUI*) malloc (num_dev * sizeof(CACUDA_PUI)));

/* get device properties */
  CaCUDALib_GetDevInfo(num_dev);

#endif  // #ifdef __CUDACC__
  return 0;
}

/* Set device to use for an MPI process */

void CaCUDALib_SetDev (CCTK_ARGUMENTS)
{
#ifdef __CUDACC__
	DECLARE_CCTK_ARGUMENTS;
	DECLARE_CCTK_PARAMETERS;
	int myproc;
    int devidx;

    myproc = CCTK_MyProc(cctkGH);
    devidx = myproc % num_dev;

    /* we set device based on the number of devices available on each node */
    CUDA_SAFE_CALL (cudaSetDevice (devidx));
    cacuda_pui[devidx].mydev = devidx;
    cacuda_pui[devidx].myproc= myproc;
    CCTK_VInfo(CCTK_THORNSTRING, "number of device %d", num_dev);
    CCTK_VInfo(CCTK_THORNSTRING, "device %d is successfully assigned to process %d", devidx, myproc);

#endif  // #ifdef __CUDACC__
  return;
}

void CaCUDALib_GetDevInfo (CCTK_INT num_dev)
{
#ifdef __CUDACC__

  CCTK_INT devidx;

  for (devidx = 0; devidx < num_dev; devidx++)
  {
/* fill in the structure */
	  cacuda_pui[devidx].mydev = -1;
	  cacuda_pui[devidx].myproc= -1;
	  CUDA_SAFE_CALL(cudaGetDeviceProperties (&(cacuda_pui[devidx].devprop), devidx));

  }
#endif  // #ifdef __CUDACC__
  return;
}

CCTK_INT CaCUDALib_OutputDevInfo (void)
{
#ifdef __CUDACC__
  DECLARE_CCTK_PARAMETERS;
  CCTK_INT devidx;
  CCTK_REAL dev_capability;

  /* updated to match the definition of cudaDeviceProp in CUDA SDK version 3.3 */
  for (devidx = 0; devidx < num_dev; devidx++)
  {
    printf ("\nDevice %d: %s\n", devidx, cacuda_pui[devidx].devprop.name);
    printf ("  Total global memory %.2f MB \n",
            (unsigned long) cacuda_pui[devidx].devprop.totalGlobalMem / (1024.0 * 1024.0));
    printf ("  Shared memory per block: %lu bytes\n",
            (unsigned long) cacuda_pui[devidx].devprop.sharedMemPerBlock);
    printf ("  Number of 32 bit registers per block: %d\n",
            cacuda_pui[devidx].devprop.regsPerBlock);
    printf ("  Warp size: %d\n", cacuda_pui[devidx].devprop.warpSize);
    printf ("  Maximum memory pitch: %.2f GB\n",
            (unsigned long) cacuda_pui[devidx].devprop.memPitch / (1024.0 * 1024.0 * 1024.0));
    printf ("  Maximum number of threads per block: %d\n",
            cacuda_pui[devidx].devprop.maxThreadsPerBlock);
    printf ("  Maximum sizes of each dimension of a block: %d x %d x %d\n",
            cacuda_pui[devidx].devprop.maxThreadsDim[0], cacuda_pui[devidx].devprop.maxThreadsDim[1],
            cacuda_pui[devidx].devprop.maxThreadsDim[2]);
    printf ("  Maximum sizes of each dimension of a grid: %d x %d x %d\n",
            cacuda_pui[devidx].devprop.maxGridSize[0], cacuda_pui[devidx].devprop.maxGridSize[1],
            cacuda_pui[devidx].devprop.maxGridSize[2]);
    printf ("  Clock rate: %.2f GHz\n", cacuda_pui[devidx].devprop.clockRate * 1.0e-6);
    printf ("  Total constant memory size: %lu bytes\n",
            (unsigned long) cacuda_pui[devidx].devprop.totalConstMem);
    printf ("  CUDA capability: %d.%d\n", cacuda_pui[devidx].devprop.major, cacuda_pui[devidx].devprop.minor);
    printf ("  Texture alignment: %lu bytes\n",
            (unsigned long) cacuda_pui[devidx].devprop.textureAlignment);
    #if CUDART_VERSION >= 2000
    printf ("  Concurrent copy and execution: %s\n",
            cacuda_pui[devidx].devprop.deviceOverlap ? "Yes" : "No");
    #endif
    printf ("  Number of multiprocessors: %d\n", cacuda_pui[devidx].devprop.multiProcessorCount);
    #if CUDART_VERSION >= 2020
    printf ("  Run time limit on kernels: %s\n",
            cacuda_pui[devidx].devprop.kernelExecTimeoutEnabled ? "Yes" : "No");
    printf ("  Device integrated: %s\n", cacuda_pui[devidx].devprop.integrated ? "Yes" : "No");
    printf ("  Device can map host memory: %s\n",
            cacuda_pui[devidx].devprop.canMapHostMemory ? "Yes" : "No");
    printf ("  Compute Mode: %s\n",
            cacuda_pui[devidx].devprop.computeMode != 0 ? (cacuda_pui[devidx].devprop.computeMode ==
                                        1 ? "Compute-exclusive" :
                                        "Compute-prohibited") : "Normal");
    #endif
    printf ("  Maximum 1D texture size: %d\n", cacuda_pui[devidx].devprop.maxTexture1D);
    printf ("  Maximum 2D texture dimensions: %d x %d\n",
            cacuda_pui[devidx].devprop.maxTexture2D[0], cacuda_pui[devidx].devprop.maxTexture2D[1]);
    printf ("  Maximum 3D texture dimensions: %d x %d x %d\n",
            cacuda_pui[devidx].devprop.maxTexture3D[0], cacuda_pui[devidx].devprop.maxTexture3D[1],
            cacuda_pui[devidx].devprop.maxTexture3D[2]);
    #if CUDART_VERSION < 4000
    printf ("  Maximum 2D texture array dimensions: %d x %d x %d\n",
            cacuda_pui[devidx].devprop.maxTexture2DArray[0], cacuda_pui[devidx].devprop.maxTexture2DArray[1],
            cacuda_pui[devidx].devprop.maxTexture2DArray[2]);
    #endif
    printf ("  Surface alignment: %lu bytes\n",
            (unsigned long) cacuda_pui[devidx].devprop.surfaceAlignment);
    #if CUDART_VERSION >= 3000
    printf ("  Concurrent kernel execution: %s\n",
            cacuda_pui[devidx].devprop.concurrentKernels ? "Yes" : "No");
    #endif
    #if CUDART_VERSION >= 3010
    printf ("  ECC enabled: %s\n", cacuda_pui[devidx].devprop.ECCEnabled ? "Yes" : "No");
    #endif
    printf ("  PCI bus ID: %d\n", cacuda_pui[devidx].devprop.pciBusID);
    printf ("  PCI device ID: %d\n", cacuda_pui[devidx].devprop.pciDeviceID);
    #if CUDART_VERSION >= 3020
    printf ("  TCC driver used: %s\n", cacuda_pui[devidx].devprop.tccDriver ? "Yes" : "No");
    #endif
    dev_capability =
      cacuda_pui[devidx].devprop.major + pow (0.1, (cacuda_pui[devidx].devprop.minor / 10 + 1.0)) * cacuda_pui[devidx].devprop.minor;

    if (min_cuda_capability > dev_capability)
    {
      CCTK_VWarn (0, __LINE__, __FILE__, CCTK_THORNSTRING,
                  "This CUDA device has a CUDA capability of %2.1lf, which does not match the minimum requirement %2.1lf set by parameter CaCUDALib::min_cuda_capability !",
                  dev_capability, min_cuda_capability);
    }
  }
  CCTK_INFO ("Finish outputting CUDA device information");
#endif  // #ifdef __CUDACC__

  return 0;
}

CCTK_INT CaCUDALib_ShutDownDev (void)
{
  free (cacuda_pui);
  return 0;
}
