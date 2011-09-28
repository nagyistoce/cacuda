/* Assume Piraha will generate this file and this file will be pushed here as well */

#include "cctk.h"
#include "cctk_Parameters.h"
#include "cctk_Arguments.h"

#include "CaCUDA/CaCUDALib/src/CaCUDAUtil.h"
#include "CaKernel__%{path_parent_lower}__vars.h"
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
  CUDA_CHECK_LAST_CALL("Memory copy to device failed");
//  CCTK_INFO("CaCUDA variables %var_loop('delimit=,','%vname') have been successfully copied to device");
}

void CaCUDA_CopyFromDev( CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;

  const size_t dataSize = sizeof(CCTK_REAL) * cctkGH->cctk_lsh[0]
      * cctkGH->cctk_lsh[1] * cctkGH->cctk_lsh[2];
%var_loop() %[
  CUDA_SAFE_CALL (cudaMemcpy(%{vname}, d_%{vname},
          dataSize, cudaMemcpyDeviceToHost));
]%
  CUDA_CHECK_LAST_CALL("Memory copy from device failed");
//  CCTK_INFO("CaCUDA variables %var_loop('delimit=,','%vname') have been successfully copied from device");
}

void CaCUDA_CopyToDev_Boundary( CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;

  long szp = sizeof(CCTK_REAL);
 

  // I USED THAT ORDER in former code; I didn't change it: N, E, F, S, W, B
  int t[] = {3, 1, 5, 2, 0, 4};
  int ghost_zones[6] = {0};
  for (int i = 0; i< 6; i++){
    if(cctkGH->cctk_bbox[t[i]] != 1) ghost_zones[i] = cctk_nghostzones[t[i] / 2]; 
  }	
  
  long  dimxg = cctk_lsh[0],
        dimyg = cctk_lsh[1],
        dimzg = cctk_lsh[2];
  long  dimxl = dimxg - ghost_zones[1] - ghost_zones[4],
        dimyl = dimyg - ghost_zones[0] - ghost_zones[3],
        dimzl = dimzg - ghost_zones[2] - ghost_zones[5];

/////  variables pointing to places in the array from/where the ghost zone should be copied // N, E, F, S, W, B
/// The data copy is desigined to minimize the amount of data copied unaligned (E and W ghost-zones).

  // I USED THAT ORDER in former code; I didn't change it: N, E, F, S, W, B
 
  long dimx_out[] = {dimxg * szp, ghost_zones[1] * szp, dimxg * szp, dimxg * szp, ghost_zones[4] * szp, dimxg * szp};
  long dimy_out[] = {ghost_zones[0], dimyl, dimyl, ghost_zones[3], dimyl, dimyl};
  long dimz_out[] = {dimzg, dimzl, ghost_zones[2], dimzg, dimzl, ghost_zones[5]};

  long offx_out[] = {0, (dimxl + ghost_zones[4]) * szp, 0, 0, 0, 0};
  long offy_out[] = {dimyl + ghost_zones[3], ghost_zones[3], ghost_zones[3], 0, ghost_zones[3], ghost_zones[3]};
  long offz_out[] = {0, ghost_zones[5], ghost_zones[5] + dimzl, 0, ghost_zones[5], 0};

  long pitchx = dimxg * szp; 
  long pitchy = pitchx * dimyg;

  CCTK_REAL * var[]   = {%var_loop() %[%{vname}, ]% 0};
  CCTK_REAL * d_var[] = {%var_loop() %[d_%{vname}, ]% 0};
  char *      names[] = {%var_loop() %["%{vname}", ]% 0};
  cudaPitchedPtr srcPtr, dstPtr;
  cudaExtent cudaExt;
  cudaPos srcPos, dstPos;
  cudaMemcpy3DParms params;
  memset(&params, 0, sizeof(cudaMemcpy3DParms));

  for(int k = 0; var[k] != 0; k++){

    memset(&params, 0, sizeof(cudaMemcpy3DParms));

    srcPtr = make_cudaPitchedPtr(var[k], pitchx, pitchy, dimyg);
    dstPtr = make_cudaPitchedPtr(d_var[k], pitchx, pitchy, dimyg);

    for(int i = 0; i < 6; i++)
    {
      if(cctkGH->cctk_bbox[t[i]] != 1 && dimx_out[i] > 0 && dimy_out[i] > 0 && dimz_out[i] > 0){
        srcPos = make_cudaPos(offx_out[i], offy_out[i], offz_out[i]);
        dstPos = make_cudaPos(offx_out[i], offy_out[i], offz_out[i]);
        cudaExt= make_cudaExtent(dimx_out[i], dimy_out[i], dimz_out[i]);
        params.srcPtr = srcPtr; params.dstPtr = dstPtr;
        params.dstPos = dstPos; params.srcPos = srcPos;
        params.kind   = cudaMemcpyHostToDevice;
        params.extent = cudaExt;

        CUDA_SAFE_CALL(cudaMemcpy3D(&params));
      }
    }
  }
 
}


void CaCUDA_CopyFromDev_Boundary(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;
  long szp = sizeof(CCTK_REAL);
 

  // I USED THAT ORDER in former code; I didn't change it: N, E, F, S, W, B
  int t[] = {3, 1, 5, 2, 0, 4};
  int ghost_zones[6] = {0};
  for (int i = 0; i< 6; i++){
    if(cctkGH->cctk_bbox[t[i]] != 1) ghost_zones[i] = cctk_nghostzones[t[i] / 2]; 
  }	

  long  dimxg = cctk_lsh[0],
        dimyg = cctk_lsh[1],
        dimzg = cctk_lsh[2];
  long  dimxl = dimxg - ghost_zones[1] - ghost_zones[4],
        dimyl = dimyg - ghost_zones[0] - ghost_zones[3],
        dimzl = dimzg - ghost_zones[2] - ghost_zones[5];

/// local dimension without inner ghost-zone
  long //dimxn = dimxl - ghost_zones[1] - ghost_zones[0],
       dimyn = dimyl -  ghost_zones[0] - ghost_zones[3],
       dimzn = dimzl -  ghost_zones[2] - ghost_zones[5];

///  variables pointing to places in the array from/where the ghost zone should be copied // N, E, F, S, W, B
/// The data copy is desigined to minimize the amount of data copied unaligned (E and W ghost-zones).

  // I USED THAT ORDER in former code; I didn't change it: N, E, F, S, W, B
 
  long dimx_in[]  = {dimxl * szp, ghost_zones[1] * szp, dimxl * szp, dimxl * szp, ghost_zones[4] * szp, dimxl * szp};
  long dimy_in[]  = {ghost_zones[0], dimyn, dimyn, ghost_zones[3], dimyn, dimyn};
  long dimz_in[]  = {dimzl, dimzn, ghost_zones[2], dimzl, dimzn, ghost_zones[5]};

  long offx_in[] = {ghost_zones[4] * szp, (ghost_zones[4] + dimxl - ghost_zones[1]) * szp, ghost_zones[4] * szp, ghost_zones[4] * szp, ghost_zones[4] * szp, ghost_zones[4] * szp};
  long offy_in[] = {dimyl + ghost_zones[3] - ghost_zones[0], ghost_zones[3] + ghost_zones[3], ghost_zones[3] + ghost_zones[3], ghost_zones[3], ghost_zones[3] + ghost_zones[3], ghost_zones[3] + ghost_zones[3]};
  long offz_in[] = {ghost_zones[5], ghost_zones[5] + ghost_zones[5], ghost_zones[5] + dimzl - ghost_zones[2], ghost_zones[5], ghost_zones[5] + ghost_zones[5], ghost_zones[5]};

  long pitchx = dimxg * szp; 
  long pitchy = pitchx * dimyg;

  CCTK_REAL * var[]   = {%var_loop() %[%{vname}, ]% 0};
  CCTK_REAL * d_var[] = {%var_loop() %[d_%{vname}, ]% 0};
  char *      names[] = {%var_loop() %["%{vname}", ]% 0};
  cudaPitchedPtr srcPtr, dstPtr;
  cudaExtent cudaExt;
  cudaPos srcPos, dstPos;
  cudaMemcpy3DParms params;
  memset(&params, 0, sizeof(cudaMemcpy3DParms));

  for(int k = 0; var[k] != 0; k++){

    memset(&params, 0, sizeof(cudaMemcpy3DParms));

    dstPtr = make_cudaPitchedPtr(var[k], pitchx, pitchy, dimyg);
    srcPtr = make_cudaPitchedPtr(d_var[k], pitchx, pitchy, dimyg);

    for(int i = 0; i < 6; i++)
    {
      if(cctkGH->cctk_bbox[t[i]] != 1 && dimx_in[i] > 0 && dimy_in[i] > 0 && dimz_in[i] > 0){
        srcPos = make_cudaPos(offx_in[i], offy_in[i], offz_in[i]);
        dstPos = make_cudaPos(offx_in[i], offy_in[i], offz_in[i]);
        cudaExt= make_cudaExtent(dimx_in[i], dimy_in[i], dimz_in[i]);
        params.srcPtr = srcPtr; params.dstPtr = dstPtr;
        params.dstPos = dstPos; params.srcPos = srcPos;
        params.kind   = cudaMemcpyDeviceToHost;
        params.extent = cudaExt;

        CUDA_SAFE_CALL(cudaMemcpy3D(&params));
      }
    }
  }
  
}


