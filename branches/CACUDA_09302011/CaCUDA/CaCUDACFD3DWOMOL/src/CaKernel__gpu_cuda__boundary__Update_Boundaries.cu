
#include "CaKernel__gpu_cuda__vars.h"

#include "CaKernel__gpu_cuda__boundary__Update_Boundaries.h"

#include "CaKernel__gpu_cuda__code__Update_Boundaries.code"



void CAKERNEL_Launch_Update_Boundaries(CCTK_ARGUMENTS)
{
    DECLARE_CCTK_ARGUMENTS;

#if CACTUS_DEBUG 
    size_t datasize = cctk_lsh[0] * cctk_lsh[1] * cctk_lsh[2] * sizeof(CCTK_REAL);
    
    static CCTK_REAL * tmp_buff = 0;
    if (!tmp_buff) tmp_buff = (CCTK_REAL *) malloc(datasize);

    cudaMemcpy(tmp_buff, d_vx, datasize, cudaMemcpyDeviceToHost);

    CCTK_VInfo(CCTK_THORNSTRING, "Printing the table before the boundary conditions");
    printPartOfTheTable3D(stdout, (CCTK_REAL *) tmp_buff, cctk_lsh[0] * sizeof(CCTK_REAL), cctk_lsh[0] * sizeof(CCTK_REAL) * cctk_lsh[1], (CCTK_REAL *)0, 0, 0, 0, 0, cctk_lsh[2] / 2, cctk_lsh[0], cctk_lsh[1], 1);
#endif /* CACTUS_DEBUG  */

    CaCUDA_Kernel_Launch_Parameters prms(cctk_iteration,
    cctk_lsh[0], cctk_lsh[1], cctk_lsh[2],
    cctk_nghostzones[0], cctk_nghostzones[1], cctk_nghostzones[2], 0,
            cctk_delta_space[0], cctk_delta_space[1], cctk_delta_space[2],
            cctk_delta_time,
            cctk_origin_space[0], cctk_origin_space[1], cctk_origin_space[2],
            cctk_time);

    CUDA_SAFE_CALL(cudaThreadSynchronize());

    if(cctkGH->cctk_bbox[1] == 1){
      CAKERNEL_Update_Boundaries<1,0,0><<<
dim3(iDivUp(prms.cagh_nj, CAKERNEL_Tiley), iDivUp(prms.cagh_nk, CAKERNEL_Tilez)),
dim3(CAKERNEL_Tiley, CAKERNEL_Tilez)>>>(  
          
          d_vx,d_vy,d_vz, 
           prms);
    }

    if(cctkGH->cctk_bbox[0] == 1){
    CAKERNEL_Update_Boundaries<-1,0,0><<<
dim3(iDivUp(prms.cagh_nj, CAKERNEL_Tiley), iDivUp(prms.cagh_nk, CAKERNEL_Tilez)),
dim3(CAKERNEL_Tiley, CAKERNEL_Tilez)>>>(  
          
          d_vx,d_vy,d_vz, 
           prms);
    }

    if(cctkGH->cctk_bbox[3] == 1){
    CAKERNEL_Update_Boundaries<0,1,0><<<
dim3(iDivUp(prms.cagh_ni, CAKERNEL_Tilex), iDivUp(prms.cagh_nk, CAKERNEL_Tilez)),
dim3(CAKERNEL_Tilex, CAKERNEL_Tilez)>>>(  
          
          d_vx,d_vy,d_vz, 
           prms);
    }

    if(cctkGH->cctk_bbox[2] == 1){
    CAKERNEL_Update_Boundaries<0,-1,0><<<
dim3(iDivUp(prms.cagh_ni, CAKERNEL_Tilex), iDivUp(prms.cagh_nk, CAKERNEL_Tilez)),
dim3(CAKERNEL_Tilex, CAKERNEL_Tilez)>>>(  
          
          d_vx,d_vy,d_vz, 
           prms);
    }

    if(cctkGH->cctk_bbox[5] == 1){
    CAKERNEL_Update_Boundaries<0,0,1><<<
dim3(iDivUp(prms.cagh_ni, CAKERNEL_Tilex), iDivUp(prms.cagh_nj, CAKERNEL_Tiley)),
dim3(CAKERNEL_Tilex, CAKERNEL_Tiley)>>>(  
          
          d_vx,d_vy,d_vz, 
           prms);
    }

    if(cctkGH->cctk_bbox[4] == 1){
    CAKERNEL_Update_Boundaries<0,0,-1><<<
dim3(iDivUp(prms.cagh_ni, CAKERNEL_Tilex), iDivUp(prms.cagh_nj, CAKERNEL_Tiley)),
dim3(CAKERNEL_Tilex, CAKERNEL_Tiley)>>>(  
          
          d_vx,d_vy,d_vz, 
           prms);
    }
//    cutilCheckMsg("failed while updating the velocity");
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    
    
#if CACTUS_DEBUG 
    cudaMemcpy(tmp_buff, d_vx, datasize, cudaMemcpyDeviceToHost);

    CCTK_VInfo(CCTK_THORNSTRING, "Printing the table after the boundary conditions");
    printPartOfTheTable3D(stdout, (CCTK_REAL *) tmp_buff, cctk_lsh[0] * sizeof(CCTK_REAL), cctk_lsh[0] * sizeof(CCTK_REAL) * cctk_lsh[1], (CCTK_REAL *)0, 0, 0, 0, 0, cctk_lsh[2] / 2, cctk_lsh[0], cctk_lsh[1], 1);
#endif /* CACTUS_DEBUG */
}

