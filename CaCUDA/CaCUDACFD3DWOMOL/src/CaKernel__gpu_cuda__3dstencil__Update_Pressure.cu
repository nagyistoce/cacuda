
#include "CaKernel__gpu_cuda__vars.h"

#include "CaKernel__gpu_cuda__3dstencil__Update_Pressure.h"

#include "CaKernel__gpu_cuda__code__Update_Pressure.code"


void CAKERNEL_Launch_Update_Pressure(CCTK_ARGUMENTS)
{
	DECLARE_CCTK_ARGUMENTS;    
    size_t datasize = cctk_lsh[0] * cctk_lsh[1] * cctk_lsh[2] * sizeof(CCTK_REAL);
#if CACTUS_DEBUG    
    static CCTK_REAL * tmp_buff = 0;
    if (!tmp_buff) tmp_buff = (CCTK_REAL *) malloc(datasize);

    cudaMemcpy(tmp_buff, d_vx, datasize, cudaMemcpyDeviceToHost);
    CCTK_VInfo(CCTK_THORNSTRING, "Printing the velocity from the GPU before pressure updating");
    printPartOfTheTable3D(stdout, (CCTK_REAL *) tmp_buff, cctk_lsh[0] * sizeof(CCTK_REAL), cctk_lsh[0] * sizeof(CCTK_REAL) * cctk_lsh[1], (CCTK_REAL *)0, 0, 0, 0, 0, cctk_lsh[2] / 2, cctk_lsh[0], cctk_lsh[1], 1);

    cudaMemcpy(tmp_buff, d_p, datasize, cudaMemcpyDeviceToHost);
    CCTK_VInfo(CCTK_THORNSTRING, "Printing the pressure from the GPU before pressure updating");
    printPartOfTheTable3D(stdout, (CCTK_REAL *) tmp_buff, cctk_lsh[0] * sizeof(CCTK_REAL), cctk_lsh[0] * sizeof(CCTK_REAL) * cctk_lsh[1], (CCTK_REAL *)0, 0, 0, 0, 0, cctk_lsh[2] / 2, cctk_lsh[0], cctk_lsh[1], 1);
    CCTK_VInfo(CCTK_THORNSTRING, "Updating PRESSURE");
#endif

     cudaMemset(d_vx_out, 0, datasize);
cudaMemset(d_vy_out, 0, datasize);
cudaMemset(d_vz_out, 0, datasize);
;

    const int blocky = iDivUp(cctk_lsh[1] - stncl_yn - stncl_yp,
                    CAKERNEL_Tiley - stncl_yn - stncl_yp);


    CaCUDA_Kernel_Launch_Parameters prms(cctk_iteration,
    		cctk_lsh[0], cctk_lsh[1], cctk_lsh[2],
    		cctk_nghostzones[0], cctk_nghostzones[1], cctk_nghostzones[2],
    		blocky,
            cctk_delta_space[0], cctk_delta_space[1], cctk_delta_space[2],
            cctk_delta_time,
            cctk_origin_space[0], cctk_origin_space[1], cctk_origin_space[2],
            cctk_time);


    CAKERNEL_Update_Pressure<<<                                                    
 dim3(iDivUp(prms.cagh_ni - stncl_xn - stncl_xp, CAKERNEL_Tilex - stncl_xn - stncl_xp), 
      iDivUp(prms.cagh_nk - stncl_zn - stncl_zp, CAKERNEL_Tilez - stncl_zn - stncl_zp) 
          * blocky),
 dim3(CAKERNEL_Tilex, CAKERNEL_Threadsy, CAKERNEL_Threadsz)>>>(
    d_vx,d_vx_out,d_vy,d_vy_out,d_vz,d_vz_out,
    d_p, 
     prms);
//    cutilCheckMsg("failed while updating the velocity");
    CUDA_SAFE_CALL(cudaThreadSynchronize());

std::swap(d_vx, d_vx_out);
std::swap(d_vy, d_vy_out);
std::swap(d_vz, d_vz_out);


}
