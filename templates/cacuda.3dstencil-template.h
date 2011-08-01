/*@@
 * @file    %{file}
 * @date    %{date}
 * @author  Marek Blazewicz
 * @desc
 * The prototype of the CaCUDA computational schema. It contains macros
 * which enable to declare, define and launch kernels as well as to copy
 * the data required for proper computations. The macros presented
 * in this file in the future will be automatically generated by the
 * Cactus parser depending on the input in interfaces.ccl file.
 * @enddesc
 * @version  $Header$
 *
 @@*/

#ifndef CCTK_CACUDA_%{name_upper}
#define CCTK_CACUDA_%{name_upper}

#include <algorithm>

/* definition of CCTK_REAL */
#include "cctk.h"

/* CaCUDAUtil.h shall be visible to all CaCUDA developers at some point */
#include "CaCUDA/CaCUDALib/src/CaCUDAUtil.h"

#include "cctk_Parameters.h"
#include "cctk_Arguments.h"

#ifdef __CUDACC__

/// !!!!!!!!!!!!! BEGIN of global definitions (not auto generated) !!!!!!!!!!!!!!!!

#define CACUDA_KERNEL_Threadsx %{tile_x}
#define CACUDA_KERNEL_Threadsy %{tile_y}
#define CACUDA_KERNEL_Threadsz 1

/* JT: 16x16x16 failed to compile on spider. 8x8x8 is ok to compile.
 * Let's make 8x8x8 as the default setting temporarily.
 * We will need to estimate the best configuration based on
 * the number of variables and the memory available.
 * */
#define CACUDA_KERNEL_Tilex %{tile_x}
#define CACUDA_KERNEL_Tiley %{tile_y}
#define CACUDA_KERNEL_Tilez %{tile_z}


/// !!!!!!!!!!!!! END of global definitions (not auto generated) !!!!!!!!!!!!!!!!

/// !!!!!!!!!!!!!!!!!!!!!!!!! BEGIN %{name} Kernel macors !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#ifdef stncl_xn
#error "You can't include two header file in one execution file"
#endif

#define stncl_xn %{stencil_xn}
#define stncl_xp %{stencil_xp}
#define stncl_yn %{stencil_yn}
#define stncl_yp %{stencil_yp}
#define stncl_zn %{stencil_zn}
#define stncl_zp %{stencil_zp}
                 
/** Element  */
#define CACUDA_KERNEL_GFINDEX3D_%{name}(ptr, i, j, k)               \
 ptr[(i + gi) + params.cagh_ni * ((j + gj) + params.cagh_nj * (k + gk))]

// indexing function; havn't been tested yet
template<int i, int j, int k, typename t, typename t2>
__device__ inline CCTK_REAL & cctk_cuda_gfindex3d_%{name}_lhelper
         (t ptr_sh, t2 ptr_v, const short & li, 
         const short & lj, const short & lk)
{
    if (k == 0)  return ptr_sh[j + lj][i + li];
    if (k > 0)   return ptr_v[k + stncl_zn - 1];
    return ptr_v[k + lk];
}

// Static derefernce of the variables 
#define CACUDA_KERNEL_GFINDEX3D_%{name}_l(ptr, i, j, k)             \
    cctk_cuda_gfindex3d_%{name}_lhelper<i, j, k>(ptr##_sh, ptr##_v, li, lj, lk)

// Dynamic derefernce of the variables (necessary to use this macro )
#  define CACUDA_KERNEL_GFINDEX3D_%{name}_ld(ptr, i, j, k)           \
    ((k == 0) ? ptr##_sh[j + lj][i + li] : (k > 0) ?                        \
    ptr##_v[k + lk - 1] : ptr##_v[k + lk])

#define I3D CACUDA_KERNEL_GFINDEX3D_%{name}
#define I3D_l CACUDA_KERNEL_GFINDEX3D_%{name}_l
#define I3D_ld CACUDA_KERNEL_GFINDEX3D_%{name}_ld

#define CACUDA_KERNEL_%{name}_Declare_Begin_s                               \
__global__ void CACUDA_KERNEL_%{name}(                                      \
/** Variables automatically added by the Cactus parser:  */                 \
%var_loop("intent=separateinout",'const CCTK_REAL *%{vname}, CCTK_REAL *%{vname}_out,')\
%var_loop("intent=inout", 'CCTK_REAL *%{vname},')\
%var_loop("intent=in", 'const CCTK_REAL *%{vname},')\
%var_loop("intent=out", 'CCTK_REAL *%{vname}_out,')\
/** Statically added variables to each kernel: */                           \
const CaCUDA_Kernel_Launch_Parameters params)                               \
{                                                                           
# define CACUDA_KERNEL_%{name}_Declare_Cached_Variables_s                   \
/** Kernel specific variables declaration. */                               \
  %var_loop("cached=yes",'__shared__ CCTK_REAL %{vname}_sh[CACUDA_KERNEL_Tiley][CACUDA_KERNEL_Tilex];')\
  %var_loop("cached=yes",'__shared__ CCTK_REAL %{vname}_v[stncl_zn + stncl_zp];')\
/** Common variables declaration; values are kernel specific. */            
# define CACUDA_KERNEL_%{name}_Declare_Flow_Variables_s                     \
  short li = threadIdx.x;                                                   \
  short lj = threadIdx.y;                                                   \
  short lk = threadIdx.z + stncl_zn;                                        \
  int gi = blockIdx.x * (CACUDA_KERNEL_Tilex - stncl_xn - stncl_xp) + li;   \
  int gj = (blockIdx.y % params.cagh_blocky) *                              \
          (CACUDA_KERNEL_Tiley - stncl_yn - stncl_yp) + lj;                 \
  int gk2= (blockIdx.y / params.cagh_blocky) *                              \
          (CACUDA_KERNEL_Tilez - stncl_zn - stncl_zp) + lk;                 \
  int gk = gk2;                                                             \
  bool fetch_data = gi < params.cagh_ni && gj < params.cagh_nj;             \
  bool compute = fetch_data && li >= stncl_xn && lj >= stncl_yn &&          \
    li < CACUDA_KERNEL_Tilex - stncl_xp &&                                  \
    lj < CACUDA_KERNEL_Tiley - stncl_yp;                                    \
  short tilez_to = min(CACUDA_KERNEL_Tilez - stncl_zp - stncl_zn,           \
                        params.cagh_nk - gk - stncl_zp);                    \
  short tmpi, tmpj;                                                         \
    /** Dynamically set fetching from global memory */                      
# define CACUDA_KERNEL_%{name}_Limit_Threads_To_LSH_Begin_s                 \
  if(fetch_data)                                                            \
  {                                                                         
#   define CACUDA_KERNEL_%{name}_Fetch_Data_To_Cache_s                      \
%for_loop(tmpi,'-%{stencil_zn}','%{stencil_zp}') %[                         \
      %var_loop("cached=yes") %[                                            \
    I3D_l(%vname, 0, 0, %var(tmpi) + 1) = I3D(%vname, 0, 0, %var(tmpi));]%]%\
                                                                            \
//    The loop is suppose to fetch the data from global to cached memory.   \
//    For cached variables only!                                            \
//    for (tmpi = -stncl_zn; tmpi < stncl_zp; tmpi++)                       \
//    {                                                                     \
//      I3D_l(vx, 0, 0, tmpi + 1) = I3D(vx, 0, 0, tmpi);                    \
//      I3D_l(vy, 0, 0, tmpi + 1) = I3D(vy, 0, 0, tmpi);                    \
//      I3D_l(vz, 0, 0, tmpi + 1) = I3D(vz, 0, 0, tmpi);                    \
//      I3D_l(p,  0, 0, tmpi + 1) = I3D(p,  0, 0, tmpi);                    \
//    }

#   define CACUDA_KERNEL_%{name}_Computations_Begin_s                       \
    for(tmpj = 0; tmpj < tilez_to; tmpj++)                                  \
    {                                                                       \
      __syncthreads();                                                      
#     define CACUDA_KERNEL_%{name}_Iterate_Local_Tile_s                     \
%for_loop(tmpi,'-%{stencil_zn}','%{stencil_zp}') %[                         \
      %var_loop("cached=yes") %[                                            \
  I3D_l(%vname, 0, 0, %var(tmpi)) = I3D_l(%vname, 0, 0, %var(tmpi) + 1);]%]%\
                                                                            \
      gk = gk2 + tmpj;                                                      
//    The loop is suppose to iterate local variables, as the tiles 'walks'  \
      through the z dimension. For cached variables only!                   \
      for (tmpi = -stncl_zn; tmpi < stncl_zp; tmpi++)                       \
      {                                                                     \
        I3D_l(vx, 0, 0, tmpi) = I3D_l(vx, 0, 0, tmpi + 1);                  \
        I3D_l(vy, 0, 0, tmpi) = I3D_l(vy, 0, 0, tmpi + 1);                  \
        I3D_l(vz, 0, 0, tmpi) = I3D_l(vz, 0, 0, tmpi + 1);                  \
      }                                                                   
#     define CACUDA_KERNEL_%{name}_Fetch_Front_Tile_To_Cache_s              \
      %var_loop("cached=yes") %[                                            \
        I3D_l(%vname, 0, 0, stncl_zp) = I3D(%vname, 0, 0, stncl_zp);]%      \
                                                                            \
      __syncthreads();                                                      
//      I3D_l(vx, 0, 0, stncl_zp) = I3D(vx, 0, 0, stncl_zp);                \
//      I3D_l(vy, 0, 0, stncl_zp) = I3D(vy, 0, 0, stncl_zp);                \
//      I3D_l(vz, 0, 0, stncl_zp) = I3D(vz, 0, 0, stncl_zp);                
#     define CACUDA_KERNEL_%{name}_Limit_Threads_To_Compute_Begin_s         \
      if(compute)                                                           \
      {                                                                     \
      /*if(threadIdx.x == 1 && threadIdx.y == 1)                            \
          printf("3cmpt [%02d, %02d, %02d]\n", gi, gj, gk);*/               \
         /** TODO Add your computations here */                             \
         /** TODO Store the results to global array ({...}_out)  */

#     define CACUDA_KERNEL_%{name}_Limit_Threads_To_Compute_End_s   \
      }                                                                     
#   define CACUDA_KERNEL_%{name}_Computations_End_s                 \
    }
# define CACUDA_KERNEL_%{name}_Limit_Threads_To_LSH_End_s           \
  }                                                                         
#define CACUDA_KERNEL_%{name}_Declare_End_s                         \
}


/* Declaration of the global function */
__global__ void CACUDA_KERNEL_%{name}(                                      
/** Variables automatically added by the Cactus parser:  */                 
%var_loop("intent=separateinout",'const CCTK_REAL *%{vname},CCTK_REAL *%{vname}_out,')
%var_loop("intent=inout",'CCTK_REAL *%{vname},')
%var_loop("intent=in",'const CCTK_REAL *%{vname},')
%var_loop("intent=out",'const CCTK_REAL *%{vname},')
/** Statically added variables to each kernel: */                           
const CaCUDA_Kernel_Launch_Parameters params);

%var_loop("intent=separateinout",'extern CCTK_REAL *d_%{vname},*d_%{vname}_out;\n')
%var_loop("intent=inout",'extern CCTK_REAL *d_%{vname};\n')
%var_loop("intent=in",'extern CCTK_REAL *d_%{vname};\n')
%var_loop("intent=out",'extern CCTK_REAL *d_%{vname}_out;\n')

inline void CACUDA_KERNEL_Launch_%{name}(CCTK_ARGUMENTS)
{
	DECLARE_CCTK_ARGUMENTS;

    const int blocky = iDivUp(cctk_lsh[1] - stncl_yn - stncl_yp,
                    CACUDA_KERNEL_Tiley - stncl_yn - stncl_yp);


    CaCUDA_Kernel_Launch_Parameters prms(cctk_iteration,
    		cctk_lsh[0], cctk_lsh[1], cctk_lsh[2],
    		cctk_nghostzones[0], cctk_nghostzones[1], cctk_nghostzones[2],
    		blocky,
            cctk_delta_space[0], cctk_delta_space[1], cctk_delta_space[2],
            cctk_delta_time,
            cctk_origin_space[0], cctk_origin_space[1], cctk_origin_space[2],
            cctk_time);

//  TODO If the seperate in & out are required the d_xxx_out variables need to be assigned to local xxx_out variables;


    CACUDA_KERNEL_%{name}<<<                                                    
 dim3(iDivUp(prms.cagh_ni - stncl_xn - stncl_xp, CACUDA_KERNEL_Tilex - stncl_xn - stncl_xp), 
      iDivUp(prms.cagh_nk - stncl_zn - stncl_zp, CACUDA_KERNEL_Tilez - stncl_zn - stncl_zp) 
          * blocky),
 dim3(CACUDA_KERNEL_Tilex, CACUDA_KERNEL_Threadsy, CACUDA_KERNEL_Threadsz)>>>(
    %var_loop("intent=separateinout",'d_%{vname},d_%{vname}_out,')
    %var_loop("intent=inout",'d_%{vname},') %var_loop("intent=in",'d_%{vname},')
    %var_loop("intent=out",'d_%{vname}_out,') prms);
//    cutilCheckMsg("failed while updating the velocity");
    CUDA_SAFE_CALL(cudaThreadSynchronize());

%var_loop("intent=separateinout",'std::swap(d_%{vname}, d_%{vname}_out);\n')

}

/// !!!!!!!!!!!!!!!!!!!!!!!!! END %{name} Kernel macors !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#endif

#endif /* %{name_upper} */
