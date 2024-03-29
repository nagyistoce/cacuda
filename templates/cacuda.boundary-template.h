/*@@
 * @file    %{file}
 * @date    %{date}
 * @author  Marek Blazewicz and Steven R. Brandt
 * @desc
 * WARNING: Automatically Generated. Do not modify.
 * The prototype of the CaCUDA computational schema. It contains macros
 * which enable to declare, define and launch kernels as well as to copy
 * the data required for proper computations. The macros presented
 * in this file in the future will be automatically generated by the
 * Cactus parser depending on the input in interfaces.ccl file.
 * @enddesc
 * @version  $Header$
 *
 @@*/

#ifndef %{name_upper}
#define %{name_upper}

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

/// !!!!!!!!!!!!!!!!!!!!!!!!! BEGIN Update Pressure Kernel macors !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef stncl_xn
#error "You can't include two header file in one execution file"
#endif

#define stncl_xn %{stencil_xn}
#define stncl_xp %{stencil_xp}
#define stncl_yn %{stencil_yn}
#define stncl_yp %{stencil_yp}
#define stncl_zn %{stencil_xn}
#define stncl_zp %{stencil_xp}

/** Element  */
#define CACUDA_KERNEL_GFINDEX3D_%{name}(ptr, i, j, k)                       \
 ptr[(i + gi) + params.cagh_ni * ((j + gj) + params.cagh_nj * (k + gk))]

#define I3D CACUDA_KERNEL_GFINDEX3D_%{name}          

/** The '_s' postfix means that the macro is more 'specific'. The user gets
more control to the compilation process, by for example skipping some of the
macros. But it requires a greater knowledge of what exactly is done inside
these macros (advanced programmers only - like Jian) */
#define CACUDA_KERNEL_%{name}_Declare_Begin_s                     \
template<short x, short y, short z>                                         \
__global__ void CACUDA_KERNEL_%{name}(                            \
/** Variables automatically added by the Cactus parser:  */                 \
%var_loop("intent=separatedinout",'/*sep in out */const CCTK_REAL *%{vname}, CCTK_REAL *%{vname}_out,')\
%var_loop("intent=inout",'/*inout*/CCTK_REAL *%{vname},')                    \
%var_loop("intent=in",'/*in*/const CCTK_REAL *%{vname},')                    \
%var_loop("intent=out",'/*in*/const CCTK_REAL *%{vname}_out,')              \
/** Statically added variables to each kernel: */                           \
const CaCUDA_Kernel_Launch_Parameters params)                               \
{                                                                           
# define CACUDA_KERNEL_%{name}_Declare_Cached_Variables_s         
# define CACUDA_KERNEL_%{name}_Declare_Flow_Variables_s                            \
/** Kernel specific variables declaration. */                               \
/** Common variables declaration; values are kernel specific. */            \
  int gi, gj, gk;                                                           \
  if(x){                                                                    \
    gj = blockIdx.x * blockDim.y + threadIdx.y;                             \
    gk = blockIdx.y * blockDim.z + threadIdx.z;                             \
    if(x > 0) gi = params.cagh_ni - 1;                                      \
    else      gi = 0;                                                       \
  }                                                                         \
  if(y){                                                                    \
    gi = blockIdx.x * blockDim.y + threadIdx.y;                             \
    gk = blockIdx.y * blockDim.z + threadIdx.z;                             \
    if(y > 0) gj = params.cagh_nj - 1;                                      \
    else      gj = 0;                                                       \
  }                                                                         \
  if(z){                                                                    \
    gi = blockIdx.x * blockDim.y + threadIdx.y;                             \
    gj = blockIdx.y * blockDim.z + threadIdx.z;                             \
    if(z > 0) gk = params.cagh_nk - 1;                                      \
    else      gk = 0;                                                       \
  }                                                                         \
  bool fetch_data = gi < params.cagh_ni && gj < params.cagh_nj              \
                    && gj < params.cagh_nj;                                 \
  short tilez_to = min(CACUDA_KERNEL_Tilez - stncl_zp - stncl_zn,           \
                        params.cagh_nk - gk - stncl_zp);                    
# define CACUDA_KERNEL_%{name}_Limit_Threads_To_LSH_Begin_s       \
  if(fetch_data)                                                            \
  {                                                                         
#   define CACUDA_KERNEL_%{name}_Computations_Begin_s               \
    
# define CACUDA_KERNEL_%{name}_Limit_Threads_To_LSH_End_s           \
  }                                                                         
#define CACUDA_KERNEL_%{name}_Declare_End_s                         \
}


#define CACUDA_KERNEL_%{name}_Begin                                 \
CACUDA_KERNEL_%{name}_Declare_Begin_s                               \
  CACUDA_KERNEL_%{name}_Declare_Cached_Variables_s                  \
  CACUDA_KERNEL_%{name}_Declare_Flow_Variables_s                    \
  CACUDA_KERNEL_%{name}_Limit_Threads_To_LSH_Begin_s                \

#define CACUDA_KERNEL_%{name}_End                                 \
  CACUDA_KERNEL_%{name}_Limit_Threads_To_LSH_End_s                \
CACUDA_KERNEL_%{name}_Declare_End_s

template<short x, short y, short z> 
__global__ void CACUDA_KERNEL_%{name}(                            
%var_loop("intent=separatedinout",'/*sep in out */const CCTK_REAL *%{vname}, CCTK_REAL *%{vname}_out,')
%var_loop("intent=inout",'/*inout*/CCTK_REAL *%{vname},')
%var_loop("intent=in",'/*in*/const CCTK_REAL *%{vname},')
%var_loop("intent=out",'/*in*/const CCTK_REAL *%{vname}_out,')
const CaCUDA_Kernel_Launch_Parameters params);                               

%var_loop("intent=separateinout",'extern CCTK_REAL *d_%{vname},*%{vname}_out;\n')
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

    CUDA_SAFE_CALL(cudaThreadSynchronize());

    if(cctkGH->cctk_bbox[1] == 1){
      CACUDA_KERNEL_%{name}<1,0,0><<<
dim3(iDivUp(prms.cagh_nj, CACUDA_KERNEL_Tiley), iDivUp(prms.cagh_nk, CACUDA_KERNEL_Tilez)),
dim3(1, CACUDA_KERNEL_Tiley, CACUDA_KERNEL_Tilez)>>>(  
          %var_loop("intent=separatedinout",'d_%{vname},d_%{vname}_out,')
          %var_loop("intent=inout",'d_%{vname},') %var_loop("intent=in",'d_%{vname},')
          %var_loop("intent=out",'d_%{vname},') prms);
    }

    if(cctkGH->cctk_bbox[0] == 1){
    CACUDA_KERNEL_%{name}<-1,0,0><<<
dim3(iDivUp(prms.cagh_nj, CACUDA_KERNEL_Tiley), iDivUp(prms.cagh_nk, CACUDA_KERNEL_Tilez)),
dim3(1, CACUDA_KERNEL_Tiley, CACUDA_KERNEL_Tilez)>>>(  
          %var_loop("intent=separatedinout",'d_%{vname},d_%{vname}_out,')
          %var_loop("intent=inout",'d_%{vname},') %var_loop("intent=in",'d_%{vname},')
          %var_loop("intent=out",'d_%{vname},') prms);
    }

    if(cctkGH->cctk_bbox[3] == 1){
    CACUDA_KERNEL_%{name}<0,1,0><<<
dim3(iDivUp(prms.cagh_ni, CACUDA_KERNEL_Tilex), iDivUp(prms.cagh_nk, CACUDA_KERNEL_Tilez)),
dim3(CACUDA_KERNEL_Tilex, 1, CACUDA_KERNEL_Tilez)>>>(  
          %var_loop("intent=separatedinout",'d_%{vname},d_%{vname}_out,')
          %var_loop("intent=inout",'d_%{vname},') %var_loop("intent=in",'d_%{vname},')
          %var_loop("intent=out",'d_%{vname},') prms);
    }

    if(cctkGH->cctk_bbox[2] == 1){
    CACUDA_KERNEL_%{name}<0,-1,0><<<
dim3(iDivUp(prms.cagh_ni, CACUDA_KERNEL_Tilex), iDivUp(prms.cagh_nk, CACUDA_KERNEL_Tilez)),
dim3(CACUDA_KERNEL_Tilex, 1, CACUDA_KERNEL_Tilez)>>>(  
          %var_loop("intent=separatedinout",'d_%{vname},d_%{vname}_out,')
          %var_loop("intent=inout",'d_%{vname},') %var_loop("intent=in",'d_%{vname},')
          %var_loop("intent=out",'d_%{vname},') prms);
    }

    if(cctkGH->cctk_bbox[5] == 1){
    CACUDA_KERNEL_%{name}<0,0,1><<<
dim3(iDivUp(prms.cagh_ni, CACUDA_KERNEL_Tilex), iDivUp(prms.cagh_nj, CACUDA_KERNEL_Tiley)),
dim3(CACUDA_KERNEL_Tilex, CACUDA_KERNEL_Tiley, 1)>>>(  
          %var_loop("intent=separatedinout",'d_%{vname},d_%{vname}_out,')
          %var_loop("intent=inout",'d_%{vname},') %var_loop("intent=in",'d_%{vname},')
          %var_loop("intent=out",'d_%{vname},') prms);
    }

    if(cctkGH->cctk_bbox[4] == 1){
    CACUDA_KERNEL_%{name}<0,0,-1><<<
dim3(iDivUp(prms.cagh_ni, CACUDA_KERNEL_Tilex), iDivUp(prms.cagh_nj, CACUDA_KERNEL_Tiley)),
dim3(CACUDA_KERNEL_Tilex, CACUDA_KERNEL_Tiley, 1)>>>(  
          %var_loop("intent=separatedinout",'d_%{vname},d_%{vname}_out,')
          %var_loop("intent=inout",'d_%{vname},') %var_loop("intent=in",'d_%{vname},')
          %var_loop("intent=out",'d_%{vname},') prms);
    }
//    cutilCheckMsg("failed while updating the velocity");
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    %var_loop("intent=separateinout",'std::swap(d_%{vname}, d_%{vname}_out);')
}

/// !!!!!!!!!!!!!!!!!!!!!!!!! END Update Pressure Kernel macors !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#endif

#endif /* CFD3D_KERNELS_Update_Boundaries_H_ */
