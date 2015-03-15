## CaCUDA - Unleash the Power of Cactus on Hybrid Systems ##
### Introduction ###
We present a massive data parallel computational framework that can be used to develop large scale scientific applications on such petascale/exascale hybrid systems. The framework is built upon the highly scalable [Cactus computational framework](http://www.cactuscode.org) that has been used by scientists and engineers from various fields to solve a wide spectrum of real world scientific problems. Within Cactus, we further explore the performance and scalability of hybrid systems by making use of the computing power of the accelerating devices, in particular GPU's connected to many-core CPU's.
### Kernel Definition ###
The task of simplifying the generation of CUDA code for a finite differencing code is not a straightforward one. Shared arrays with appropriate stencil sizes have to be carefully managed, and data needed by the stencil has to be streamed in while calculations proceed. It is possible to abstract away much of the difficult work into boiler plate code, but doing so requires some extra machinery. We design and implement a programming abstraction in the Cactus framework to enable automatic generation from a set of highly optimized templates to simplify code construction.

```
CCTK_CUDA_KERNEL UPDATE_VELOCITY
   TYPE=3DBLOCK
   STENCIL="1,1,1,1,1,1"
   TILE="16,16,16"
{
  CCTK_CUDA_KERNEL_VARIABLE CACHED=YES INTENT=SEPARATEINOUT 
  {
    vx, vy, vz
  } "VELOCITY"
  CCTK_CUDA_KERNEL_VARIABLE CACHED=YES INTENT=IN
  {
    p
  } "PRESSURE"
  CCTK_CUDA_KERNEL_PARAMETER
  {
    density
  } "DENSITY"
}
```
### Stencil Computation Templates ###
CaCUDA Templates are a set of templates which are highly optimized for particular types of computational tasks and optimization strategies.

```
#   define CACUDA_KERNEL_%{name}_Computations_Begin_s                       \
    for(tmpj = 0; tmpj < tilez_to; tmpj++)                                  \
    {                                                                       \
      __syncthreads();                                                      
#     define CACUDA_KERNEL_%{name}_Iterate_Local_Tile_s                     \
      %for_loop(tmpi,'-%{stencil_zn}','%{stencil_zp}') %[                   \
      %var_loop("cached=yes") %[                                            \
  I3D_l(%vname, 0, 0, %var(tmpi)) = I3D_l(%vname, 0, 0, %var(tmpi) + 1);]%]%\
      gk = gk2 + tmpj;                                                      
#     define CACUDA_KERNEL_%{name}_Fetch_Front_Tile_To_Cache_s              \
      %var_loop("cached=yes") %[                                            \
          I3D_l(%vname, 0, 0, stncl_zp) = I3D(%vname, 0, 0, stncl_zp);]%    \
      __syncthreads();                                                      
#     define CACUDA_KERNEL_%{name}_Limit_Threads_To_Compute_Begin_s         \
      if(compute)                                                           \
      {                                                                     \
      /*if(threadIdx.x == 1 && threadIdx.y == 1)                            \
          printf("3cmpt [%02d, %02d, %02d]\n", gi, gj, gk);*/               \
         /** TODO Add your computations here */                             \
         /** TODO Store the results to global array ({...}_out)  */
#     define CACUDA_KERNEL_%{name}_Limit_Threads_To_Compute_End_s           \
      }
```

### Piraha-Based Kernel Parser and Code Generator ###
A [Piraha](http://code.google.com/p/piraha/)-based parser and code generator is used to
parse the descriptors and automatically generate CUDA-based macros.
Piraha implements a type of parsing expression grammar\cite{Ford2004}.
The grammar used to parse the whole CaCUDA kernel definitions is the following:
```
    g.compile("w", "([ \t\r\n]|#.*)*");
    g.compile("w1", "([ \t\r\n]|#.*)+");
    g.compile("KERNEL",
        "CCTK_CUDA_KERNEL{-w1}{name}{-w1}({key}{-w}={-w}{value}{-w})*\\{{-w}({VAR}{-w}|{PAR}{-w})*\\}{-w}");
    g.compile("KERNELS", "^{-w}({KERNEL})*$");
    g.compile("name", "[A-Za-z0-9_]+");
    g.compile("key", "{name}");
    g.compile("value", "{name}|{dquote}|{squote}");
    g.compile("dquote", "\"(\\\\[^]|[^\\\\\"])*\"");
    g.compile("squote", "'(\\\\[^]|[^\\\\'])*'");
    g.compile("VAR",
        "CCTK_CUDA_KERNEL_VARIABLE({-w1}({key}{-w}={-w}{value}{-w})*|)\\{{-w}{name}({-w},{-w}{name})*{-w}\\}{-w}{dquote}");
    g.compile("PAR",
        "CCTK_CUDA_KERNEL_PARAMETER({-w1}({key}{-w}={-w}{value}{-w})*|)\\{{-w}{name}({-w},{-w}{name})*{-w}\\}{-w}{dquote}");
    g.compile("digit", "[0-9]+");
    g.compile("any","[^]*");
    g.compile("par","{key}{-w}={-w}{any}");
```

### User's Numerical Kernel ###
A user will have to write the code to carry out computations on the given stencil for finite
difference calculations to solve particular problems with preferred algorithms. The macros
that loop around all the points that are necessary to carry out local computations are
in the automatically generated kernel header files.

A user's code will look like:
```
#include "cctk_CaCUDA_Update_Velocity_3DBlock.h"

CACUDA_KERNEL_Update_Velocity_Begin

/* users temporary variables */
    CCTK_REAL tmpf = 0, v_sum, v, va, vb;

    CACUDA_KERNEL_Update_Velocity_Computations_Begin

/* more user's code above */
        v = (I3D_l(vx,0,0,0) + I3D_l(vx,-1,0,0) + I3D_l(vx,0,1,0) + I3D_l(vx,-1,1,0)) / 4;
        va = (I3D_l(vy,1,0,0) - I3D_l(vy,0,0,0)) ;
        vb = (I3D_l(vy,0,0,0) - I3D_l(vy,-1,0,0)) ;
        tmpf = params.cagh_dx * 2 ;
        v_sum -= v / tmpf * (va + vb + COPYSIGN(/*alpha*/0.2, v) * (vb - va));
/* more user's code below */

    CACUDA_KERNEL_Update_Velocity_Computations_End

CACUDA_KERNEL_Update_Velocity_End
```
where I3D\_l are local stencil given in the shared memory and macros whose names prefixed with CACUDA are looping macros defined in the kernerl header file.