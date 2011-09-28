#include <stdio.h>
#include <math.h>
#include "cctk.h"
#include "cctk_Parameters.h"

#include "CaCUDALib.h"
#include "CaCUDAUtil.h"

#ifdef __CUDACC__
#  include <cuda.h>
#  include <cuda_runtime.h>
#endif

/* register variables */

CCTK_INT CaCUDALib_RegVar(const char * varname)
{

  DECLARE_CCTK_PARAMETERS
  CCTK_INT idx, varused;
  CCTK_INT numtimelevs;
  CCTK_INT varidx;

  varidx = CCTK_VarIndex(varname);

#ifdef CACUDADEBUG
    CCTK_INFO("register CaCUDA variables");
    printf("\tthe index is %d and the name is %s\n", varidx, varname);
#endif

  if ( varidx < 0 )
  {
    CCTK_WARN(0, "variable index is smaller than zero !");
  }

  numtimelevs = CCTK_MaxTimeLevelsVI(varidx);

  if ( numtimelevs < 0 )
  {
    CCTK_WARN(0, "variable registration failed in CaCUDALib !");
  }

  if ( numtimelevs < 2 )
  {
    CCTK_WARN(0, "the GF needs to have at least two time levels");
  }

  varused = 0;

  for (idx = 0; (idx < num_var_reg)&&(!varused); idx++)
  {
    varused = (varidx == idx_regvar[idx]);


#ifdef CACUDADEBUG
    printf("\tchecking idx %d which is %d\n", idx, idx_regvar[idx]);
#endif

  }

  if (varused)
  {

    CCTK_VWarn(2,__LINE__,__FILE__,"CaCUDALib",
               "the variable %s has already been registered ",
               varname);
  }
  else
  {

    if (num_var_reg+1 > num_dev_vars)
    {
      CCTK_WARN(0,"you have tried to register more device "
                "variables than the accumulator parameter "
                "num_dev_vars allows.");
    }

    idx_regvar[num_var_reg] = varidx;

    num_var_reg ++;

#ifdef CACUDADEBUG
    printf("\tthe max device variable number is now %d. Just added %d (%s).\n",
           num_var_reg, varidx, varname);
#endif

  }

  return num_var_reg;
}

/* register groups
 * only double precision is considered */

CCTK_INT CaCUDALib_RegGrp(const char * grpname)
{

 CCTK_INT grpidx, firstvar, nvar, idx;
 CCTK_INT retval = 0;

 grpidx = CCTK_GroupIndex(grpname);

 if ( grpidx < 0 )
 {
   CCTK_WARN(0, "variable index is smaller than zero !");
 }

 firstvar = CCTK_FirstVarIndexI(grpidx);

 if (firstvar < 0)
 {
   CCTK_VWarn(0, __LINE__, __FILE__, CCTK_THORNSTRING,
              "Evolved group idx %i is not a real group idx.",
             grpidx);
 }

 nvar = CCTK_NumVarsInGroupI(grpidx);

 for (idx = firstvar; idx < firstvar + nvar; idx++)
 {
   retval += CaCUDALib_RegVar(CCTK_VarName(idx));
 }

 return retval;
}
