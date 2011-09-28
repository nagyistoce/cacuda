#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"

void CaCUDACFD3D_Pressure_Solver_Iterator_Start(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  *psolver_step = num_psolver_steps;
}

void CaCUDACFD3D_Pressure_Solver_Iterate(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  (*psolver_step)--;
}

