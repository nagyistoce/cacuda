/*@@
 * @file    InitData.c
 * @date    Fri Jul 29, 2011
 * @author  Jian Tao
 * @desc
 *          Initial data for CaCUDACFD3D
 * @enddesc
 * @version  $Header$
 *
 @@*/

#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"

#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"

void CACUDACFD3D_Init_LDC( CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  const n = cctk_lsh[0] * cctk_lsh[1] * cctk_lsh[2];
  for (int i = 0; i < n; i++)
  {
    vx[i] = 0.0;
    vy[i] = 0.0;
    vz[i] = 0.0;
    p[i] = 0.0;
  }
}
