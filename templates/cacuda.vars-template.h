#ifndef %{name_upper}
#define %{name_upper}

/* definition of CCTK_REAL */
#include "cctk.h"

#ifdef __CUDACC__

/* device pointers */
/* All the external variables will be declared here.
 * We will generate the this file from the parser directly but we
 * need to generate the memory allocation routines as well.
 */

%var_loop() %[
extern const CCTK_REAL * d_%{vname};
extern CCTK_REAL * d_%{vname}_out;
]%

#endif

#endif /* %{name_upper} */
