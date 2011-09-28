#ifndef _CACUDACONST_H_
#define _CACUDACONST_H_
#include "cctk.h"

/* enable/disable debugging information */
#ifndef CACUDADEBUG
#define CACUDADEBUG 1
#endif

/* tile size is the same as the block dim so that we have one grid point per thread */
#define BLOCKDIMX 16
#define BLOCKDIMY 16
#define BLOCKDIMZ 1

/* stencil size
 * 1. stencil size is half of the finite different order
 * 2. nghostzones size will be ckecked at paramcheck to make sure
 *    nghostzones >= STENCILESIZE
 * */
#define STENCILSIZE 1

#endif                          /* _CACUDACONST_H_ */
