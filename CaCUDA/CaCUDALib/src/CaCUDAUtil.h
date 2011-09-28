#ifndef _CACUDAUTIL_H_
#define _CACUDAUTIL_H_

/* CaCUDAUtil.h shall be visible to all CaCUDA developers at some point */

#include "cctk.h"
#include <typeinfo>
#include <stdio.h>
#ifdef CCTK_MPI
#include "mpi.h"
#endif

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
extern "C"
{
#endif


#define CUDA_SAFE_CALL( call )                                             \
  {                                                                        \
    cudaError err = call;                                                  \
    if(err != cudaSuccess)                                                 \
    {                                                                      \
      CCTK_VWarn (CCTK_WARN_ABORT, __LINE__, __FILE__, CCTK_THORNSTRING,   \
            "CUDA error: %s", cudaGetErrorString( err));                   \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  }

#define CUDA_CHECK_LAST_CALL(msg)           __cutilCheckMsg(msg, __FILE__, __LINE__)

inline void __cutilCheckMsg(const char *errorMessage, const char *file,
    const int line)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "%s(%i) : cutilCheckMsg() CUTIL CUDA error : %s : %s.\n",
        file, line, errorMessage, cudaGetErrorString(err));
    exit(-1);
  }
  err = cudaThreadSynchronize();
  if (cudaSuccess != err)
  {
    fprintf(stderr,
        "%s(%i) : cutilCheckMsg cudaThreadSynchronize error: %s : %s.\n", file,
        line, errorMessage, cudaGetErrorString(err));
    exit(-1);
  }
}

#ifdef __CUDACC__
}
#endif

/* MPI Util */

#ifdef CCTK_MPI

#define MPI_SAFE_CALL( call )                                              \
  {                                                                        \
    int err = call;                                                        \
    if (err != MPI_SUCCESS)                                                \
    {                                                                      \
      char mpi_error_string[MPI_MAX_ERROR_STRING+1];                       \
      int resultlen;                                                       \
      MPI_Error_string (err, mpi_error_string, &resultlen);                \
      CCTK_VWarn (CCTK_WARN_ABORT, __LINE__, __FILE__, CCTK_THORNSTRING,   \
            "MPI error: %s", mpi_error_string);                            \
      exit(-1);                                                            \
     }                                                                     \
   }
#endif

/* Malloc Util */

#define MALLOC_SAFE_CALL( call )                                           \
  {                                                                        \
    void *err = call;                                                      \
    if(err == NULL)                                                        \
    {                                                                      \
      CCTK_VWarn (CCTK_WARN_ABORT, __LINE__, __FILE__, CCTK_THORNSTRING,   \
            "Malloc error: %s", "failed to allocate memory !");            \
      exit(-1);                                                            \
    }                                                                      \
  }

#ifdef CCTK_REAL_PRECISION_4
  #define COPYSIGN copysignf
#else
  #define COPYSIGN copysign
#endif

/* util functions used in Kerner launcher */

int inline iDivUp(int a, int b){
  return (a + b - 1) / b;
}


/* CaCUDA parameters to lauch CaCUDA kernel */

struct CaCUDA_Kernel_Launch_Parameters
{
  int cagh_it;
  int cagh_ni;
  int cagh_nj;
  int cagh_nk;
  int cagh_nghostsi;
  int cagh_nghostsj;
  int cagh_nghostsk;
  int cagh_blocky;
  CCTK_REAL cagh_dx;
  CCTK_REAL cagh_dy;
  CCTK_REAL cagh_dz;
  CCTK_REAL cagh_dt;
  CCTK_REAL cagh_xmin;
  CCTK_REAL cagh_ymin;
  CCTK_REAL cagh_zmin;
  CCTK_REAL cagh_time;
  CaCUDA_Kernel_Launch_Parameters ():cagh_it(0), cagh_ni(0), cagh_nj(0),
    cagh_nk (0), cagh_nghostsi (0), cagh_nghostsj (0), cagh_nghostsk (0), cagh_blocky(0),
    cagh_dx (0), cagh_dy (0), cagh_dz (0), cagh_dt (0), cagh_xmin (0), cagh_ymin (0),
    cagh_zmin (0), cagh_time (0)
  {
  }

  CaCUDA_Kernel_Launch_Parameters(int _cagh_it, int _cagh_ni, int _cagh_nj, int _cagh_nk,
          int _cagh_nghostsi, int _cagh_nghostsj, int _cagh_nghostsk, int _cagh_blocky,
          CCTK_REAL _cagh_dx, CCTK_REAL _cagh_dy, CCTK_REAL _cagh_dz, CCTK_REAL _cagh_dt,
          CCTK_REAL _cagh_xmin, CCTK_REAL _cagh_ymin, CCTK_REAL _cagh_zmin, CCTK_REAL _cagh_time) :
    cagh_it (_cagh_it), cagh_ni (_cagh_ni), cagh_nj (_cagh_nj), cagh_nk (_cagh_nk),
    cagh_nghostsi (_cagh_nghostsi), cagh_nghostsj (_cagh_nghostsj), cagh_nghostsk (_cagh_nghostsk),
    cagh_blocky(_cagh_blocky), cagh_dx (_cagh_dx), cagh_dy (_cagh_dy), cagh_dz (_cagh_dz),
    cagh_dt (_cagh_dt), cagh_xmin (_cagh_xmin), cagh_ymin (_cagh_ymin), cagh_zmin (_cagh_zmin),
    cagh_time (_cagh_time)
  {
  }
};
  struct P3 {
   CCTK_REAL x, y, z;
   inline P3(CCTK_REAL _x, CCTK_REAL _y, CCTK_REAL _z):x(_x), y(_y), z(_z){}
   inline P3():x(0.0f), y(0.0f), z(0.0f){}
 };
 typedef CCTK_REAL P;

#ifndef T3Dx
# define T3Dx(z, y, x) (((z) * DIMY + (y)) * DIMX + (x))
# define T3Dy(z, y, x) (T3Dx(z, y, x) + (DIMX * DIMY * DIMZ))
# define T3Dz(z, y, x) (T3Dx(z, y, x) + (DIMX * DIMY * DIMZ * 2))
#endif

#define T3Dx_val(ptr, z, y, x, pitchx, pitchy)             (((__typeof__(ptr))((char *)(ptr) + (z) * (pitchy)  + (y) * (pitchx)))[(x)])
#define T3Dy_val(ptr, z, y, x, pitchx, pitchy, next_table) (__typeof__(ptr)(((char *)&T3Dx_val(ptr, z, y, x, pitchx, pitchy)) + (next_table)))[0]
#define T3Dz_val(ptr, z, y, x, pitchx, pitchy, next_table) (__typeof__(ptr)(((char *)&T3Dx_val(ptr, z, y, x, pitchx, pitchy)) + (next_table) * 2))[0]

inline void printPartOfTheTable3Dxzy(FILE * f, P3 * tab, long pitchx, long pitchy, P3 * tab_ref, long pitchx_ref, long pitchy_ref, int startx, int starty, int startz, int offx, int offy, int offz, double thresh, int elemsize = 4)
{
	char infostring[50] = "[%8.2g,%8.2g,%8.2g]\t";
	char infostring2[50] = "%27d\t";
	elemsize /= sizeof(P3);


	for(int j = starty; j < starty + offy; j++){
		fprintf(f, "y= %d\n", j);
		fprintf(f, "z\\x \t");
		for(int i = startx; i < startx + offx; i++)
			fprintf(f, infostring2, i);
		fprintf(f, "\n");

		for(int k = startz + offz - 1; k >= startz; k--){
			fprintf(f, "%3d:\t", j);
			for(int i = startx; i < startx + offx; i++){
					fprintf(f, infostring, T3Dx_val(tab, k, j, i, pitchx, pitchy).x,
										   T3Dx_val(tab, k, j, i, pitchx, pitchy).y,
										   T3Dx_val(tab, k, j, i, pitchx, pitchy).z);
			}
			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}
}

inline void printPartOfTheTable3Dxyz(FILE * f, P3 * tab, long pitchx, long pitchy, P3 * tab_ref, long pitchx_ref, long pitchy_ref, int startx, int starty, int startz, int offx, int offy, int offz, double thresh, int elemsize = 4)
{
	char infostring[50] = "[%8.2g,%8.2g,%8.2g]\t";
	char infostring2[50] = "%27d\t";
	elemsize /= sizeof(P3);

	for(int k = startz; k < startz + offz; k++){
		fprintf(f, "z= %d\n", k);
		fprintf(f, "y\\x \t");
		for(int j = startx; j < startx + offx; j++)
			fprintf(f, infostring2, j);
		fprintf(f, "\n");

		for(int j = starty + offy - 1; j >= starty; j--){
			fprintf(f, "%3d:\t", j);
			for(int i = startx; i < startx + offx; i++){
					fprintf(f, infostring, T3Dx_val(tab, k, j, i, pitchx, pitchy).x,
										   T3Dx_val(tab, k, j, i, pitchx, pitchy).y,
										   T3Dx_val(tab, k, j, i, pitchx, pitchy).z);
			}
			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}
}

inline void printPartOfTheTable3Dyzx(FILE * f, P3 * tab, long pitchx, long pitchy, P3 * tab_ref, long pitchx_ref, long pitchy_ref, int startx, int starty, int startz, int offx, int offy, int offz, double thresh, int elemsize = 4)
{
	char infostring[50] = "[%8.2g,%8.2g,%8.2g]\t";
	char infostring2[50] = "%27d\t";
	elemsize /= sizeof(P3);

	for(int i = startx; i < startx + offx; i++){
		fprintf(f, "x= %d\n", i);
		fprintf(f, "y\\x \t");
		for(int j = startx; j < startx + offx; j++)
			fprintf(f, infostring2, j);
		fprintf(f, "\n");

		for(int j = starty + offy - 1; j >= starty; j--){
			fprintf(f, "%3d:\t", j);
			for(int k = startz; k < startz + offz; k++){
					fprintf(f, infostring, T3Dx_val(tab, k, j, i, pitchx, pitchy).x,
										   T3Dx_val(tab, k, j, i, pitchx, pitchy).y,
										   T3Dx_val(tab, k, j, i, pitchx, pitchy).z);
			}
			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}
}

template <class T>
inline void printPartOfTheTable3Dxyz(FILE * f, T * tab, long pitchx, long pitchy, T * tab_ref, long pitchx_ref, long pitchy_ref, int startx, int starty, int startz, int offx, int offy, int offz, double thresh, int elemsize = sizeof(P))
{
	char infostring[50] = {0};
	char infostring_c[50] = {0};
	char infostring2[50] = "%5d|\t";

	if(typeid(T) == typeid(P3)){
		throw "shit";
	}
	else if (typeid(T) == typeid(int) || typeid(T) == typeid(unsigned int)){
		strcpy(infostring,   "%5d\t");
		strcpy(infostring_c, "\x1b[1;31m%5d\x1b[1;0m\t");
	}
	else if (typeid(T) == typeid(double) || typeid(T) == typeid(float)){
		strcpy(infostring,   "%10e\t");
		strcpy(infostring_c, "\x1b[1;31m%10e\x1b[1;0m\t");
		strcpy(infostring2,"%10d\t");
	}
	else{
		strcpy(infostring,   "%10.7f\t");
		strcpy(infostring_c, "\x1b[1;31m%10.7f\x1b[1;0m\t");
		strcpy(infostring2,  "%10d\t");
	}

	elemsize /= sizeof(T);

	for(int k = startz; k < startz + offz; k++){
		fprintf(f, "z= %d\n", k);
		fprintf(f, "y\\x \t");
		for(int j = startx; j < startx + offx; j++)
			fprintf(f, infostring2, j);
		fprintf(f, "\n");

		for(int j = starty + offy - 1; j >= starty; j--){
			fprintf(f, "%3d:\t", j);
			for(int i = startx; i < startx + offx; i++){
				if (typeid(T) == typeid(double) || typeid(T) == typeid(float)){
					bool coloured = false;
					if(!tab_ref){
						if(abs(T3Dx_val(tab, k, j, i, pitchx, pitchy)) > thresh) coloured = true;
					}else{
						double res = abs((T3Dx_val(tab, k, j, i, pitchx, pitchy) - T3Dx_val(tab_ref, k, j, i, pitchx_ref, pitchy_ref)) / T3Dx_val(tab_ref, k, j, i, pitchx_ref, pitchy_ref));
						if(!isnan(res) && !isinf(res) && res > thresh)
							coloured = true;
					}
					if(coloured)fprintf(f, infostring, T3Dx_val(tab, k, j, i, pitchx, pitchy));
					else 		fprintf(f, infostring,   T3Dx_val(tab, k, j, i, pitchx, pitchy));
				}
				else{
					if(tab_ref && T3Dx_val(tab, k, j, i, pitchx, pitchy) != T3Dx_val(tab_ref, k, j, i, pitchx_ref, pitchy_ref) ||
							!tab_ref && T3Dx_val(tab, k, j, i, pitchx, pitchy) != 0)
						 fprintf(f, infostring, T3Dx_val(tab, k, j, i, pitchx, pitchy));
					else fprintf(f, infostring,   T3Dx_val(tab, k, j, i, pitchx, pitchy));
				}
			}

			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}
}

template <class T>
inline void printPartOfTheTable3Dxzy(FILE * f, T * tab, long pitchx, long pitchy, T * tab_ref, long pitchx_ref, long pitchy_ref, int startx, int starty, int startz, int offx, int offy, int offz, double thresh, int elemsize = sizeof(P))
{
	char infostring[50] = {0};
	char infostring_c[50] = {0};
	char infostring2[50] = "%5d|\t";

	if(typeid(T) == typeid(P3)){
		throw "shit";
	}
	else if (typeid(T) == typeid(int) || typeid(T) == typeid(unsigned int)){
		strcpy(infostring,   "%5d\t");
		strcpy(infostring_c, "\x1b[1;31m%5d\x1b[1;0m\t");
	}
	else if (typeid(T) == typeid(double) || typeid(T) == typeid(float)){
		strcpy(infostring,   "%10e\t");
		strcpy(infostring_c, "\x1b[1;31m%10e\x1b[1;0m\t");
		strcpy(infostring2,"%10d\t");
	}
	else{
		strcpy(infostring,   "%10.7f\t");
		strcpy(infostring_c, "\x1b[1;31m%10.7f\x1b[1;0m\t");
		strcpy(infostring2,  "%10d\t");
	}

	elemsize /= sizeof(T);

	for(int j = starty; j < starty + offy; j++){
		fprintf(f, "y= %d\n", j);
		fprintf(f, "z\\x \t");
		for(int i = startx; i < startx + offx; i++)
			fprintf(f, infostring2, i);
		fprintf(f, "\n");

		for(int k = startz + offz - 1; k >= startz; k--){
			fprintf(f, "%3d:\t", k);
			for(int i = startx; i < startx + offx; i++){
				if (typeid(T) == typeid(double) || typeid(T) == typeid(float)){
					bool coloured = false;
					if(!tab_ref){
						if(abs(T3Dx_val(tab, k, j, i, pitchx, pitchy)) > thresh) coloured = true;
					}else{
						double res = abs((T3Dx_val(tab, k, j, i, pitchx, pitchy) - T3Dx_val(tab_ref, k, j, i, pitchx_ref, pitchy_ref)) / T3Dx_val(tab_ref, k, j, i, pitchx_ref, pitchy_ref));
						if(!isnan(res) && !isinf(res) && res > thresh)
							coloured = true;
					}
					if(coloured)fprintf(f, infostring, T3Dx_val(tab, k, j, i, pitchx, pitchy));
					else 		fprintf(f, infostring,   T3Dx_val(tab, k, j, i, pitchx, pitchy));
				}
				else{
					if(tab_ref && T3Dx_val(tab, k, j, i, pitchx, pitchy) != T3Dx_val(tab_ref, k, j, i, pitchx_ref, pitchy_ref) ||
							!tab_ref && T3Dx_val(tab, k, j, i, pitchx, pitchy) != 0)
						 fprintf(f, infostring, T3Dx_val(tab, k, j, i, pitchx, pitchy));
					else fprintf(f, infostring,   T3Dx_val(tab, k, j, i, pitchx, pitchy));
				}
			}
			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}
}

template <class T>
inline void printPartOfTheTable3Dyzx(FILE * f, T * tab, long pitchx, long pitchy, T * tab_ref, long pitchx_ref, long pitchy_ref, int startx, int starty, int startz, int offx, int offy, int offz, double thresh, int elemsize = sizeof(P))
{
	char infostring[50] = {0};
	char infostring_c[50] = {0};
	char infostring2[50] = "%5d|\t";

	if(typeid(T) == typeid(P3)){
		throw "shit";
	}
	else if (typeid(T) == typeid(int) || typeid(T) == typeid(unsigned int)){
		strcpy(infostring,   "%5d\t");
		strcpy(infostring_c, "\x1b[1;31m%5d\x1b[1;0m\t");
	}
	else if (typeid(T) == typeid(double) || typeid(T) == typeid(float)){
		strcpy(infostring,   "%10e\t");
		strcpy(infostring_c, "\x1b[1;31m%10e\x1b[1;0m\t");
		strcpy(infostring2,"%10d\t");
	}
	else{
		strcpy(infostring,   "%10.7f\t");
		strcpy(infostring_c, "\x1b[1;31m%10.7f\x1b[1;0m\t");
		strcpy(infostring2,  "%10d\t");
	}
	elemsize /= sizeof(T);

	for(int i = startx; i < startx + offx; i++){
		fprintf(f, "x= %d\n", i);
		fprintf(f, "y\\z \t");
		for(int k = startz; k < startz + offz; k++)
			fprintf(f, infostring2, k);
		fprintf(f, "\n");

		for(int j = starty + offy - 1; j >= starty; j--){
			fprintf(f, "%3d:\t", j);
			for(int k = startz; k < startz + offz; k++){
				if (typeid(T) == typeid(double) || typeid(T) == typeid(float)){
					bool coloured = false;
					if(!tab_ref){
						if(abs(T3Dx_val(tab, k, j, i, pitchx, pitchy)) > thresh) coloured = true;
					}else{
						double res = abs((T3Dx_val(tab, k, j, i, pitchx, pitchy) - T3Dx_val(tab_ref, k, j, i, pitchx_ref, pitchy_ref)) / T3Dx_val(tab_ref, k, j, i, pitchx_ref, pitchy_ref));
						if(!isnan(res) && !isinf(res) && res > thresh)
							coloured = true;
					}
					if(coloured)fprintf(f, infostring, T3Dx_val(tab, k, j, i, pitchx, pitchy));
					else 		fprintf(f, infostring,   T3Dx_val(tab, k, j, i, pitchx, pitchy));
				}
				else{
					if(tab_ref && T3Dx_val(tab, k, j, i, pitchx, pitchy) != T3Dx_val(tab_ref, k, j, i, pitchx_ref, pitchy_ref) ||
							!tab_ref && T3Dx_val(tab, k, j, i, pitchx, pitchy) != 0)
						 fprintf(f, infostring, T3Dx_val(tab, k, j, i, pitchx, pitchy));
					else fprintf(f, infostring,   T3Dx_val(tab, k, j, i, pitchx, pitchy));
				}
			}
			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}
}

template <class T>
inline void printPartOfTheTable3D(FILE * f, T * tab, long pitchx, long pitchy, T * tab_ref, long pitchx_ref, long pitchy_ref, int startx, int starty, int startz, int lenx, int leny, int lenz, double thresh = 0.01, long elemsize = sizeof(T)){
	if(lenz < leny){
		if(lenz <= lenx)	printPartOfTheTable3Dxyz(f, tab, pitchx, pitchy, tab_ref, pitchx_ref, pitchy_ref, startx, starty, startz, lenx, leny, lenz, thresh, elemsize);
		else				printPartOfTheTable3Dyzx(f, tab, pitchx, pitchy, tab_ref, pitchx_ref, pitchy_ref, startx, starty, startz, lenx, leny, lenz, thresh, elemsize);
	}
	else
	{
		if(leny <= lenx)	printPartOfTheTable3Dxzy(f, tab, pitchx, pitchy, tab_ref, pitchx_ref, pitchy_ref, startx, starty, startz, lenx, leny, lenz, thresh, elemsize);
		else				printPartOfTheTable3Dyzx(f, tab, pitchx, pitchy, tab_ref, pitchx_ref, pitchy_ref, startx, starty, startz, lenx, leny, lenz, thresh, elemsize);
	}
}

#endif                          /* _CACUDAUTIL_H_ */
