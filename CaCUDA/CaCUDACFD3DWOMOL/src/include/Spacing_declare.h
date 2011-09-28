/*@@
  @header    Spacing_declare.h
  @date      Tue Feb 27 2007
  @author    Jian Tao
  @desc
  declare temporary variables for derivative calculations
  @enddesc
  @version $Header
  @@*/

/* the preprocessor should know which order you want to use when you
 * include this file*/

      CCTK_REAL dx, dy, dz;
      CCTK_REAL idx, idy, idz;

#if FD_CENTRAL_ORDER == 2

      CCTK_REAL i2dx, i2dy, i2dz;
      CCTK_REAL idxx, idyy, idzz, idxy, idxz, idyz;

#elif FD_CENTRAL_ORDER == 4

      CCTK_REAL i4dx, i4dy, i4dz;
      CCTK_REAL i4dxx, i4dyy, i4dzz, i4dxy, i4dxz, i4dyz;

#elif FD_CENTRAL_ORDER == 8

      CCTK_REAL i8dx, i8dy, i8dz;
      CCTK_REAL i8dxx, i8dyy, i8dzz, i8dxy, i8dxz, i8dyz;

#endif
