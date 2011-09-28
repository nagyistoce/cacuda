/*@@
  @header    Spacing.h
  @date      Tue Feb 27 2007
  @author    Jian Tao
  @desc 
  Please read the documentation of the thorn for more information
  @enddesc
  @version $Header
  @@*/


#ifndef _SPACING_H_
#define _SPACING_H_

/* here wo initialize the spacing with the cacuda parameters */

      dx=params.cagh_dx;
      dy=params.cagh_dy;
      dz=params.cagh_dz;

      idx=1.0/dx;
      idy=1.0/dy;
      idz=1.0/dz;

#if FD_CENTRAL_ORDER == 2
      
      i2dx=idx/2.0;
      i2dy=idy/2.0;
      i2dz=idz/2.0;

      idxx=idx*idx;
      idyy=idy*idy;
      idzz=idz*idz;

      idxy=i2dx*i2dy;
      idxz=i2dx*i2dz;
      idyz=i2dy*i2dz;


#elif FD_CENTRAL_ORDER == 4

      i4dx=idx/12.0;
      i4dy=idy/12.0;
      i4dz=idz/12.0;

      i4dxx=idx*i4dx;
      i4dyy=idy*i4dy;
      i4dzz=idz*i4dz;

      i4dxy=i4dx*i4dy;
      i4dxz=i4dx*i4dz;
      i4dyz=i4dy*i4dz;


#elif FD_CENTRAL_ORDER == 8

      i8dx=idx/840.0;
      i8dy=idy/840.0;
      i8dz=idz/840.0;

      i8dxx=idx*i8dx/6.0;
      i8dyy=idy*i8dy/6.0;
      i8dzz=idz*i8dz/6.0;

      i8dxy=i8dx*i8dy;
      i8dxz=i8dx*i8dz;
      i8dyz=i8dy*i8dz;

#endif /*FD_CENTRAL_ORDER*/
#endif /*_SPACING_H_*/
