/*@@
  @header    Derivative.h
  @date      Tue Feb 27 2007
  @author    Jian Tao
  @desc
  Please read the documentation of the thorn for more information
  @enddesc
  @version $Header
  @@*/

#ifndef _DERIVATIVE_H_
#define _DERIVATIVE_H_

/* the preprocessor should know which order you want to use when you
 * include this file*/

#if FD_CENTRAL_ORDER == 2

#define DX(var)   i2dx*(I3D_l(var,+1,0,0) - I3D_l(var,-1,0,0))
#define DY(var)   i2dy*(I3D_l(var,0,+1,0) - I3D_l(var,0,-1,0))
#define DZ(var)   i2dz*(I3D_l(var,0,0,+1) - I3D_l(var,0,0,-1))

#define DXX(var)  idxx*(I3D_l(var,+1,0,0) - 2.0*I3D_l(var,0,0,0) \
                            + I3D_l(var,-1,0,0))
#define DYY(var)  idyy*(I3D_l(var,0,+1,0) - 2.0*I3D_l(var,0,0,0) \
                            + I3D_l(var,0,-1,0))
#define DZZ(var)  idzz*(I3D_l(var,0,0,+1) - 2.0*I3D_l(var,0,0,0) \
                            + I3D_l(var,0,0,-1))

#define DXY(var)  idxy*(I3D_l(var,1,1,0) - I3D_l(var,1,-1,0) \
                            - I3D_l(var,-1,1,0) + I3D_l(var,-1,-1,0))
#define DXZ(var)  idxz*(I3D_l(var,1,0,1) - I3D_l(var,1,0,-1) \
                            - I3D_l(var,-1,0,1) + I3D_l(var,-1,0,-1))
#define DYZ(var)  idyz*(I3D_l(var,0,1,1) - I3D_l(var,0,1,-1) \
                            - I3D_l(var,0,-1,1) + I3D_l(var,0,-1,-1))

#elif FD_CENTRAL_ORDER == 4

#define DX(var)   i4dx*(-I3D_l(var,2,0,0) + I3D_l(var,-2,0,0) \
                  + 8.0*(I3D_l(var,1,0,0) - I3D_l(var,-1,0,0)))
#define DY(var)   i4dy*(-I3D_l(var,0,2,0) + I3D_l(var,0,-2,0) \
                  + 8.0*(I3D_l(var,0,1,0) - I3D_l(var,0,-1,0)))
#define DZ(var)   i4dz*(-I3D_l(var,0,0,2) + I3D_l(var,0,0,-2) \
                  + 8.0*(I3D_l(var,0,0,1) - I3D_l(var,0,0,-1)))

#define DXX(var)  i4dxx*(-I3D_l(var,2,0,0) - I3D_l(var,-2,0,0) \
                  + 16.0*(I3D_l(var,1,0,0) + I3D_l(var,-1,0,0)) \
                  - 30.0*I3D_l(var,0,0,0))
#define DYY(var)  i4dyy*(-I3D_l(var,0,2,0) - I3D_l(var,0,-2,0) \
                  + 16.0*(I3D_l(var,0,1,0) + I3D_l(var,0,-1,0)) \
                  - 30.0*I3D_l(var,0,0,0))
#define DZZ(var)  i4dzz*(-I3D_l(var,0,0,2) - I3D_l(var,0,0,-2) \
                  + 16.0*(I3D_l(var,0,0,1) + I3D_l(var,0,0,-1)) \
                  - 30.0*I3D_l(var,0,0,0))

#define DXY(var)  i4dxy* \
        (I3D_l(var,2,2,0) - I3D_l(var,2,-2,0) \
       - I3D_l(var,-2,2,0) + I3D_l(var,-2,-2,0) \
 + 8.0*(-I3D_l(var,2,1,0) + I3D_l(var,2,-1,0) \
       - I3D_l(var,1,2,0) + I3D_l(var,1,-2,0) \
        +I3D_l(var,-2,1,0) - I3D_l(var,-2,-1,0) \
       - I3D_l(var,-1,-2,0) + I3D_l(var,-1,2,0)) \
 + 64.0*(I3D_l(var,1,1,0) - I3D_l(var,1,-1,0) \
       - I3D_l(var,-1,1,0) + I3D_l(var,-1,-1,0)))

#define DXZ(var)  i4dxz* \
        (I3D_l(var,2,0,2) - I3D_l(var,2,0,-2) \
       - I3D_l(var,-2,0,2) + I3D_l(var,-2,0,-2) \
 + 8.0*(-I3D_l(var,2,0,1) + I3D_l(var,2,0,-1) \
       - I3D_l(var,1,0,2) + I3D_l(var,1,0,-2) \
        +I3D_l(var,-2,0,1) - I3D_l(var,-2,0,-1) \
       - I3D_l(var,-1,0,-2) + I3D_l(var,-1,0,2)) \
 + 64.0*(I3D_l(var,1,0,1) - I3D_l(var,1,0,-1) \
       - I3D_l(var,-1,0,1) + I3D_l(var,-1,0,-1)))

#define DYZ(var)  i4dyz* \
        (I3D_l(var,0,2,2) - I3D_l(var,0,2,-2) \
       - I3D_l(var,0,-2,2) + I3D_l(var,0,-2,-2) \
 + 8.0*(-I3D_l(var,0,2,1) + I3D_l(var,0,2,-1) \
       - I3D_l(var,0,1,2) + I3D_l(var,0,1,-2) \
        +I3D_l(var,0,-2,1) - I3D_l(var,0,-2,-1) \
       - I3D_l(var,0,-1,-2) + I3D_l(var,0,-1,2)) \
 + 64.0*(I3D_l(var,0,1,1) - I3D_l(var,0,1,-1) \
       - I3D_l(var,0,-1,1) + I3D_l(var,0,-1,-1)))

#elif FD_CENTRAL_ORDER == 8

#define DX(var) i8dx* \
        (  3.0*(I3D_l(var,-4,0,0) - I3D_l(var,4,0,0)) \
        - 32.0*(I3D_l(var,-3,0,0) - I3D_l(var,3,0,0)) \
        +168.0*(I3D_l(var,-2,0,0) - I3D_l(var,2,0,0)) \
        -672.0*(I3D_l(var,-1,0,0) - I3D_l(var,1,0,0)))
          
#define DY(var) i8dy* \
        (  3.0*(I3D_l(var,0,-4,0) - I3D_l(var,0,4,0)) \
        - 32.0*(I3D_l(var,0,-3,0) - I3D_l(var,0,3,0)) \
        +168.0*(I3D_l(var,0,-2,0) - I3D_l(var,0,2,0)) \
        -672.0*(I3D_l(var,0,-1,0) - I3D_l(var,0,1,0)))
        
#define DZ(var) i8dz* \
        (  3.0*(I3D_l(var,0,0,-4) - I3D_l(var,0,0,4)) \
        - 32.0*(I3D_l(var,0,0,-3) - I3D_l(var,0,0,3)) \
        +168.0*(I3D_l(var,0,0,-2) - I3D_l(var,0,0,2)) \
        -672.0*(I3D_l(var,0,0,-1) - I3D_l(var,0,0,1)))


#define DXX(var) i8dxx* \
        (-   9.0*(I3D_l(var,-4,0,0) + I3D_l(var,4,0,0)) \
        +  128.0*(I3D_l(var,-3,0,0) + I3D_l(var,3,0,0)) \
        - 1008.0*(I3D_l(var,-2,0,0) + I3D_l(var,2,0,0)) \
        + 8064.0*(I3D_l(var,-1,0,0) + I3D_l(var,1,0,0)) \
        -14350.0*I3D_l(var,0,0,0))

#define DYY(var) i8dyy* \
        (-   9.0*(I3D_l(var,0,-4,0) + I3D_l(var,0,4,0)) \
        +  128.0*(I3D_l(var,0,-3,0) + I3D_l(var,0,3,0)) \
        - 1008.0*(I3D_l(var,0,-2,0) + I3D_l(var,0,2,0)) \
        + 8064.0*(I3D_l(var,0,-1,0) + I3D_l(var,0,1,0)) \
        -14350.0*I3D_l(var,0,0,0))

#define DZZ(var) i8dzz* \
        (-   9.0*(I3D_l(var,0,0,-4) + I3D_l(var,0,0,4)) \
        +  128.0*(I3D_l(var,0,0,-3) + I3D_l(var,0,0,3)) \
        - 1008.0*(I3D_l(var,0,0,-2) + I3D_l(var,0,0,2)) \
        + 8064.0*(I3D_l(var,0,0,-1) + I3D_l(var,0,0,1)) \
        -14350.0*I3D_l(var,0,0,0))

#define DXY(var) i8dxy* \
      (9.0*(I3D_l(var,-4,-4,0) - I3D_l(var,-4,4,0)  \
          - I3D_l(var, 4,-4,0) + I3D_l(var, 4,4,0)) \
     -96.0*(I3D_l(var,-4,-3,0) - I3D_l(var,-4,3,0)  \
          + I3D_l(var,-3,-4,0) - I3D_l(var,-3,4,0)  \
          - I3D_l(var, 3,-4,0) + I3D_l(var, 3,4,0)  \
          - I3D_l(var, 4,-3,0) + I3D_l(var, 4,3,0)) \
    +504.0*(I3D_l(var,-4,-2,0) - I3D_l(var,-4,2,0)  \
          + I3D_l(var,-2,-4,0) - I3D_l(var,-2,4,0)  \
          - I3D_l(var, 2,-4,0) + I3D_l(var, 2,4,0)  \
          - I3D_l(var, 4,-2,0) + I3D_l(var, 4,2,0)) \
   +1024.0*(I3D_l(var,-3,-3,0) - I3D_l(var,-3,3,0)  \
          - I3D_l(var, 3,-3,0) + I3D_l(var, 3,3,0)) \
   -2016.0*(I3D_l(var,-4,-1,0) - I3D_l(var,-4,1,0)  \
          + I3D_l(var,-1,-4,0) - I3D_l(var,-1,4,0)  \
          - I3D_l(var, 1,-4,0) + I3D_l(var, 1,4,0)  \
          - I3D_l(var, 4,-1,0) + I3D_l(var, 4,1,0)) \
   -5376.0*(I3D_l(var,-3,-2,0) - I3D_l(var,-3,2,0)  \
          + I3D_l(var,-2,-3,0) - I3D_l(var,-2,3,0)  \
          - I3D_l(var, 2,-3,0) + I3D_l(var, 2,3,0)  \
          - I3D_l(var, 3,-2,0) + I3D_l(var, 3,2,0)) \
  +21504.0*(I3D_l(var,-3,-1,0) - I3D_l(var,-3,1,0)  \
          + I3D_l(var,-1,-3,0) - I3D_l(var,-1,3,0)  \
          - I3D_l(var, 1,-3,0) + I3D_l(var, 1,3,0)  \
          - I3D_l(var, 3,-1,0) + I3D_l(var, 3,1,0)) \
  +28224.0*(I3D_l(var,-2,-2,0) - I3D_l(var,-2,2,0)  \
          - I3D_l(var, 2,-2,0) + I3D_l(var, 2,2,0)) \
 -112896.0*(I3D_l(var,-2,-1,0) - I3D_l(var,-2,1,0)  \
          + I3D_l(var,-1,-2,0) - I3D_l(var,-1,2,0)  \
          - I3D_l(var, 1,-2,0) + I3D_l(var, 1,2,0)  \
          - I3D_l(var, 2,-1,0) + I3D_l(var, 2,1,0)) \
 +451584.0*(I3D_l(var,-1,-1,0) - I3D_l(var,-1,1,0)  \
          - I3D_l(var, 1,-1,0) + I3D_l(var, 1,1,0)))

#define DXZ(var) i8dxz* \
      (9.0*(I3D_l(var,-4, 0, -4) - I3D_l(var,-4, 0, 4)  \
          - I3D_l(var, 4, 0, -4) + I3D_l(var, 4, 0, 4)) \
     -96.0*(I3D_l(var,-4, 0, -3) - I3D_l(var,-4, 0, 3)  \
          + I3D_l(var,-3, 0, -4) - I3D_l(var,-3, 0, 4)  \
          - I3D_l(var, 3, 0, -4) + I3D_l(var, 3, 0, 4)  \
          - I3D_l(var, 4, 0, -3) + I3D_l(var, 4, 0, 3)) \
    +504.0*(I3D_l(var,-4, 0, -2) - I3D_l(var,-4, 0, 2)  \
          + I3D_l(var,-2, 0, -4) - I3D_l(var,-2, 0, 4)  \
          - I3D_l(var, 2, 0, -4) + I3D_l(var, 2, 0, 4)  \
          - I3D_l(var, 4, 0, -2) + I3D_l(var, 4, 0, 2)) \
   +1024.0*(I3D_l(var,-3, 0, -3) - I3D_l(var,-3, 0, 3)  \
          - I3D_l(var, 3, 0, -3) + I3D_l(var, 3, 0, 3)) \
   -2016.0*(I3D_l(var,-4, 0, -1) - I3D_l(var,-4, 0, 1)  \
          + I3D_l(var,-1, 0, -4) - I3D_l(var,-1, 0, 4)  \
          - I3D_l(var, 1, 0, -4) + I3D_l(var, 1, 0, 4)  \
          - I3D_l(var, 4, 0, -1) + I3D_l(var, 4, 0, 1)) \
   -5376.0*(I3D_l(var,-3, 0, -2) - I3D_l(var,-3, 0, 2)  \
          + I3D_l(var,-2, 0, -3) - I3D_l(var,-2, 0, 3)  \
          - I3D_l(var, 2, 0, -3) + I3D_l(var, 2, 0, 3)  \
          - I3D_l(var, 3, 0, -2) + I3D_l(var, 3, 0, 2)) \
  +21504.0*(I3D_l(var,-3, 0, -1) - I3D_l(var,-3, 0, 1)  \
          + I3D_l(var,-1, 0, -3) - I3D_l(var,-1, 0, 3)  \
          - I3D_l(var, 1, 0, -3) + I3D_l(var, 1, 0, 3)  \
          - I3D_l(var, 3, 0, -1) + I3D_l(var, 3, 0, 1)) \
  +28224.0*(I3D_l(var,-2, 0, -2) - I3D_l(var,-2, 0, 2)  \
          - I3D_l(var, 2, 0, -2) + I3D_l(var, 2, 0, 2)) \
 -112896.0*(I3D_l(var,-2, 0, -1) - I3D_l(var,-2, 0, 1)  \
          + I3D_l(var,-1, 0, -2) - I3D_l(var,-1, 0, 2)  \
          - I3D_l(var, 1, 0, -2) + I3D_l(var, 1, 0, 2)  \
          - I3D_l(var, 2, 0, -1) + I3D_l(var, 2, 0, 1)) \
 +451584.0*(I3D_l(var,-1, 0, -1) - I3D_l(var,-1, 0, 1)  \
          - I3D_l(var, 1, 0, -1) + I3D_l(var, 1, 0, 1)))

#define DYZ(var) i8dyz* \
      (9.0*(I3D_l(var,0, -4, -4) - I3D_l(var,0, -4, 4)  \
          - I3D_l(var,0,  4, -4) + I3D_l(var,0,  4, 4)) \
     -96.0*(I3D_l(var,0, -4, -3) - I3D_l(var,0, -4, 3)  \
          + I3D_l(var,0, -3, -4) - I3D_l(var,0, -3, 4)  \
          - I3D_l(var,0,  3, -4) + I3D_l(var,0,  3, 4)  \
          - I3D_l(var,0,  4, -3) + I3D_l(var,0,  4, 3)) \
    +504.0*(I3D_l(var,0, -4, -2) - I3D_l(var,0, -4, 2)  \
          + I3D_l(var,0, -2, -4) - I3D_l(var,0, -2, 4)  \
          - I3D_l(var,0,  2, -4) + I3D_l(var,0,  2, 4)  \
          - I3D_l(var,0,  4, -2) + I3D_l(var,0,  4, 2)) \
   +1024.0*(I3D_l(var,0, -3, -3) - I3D_l(var,0, -3, 3)  \
          - I3D_l(var,0,  3, -3) + I3D_l(var,0,  3, 3)) \
   -2016.0*(I3D_l(var,0, -4, -1) - I3D_l(var,0, -4, 1)  \
          + I3D_l(var,0, -1, -4) - I3D_l(var,0, -1, 4)  \
          - I3D_l(var,0,  1, -4) + I3D_l(var,0,  1, 4)  \
          - I3D_l(var,0,  4, -1) + I3D_l(var,0,  4, 1)) \
   -5376.0*(I3D_l(var,0, -3, -2) - I3D_l(var,0, -3, 2)  \
          + I3D_l(var,0, -2, -3) - I3D_l(var,0, -2, 3)  \
          - I3D_l(var,0,  2, -3) + I3D_l(var,0,  2, 3)  \
          - I3D_l(var,0,  3, -2) + I3D_l(var,0,  3, 2)) \
  +21504.0*(I3D_l(var,0, -3, -1) - I3D_l(var,0, -3, 1)  \
          + I3D_l(var,0, -1, -3) - I3D_l(var,0, -1, 3)  \
          - I3D_l(var,0,  1, -3) + I3D_l(var,0,  1, 3)  \
          - I3D_l(var,0,  3, -1) + I3D_l(var,0,  3, 1)) \
  +28224.0*(I3D_l(var,0, -2, -2) - I3D_l(var,0, -2, 2)  \
          - I3D_l(var,0,  2, -2) + I3D_l(var,0,  2, 2)) \
 -112896.0*(I3D_l(var,0, -2, -1) - I3D_l(var,0, -2, 1)  \
          + I3D_l(var,0, -1, -2) - I3D_l(var,0, -1, 2)  \
          - I3D_l(var,0,  1, -2) + I3D_l(var,0,  1, 2)  \
          - I3D_l(var,0,  2, -1) + I3D_l(var,0,  2, 1)) \
 +451584.0*(I3D_l(var,0, -1, -1) - I3D_l(var,0, -1, 1)  \
          - I3D_l(var,0,  1, -1) + I3D_l(var,0,  1, 1)))

#endif /*FD_CENTRAL_ORDER*/
#endif /*_DERIVATIVE_H_*/
