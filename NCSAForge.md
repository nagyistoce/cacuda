## Forge at NCSA ##
[Forge](http://www.ncsa.illinois.edu/UserInfo/Resources/Hardware/DellNVIDIACluster/) has 18 Dell [PowerEdge C6145s](http://www.dell.com/us/enterprise/p/poweredge-c6145/pd.aspx) that contain 36 nodes of dual-socket/eight-core AMD processors, with M2070 NVIDIA Fermi GPU units.
There are eight Fermi units for each node, for a total of 288 while
the maximum number of nodes a common user can use is only 18 for a total
of 144.

## Queues on Forge ##
```
Queue            Memory CPU Time Walltime Node  Run Que Lm  State
---------------- ------ -------- -------- ----  --- --- --  -----
industrial         --      --    336:00:0    18   0   0 --   D S
debug              --      --    00:30:00     4   0   0 --   E R
wide               --      --    48:00:00    35   0   0 --   E R
indprio            --      --    48:00:00    18   0   0 --   D S
small              --      --    48:00:00     4   0   0 --   D S
nomss              --      --    48:00:00    18   0   0 --   D S
long               --      --    168:00:0    18   0   0 --   D S
normal             --      --    12:00:00    18   2   1 --   E R
```

## Device Query Output ##
The head node has 4 Fermi cards Tesla M2070, while each computing node has
8 M2070.
```
Device 0: Tesla M2070
  Total global memory 5375.44 MB 
  Shared memory per block: 49152 bytes
  Number of 32 bit registers per block: 32768
  Warp size: 32
  Maximum memory pitch: 2.00 GB
  Maximum number of threads per block: 1024
  Maximum sizes of each dimension of a block: 1024 x 1024 x 64
  Maximum sizes of each dimension of a grid: 65535 x 65535 x 65535
  Clock rate: 1.15 GHz
  Total constant memory size: 65536 bytes
  CUDA capability: 2.0
  Texture alignment: 512 bytes
  Concurrent copy and execution: Yes
  Number of multiprocessors: 14
  Run time limit on kernels: No
  Device integrated: No
  Device can map host memory: Yes
  Compute Mode: Normal
  Maximum 1D texture size: 65536
  Maximum 2D texture dimensions: 65536 x 65535
  Maximum 3D texture dimensions: 2048 x 2048 x 2048
  Surface alignment: 512 bytes
  Concurrent kernel execution: Yes
  ECC enabled: Yes
  PCI bus ID: 86
  PCI device ID: 0
  TCC driver used: No
```

## Running CaCUDA Code on Forge ##

You can simply follow [GetStarted](GetStarted.md) to download the code. When you compile
you shall start with a sample option forge\_gnu\_options distributed with CaCUDA by
```
make -j 16 cacuda-config options=manifest/forge_gnu_options THORNLIST=manifest/ThornList PROMPT=no
```
Since the Java module is not loaded by default, in order to generate CUDA code on the fly, you should load java by
```
$module load java-sun-1.6
```

A sample PBS script for those who use OpenMPI can be found at
```
/usr/local/doc/batch_scripts/openmpi.pbs
```

If you use other MPI libraries please pick the corresponding sample
PBS script to get started. Other scripts can be found under
```
/uf/ac/jtao/CactusCaCUDA
```

## Cactus Timing for CaCUDA ##

CaCUDACFD3D (single precision) running on one GPU card with 128x128x128 local grid size for 1000 timesteps.

```
===================================================================================================
Thorn           | Scheduled routine in time bin           | gettimeofday [secs] | getrusage [secs] 
===================================================================================================
  CaCUDALib       | Output CaCUDA banner                    |          0.00000800 |       0.00000000 
  CaCUDALib       | Initialize CaCUDALib                    |         22.58169100 |       0.00799900 
  CaCUDALib       | Output device information               |          0.00012100 |       0.00000000 
  CartGrid3D      | Register GH Extension for GridSymmetry  |          0.00000300 |       0.00000000 
  CoordBase       | Register a GH extension to store the coo|          0.00000100 |       0.00000000 
  PUGH            | Startup routine                         |          0.00000700 |       0.00000000 
  IOUtil          | Startup routine                         |          0.00000500 |       0.00000000 
  IOASCII         | Startup routine                         |          0.00000200 |       0.00000000 
  LocalReduce     | Startup routine                         |          0.00001800 |       0.00000000 
  IOBasic         | Startup routine                         |          0.00000200 |       0.00000000 
  PUGH            | Register Physical to Logical process map|          0.00000300 |       0.00000000 
  PUGH            | Register topology generation routines ro|          0.00000200 |       0.00000000 
  PUGHReduce      | Startup routine                         |          0.00002600 |       0.00000000 
  SymBase         | Register GH Extension for SymBase       |          0.00000300 |       0.00000000 
  ---------------------------------------------------------------------------------------------------
                | Total time for CCTK_STARTUP             |         22.58189200 |       0.00799900 
===================================================================================================
  Boundary        | Register boundary conditions that this t|          0.00008800 |       0.00000000 
  CartGrid3D      | Register coordinates for the Cartesian g|          0.00019200 |       0.00100000 
  CartGrid3D      | Register symmetry boundaries            |          0.00003200 |       0.00000000 
  SymBase         | Print symmetry boundary face description|          0.00000300 |       0.00000000 
  ---------------------------------------------------------------------------------------------------
                | Total time for CCTK_WRAGH               |          0.00031500 |       0.00100000 
===================================================================================================
  Boundary        | Check dimension of grid variables       |          0.00000000 |       0.00000000 
  CartGrid3D      | Check coordinates for CartGrid3D        |          0.00000200 |       0.00000000 
  ---------------------------------------------------------------------------------------------------
                | Total time for CCTK_PARAMCHECK          |          0.00000200 |       0.00000000 
===================================================================================================
  CaCUDALib       | Set device                              |          0.00007500 |       0.00000000 
  CaCUDACFD3DWOMOL| Allocate memory for variables on devices|          0.31104000 |       0.00199900 
  CaCUDACFD3DWOMOL| Allocate memory for variables on devices|          0.02696800 |       0.01499800 
  CartGrid3D      | Set up ranges for spatial 3D Cartesian c|          0.00011400 |       0.00100000 
  CartGrid3D      | Set up spatial 3D Cartesian coordinates |          0.09460900 |       0.08498700 
  IOASCII         | Choose 1D output lines                  |          0.00004800 |       0.00000000 
  IOASCII         | Choose 2D output planes                 |          0.00000400 |       0.00000000 
  PUGH            | Report on PUGH set up                   |          0.00002100 |       0.00000000 
  SymBase         | Check whether the driver set up the grid|          0.00000300 |       0.00000000 
  Time            | Initialise Time variables               |          0.00000300 |       0.00000000 
  Time            | Set timestep based on Courant condition |          0.00000500 |       0.00000000 
  ---------------------------------------------------------------------------------------------------
                | Total time for CCTK_BASEGRID            |          0.43289000 |       0.10298400 
===================================================================================================
  CaCUDACFD3DWOMOL| Initialize with the lid-driven cavity in|          0.01331900 |       0.01199800 
  IOBasic         | Initialisation routine                  |          0.00000500 |       0.00000000 
  ---------------------------------------------------------------------------------------------------
                | Total time for CCTK_INITIAL             |          0.01332400 |       0.01199800 
===================================================================================================
  CaCUDACFD3DWOMOL| Launch CaCUDA Kernel Update_Velocity    |          9.86934300 |       9.81653100 
  CaCUDACFD3DWOMOL| Launch CaCUDA Kernel Update_Boundaries a|          0.34422500 |       0.25896000 
  CaCUDACFD3DWOMOL| Start the iterator for the pressure solv|          0.00111500 |       0.00000000 
  CaCUDACFD3DWOMOL| Update the number of iterations for the |          0.00945100 |       0.00699900 
  CaCUDACFD3DWOMOL| Iteratively solves the conservation of m|         28.67740300 |      28.21167900 
  CaCUDACFD3DWOMOL| Update the velocity on the boundaries   |          3.45935100 |       2.52762200 
  CaCUDACFD3DWOMOL| Copy variables (boundary only) from devi|          0.45957700 |       0.28295200 
  CaCUDACFD3DWOMOL| Copy variables to device (boundary only)|          0.67973400 |       0.23496200 
  ---------------------------------------------------------------------------------------------------
                | Total time for CCTK_EVOL                |         43.50019900 |      41.33970500 
===================================================================================================
  TimerReport     | Print the timer report                  |          0.00057500 |       0.00000000 
  ---------------------------------------------------------------------------------------------------
                | Total time for CCTK_CHECKPOINT          |          0.00057500 |       0.00000000 
===================================================================================================
  CaCUDACFD3DWOMOL| Copy variables from devices             |          0.25645300 |       0.20897000 
  TimerReport     | Print the timer report                  |          0.00056900 |       0.00000000 
  ---------------------------------------------------------------------------------------------------
                | Total time for CCTK_ANALYSIS            |          0.25702200 |       0.20897000 
===================================================================================================
  TimerReport     | Print the timer report                  |          0.00000000 |       0.00000000 
  ---------------------------------------------------------------------------------------------------
                | Total time for CCTK_TERMINATE           |          0.00000000 |       0.00000000 
===================================================================================================
  ---------------------------------------------------------------------------------------------------
                | Total time for simulation               |         47.66059500 |      43.90232600 
===================================================================================================
```

CaCUDACFD3D (single precision) running on a 16 GPU cards with 128x128x128 local grid size for 1000 timesteps.

```

===================================================================================================
Thorn           | Scheduled routine in time bin           | gettimeofday [secs] | getrusage [secs] 
===================================================================================================
  CaCUDALib       | Output CaCUDA banner                    |          0.00000900 |       0.00000000 
  CaCUDALib       | Initialize CaCUDALib                    |         27.54636800 |       0.00799900 
  CaCUDALib       | Output device information               |          0.00012100 |       0.00000000 
  CartGrid3D      | Register GH Extension for GridSymmetry  |          0.00000400 |       0.00000000 
  CoordBase       | Register a GH extension to store the coo|          0.00000100 |       0.00000000 
  PUGH            | Startup routine                         |          0.00000700 |       0.00000000 
  IOUtil          | Startup routine                         |          0.00000500 |       0.00000000 
  IOASCII         | Startup routine                         |          0.00000200 |       0.00000000 
  LocalReduce     | Startup routine                         |          0.00001600 |       0.00000000 
  IOBasic         | Startup routine                         |          0.00000200 |       0.00000000 
  PUGH            | Register Physical to Logical process map|          0.00000300 |       0.00000000 
  PUGH            | Register topology generation routines ro|          0.00000200 |       0.00000000 
  PUGHReduce      | Startup routine                         |          0.00002500 |       0.00000000 
  SymBase         | Register GH Extension for SymBase       |          0.00001000 |       0.00000000 
  ---------------------------------------------------------------------------------------------------
                | Total time for CCTK_STARTUP             |         27.54657500 |       0.00799900 
===================================================================================================
  Boundary        | Register boundary conditions that this t|          0.03041100 |       0.00000000 
  CartGrid3D      | Register coordinates for the Cartesian g|          0.04359200 |       0.00000000 
  CartGrid3D      | Register symmetry boundaries            |          0.00481200 |       0.00000000 
  SymBase         | Print symmetry boundary face description|          0.00000300 |       0.00000000 
  ---------------------------------------------------------------------------------------------------
                | Total time for CCTK_WRAGH               |          0.07881800 |       0.00000000 
===================================================================================================
  Boundary        | Check dimension of grid variables       |          0.00000100 |       0.00000000 
  CartGrid3D      | Check coordinates for CartGrid3D        |          0.00000300 |       0.00000000 
  ---------------------------------------------------------------------------------------------------
                | Total time for CCTK_PARAMCHECK          |          0.00000400 |       0.00000000 
===================================================================================================
  CaCUDALib       | Set device                              |          0.00011500 |       0.00000000 
  CaCUDACFD3DWOMOL| Allocate memory for variables on devices|          2.34137500 |       0.00599900 
  CaCUDACFD3DWOMOL| Allocate memory for variables on devices|          0.07256300 |       0.02799500 
  CartGrid3D      | Set up ranges for spatial 3D Cartesian c|          0.00028600 |       0.00000000 
  CartGrid3D      | Set up spatial 3D Cartesian coordinates |          0.09712100 |       0.08598700 
  IOASCII         | Choose 1D output lines                  |          0.00005300 |       0.00000000 
  IOASCII         | Choose 2D output planes                 |          0.00000500 |       0.00000000 
  PUGH            | Report on PUGH set up                   |          0.00002200 |       0.00000000 
  SymBase         | Check whether the driver set up the grid|          0.00000300 |       0.00000000 
  Time            | Initialise Time variables               |          0.00000400 |       0.00000000 
  Time            | Set timestep based on Courant condition |          0.00000400 |       0.00000000 
  ---------------------------------------------------------------------------------------------------
                | Total time for CCTK_BASEGRID            |          2.51155100 |       0.11998100 
===================================================================================================
  CaCUDACFD3DWOMOL| Initialize with the lid-driven cavity in|          0.01165000 |       0.01199900 
  IOBasic         | Initialisation routine                  |          0.00000400 |       0.00000000 
  ---------------------------------------------------------------------------------------------------
                | Total time for CCTK_INITIAL             |          0.01165400 |       0.01199900 
===================================================================================================
  CaCUDACFD3DWOMOL| Launch CaCUDA Kernel Update_Velocity    |         13.16085100 |      13.09203600 
  CaCUDACFD3DWOMOL| Launch CaCUDA Kernel Update_Boundaries a|          7.32819800 |       7.16389800 
  CaCUDACFD3DWOMOL| Start the iterator for the pressure solv|          0.00133900 |       0.00200000 
  CaCUDACFD3DWOMOL| Update the number of iterations for the |          0.00974400 |       0.01199800 
  CaCUDACFD3DWOMOL| Iteratively solves the conservation of m|         38.29541100 |      37.67623300 
  CaCUDACFD3DWOMOL| Update the velocity on the boundaries   |         22.83294100 |      21.59974000 
  CaCUDACFD3DWOMOL| Copy variables (boundary only) from devi|         27.99468700 |      11.23630400 
  CaCUDACFD3DWOMOL| Copy variables to device (boundary only)|         15.35790900 |      12.11915000 
  ---------------------------------------------------------------------------------------------------
                | Total time for CCTK_EVOL                |        124.98108000 |     102.90135900 
===================================================================================================
  TimerReport     | Print the timer report                  |          0.00096700 |       0.00000000 
  ---------------------------------------------------------------------------------------------------
                | Total time for CCTK_CHECKPOINT          |          0.00096700 |       0.00000000 
===================================================================================================
  CaCUDACFD3DWOMOL| Copy variables from devices             |          2.04147800 |       0.28495700 
  TimerReport     | Print the timer report                  |          0.00127200 |       0.00000000 
  ---------------------------------------------------------------------------------------------------
                | Total time for CCTK_ANALYSIS            |          2.04275000 |       0.28495700 
===================================================================================================
  TimerReport     | Print the timer report                  |          0.00000000 |       0.00000000 
  ---------------------------------------------------------------------------------------------------
                | Total time for CCTK_TERMINATE           |          0.00000000 |       0.00000000 
===================================================================================================
  ---------------------------------------------------------------------------------------------------
                | Total time for simulation               |        174.00292500 |     123.97015400 
===================================================================================================
```