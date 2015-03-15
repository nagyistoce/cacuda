
```
===========================================================
           o-o       o-o o   o o-o     O
          /         /    |   | |  \   / \
         O      oo O     |   | |   O o---o
          \    | |  \    |   | |  /  |   |
           o-o o-o-  o-o  o-o  o-o   o   o

     Unleash the Power of Cactus on Hybrid Systems !
          http://code.google.com/p/cacuda

             (c) Copyright The Authors
             GNU Licensed. No Warranty
=========================================================== 
```
## Prepare CUDA Programming Environment ##

CaCUDA has been tested with CUDA toolkit 3.0+. Before you go through the following steps, make sure you set up your CUDA programming environment properly.

The latest CUDA toolkit can be found at [NVIDIA CUDA Zone](http://developer.nvidia.com/category/zone/cuda-zone).
You need to
  1. Download and install the latest graphic card driver from NVIDIA.
  1. Download and install the CUDA toolkit Version 3.0+
You can then download and try to compile the sample programs in NVIDIA\_GPU\_Computing\_SDK. If you manage to compile most of the sample codes, you should be ready to test-drive CaCUDA. Otherwise, please continue working on your CUDA programming environment.

## Download Cactus and CaCUDA ##

One can easily check out everything required to run a 3D CFD application with a script [GetComponents](http://www.cactuscode.org/download/GetComponents), which can be downloaded by
```
$wget http://www.cactuscode.org/download/GetComponents 
```
You can then checkout everything you need with
```
$chmod 755 GetComponents
$./GetComponents -a http://cacuda.googlecode.com/svn/branches/CACUDA_09302011/manifest/ThornList 
```
Everything, including Cactus flesh and the computational toolkit required for CaCUDA, will be checked out in a directory called
```
CactusCaCUDA
```

## Compile CaCUDA ##

The command line to start your first compilation is:
```
$cd CactusCaCUDA
$make cacuda-config options=manifest/gnu_options THORNLIST=manifest/ThornList PROMPT=no
```

You may want to speed up the compilation with "-j" option.
```
$make -j 16 cacuda-config options=manifest/gnu_options THORNLIST=manifest/ThornList PROMPT=no
```

If you notice any errors, please take a look at the Cactus compilation options in CactusCaCUDA/manifest/gnu\_options to make sure they match your system.
You probably don't need to do anything if you have the CUDA compiler installed at the default location (assuming you installed the 64 bit version). If not, you need to modify
```
LIBDIRS=/usr/lib64 /usr/local/cuda/lib64 /usr/include/cuda/lib64 /usr/lib64/nvidia
```
to match what you have. You can also use the 32 bit installation by setting LIBDIRS to point to your 32 bit installation.
You may also want to change
```
CUCCFLAGS= -arch=sm_11 --ptxas-options=-v
```
to better match your GPU capability. E.g.,
```
CUCCFLAGS= -arch=sm_20 --ptxas-options=-v
```

The compilation will create an executable
```
CactusCaCUDA/exe/cactus_cacuda
```

After the first successful compilation with the lengthy compilation command, hereafter you can simply run
```
$make cacuda-rebuild
```
to generate the CUDA code and compile,
or
```
$make cacuda
```
to compile without regenerating the CUDA code.

## Run a 3D CFD Demo on GPU ##
You can copy a sample parameter file cacuda.par under CactusCaCUDA/manifest to CactusCaCUDA/exe.
```
$cd CactusCaCUDA/exe
$./cactus_cacuda cacuda.par
```