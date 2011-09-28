#include <cctk.h>

int CaCUDALib_Banner (void)
{
  const char *const banner =
"\n=================================================\n"
"       o-o       o-o o   o o-o     O   \n"
"      /         /    |   | |  \\   / \\  \n"
"     O      oo O     |   | |   O o---o \n"
"      \\    | |  \\    |   | |  /  |   | \n"
"       o-o o-o-  o-o  o-o  o-o   o   o  \n\n"

" Unleash the Power of Cactus on Hybrid Systems ! \n\n"
"         (c) Copyright The Authors \n"
"         GNU Licensed. No Warranty \n"
"=================================================\n\n";
  CCTK_RegisterBanner(banner);
  return 0;
}
