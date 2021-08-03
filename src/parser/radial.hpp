#ifndef RADIAL_VPARSE_HPP
#define RADIAL_VPARSE_HPP

struct  t_modifier{

    // 1: for each site, take corresponding PAW
    // 2: take PAW function from POTCAR 
    // 3: take Wannier 90 function
    int radial_source;

    int index;
    int shell;

    // width of Wannier90 sigma
    double sigma;
   
    t_modifier(int radsrc, int ind, int rdshell, double rdsigma = 1.): 
      radial_source(radsrc), 
      index(ind),
      shell(rdshell),
      sigma(rdsigma){};
};

#endif
