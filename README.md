# DeltaSpin

![version](https://img.shields.io/badge/version-1.0.0-blue)

DeltaSpin is a self-adaptive spin-constraining method based on cDFT, embedded in VASP.

Please cite  
Cai, Z., & Xu, B. (2022). First-principles Study of Spin Fluctuations Using Self-adaptive Spin-constrained DFT.
https://doi.org/10.48550/arXiv.2208.04551

## SOURCE CODE ACCESS
To obtain the source code, please send an email to bxu@gscaep.ac.cn.

### Requirements and Installation Guide
The system requirements, including all software dependencies and operating systems, are the **same** as those for the original Vienna Ab initio Simulation Package (VASP).

In addition to following the original routine for compiling `vasp_ncl`, which may take several minutes, please make sure to:
1. Clone the entire repository, not just the `deltaspin` directory. 
2. Use **Intel toolchain >= 2019 / oneAPI >= 2020**.
3. In some version of the Intel compiler, you may need to modify `FFLAGS` in `makefile.include` as `FFLAGS     = -assume byterecl -w -warn nointerfaces` to turn off type checking due to some old type casting techniques used in the VASP codebase.

### Getting Started
1. Change your working directory to `examples/metal/Fe`, and simply run the binary `bin/vasp_ncl`, or submit it to your own cluster like a normal VASP task.
2. Wait for the calculation to complete. The time required may depend on your system specifications (approximately 20 minutes on a 56-core Intel machine).
3. Once the calculation is complete, the obtained magnetic moments `MW_current` in the output `OSZICAR` should equal the value `M_CONSTR` set in the input `INCAR`. All VASP outputs, such as the total energy and electronic structure, correspond to the obtained constrained state.

### More Instructions
Please download the manual [DeltaSpin_manual.pdf](https://github.com/caizefeng/DeltaSpin/files/11144318/DeltaSpin_Manual.2.pdf) for additional information.


