# DeltaSpin

![version](https://img.shields.io/badge/version-1.0.0-blue)

A self-adaptive spin-constraining method based on cDFT, embedd in VASP.

Please cite  
Cai, Z., & Xu, B. (2022). First-principles Study of Spin Fluctuations Using Self-adaptive Spin-constrained DFT.
https://doi.org/10.48550/arXiv.2208.04551

## SOURCE CODE ACCESS
Please send an email to bxu@gscaep.ac.cn asking for the source code.

### Requirements and Installation Guide
System requirements, including all software dependencies and operating systems, are the **same** as the original Vienna Ab initio Simulation Package (VASP).

Other than following the original routine for compiling `vasp_ncl`, which may take severl minutes, please make sure you also
1. Clone the whole repo, not just the `deltaspin` directory. 
2. Use Intel **toolchain >= 2019 / oneAPI >= 2020**.

### Getting Started
1. Change your working directory to `examples/insulator/Fe`, simply run the binary `bin/vasp_ncl`, or submit it to your own cluster like a normal VASP task.
2. Wait until the calculation is complete. The time cost may depend on your specs (20 min on a 56 cores Intel machine).
3. Eventually, the obtained magnetic moments `MW_current` in the output `OSZICAR` should equal to the value `M_CONSTR` you set in the input `INCAR`. All VASP outputs, such as the total energy and electronic structure, are for the obtained constrained state.

### More Instructions
Please download the manual [DeltaSpin_manual.pdf](https://github.com/caizefeng/DeltaSpin/files/10611430/DeltaSpin_manual.pdf).

