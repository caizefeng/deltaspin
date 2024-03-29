# DeltaSpin

![version](https://img.shields.io/badge/version-1.0.2-blue)

DeltaSpin is a **self-adaptive spin-constraining method** based on constrained Density Functional Theory (cDFT). It operates as an extension to the [Vienna Ab-initio Simulation Package (VASP)](https://www.vasp.at/).

## How to cite

```
@article{cai2023self,
  title={A self-adaptive first-principles approach for magnetic excited states},
  author={Cai, Zefeng and Wang, Ke and Xu, Yong and Wei, Su-Huai and Xu, Ben},
  journal={Quantum Frontiers},
  volume={2},
  number={1},
  pages={21},
  year={2023},
  publisher={Springer}
}
```

<!-- ## ATTENTION
To ensure the submitted binary `bin\vasp_ncl` operates correctly, it needs a container with ubuntu 18.04 and oneapi ~= 2021 for runtime libraries. For instance, `ghcr.io/caizefeng/oneapi-hpckit:2021.2.0-ubuntu18.04`. Note, it is necessary to increase the shared memory size by passing the optional parameter `--shm-size` to the `docker run` command. -->

## SOURCE CODE ACCESS
To acquire the source code, please send an email to bxu@gscaep.ac.cn.

## Requirements and Installation Guide

The system requirements, including all software dependencies and supported operating systems, are **similar** to those of the original Vienna Ab initio Simulation Package (VASP).

1. Clone the entire repository, not just the `deltaspin` directory. 

2. Install **Intel Parallel Studio >= 2019 / oneAPI >= 2020** and load environment variables correctly.

3. (Optional) In some versions of the Intel compiler, you might need to modify `FFLAGS` in `makefile.include` as 
    ```makefile
    FFLAGS = -assume byterecl -w -warn nointerfaces
    ```
    to disable type checking due to the usage of some outdated type casting techniques in the VASP codebase.

4. Compile `vasp_deltaspin` specifically using one single thread. This is crucial, as multi-threading is not supported.
    ```shell
    make deltaspin
    ```
    Please note that the compilation might take several minutes.

## Getting Started

For the following commands, replace the values within the angle brackets (`< >`) with relevant values for your system.

1. **Inspect your system configurations**, such as Intel runtime libraries and system stack size. Configure the stack size to unlimited if it hasn't been configured as such already. This is necessary because VASP uses numerous stack-based variables and arrays.
    ```shell
    ulimit -s unlimited
    ```

2. **Change your working directory** to `examples/insulator/NiO`.
    ```shell
    cd <repository root>/examples/insulator/NiO
    ```

3. **Review the INCAR file**. Pay particular attention to the following tags and their respective meanings. As a reference, the ground-state magnetic configuration for the provided NiO example is `-0.424907764 1.024101942 0.627479909 0.424903960 -1.024103888 -0.627479400`.
    ```
    M_CONSTR = 0.75914 0.16146 -1.00688 -0.67954 -0.91284 0.56668 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000 
    SCTYPE = 1
    CONSTRL  = 6*1 6*0
    ```

4. **Execute the binary `bin/vasp_deltaspin`**, or submit it to your cluster as you would **a regular VASP task**.
    ```shell
    mpirun -np <number of threads> ../../../bin/vasp_deltaspin
    ```

5. **Wait for the calculation to finish**. The time this takes can vary based on your system specifications. For reference, it takes approximately 1 hour and 7 minutes on a 56-thread compute node powered by two Intel Xeon Gold 6258R CPUs.

6. After the calculation is complete, the obtained magnetic moments, **`MW_current` in the output `OSZICAR`, should match the value `M_CONSTR` set in the `INCAR` input**. 
    ```shell
    grep "M_CONSTR = " INCAR
    grep "MW_current" OSZICAR -A 2 | tail -2 | awk '{print$2,$3,$4}'
    ```
    All VASP outputs, such as the total energy and electronic structure, correspond to the achieved constrained state.

7. (Optional) For convenience, you can use the provided `energy_force.sh` script to **inspect the critical properties of the achieved constrained state**, including the magnetic moments and magnetic forces (also known as magnetic effective fields).
      ```shell
      bash ../../../scripts/energy_force.sh
      ```

## More Instructions
For additional information, please download the manual [DeltaSpin_Manual.pdf](docs/DeltaSpin_Manual.pdf).

## Disclaimer
VASP is a proprietary software. Ensure that you possess an appropriate license to use it.

