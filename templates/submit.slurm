#!/bin/bash
#SBATCH -p foo
#SBATCH -N 1
#SBATCH -n 56
#SBATCH -J bar_BFO

source <path-to-oneapi>/setvars.sh

free -m
df -Th
cat INCAR
ulimit -s unlimited

echo "the start time is:"   $(date)  >> timing.log
DATE1=$(date +%s)

unset I_MPI_PMI_LIBRARY 
export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0
mpirun -np 56 <path-to-deltaspin>/bin/vasp_deltaspin

DATE2=$(date +%s)
echo "the end time is:"   $(date)   >> timing.log

diff=$((DATE2-DATE1))
printf "TIME COST: %d DAYS %02d:%02d:%02d" \
$((diff/86400)) $(((diff/3600)%24)) $(((diff/60)%60)) $(($diff %60)) >> timing.log
echo -e "\n\n" >> timing.log

sh <path-to-deltaspin>/scripts/energy_force.sh

