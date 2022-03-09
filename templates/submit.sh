#!/bin/bash

cat INCAR
ulimit -s unlimited

echo "the start time is:"   $(date)  >> timing.log
DATE1=$(date +%s)

mpirun -np 32 <path-to-deltaspin>/bin/vasp_ncl

DATE2=$(date +%s)
echo "the end time is:"   $(date)   >> timing.log

diff=$((DATE2-DATE1))
printf "TIME COST: %d DAYS %02d:%02d:%02d" \
$((diff/86400)) $(((diff/3600)%24)) $(((diff/60)%60)) $(($diff %60)) >> timing.log
echo -e "\n\n" >> timing.log

sh <path-to-deltaspin>/scripts/energy_force.sh

