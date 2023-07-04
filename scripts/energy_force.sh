# Only for "SCTYPE = 1" case
alias grep="grep --text"
E0=$(grep '  without' OUTCAR | awk '{print $7}')
N=$(sed -n '7p' POSCAR | awk '{for(i=1;i<=NF;i++)a+=$i; print a}')
echo "RMS: (muB)"
grep "RMS" OSZICAR | tail -1 | awk '{print $3}'
echo "Energy: (eV)"
echo "$E0"
echo "Cartesian Coordinates: (Angstrom)"
grep "position of ions in cartesian coordinates" OUTCAR -A $N | tail -$N | awk '{print $1, $2, $3}' | awk '{b[NR]=$0; }END{for(i=1;i<=NR;i++) if(i==NR){printf b[i] "\n"}else{printf "%s ",b[i]}}'
echo "Magnetization: (muB)"
grep 'MW_current' OSZICAR -A $N | tail -$N | awk '{print $2, $3, $4}' | awk '{b[NR]=$0; }END{for(i=1;i<=NR;i++) if(i==NR){printf b[i] "\n"}else{printf "%s ",b[i]}}'
echo "Magnetization Difference from Target: (muB)"
grep 'MW_current' OSZICAR -A $N | tail -$N | awk '{print $5, $6, $7}' | awk '{b[NR]=$0; }END{for(i=1;i<=NR;i++) if(i==NR){printf b[i] "\n"}else{printf "%s ",b[i]}}'
echo "Atomic Force: (eV/A)"
grep 'TOTAL-FORCE' OUTCAR -A $(($N + 1)) | tail -$N | awk '{print $4, $5, $6}' | awk '{b[NR]=$0; }END{for(i=1;i<=NR;i++) if(i==NR){printf b[i] "\n"}else{printf "%s ",b[i]}}'
echo "Magnetic Force: (eV/muB)"
grep 'Magnetic Force' OSZICAR -A $N | tail -$N | awk '{print $5, $6, $7}' | awk '{b[NR]=$0; }END{for(i=1;i<=NR;i++) if(i==NR){printf b[i] "\n"}else{printf "%s ",b[i]}}'
echo "Virial: (eV)"
read v_0 v_1 v_2 v_3 v_4 v_5 <<< $(grep "Total" OUTCAR | grep -v "CPU" | awk '{print $2,$3,$4,$5,$6,$7}')
echo "$v_0 $v_3 $v_5 $v_3 $v_1 $v_4 $v_5 $v_4 $v_2"
echo "Time Cost:"
time_cost_sec=$(grep Elapsed OUTCAR | awk '{print $4}' | awk '{printf("%.0f", $1)}')
printf "%d DAYS %02d:%02d:%02d" $((time_cost_sec/86400)) $(((time_cost_sec/3600)%24)) $(((time_cost_sec/60)%60)) $(($time_cost_sec%60))
echo ""
