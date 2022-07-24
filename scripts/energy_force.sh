# Only for "SCTYPE = 1" case
alias grep="grep --text"
E0=$(grep '  without' OUTCAR | awk '{print $7}')
N=$(sed -n '7p' POSCAR | awk '{for(i=1;i<=NF;i++)a+=$i; print a}')
echo "RMS:"
grep "RMS" OSZICAR | tail -1 | awk '{print $3}'
echo "Energy:"
echo "$E0"
echo "Cartesian Coordinates:"
grep "position of ions in cartesian coordinates" OUTCAR -A $N | tail -$N | awk '{print $1, $2, $3}' | awk '{b[NR]=$0; }END{for(i=1;i<=NR;i++) if(i==NR){printf b[i] "\n"}else{printf "%s ",b[i]}}'
echo "Magnetization:"
grep 'MW_current' OSZICAR -A $N | tail -$N | awk '{print $2, $3, $4}' | awk '{b[NR]=$0; }END{for(i=1;i<=NR;i++) if(i==NR){printf b[i] "\n"}else{printf "%s ",b[i]}}'
echo "Magnetization Difference from Target:"
grep 'MW_current' OSZICAR -A $N | tail -$N | awk '{print $5, $6, $7}' | awk '{b[NR]=$0; }END{for(i=1;i<=NR;i++) if(i==NR){printf b[i] "\n"}else{printf "%s ",b[i]}}'
echo "Atomic Force:"
grep 'TOTAL-FORCE' OUTCAR -A $(($N + 1)) | tail -$N | awk '{print $4, $5, $6}' | awk '{b[NR]=$0; }END{for(i=1;i<=NR;i++) if(i==NR){printf b[i] "\n"}else{printf "%s ",b[i]}}'
echo "Magnetic Force:"
grep 'Magnetic Force' OSZICAR -A $N | tail -$N | awk '{print $5, $6, $7}' | awk '{b[NR]=$0; }END{for(i=1;i<=NR;i++) if(i==NR){printf b[i] "\n"}else{printf "%s ",b[i]}}'
echo "Time Cost:"
grep 'TIME COST' timing.log | tail -1 | awk '{print $3, $4, $5}' 
echo ""
