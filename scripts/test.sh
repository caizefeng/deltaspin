range=(2.{1..9} 3.{0..9})

for i in ${range[@]}
do 
        mkdir $i 
        cd $i 
        cp ../INCAR ../POSCAR ../POTCAR ../KPOINTS ../submit.pbs . 
        sed -i "s/SCCONVB_GRAD = 1.9 -1/SCCONVB_GRAD = $i -1/" INCAR
        qsub submit.pbs
        cd $OLDPWD
done

#for i in ${range[@]}
#do 
        #cd $i 
        #echo $i
        #grep -A 1 Time log* 
        #grep -A 1 RMS log* | tail -2 
        #echo ""
        #cd $OLDPWD
#done
