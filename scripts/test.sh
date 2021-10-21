for i in 1.{0..9}
do 
        mkdir $i 
        cd $i 
        cp ../INCAR ../POSCAR ../POTCAR ../KPOINTS ../submit.pbs . 
        sed -i "s/SCCONVB_GRAD = 2.0 -1/SCCONVB_GRAD = $i -1/" INCAR
        qsub submit.pbs
        cd $OLDPWD
done
