C
C     bechmark param.inc                     
C

C-----General parameters always needed ...
      PARAMETER(NGX=16,NGY=16,NGZ=16,NGXF=32,NGYF=32,NGZF=32)
      PARAMETER(NTYPD=1,NIOND=8,NBANDS=48,NKDIM=1)
      PARAMETER(NRPLWV=1001,NPLINI=100)
      PARAMETER(NRPLWL=1,NBLK=16,MCPU=1)

C-----Parameter for non-local contribution 
      PARAMETER(LDIM=3,LMDIM=18,LDIM2=(LDIM*(LDIM+1))/2,LMYDIM=10)
      PARAMETER(IRECIP=1,IRMAX=1001,IRDMAX=7000)

C-----Special extra parameters: 
C     MIXD =dimension additional mixing arrays or not (1 or 0)a
C     IALLD=dimension additional arrays for Davidson/all band (0-2)
C     ISELF=dimension additional arrays for 'selfconsistent schemes'
      PARAMETER(IALLD=0,IRMMD=4,NEMPTY=0,ISPIND=1,IGRAD=0)
