          TYPE gadget_io
            INTEGER :: REPORT        !=66
	    INTEGER :: REFCOORD      !=67
	    INTEGER :: ACTCOORD      !=68
	    INTEGER :: CONSTRAINTS   !=69
            INTEGER :: PENALTY
            INTEGER :: STRUCTINPUT
	  END TYPE gadget_io

          TYPE gadget_input
	    INTEGER :: RELAX=0         !c type of relaxation (0-ionic,1-full)

	    INTEGER :: HESSIAN=1       !c model for initial hessian
            INTEGER :: HUPDATE=1       !c Hesse-update algorithm
            INTEGER :: HREFRESH=100    !c refreshing of Hesse matrix

            INTEGER :: OPTENGINE=1     !c optimization engine

            REAL(q) :: STEPLIM=0.4     !c step limitation (in a.u.)

	    REAL(q) :: ECRITER=1.0     !c energy convergence criterion (a.u.)
	    REAL(q) :: GCRITER=1.0     !c gradient convergence criterion (a.u.)
	  END TYPE

	  TYPE coordinate
            CHARACTER (LEN=2) :: TAG         !c
	    CHARACTER (LEN=2)  :: ATOMS(4)   !c atom label
            CHARACTER (LEN=1)  :: TYPE='s'   !c (s)imple or (c)omplex
            INTEGER :: NATOMS(4)             !c number of atom label
            INTEGER :: WHAT(4)               !c which atoms int. coord. consists of
            INTEGER :: WHERE(4,3)            !c where the int. coord. is located
	    INTEGER :: STATUS                !c 0-fixed, 1-normal,2-revert
            REAL(q) :: VALUE                 !c int. coords.
            REAL(q) :: DVALUE                !c int. coords.
	    REAL(q),POINTER :: COEFS(:)      !c coeficients if TYPE=='c'
          END TYPE

          TYPE coordstructure
            TYPE(coordinate),POINTER :: COORDSTRUCT(:)
	    INTEGER :: NUMBONDS
            INTEGER :: NUMANGLES
            INTEGER :: NUMTORSIONS
            INTEGER :: NUMINTERNALS
            INTEGER,POINTER :: TOPOLOGY(:,:)
            INTEGER,POINTER :: FRAGMENTS(:,:)
	  END TYPE


	  TYPE elemtable
            CHARACTER (LEN=2),DIMENSION(98) :: ELEMENTS=&       !c list of elements
	                            (/'Ru','Re','Ra', 'Rb', 'Rn', 'Rh',&
                            'Be', 'Ba', 'Bi', 'Bk', 'Br', 'H ', 'P ',&
                             'Os', 'Hg', 'Ge', 'Gd', 'Ga', 'Pr', 'Pt',&
                             'Pu', 'C ', 'Pb', 'Pa', 'Pd', 'Cd', 'Po',&
                             'Pm', 'Ho', 'Hf', 'K ', 'He', 'Mg', 'Mo',&
                             'Mn', 'O ', 'S ', 'W ', 'Zn', 'Eu', 'Zr',&
                             'Er', 'Ni', 'Na', 'Nb', 'Nd', 'Ne', 'Np',&
                             'Fr', 'Fe', 'B ', 'F ', 'Sr', 'N ', 'Kr',&
                             'Si', 'Sn', 'Sm', 'V ', 'Sc', 'Sb', 'Se',&
                             'Co', 'Cm', 'Cl', 'Ca', 'Cf', 'Ce', 'Xe',&
                             'Lu', 'Cs', 'Cr', 'Cu', 'La', 'Li', 'Tl',&
                             'Tm', 'Th', 'Ti', 'Te', 'Tb', 'Tc', 'Ta',&
                             'Yb', 'Dy', 'I ', 'U ', 'Y ', 'Ac', 'Ag',&
                             'Ir', 'Am', 'Al', 'As', 'Ar', 'Au', 'At',&
                             'In'/)
            REAL(q),DIMENSION(98) :: COVALENT_RADII=&          !c list of covalent
	        (/1.5,   1.55,  2.100, 1.670, 0.200, 1.650,&   !c radii for elements
                                  0.550, 1.540, 1.740, 0.200, 1.410, 0.430, 1.250,&
                                  1.570, 1.900, 1.370, 1.990, 1.420, 2.020, 1.700,&
                                  0.200, 0.900, 1.740, 1.810, 1.000, 1.890, 1.880,&
                                  2.000, 1.940, 1.770, 1.530, 0.741, 1.300, 1.670,&
                                  1.550, 0.880, 1.220, 1.570, 1.650, 2.190, 1.760,&
                                  1.930, 1.700, 1.170, 1.680, 2.010, 0.815, 1.750,&
                                  0.200, 1.540, 1.030, 0.840, 1.320, 0.880, 1.069,&
                                  1.000, 1.660, 2.000, 1.530, 1.640, 1.660, 1.420,&
                                  1.530, 1.200, 1.190, 1.190, 1.730, 2.030, 1.750,&
                                  1.920, 1.870, 1.550, 1.000, 2.070, 0.880, 1.750,&
                                  1.920, 1.990, 1.670, 1.670, 1.960, 1.550, 1.630,&
                                  2.140, 1.950, 1.600, 1.780, 1.980, 2.080, 1.790,&
                                  1.520, 1.710, 0.950, 1.410, 0.995, 1.700, 0.200,&
                                  1.830/)/0.5291772083

	    INTEGER,DIMENSION(98) :: ELEM_TABLE=&
	                          (/5, 6, 7, 5, 6, 5, 2, 6,&
                               6, 7, 4, 1, 3, 6, 6, 4, 6, 4,&
                               6, 6, 7, 2, 6, 7, 5, 5, 6, 6,&
                               6, 6, 4, 1, 3, 5, 4, 2, 3, 6,&
                               4, 6, 5, 6, 4, 3, 5, 6, 2, 7,&
                               7, 4, 2, 2, 5, 2, 4, 3, 5, 6,&
                               4, 4, 5, 4, 4, 7, 3, 4, 7, 6,&
                               5, 6, 6, 4, 4, 6, 2, 6, 6, 7,&
                               4, 5, 6, 5, 6, 6, 6, 5, 7, 5,&
                               7, 5, 6, 7, 3, 4, 3, 6, 6, 5/)
	    REAL(q),DIMENSION(98) :: ELEM_MASSES=&
	                          (/101.07,186.2,226.0, 85.47, 222.0, 102.90,&
                             9.0122, 137.34, 208.98, 247.0,79.91, 1.0079, 30.974,&
                             190.20, 200.59, 72.59, 157.25, 69.72, 140.91, 195.09,&
                             244.0, 12.01, 207.19, 231.0, 106.40, 112.40, 210.0,&
                             145.0, 164.93, 178.49, 39.09, 4.0026, 24.305, 95.94,&
                             54.938, 15.999, 32.064, 183.85, 65.38, 151.96, 91.22,&
                             167.26, 58.71, 22.9898, 92.91, 144.24, 20.18, 237.05,&
                             223.0, 55.85, 10.81, 18.998, 87.62, 14.007, 39.948,&
                             28.086, 118.69, 150.35, 50.942, 44.956, 121.75, 78.96,&
                             58.93, 247.0, 35.453, 40.08, 251.0, 140.12, 131.30,&
                             174.97, 132.91, 52.00, 63.55, 138.91, 6.941, 204.37,&
                             168.93, 232.04, 47.90, 127.60, 158.92, 98.91, 180.95,&
                             173.04, 162.50, 126.90, 238.03, 88.91, 227.0, 107.87,&
                             192.22, 243.0, 26.982, 74.922, 39.948, 196.97, 210.0,&
                             114.82/)
	  END TYPE

          TYPE internal
            INTEGER,POINTER :: WHAT(:,:)   ! which atoms int. coord. consists of
            INTEGER,POINTER :: WHERE(:,:)  ! where the int. coord. is located
            INTEGER,POINTER :: NUM(:)      ! number of int. coords. per atoms (for all atoms)
            INTEGER         :: COUNT       ! total number of int.coords
            REAL(q),POINTER :: DIST(:,:)   ! int. coords.
          END TYPE

          TYPE ptr_to_arr
            INTEGER,POINTER :: arr(:,:)
          END TYPE

	
