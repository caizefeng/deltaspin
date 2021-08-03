!
! nv_profile timer ids
!

! should match elements of timers array in Cuda_Init.cu
#ifdef USE_NVP
#define nv_profile_malloc(n)	call nvp_malloc(n)
#define nv_profile_free()	call nvp_free()
#define nv_profile_start(tid)	call nvp_start(tid)
#define nv_profile_stop(tid)	call nvp_stop(tid)
#define nv_profile_print(nid)	call nvp_print(nid)
#else
#define nv_profile_malloc(n)    call nvp_malloc(n)
#define nv_profile_free()       call nvp_free()
#define nv_profile_start(tid)
#define nv_profile_stop(tid)
#define nv_profile_print(nid)
#endif

#define NUM_TIMERS	200

#define NVP_TOTAL       0
#define NVP_CPU         1
#define NVP_MALLOC      2
#define NVP_MEMCPY      3
#define NVP_GEMM        4
#define NVP_FFT         5
#define NVP_CUSTOM      6
#define NVP_MPI         7

#define NVP_VASP                8
#define NVP_VASP_SPHER          9
#define NVP_VASP_DENINI         10
#define NVP_VASP_DENSTA         11
#define NVP_VASP_SETDIJ         12
#define NVP_VASP_PROALL         13
#define NVP_VASP_ORTHCH         14
#define NVP_VASP_PARDENS        15
#define NVP_VASP_CHARGEDEN      16
#define NVP_VASP_WAVPRE         17
#define NVP_VASP_EWALD          18
#define NVP_VASP_CALCSV         19
#define NVP_VASP_IONLOOP        20

#define NVP_IONLOOP_ELMIN       21
#define NVP_IONLOOP_FSTRESS     22
#define NVP_IONLOOP_POTLOK      23

#define NVP_ELMIN               24
#define NVP_ELMIN_POTLOK        25
#define NVP_ELMIN_SETDIJ        26
#define NVP_ELMIN_EDDIAG        27
#define NVP_ELMIN_ALGO          28
#define NVP_ELMIN_ORTHCH        29
#define NVP_ELMIN_DOS           30
#define NVP_ELMIN_CHARGE        31
#define NVP_ELMIN_MIXING        32

#define NVP_EDDAV               33
#define NVP_EDDAV_INIT          34
#define NVP_EDDAV_SKINIT        35
#define NVP_EDDAV_NEWBAND       36
#define NVP_EDDAV_HF            37
#define NVP_EDDAV_BANDOPT       38
#define NVP_EDDAV_UPDATEHO      39
#define NVP_EDDAV_TOTE          40
#define NVP_EDDAV_BREAKOPT      41
#define NVP_EDDAV_APPLYPRECOND  42
#define NVP_EDDAV_CALCWAVEC     43
#define NVP_EDDAV_OO            44
#define NVP_EDDAV_W1FFT         45
#define NVP_EDDAV_SUBSPACEROT   46
#define NVP_EDDAV_SKEND         47
#define NVP_EDDAV_END           48

#define NVP_NEWBAND_MEMCPY	49
#define NVP_NEWBAND_FFT		50
#define NVP_NEWBAND_ECCP	51
#define NVP_NEWBAND_CPU		52
#define NVP_NEWBAND_PRECOND	53

#define NVP_BANDOPT_HAMILTMU    54
#define NVP_BANDOPT_ADDFOCK     55
#define NVP_BANDOPT_OVERL	56
#define NVP_BANDOPT_TRUNCATE	57
#define NVP_BANDOPT_NORM	58
#define NVP_BANDOPT_COPY	59
#define NVP_BANDOPT_END         60

#define NVP_UPDATEHO_MEMCPY     61
#define NVP_UPDATEHO_CHAM	62
#define NVP_UPDATEHO_COVL	63
#define NVP_UPDATEHO_ADD        64
#define NVP_UPDATEHO_POTRF      65
#define NVP_UPDATEHO_POCON	66
#define NVP_UPDATEHO_ESOLVE	67

#define NVP_BREAKOPT_MALLOC	68
#define NVP_BREAKOPT_MEMCPY	69
#define NVP_BREAKOPT_WACW	70
#define NVP_BREAKOPT_WACPROJ    71
#define NVP_BREAKOPT_WOPTCW	72
#define NVP_BREAKOPT_WOPTCPROJ	73
#define NVP_BREAKOPT_CHAM	74

#define NVP_CALCWAVEC_FFT       75
#define NVP_CALCWAVEC_PROJALL   76
#define NVP_CALCWAVEC_NORM	77
#define NVP_CALCWAVEC_CPU       78
#define NVP_CALCWAVEC_REDIS	79

#define NVP_OO_MEMCPY           80
#define NVP_OO_CORTHO		81
#define NVP_OO_WOPTCW		82
#define NVP_OO_WOPTCPROJ	83
#define NVP_OO_COMPUTE          84

#define NVP_SUBSPACEROT_MEMCPY	85
#define NVP_SUBSPACEROT_ADD	86
#define NVP_SUBSPACEROT_ZHEEVX	87
#define NVP_SUBSPACEROT_DSYEV	88
#define NVP_SUBSPACEROT_ZHEEV	89
#define NVP_SUBSPACEROT_LINCOM	90

#define NVP_PROALL              91
#define NVP_RPRO                92
#define NVP_RPRO_MALLOC         93
#define NVP_RPRO_MEMCPY         94
#define NVP_RPRO_COMPUTE        95
#define NVP_RPRO_RPROMUISP      96

#define NVP_RPROMUISP           97
#define NVP_RPROMUISP_MALLOC    98
#define NVP_RPROMUISP_MEMCPY    99
#define NVP_RPROMUISP_COMPUTE   100

#define NVP_RPROMU		101
#define NVP_RPROMU_MALLOC	102
#define NVP_RPROMU_MEMCPY	103
#define NVP_RPROMU_COMPUTE	104

#define NVP_ECCP		105
#define NVP_ECCP_COMPUTE	106
#define NVP_ECCP_CPU		107

#define NVP_HAMILTMU            108
#define NVP_HAMILTMU_MALLOC     109
#define NVP_HAMILTMU_VHAMIL     110
#define NVP_HAMILTMU_RACCMU     111
#define NVP_HAMILTMU_KINHAMIL   112
#define NVP_HAMILTMU_FREE       113

#define NVP_RACC0MU             114
#define NVP_RACC0MU_MALLOC      115
#define NVP_RACC0MU_MERGE       116
#define NVP_RACC0MU_MEMCPY      117
#define NVP_RACC0MU_COMPUTE     118

#define NVP_ORTHCH		119
#define NVP_ORTHCH_INIT		120
#define NVP_ORTHCH_MALLOC	121
#define NVP_ORTHCH_MEMCPY	122
#define NVP_ORTHCH_COMPUTE	123
#define NVP_ORTHCH_CHOLESKI	124
#define NVP_ORTHCH_LINCOM	125

#define NVP_LINCOM		126
#define NVP_LINCOM_MALLOC	127
#define NVP_LINCOM_LINBAS	128

#define NVP_LINBAS		129
#define NVP_LINBAS_MEMCPY	130
#define NVP_LINBAS_COMPUTE	131

#define NVP_POTLOK              132
#define NVP_POTLOK_ECDC		133
#define NVP_POTLOK_CPOT		134
#define NVP_POTLOK_TOTPOT	135
#define NVP_POTLOK_FFTEXC	136
#define NVP_POTLOK_POTHAR	137
#define NVP_POTLOK_LPPOT	138
#define NVP_POTLOK_CVTOTSV	139

#define NVP_GGAALLGRID		140
#define NVP_GGAALLGRID_MALLOC   141
#define NVP_GGAALLGRID_MEMCPY   142
#define NVP_GGAALLGRID_COMPUTE  143

#define NVP_REDIS_MPI		144
#define NVP_REDIS_COPY		145

#define NVP_REDIS_BANDS		146
#define NVP_REDIS_NEWBAND	147
#define NVP_REDIS_BANDOPT	148
#define NVP_REDIS_CALCWAVEC	149
#define NVP_REDIS_W1FFT		150

#define NVP_GPUDIRECT		151

#define NVP_EDDIAG		152
#define NVP_EDDIAG_MEMCPY	153
#define NVP_EDDIAG_ZHEEVD	154

#define NVP_FSTRESS             155
#define NVP_FSTRESS_CHARGE      156
#define NVP_FSTRESS_FORLOC      157
#define NVP_FSTRESS_DIRKAR      158
#define NVP_FSTRESS_FOCK        159
#define NVP_FSTRESS_CDIJ        160
#define NVP_FSTRESS_UNITCELL    161
#define NVP_FSTRESS_COREC       162
#define NVP_FSTRESS_HARRIS      163
#define NVP_FSTRESS_MIX         164
#define NVP_FSTRESS_SUMCELL     165
#define NVP_FSTRESS_VDW         166
#define NVP_FSTRESS_SYMM        167

#define NVP_FOCK_KPAR           168
#define NVP_FOCK_FOCKFORCE      169
#define NVP_FOCK_FORSYM         170
#define NVP_FOCK_FORNLR         171
#define NVP_FOCK_FORNL          172
#define NVP_FOCK_FORDEP         173

#define NVP_UNITCELL_STRKIN     174
#define NVP_UNITCELL_STRELO     175
#define NVP_UNITCELL_STRNLR     176
#define NVP_UNITCELL_STRENL     177
#define NVP_UNITCELL_STRDEP     178
#define NVP_UNITCELL_STRETAU    179

#define NVP_HARRIS_CHGGRA       180
#define NVP_HARRIS_FORHAR       181
#define NVP_HARRIS_GAUSSIAN     182
