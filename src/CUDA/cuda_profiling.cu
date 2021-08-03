// File: cuda_fft.cu
// C/Fortran interface to profiling.

// includes standard headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// includes linux headers
#include <time.h>
// includes cuda headers
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvToolsExtCuda.h>
#include <nvToolsExtCudaRt.h>
#include "nvToolsExt.h"
// includes project headers
#include "cuda_globals.h"

/******************************************************/

extern "C"
void cuda_profilerstart_C(void)
{
    CUDA_ERROR( cudaProfilerStart(), "Failed to start profiling!" );
}

extern "C"
void cuda_profilerstop_C(void)
{
    CUDA_ERROR( cudaProfilerStop(), "Failed to stop profiling!" );
}

// TODO: remove the two functions below from all fortran files
extern
void nvtx_range_pusha_(char *string)
{
    nvtxRangePushA(string);
}

extern
void nvtx_range_pop_(char *string)
{
    nvtxRangePop();
}

/******************************************************/

#ifdef USE_NVTX
// nvtx markers
const uint32_t colors[] = { 0xff6b8e23,		//  0 olive green
			    0xff228b22,		//  1 forest green
			    0xff9acd32,         //  2 yellow green
			    0xffff4500,		//  3 orange red
			    0xffff7f50,		//  4 coral
			    0xff8b0000,		//  5 dark red
			    0xffff00ff,		//  6 magenta
			    0xff9400d3,		//  7 dark violet
			    0xff00008b,		//  8 dark blue
                            0x000000ff,         //  9 blue
			    0xff4169e1,		// 10 royal blue
			    0xff00bfff,		// 11 sky blue
			    0xffffd700,		// 12 gold
			    0xff00ced1,		// 13 dark turquoise
			    0xffafeeee,		// 14 pale turquoise
			    0xff2f4f4f,		// 15 dark slate gray
			    0xffa0522d,		// 16 sienna brown
			    0xffff69b4,		// 17 hot pink
			    0xffc0c0c0,		// 18 silver
			    0xfff5f5f5 };	// 19 white smoke

const int num_colors = sizeof(colors)/sizeof(uint32_t);
#endif

#ifdef USE_CPUTIMERS
// cpu timers
#define NUM_TIMERS	256
#define BILLION         1000000000L
//struct timespec t0[NUM_TIMERS], t1[NUM_TIMERS];
//double tsum[NUM_TIMERS];
struct timespec *t0, *t1, faket;
double *tsum;
char *timers[] = { "TOTAL",		// NVP_TOTAL	0
		   "CPU",		// NVP_CPU	1
  	    	   "MALLOC",		// NVP_MALLOC	2
		   "MEMCPY",		// NVP_MEMCPY	3
		   "GEMM",		// NVP_GEMM	4
		   "FFT",		// NVP_FFT	5
		   "CUSTOM",		// NVP_CUSTOM	6
		   "MPI",		// NVP_MPI	7

		   "VASP",		// NVP_VASP		8
		   "VASP_SPHER",	// NVP_VASP_SPHER	9
		   "VASP_DENINI",	// NVP_VASP_DENINI	10
		   "VASP_DENSTA",	// NVP_VASP_DENSTA	11
		   "VASP_SETDIJ",	// NVP_VASP_SETDIJ	12
		   "VASP_PROALL",	// NVP_VASP_PROALL	13
		   "VASP_ORTHCH",	// NVP_VASP_ORTHCH	14
		   "VASP_PARDENS",	// NVP_VASP_PARDENS	15
		   "VASP_CHARGEDEN",	// NVP_VASP_CHARGEDEN	16
		   "VASP_WAVPRE",	// NVP_VASP_WAVPRE	17
		   "VASP_EWALD",	// NVP_EWALD_PRE	18
		   "VASP_CALCSV",	// NVP_CALCSV		19
		   "VASP_IONLOOP",	// NVP_IONLOOP		20

		   "IONLOOP_ELMIN",	// NVP_IONLOOP_ELMIN	21
		   "IONLOOP_FSTRESS",	// NVP_IONLOOP_FSTRESS	22
		   "IONLOOP_POTLOK",	// NVP_IONLOOP_POTLOK	23

		   "ELMIN",		// NVP_ELMIN		24
		   "ELMIN_POTLOK",	// NVP_ELMIN_POTLOK	25
		   "ELMIN_SETDIJ",	// NVP_ELMIN_SETDIJ	26
		   "ELMIN_EDDIAG",	// NVP_ELMIN_EDDAIG	27
		   "ELMIN_ALGO",	// NVP_ELMIN_ALGO	28
		   "ELMIN_ORTHCH",	// NVP_ELMIN_ORTHCH	29
		   "ELMIN_DOS",		// NVP_ELMIN_DOS	30
		   "ELMIN_CHARGE",	// NVP_ELMIN_CHARGE	31
		   "ELMIN_MIXING",	// NVP_ELMIN_MIXING   	32

		   "EDDAV",		// NVP_EDDAV			33
		   "EDDAV_INIT",	// NVP_EDDAV_INIT		34
		   "EDDAV_SKINIT",	// NVP_EDDAV_SKINIT		35
		   "EDDAV_NEWBAND",	// NVP_EDDAV_NEWBAND		36
		   "EDDAV_HF",		// NVP_EDDAV_HF			37
		   "EDDAV_BANDOPT",	// NVP_EDDAV_BANDOPT		38
		   "EDDAV_UPDATEHO",	// NVP_EDDAV_UPDATEHO		39
		   "EDDAV_TOTE",	// NVP_EDDAV_TOTE		40
		   "EDDAV_BREAKOPT",	// NVP_EDDAV_BREAKOPT		41
		   "EDDAV_APPLYPRECOND",// NVP_EDDAV_APPLYPRECOND	42
		   "EDDAV_CALCWAVEC",	// NVP_EDDAV_CALCWAVEC		43
		   "EDDAV_OO",		// NVP_EDDAV_OO			44
		   "EDDAV_W1FFT",	// NVP_EDDAV_W1FFT		45
		   "EDDAV_SUBSPACEROT",	// NVP_EDDAV_SUBSPACEROT	46
		   "EDDAV_SKEND",	// NVP_EDDAV_SKEND		47
		   "EDDAV_END",		// NVP_EDDAV_END		48

		   "NEWBAND_MEMCPY",	// NVP_NEWBAND_MEMCPY		49
		   "NEWBAND_FFT",	// NVP_NEWBAND_FFT		50
		   "NEWBAND_ECCP",	// NVP_NEWBAND_ECCP		51
		   "NEWBAND_CPU",	// NVP_NEWBAND_CPU		52
		   "NEWBAND_PRECOND",	// NVP_NEWBAND_PRECOND		53

		   "BANDOPT_HAMILTMU",	// NVP_BANDOPT_HAMILTMU		54
		   "BANDOPT_ADDFOCK",	// NVP_BANDOPT_ADDFOCK		55
		   "BANDOPT_OVERL",	// NVP_BANDOPT_OVERL		56
		   "BANDOPT_TRUNCATE",	// NVP_BANDOPT_TRUNCATE		57
		   "BANDOPT_NORM",	// NVP_BANDOPT_NORM		58
		   "BANDOPT_COPY",	// NVP_BANDOPT_COPY		59
		   "BANDOPT_END",	// NVP_BANDOPT_END		60

		   "UPDATEHO_MEMCPY",	// NVP_UPDATEHO_MEMCPY		61
		   "UPDATEHO_CHAM",	// NVP_UPDATEHO_CHAM		62
		   "UPDATEHO_COVL",	// NVP_UPDATEHO_COVL		63
		   "UPDATEHO_ADD",	// NVP_UPDATEHO_ADD		64
		   "UPDATEHO_POTRF",	// NVP_UPDATEHO_POTRF		65
		   "UPDATEHO_POCON",	// NVP_UPDATEHO_POCON		66
		   "UPDATEHO_ESOLVE",	// NVP_UPDATEHO_ESOLVE		67

		   "BREAKOPT_MALLOC",	// NVP_BREAKOPT_MALLOC		68
		   "BREAKOPT_MEMCPY",	// NVP_BREAKOPT_MEMCPY		69
		   "BREAKOPT_WACW",	// NVP_BREAKOPT_WACW		70
		   "BREAKOPT_WACPROJ",	// NVP_BREAKOPT_WACPROJ		71
		   "BREAKOPT_WOPTCW",	// NVP_BREAKOPT_WOPTCW		72
		   "BREAKOPT_WOPTCPROJ",// NVP_BREAKOPT_WOPTCPROJ	73
		   "BREAKOPT_CHAM",	// NVP_BREAKOPT_CHAM		74

		   "CALCWAVEC_FFT",	// NVP_CALCWAVEC_FFT		75
		   "CALCWAVEC_PROJALL",	// NVP_CALCWAVEC_PROJALL	76
		   "CALCWAVEC_NORM",	// NVP_CALCWAVEC_NORM		77
		   "CALCWAVEC_CPU",	// NVP_CALCWAVEC_CPU		78
		   "CALCWAVEC_REDIS",	// NVP_CALCWAVEC_REDIS		79

		   "OO_MEMCPY",		// NVP_OO_MEMCPY		80
		   "OO_CORTHO",		// NVP_OO_CORTHO		81
		   "OO_WOPTCW",		// NVP_OO_WOPTCW		82
		   "OO_WOPTCPROJ",	// NVP_OO_WOPTCPROJ		83
		   "OO_COMPUTE",	// NVP_OO_COMPUTE		84

		   "SUBSPACEROT_MEMCPY",  // NVP_SUBSPACEROT_MEMCPY	85
		   "SUBSPACEROT_ADD",	  // NVP_SUBSPACEROT_ADD	86
		   "SUBSPACEROT_ZHEEVX",  // NVP_SUBSPACEROT_ZHEEVX	87
		   "SUBSPACEROT_DSYEV",	  // NVP_SUBSPACEROT_DSYEV	88
		   "SUBSPACEROT_ZHEEV",   // NVP_SUBSPACEROT_ZHEEV	89
		   "SUBSPACEROT_LINCOM",  // NVP_SUBSPACEROT_LINCOM	90

		   "PROALL",		// NVP_PROALL			91
		   "RPRO",		// NVP_RPRO			92
		   "RPRO_MALLOC",	// NVP_RPRO_MALLOC		93
		   "RPRO_MEMCPY",	// NVP_RPRO_MEMCPY		94
		   "RPRO_COMPUTE",	// NVP_RPRO_COMPUTE		95
		   "RPRO_RPROMUISP",	// NVP_RPROMUISP		96

		   "RPROMUISP",		// NVP_RPROMUISP		97
		   "RPROMUISP_MALLOC",	// NVP_RPROMUISP_MALLOC		98
		   "RPROMUISP_MEMCPY",	// NVP_RPROMUISP_MEMCPY		99
		   "RPROMUISP_COMPUTE",	// NVP_RPROMUISP_COMPUTE	100

		   "RPROMU",		// NVP_RPROMU			101
		   "RPROMU_MALLOC",	// NVP_RPROMU_MALLOC		102
		   "RPROMU_MEMCPY",	// NVP_RPROMU_MEMCPY		103
		   "RPROMU_COMPUTE",	// NVP_RPROMU_COMPUTE		104

		   "ECCP",		// NVP_ECCP			105
		   "ECCP_COMPUTE",	// NVP_ECCP_COMPUTE		106
		   "ECCP_CPU",		// NVP_ECCP_CPU			107

		   "HAMILTMU",		// NVP_HAMILTMU			108
		   "HAMILTMU_MALLOC",	// NVP_HAMILTMU_MALLOC		109
		   "HAMILTMU_VHAMIL",	// NVP_HAMILTMU_VHAMIL		110
		   "HAMILTMU_RACCMU",	// NVP_HAMILTMU_RACCMU		111
		   "HAMILTMU_KINHAMIL",	// NVP_HAMILTMU_KINHAMIL	112
		   "HAMILTMU_FREE",	// NVP_HAMILTMU_FREE		113

		   "RACC0MU",		// NVP_RACC0MU			114
		   "RACC0MU_MALLOC",	// NVP_RACC0MU_MALLOC		115
		   "RACC0MU_MERGE",	// NVP_RACC0MU_MERGE		116
		   "RACC0MU_MEMCPY",	// NVP_RACC0MU_MEMCPY		117
		   "RACC0MU_COMPUTE",	// NVP_RACC0MU_COMPUTE		118

		   "ORTHCH",		// NVP_ORTHCH			119
		   "ORTHCH_INIT",	// NVP_ORTHCH_INIT		120
		   "ORTHCH_MALLOC",	// NVP_ORTHCH_MALLOC		121
		   "ORTHCH_MEMCPY",	// NVP_ORTHCH_MEMCPY		122
		   "ORTHCH_COMPUTE",	// NVP_ORTHCH_COMPUTE		123
		   "ORTHCH_CHOLESKI",	// NVP_ORTHCH_CHOLESKI		124
		   "ORTHCH_LINCOM",	// NVP_ORTHCH_LINCOM		125

		   "LINCOM",		// NVP_LINCOM			126
		   "LINCOM_MALLOC",	// NVP_LINCOM_MALLOC		127
		   "LINCOM_LINBAS",	// NVP_LINCOM_LINBAS		128

		   "LINBAS",		// NVP_LINBAS			129
		   "LINBAS_MEMCPY",	// NVP_LINBAS_MEMCPY		130
		   "LINBAS_COMPUTE",	// NVP_LINBAS_COMPUTE		131

		   "POTLOK",		// NVP_POTLOK			132
		   "POTLOK_ECDC",	// NVP_POTLOK_ECDC		133
		   "POTLOK_CPOT",	// NVP_POTLOK_CPOT		134
		   "POTLOK_TOTPOT",	// NVP_POTLOK_TOTPOT		135
		   "POTLOK_FFTEXC",	// NVP_POTLOK_FFTEXC		136
		   "POTLOK_POTHAR",	// NVP_POTLOK_POTHAR		137
		   "POTLOK_LPPOT",	// NVP_POTLOK_LPPOT		138
		   "POTLOK_CVTOTSV",	// NVP_POTLOK_CVTOTSV		139

		   "GGAALLGRID",	// NVP_GGAALLGRID		140
		   "GGAALLGRID_MALLOC", // NVP_GGAALLGRID_MALLOC	141
		   "GGAALLGRID_MEMCPY",	// NVP_GGAALLGRID_MEMCPY	142
		   "GGAALLGRID_COMPUTE",// NVP_GGAALLGRID_COMPUTE	143
		   
		   "REDIS_MPI",		// NVP_REDIS_MPI		144
		   "REDIS_COPY",	// NVP_REDIS_COPY		145

		   "REDIS_BANDS",	// NVP_REDIS_BANDS		146
		   "REDIS_NEWBAND",	// NVP_REDIS_NEWBAND		147
		   "REDIS_BANDOPT",	// NVP_REDIS_BANDOPT		148
		   "REDIS_CALCWAVEC",	// NVP_REDIS_CALCWAVEC		149
		   "REDIS_W1FFT",	// NVP_REDIS_W1FFT		150

		   "GPUDIRECT",		// NVP_GPUDIRECT		151

	 	   "EDDIAG",		// NVP_EDDIAG			152
		   "EDDIAG_MEMCPY",	// NVP_EDDIAG_MEMCPY		153
		   "EDDIAG_ZHEEVD",	// NVP_EDDIAG_ZHEEVD		154

                   "FSTRESS",           // NVP_FSTRESS                  155
                   "FSTRESS_CHARGE",    // NVP_FSTRESS_CHARGE           156
                   "FSTRESS_FORLOC",    // NVP_FSTRESS_FORLOC           157
                   "FSTRESS_DIRKAR",    // NVP_FSTRESS_DIRKAR           158
                   "FSTRESS_FOCK",      // NVP_FSTRESS_FOCK             159
                   "FSTRESS_CDIJ",      // NVP_FSTRESS_CDIJ             160
                   "FSTRESS_UNITCELL",  // NVP_FSTRESS_UNITCELL         161
                   "FSTRESS_COREC",     // NVP_FSTRESS_COREC            162
                   "FSTRESS_HARRIS",    // NVP_FSTRESS_HARRIS           163
                   "FSTRESS_MIX",       // NVP_FSTRESS_MIX              164
                   "FSTRESS_SUMCELL",   // NVP_FSTRESS_SUMCELL          165
                   "FSTRESS_VDW",       // NVP_FSTRESS_VDW              166
                   "FSTRESS_SYMM",      // NVP_FSTRESS_SYMM             167

                   "FOCK_KPAR",         // NVP_FOCK_KPAR                168
                   "FOCK_FOCKFORCE",    // NVP_FOCK_FOCKFORCE           169
                   "FOCK_FORSYM",       // NVP_FOCK_FORSYM              170
                   "FOCK_FORNLR",       // NVP_FOCK_FORNLR              171
                   "FOCK_FORNL",        // NVP_FOCK_FORNL               172
                   "FOCK_FORDEP",       // NVP_FOCK_FORDEP              173

                   "UNITCELL_STRKIN",   // NVP_UNITCELL_STRKIN          174
                   "UNITCELL_STRELO",   // NVP_UNITCELL_STRELO          175
                   "UNITCELL_STRNLR",   // NVP_UNITCELL_STRNLR          176
                   "UNITCELL_STRENL",   // NVP_UNITCELL_STRENL          177
                   "UNITCELL_STRDEP",   // NVP_UNITCELL_STRDEP          178
                   "UNITCELL_STRETAU",  // NVP_UNITCELL_STRETAU         179

                   "HARRIS_CHGGRA",     // NVP_HARRIS_CHGGRA            180
                   "HARRIS_FORHAR",     // NVP_HARRIS_FORHAR            181
                   "HARRIS_GAUSSIAN"    // NVP_HARRIS_GAUSSIAN          182
		  };
int num_timers = sizeof(timers)/sizeof(char *);
#endif

extern "C"
void nvp_malloc_C(int *n)
{
#ifdef USE_CPUTIMERS
    // allocate cpu timer arrays for each mpi process
    t0 = (struct timespec*)calloc(*n,sizeof(faket));
    t1 = (struct timespec*)calloc(*n,sizeof(faket));
    tsum = (double *)calloc(*n,sizeof(double));
#endif
}
extern "C"
void nvp_free_C()
{
#ifdef USE_CPUTIMERS
    // free cpu timer arrays for each mpi process
    free(t0); free(t1); free(tsum);
#endif
}

extern "C"
void nvp_start_C(int *tid)
{
#ifdef USE_NVTX
    // nvtx markers
    int cid = (*tid)%num_colors;
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = colors[cid];
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = timers[*tid];
    nvtxRangePushEx(&eventAttrib);
#endif

#ifdef USE_CPUTIMERS
    // cpu timers
    clock_gettime(CLOCK_REALTIME,&t0[*tid]);
#endif
}

extern "C"
void nvp_stop_C(int *tid)
{
#ifdef USE_NVTX
    // nvtx markers
    nvtxRangePop();
#endif

#ifdef USE_CPUTIMERS
    // cpu timers
    double elapsed;
    clock_gettime(CLOCK_REALTIME,&t1[*tid]);
    elapsed=(t1[*tid].tv_sec-t0[*tid].tv_sec)+1.0*(t1[*tid].tv_nsec-t0[*tid].tv_nsec)/BILLION;
    tsum[*tid] += elapsed;
#endif
}

extern "C"
void nvp_print_C(int *nid)
{
#ifdef USE_CPUTIMERS
    FILE *fp;
    char filename[32];
    double total;
    int i;

    sprintf(filename,"nvp%d.dat",*nid);
    printf("rank %d dumping timers to file %s...\n",*nid,filename);
    printf("nvp_print: nid=%d, t0=0x%x, tsum=0x%x\n",*nid,&t0[0],&tsum[0]);

    fp=fopen(filename,"w");
    fprintf(fp,"\n---------------------------------------------\n");
    fprintf(fp,"Marker:\tTime(s)\tPercentage\n");
    total = tsum[0];
    for(i=0;i<num_timers;i++)
        fprintf(fp,"%s\t  %f\t %.1f\n",timers[i],tsum[i],tsum[i]/total*100);
    fprintf(fp,"---------------------------------------------\n");
    fclose(fp);
#endif
}

/******************************************************/
