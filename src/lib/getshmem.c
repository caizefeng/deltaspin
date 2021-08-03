/*
 * Some functions, callable from fortran to use shared memory
 * wv 2011
 * shmem stuff from Willem Vermin, SARA
 */
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/sem.h>
/*
 * get a shared memory segment
 * input: size (fortran integer*8)
 * output: shmem id
 */
void getshmem_C(size_t *size, int*id)
{
  key_t key;
  int shmflg;
  int shmid;
  key = IPC_PRIVATE;
  shmflg = IPC_CREAT | IPC_EXCL | 0600 | SHM_NORESERVE ;
  shmid = shmget(key, *size, shmflg);
  if (shmid == -1)
  {
    fprintf(stderr,"%s in %s: cannot create shared segment %ld \n",__FUNCTION__,__FILE__,*size);
    perror(0);
    exit(1);
  }
  *id = shmid;
}

void getshmem_error_C(size_t *size, int*id)
{
  key_t key;
  int shmflg;
  int shmid;
  key = IPC_PRIVATE;
  shmflg = IPC_CREAT | IPC_EXCL | 0600 | SHM_NORESERVE ;
  shmid = shmget(key, *size, shmflg);
  *id = shmid;
}
/*
 * attach shared memory to a pointer
 * input: shhmid: shared memory id
 * output: address (fortran integer*8)
 */
void attachshmem_C(int *shmid, void **address)
{
  void *shmaddr,*r;
  int shmflg;
  shmflg = 0;
  shmaddr = 0;

  r = shmat(*shmid, shmaddr, shmflg);
  if (r == (void*) -1)
  {
    fprintf(stderr,"%s in %s: cannot get address of shared segment\n",__FUNCTION__,__FILE__);
    fprintf(stderr," this often means the cores do not share memory ");
    fprintf(stderr," maybe you have used -bynode, or NCSHMEM is too large ");
    perror(0);
    exit(1);
  }
  *address = r;
}
/*
 * detach shared memory from pointer
 * input: address (fortran integer*8)
 */
void detachshmem_C(void **address)
{
  int r;
  r = shmdt(*address);
  if (r == -1)
  {
    fprintf(stderr,"%s in %s:%d: cannot detach shared segment\n",__FUNCTION__,__FILE__,__LINE__);
    perror(0);
    exit(1);
  }
}
/*
 * destroy shared memory
 * input: shmid: shared memory id
 */
void destroyshmem_C(int *shmid)
{
  struct shmid_ds buf;
  int r = shmctl(*shmid,IPC_RMID,&buf);
  if (r == -1)
  {
    fprintf(stderr,"%s in %s:%d: cannot destroy shared segment\n",__FUNCTION__,__FILE__,__LINE__);
    perror(0);
    exit(1);
  }
}


/*
 * get a set of semaphores
 * input: size 
 * output: semaphores id
 */
void getsem_C(int *size, int*id)
{
  key_t key;
  int semflg;
  int semid;
  int mysize;
  int i, rc;
  short  sarray[(*size)+1];

  key = IPC_PRIVATE;
  semflg = IPC_CREAT | IPC_EXCL | 0600;
  semflg = IPC_CREAT | 0600;
  mysize=(*size)+1 ;
  /* here I have an odd problem: using *size+1 instead of mysize 
     sometimes failed
     very odd, some cross C-Fortran problem, no idea          */
  semid = semget(key, mysize, semflg);
  if (semid == -1)
  {
    perror(0);
    fprintf(stderr,"internal error in VASP %s in %s: cannot create semaphore \n",__FUNCTION__,__FILE__);
    fprintf(stderr,"requested size was %d\n", mysize);
    fprintf(stderr,"you need to increase the number of system V semaphores:\n");
    fprintf(stderr,"e.g. in LINUX try to increase SEMMNS in /proc/sys/kernel\n");
    exit(1);
  }
  /*        '1' --  The shared memory segment is being used.       */
  /*        '0' --  The shared memory segment is freed.            */
  /* the very first sempaphore can be  used to loc all others 
     currently this is not entirely thread save and might fail if no barrier
     is used before this lock, since we do not lock the slots individually  */
  for (i=0; i< mysize ; i++ )  sarray[i]=0 ;

  rc = semctl( semid, 1, SETALL, sarray);
  if(rc == -1)
  {
    fprintf(stderr,"%s in %s: cannot initialize semaphores \n",__FUNCTION__,__FILE__);
    perror(0);
    exit(1);
  }

  *id = semid;

}
/*
 * loc a specific semaphores in the set
 * we only support two states
 *        '1' --  The semaphore/shared memory segment is being used.
 *        '0' --  The semaphore/shared memory segment is freed.
 */
void locksem_C(int *semid, int* id)
{
  struct sembuf operations[2];
#ifdef debug
  fprintf(stderr,"%d: semaphores lock\n",*id) ;
#endif
  /* maybe the semphore is not properly set, then stop with internal error   */
  if (*semid==-1)
  {
    fprintf(stderr,"internal error in VASP: locksem was called with a uninitialize semaphores\n");
    fprintf(stderr,"response functions allocated in wpot.F are not protected by semaphores\n");
    perror(0);
    exit(1);
  }
  operations[0].sem_num = *id; /* Operate on the  sem id        */
  operations[0].sem_op =  0;   /* Wait for the value 0          */
  operations[0].sem_flg = 0;   /* Allow a wait to occur         */
  
  operations[1].sem_num = *id; /* Operate on the  sem is        */
  operations[1].sem_op =  1;   /* Increment the semval by 1     */
  operations[1].sem_flg = 0;   /* Allow a wait to occur         */
  
  int  rc = semop( *semid, operations, 2 );
  if (rc == -1)
  {
    fprintf(stderr,"%s in %s:%d: cannot lock semaphores \n",__FUNCTION__,__FILE__,__LINE__);
    perror(0);
    exit(1);
  }
#ifdef debug
  fprintf(stderr,"%d: semaphores done \n",*id) ;
#endif
}
/*
 * unloc a specific semaphores in the set
 * this will only succeed if the semaphores was previously locked
 * should be called only by the locing process
 */
void unlocksem_C(int *semid, int* id)
{
  struct sembuf operations[2];
  operations[0].sem_num = *id; /* Operate on the  sem id        */
  operations[0].sem_op =  -1;  /* decrease to value 0           */
  operations[0].sem_flg = IPC_NOWAIT;   /* Allow no  wait to occur       */
#ifdef debug
  fprintf(stderr,"%d: semaphores unlock \n",*semid) ;
#endif
  /* maybe the semphore is not properly set, then stop with internal error   */
  if (*semid==-1)
  {
    fprintf(stderr,"internal error in VASP: unlocksem was called with a uninitialize semaphores\n");
    fprintf(stderr,"response functions allocated in wpot.F are not protected by semaphores\n");
    perror(0);
    exit(1);
  }
  int  rc = semop( *semid, operations, 1 );
  if (rc == -1)
  {
    fprintf(stderr,"%s in %s:%d: cannot unlock semaphores \n",__FUNCTION__,__FILE__,__LINE__);
    perror(0);
    exit(1);
  }
#ifdef debug
  fprintf(stderr,"%d: semaphores clear done \n",*id) ;
#endif
}
/*
 * destroy semaphores
 * input: semid: semaphores id
 */
void destroysem_C(int *semid)
{
  int rc = semctl(*semid,1, IPC_RMID);
  if (rc == -1)
  {
    fprintf(stderr,"%s in %s:%d: cannot destroy semaphores \n",__FUNCTION__,__FILE__,__LINE__);
    perror(0);
    exit(1);
  }
}
