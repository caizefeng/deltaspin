    module c2f_interface
    use iso_c_binding
    implicit none

    type, bind(c) :: parser_function
        integer(c_int) :: radial_type
        integer(c_int) :: species
        integer(c_int) :: n
        integer(c_int) :: l
        integer(c_int) :: m
        integer(c_int) :: poscar_site
        real(c_double) :: posx
        real(c_double) :: posy
        real(c_double) :: posz
        real(c_double) :: sigma
    end type parser_function

    interface
        ! void free_parser_C()
        subroutine free_parser() bind(c,name="free_parser_C")
        end subroutine free_parser

        ! void fill_basis_info_C(t_basis *result, const int &i)
        subroutine fill_basis_info(theresult,i) bind(c,name="fill_basis_info_C")
            import
            type(parser_function) :: theresult
            integer(c_int) :: i
        end subroutine fill_basis_info

        ! void parse_file_C(int &n_basis, int &strlen, char *filename)
        subroutine parse_file(n_basis,strlen,filename) bind(c,name="parse_file_C")
            import
            integer(c_int) :: n_basis
            integer(c_int) :: strlen
            character(kind=c_char) :: filename
        end subroutine parse_file

        ! void vtime_C(double *vputim, double *cputim)
        subroutine vtime(vputim, cputim) bind(c,name="vtime_C")
            import
            real(c_double) :: vputim
            real(c_double) :: cputim
        end subroutine vtime

        ! void vtime2_C(double *cputim)
        subroutine vtime2(cputim) bind(c,name="vtime2_C")
            import
            real(c_double) :: cputim
        end subroutine vtime2

        ! void timing_C(int *mode, double *utime, double *stime, double *now, int *minpgf, int *majpgf,
        !               double *maxrsize, double *avsize, int *swaps, int *ios, int *cswitch, int *ierr)
        subroutine timing(mode,utime,stime,now,minpgf,majpgf,maxrsize,avsize,swaps,ios,cswitch,ierr) bind(c,name="timing_C")
            import
            real(c_double) :: utime
            real(c_double) :: stime
            real(c_double) :: now
            real(c_double) :: maxrsize
            real(c_double) :: avsize
            integer(c_int) :: mode
            integer(c_int) :: minpgf
            integer(c_int) :: majpgf
            integer(c_int) :: swaps
            integer(c_int) :: ios
            integer(c_int) :: cswitch
            integer(c_int) :: ierr
        end subroutine timing

        ! double erf(double x)
        function errf(x) bind(c,name="erf")
            import
            real(c_double), value :: x
            real(c_double) :: errf
        end function errf

        ! double erfc(double x)
        function errfc(x) bind(c,name="erfc")
            import
            real(c_double), value :: x
            real(c_double) :: errfc
        end function errfc

!        ! double errf_C(double *x)
!        function errf(x) bind(c,name="errf_C")
!            import
!            real(c_double) :: x
!            real(c_double) :: errf
!        end function errf
!
!        ! double errfc_C(double *x)
!        function errfc(x) bind(c,name="errfc_C")
!            import
!            real(c_double) :: x
!            real(c_double) :: errfc
!        end function errfc

        !void getshmem_C(size_t *size, int*id)
        subroutine getshmem(thesize, id) bind(c,name="getshmem_C")
            import
            integer(c_size_t) :: thesize
            integer(c_int)    :: id
        end subroutine getshmem

        !void getshmem_error_C(size_t *size, int*id)
        subroutine getshmem_error(thesize, id) bind(c,name="getshmem_error_C")
            import
            integer(c_size_t) :: thesize
            integer(c_int)    :: id
        end subroutine getshmem_error
        
        !void attachshmem_C(int *shmid, void **address)
        subroutine attachshmem(shmid, address) bind(c,name="attachshmem_C")
            import
            integer(c_int)  :: shmid
            type(c_ptr)     :: address
        end subroutine attachshmem

        !void detachshmem_C(void **address)
        subroutine detachshmem(address) bind(c,name="detachshmem_C")
            import
            type(c_ptr)     :: address
        end subroutine detachshmem

        !void destroyshmem_C(int *shmid)
        subroutine destroyshmem(shmid) bind(c,name="destroyshmem_C")
            import
            integer(c_int)    :: shmid
        end subroutine destroyshmem

        !void getsem_C(int *size, int*id)
        subroutine getsem(thesize, id) bind(c,name="getsem_C")
            import
            integer(c_int)    :: thesize
            integer(c_int)    :: id
        end subroutine getsem

        !void locksem_C(int *semid, int* id)
        subroutine locksem(semid, id) bind(c,name="locksem_C")
            import
            integer(c_int)  :: semid
            integer(c_int)  :: id
        end subroutine locksem

        !void unlocksem_C(int *semid, int* id)
        subroutine unlocksem(semid, id) bind(c,name="unlocksem_C")
            import
            integer(c_int)  :: semid
            integer(c_int)  :: id
        end subroutine unlocksem

        !void destroysem_C(int *semid)
        subroutine destroysem(semid) bind(c,name="destroysem_C")
            import
            integer(c_int)  :: semid
        end subroutine destroysem

    end interface

    end module c2f_interface
