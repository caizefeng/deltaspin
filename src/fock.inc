!
!  structure required for kpoints generation
!
      TYPE kpoints_fock
!only  KPOINTS_F
        INTEGER NKPTS                   ! actual number of k-points
        REAL(q),POINTER :: VKPT(:,:)    ! coordinate of k-point
        REAL(q),POINTER :: WTKPT(:)     ! symmetry weight-factor for each k-point
        INTEGER,POINTER :: NEQUIV(:)    ! equivlist full kp-ibzkpt
        INTEGER,POINTER :: IROTOP(:,:,:)! rotation from ibzkpt to full pt
        REAL(q),POINTER :: TRANS(:,:)   ! possible nontrivial translation
        LOGICAL,POINTER :: LINV(:)      ! mirror point?
        COMPLEX(q),POINTER:: PHASE(:,:) ! phase shift for PW components 
        INTEGER,POINTER :: NOP(:)       ! index of symmetry operation
        INTEGER  NKPX, NKPY, NKPZ       ! integer division along rec. lattice vectors for generating the mesh
        INTEGER NKPTS_RED               ! number of k-points in the reduced mesh
        INTEGER,POINTER :: INDEX(:)     ! index array pointing to the full mesh
      END TYPE
