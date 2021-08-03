      TYPE dyn_storage
         INTEGER :: entry
         COMPLEX(q), POINTER, DIMENSION(:) :: store
         TYPE(dyn_storage), POINTER :: next
      END TYPE
      TYPE (dyn_storage),POINTER :: tmp_storage
      TYPE (dyn_storage),POINTER :: broyden_storage

