          TYPE hills_io
            INTEGER :: STRUCTINPUT
            INTEGER :: PENALTY
          END TYPE hills_io
          
          TYPE penalty_data
            INTEGER         :: number
            TYPE(gauss_peak),POINTER :: gauss(:)
            REAL(q),POINTER :: force(:)
            REAL(q),POINTER :: wall(:,:)
          END TYPE penalty_data

          TYPE gauss_peak
            REAL(q),POINTER :: position(:)
            REAL(q)         :: high
            REAL(q)         :: width
          END TYPE gauss_peak


	  TYPE hills_data
            TYPE(gauss_peak),POINTER :: gauss(:)
            REAL(q),POINTER :: velocity(:)
            REAL(q),POINTER :: position(:)
            REAL(q),POINTER :: mass(:)
            REAL(q),POINTER :: force_constant(:)
            REAL(q),POINTER :: force(:)
            REAL(q)         :: SNOSE(4)
            REAL(q)         :: SQQ
            REAL(q)         :: potential
            REAL(q)         :: stride 
            REAL(q)         :: andersen_prob
            REAL(q)         :: temperature        !temperature of fict. particles
            INTEGER         :: number
            INTEGER         :: bin
            INTEGER         :: maxstride
            LOGICAL         :: variable_width
          END TYPE
