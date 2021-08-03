	block data block_petimer
c
        INCLUDE 'ptimers.h'
        data apetname /maxpet*'NIL'/

        end
c
	subroutine petimer (i,aname)
c
c	initialize timer for code chunk "aname"
c
	character*(*) aname

        INCLUDE 'ptimers.h'
c
c
	if (i.gt.maxpet) then
	   write(6,*) 'petimer:  i-index exceeds timing dimension'
	   write(6,*) 'petimer:  i=',i
	else
       	   apetname(i) = aname
	   mflop(i)    = 0.0D+00
	   cputot(i)   = 0.0D+00
           numcal(i)   = 0
	endif
	return
	end
	subroutine peton (i)
c
c	turn on timing for code chunk "aname"
c
        INCLUDE 'ptimers.h'
c
        cpu0(i) = etime_(tarray)	
	return
	end
	subroutine petoff (i,dmflop)
c
c	turn off timing for code chunk "aname"
c
	real*8 dmflop

        INCLUDE 'ptimers.h'
c
c
	t_cpu = etime_(tarray)
	cputot(i) = cputot(i) + t_cpu - cpu0(i)	
	mflop(i)  = mflop(i) + dmflop
        numcal(i) = numcal(i)+1
	return
	end
	subroutine petprint ()
c
c	print out results
c
	real*8 mflops
	character*32 amessag

        INCLUDE 'ptimers.h'
c
c
	write(6,1010)
	do 100 i = 1,maxpet
	   if(apetname(i).ne.'NIL') then
	      if(numcal(i).gt.0.and.cputot(i).ne.0.D+00)  then
	         mflops = mflop(i)/cputot(i)
                 tcall  = cputot(i)/numcal(i)
		 amessag= ' '
	      else
		 mflops = 0.D+00
                 tcall  = 0.D+00
		 amessag= '...too short to be timed'
                 if (numcal(i).eq.0) amessag= '...never called'
	      endif
	      write(6,1000) apetname(i),cputot(i),tcall,mflops,amessag
	   endif
  100   continue

	return
c
 1010   format(/,3x,'        CODE CHUNK TIMED        ',
     &          '   ELAPSED  PER CALL',
     &          '    MFLOPS',/)

 1000   format(3x,a,f10.1,f10.2,f10.1,a24)
	end
	function   etime_ (tarray)
        REAL*8 TV,TC
        INCLUDE 'ptimers.h'

        CALL VTIME(TV,TC)

        etime_ = TV 
        return
        end
