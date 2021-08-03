#include "sites.hpp"
#include <iostream>

void push_back_fromto(std::vector<site_info>* psi, int from, int to){
    if (from > to) return;
    for (int i = from; i <= to; ++i){
	psi->push_back(site_info(i));
    }
}

void output_sitelist(std::vector<site_info>* vp){
    for (std::vector<site_info>::const_iterator 
	     p = vp->begin(); p != vp->end(); ++p)
    {
	if (p->poscar == -1){
	    std::cout << "A site on position (" << p->posx << ',' 
		      << p->posy << ',' << p->posz << ")\n";
	}else{
	    std::cout << "A site on index " << p->poscar << " of the POSCAR\n";
	}
    }
}




