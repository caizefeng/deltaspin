#include "basis.hpp"
#include <iostream>

void assemble(t_basis_vector* bvector, 
	      const site_info& site,
	      const func_info& func,
	      const t_modifier& mod){
    
    t_basis tb;

    tb.radial_source = mod.radial_source;
    tb.index = mod.index;
    tb.shell = mod.shell;
    tb.sigma = mod.sigma;

    tb.posx = site.posx;
    tb.posy = site.posy;
    tb.posz = site.posz;
    tb.poscar = site.poscar;

    tb.l = func.l;
    tb.m = func.m;

    bvector->push_back(tb);
}

void assemble_all(t_basis_vector*  pbv, 
		  t_site_vector*   psv,
		  t_func_vector*   pfv,
		  const t_modifier& mod
    ){

    for (t_site_vector::const_iterator ps = psv->begin(); ps != psv->end(); ++ps)
	for (t_func_vector::const_iterator pf = pfv->begin(); pf != pfv->end(); ++pf){
	    assemble(pbv, *ps, *pf, mod);
	}

    delete psv;
    delete pfv;
}

void output_basisfunctions(t_basis_vector* pbv){
    for (t_basis_vector::const_iterator 
	     p = pbv->begin(); p != pbv->end(); ++p)
    {
	if (p->poscar == -1){
	    std::cout << "A site on position (" << p->posx << ',' 
		      << p->posy << ',' << p->posz << "), ";
	}else{
	    std::cout << "A site on index " << p->poscar << " of the POSCAR, ";
	}

	std::cout << " Y_" << p->l << '_' << p->m << " ";

	switch (p->radial_source){
	case 1: 
	    std::cout << "Using the PAW function ";
	    if (p->shell != -1){
		std::cout << "of shell " << p->shell << '\n';
	    }else{
		std::cout << "of the outermost shell\n"; 
	    };
	    break;
	case 2: 
	    std::cout << "Using the POTCAR potential number " << p->index;
	    if (p->shell != -1){
		std::cout << " of shell " << p->shell << '\n';
	    }else{
		std::cout << " of the outermost shell\n";
	    }
	    break;
	case 3:
	    std::cout << "Using the WANNIER function\n";
	    std::cout << "with shell " << p->shell << '\n';
	    std::cout << "and sigma "  << p->sigma << '\n';
	}
    }
}
