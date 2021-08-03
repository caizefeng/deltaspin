#ifndef BASIS_VPARSE_HPP
#define BASIS_VPARSE_HPP

#include "radial.hpp"
#include "sites.hpp"
#include "functions.hpp"


struct t_basis{
// radial part

    int radial_source;
    int index;
    int shell;

// spherical part

    int l;
    int m;

// site part
    
    int poscar;

    double posx;
    double posy;
    double posz;

// radial part, double
    double sigma;
};
    
    typedef std::vector<t_basis> t_basis_vector;

    void output_basisfunctions(
    t_basis_vector* pbv);

    void assemble(t_basis_vector*, 
	const site_info&,
	const func_info&,
	const t_modifier&);

    void assemble_all(t_basis_vector*, 
	t_site_vector*,
	t_func_vector*,
	const t_modifier&
	);

#endif
