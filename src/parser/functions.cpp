#include "functions.hpp"
#include "basis.hpp"
#include <iostream>
#include <cstdlib>

void yyerror(t_basis_vector*,const char *s);

int interpret_function_string(t_func_vector* pfv, const std::string& s){

// Table 3.1 and 3.2 in Wannier 90 user guide 
    if (s == "s"){
	pfv->push_back(func_info(0,1)); return 0;};

    if (s == "p"){
	pfv->push_back(func_info(1,1));
	pfv->push_back(func_info(1,2));
	pfv->push_back(func_info(1,3)); return 0;};
    if (s == "px"){pfv->push_back(func_info(1,1)); return 0;}
    if (s == "py"){pfv->push_back(func_info(1,2)); return 0;}
    if (s == "pz"){pfv->push_back(func_info(1,3)); return 0;}

    if (s == "d"){
	pfv->push_back(func_info(2,1));
	pfv->push_back(func_info(2,2));
	pfv->push_back(func_info(2,3));
	pfv->push_back(func_info(2,4));
	pfv->push_back(func_info(2,5)); return 0;};
    if (s == "dz2"   ){pfv->push_back(func_info(2,1)); return 0;}
    if (s == "dxz"   ){pfv->push_back(func_info(2,2)); return 0;}
    if (s == "dyz"   ){pfv->push_back(func_info(2,3)); return 0;}
    if (s == "dx2-y2"){pfv->push_back(func_info(2,4)); return 0;}
    if (s == "dxy"   ){pfv->push_back(func_info(2,5)); return 0;}

    if (s == "f"){
	pfv->push_back(func_info(3,1));
	pfv->push_back(func_info(3,2));
	pfv->push_back(func_info(3,3));
	pfv->push_back(func_info(3,4));
	pfv->push_back(func_info(3,5));
	pfv->push_back(func_info(3,6));
	pfv->push_back(func_info(3,7)); return 0;};
    if (s == "fz3"       ){pfv->push_back(func_info(3,1)); return 0;}
    if (s == "fxz2"      ){pfv->push_back(func_info(3,2)); return 0;}
    if (s == "fyz2"      ){pfv->push_back(func_info(3,3)); return 0;}
    if (s == "fz(fx2-y2)"){pfv->push_back(func_info(3,4)); return 0;}
    if (s == "fxyz"      ){pfv->push_back(func_info(3,5)); return 0;}
    if (s == "fx(x2-3y2)"){pfv->push_back(func_info(3,6)); return 0;}
    if (s == "fy(3x2-y2)"){pfv->push_back(func_info(3,7)); return 0;}

    if (s == "sp"){
	pfv->push_back(func_info(-1,1));
	pfv->push_back(func_info(-1,2)); return 0;};
    if (s == "sp-1"){pfv->push_back(func_info(-1,1)); return 0;};
    if (s == "sp-2"){pfv->push_back(func_info(-1,2)); return 0;};

    if (s == "sp2"){
	pfv->push_back(func_info(-2,1));
	pfv->push_back(func_info(-2,2));
	pfv->push_back(func_info(-2,3)); return 0;};
    if (s == "sp2-1"){pfv->push_back(func_info(-2,1)); return 0;};
    if (s == "sp2-2"){pfv->push_back(func_info(-2,2)); return 0;};
    if (s == "sp2-3"){pfv->push_back(func_info(-2,3)); return 0;};

    if (s == "sp3"){
	pfv->push_back(func_info(-3,1));
	pfv->push_back(func_info(-3,2));
	pfv->push_back(func_info(-3,3));
	pfv->push_back(func_info(-3,4)); return 0;};
    if (s == "sp3-1"){pfv->push_back(func_info(-3,1)); return 0;};
    if (s == "sp3-2"){pfv->push_back(func_info(-3,2)); return 0;};
    if (s == "sp3-3"){pfv->push_back(func_info(-3,3)); return 0;};
    if (s == "sp3-4"){pfv->push_back(func_info(-3,4)); return 0;};

    if (s == "sp3d"){
 	pfv->push_back(func_info(-4,1));
	pfv->push_back(func_info(-4,2));
	pfv->push_back(func_info(-4,3));
	pfv->push_back(func_info(-4,4));
	pfv->push_back(func_info(-4,5)); return 0;};
    if (s == "sp3d-1"){pfv->push_back(func_info(-4,1)); return 0;};
    if (s == "sp3d-2"){pfv->push_back(func_info(-4,2)); return 0;};
    if (s == "sp3d-3"){pfv->push_back(func_info(-4,3)); return 0;};
    if (s == "sp3d-4"){pfv->push_back(func_info(-4,4)); return 0;};
    if (s == "sp3d-5"){pfv->push_back(func_info(-4,5)); return 0;};

    if (s == "sp3d2"){
 	pfv->push_back(func_info(-5,1));
	pfv->push_back(func_info(-5,2));
	pfv->push_back(func_info(-5,3));
	pfv->push_back(func_info(-5,4));
	pfv->push_back(func_info(-5,5));
	pfv->push_back(func_info(-5,6)); return 0;};
    if (s == "sp3d2-1"){pfv->push_back(func_info(-5,1)); return 0;};
    if (s == "sp3d2-2"){pfv->push_back(func_info(-5,2)); return 0;};
    if (s == "sp3d2-3"){pfv->push_back(func_info(-5,3)); return 0;};
    if (s == "sp3d2-4"){pfv->push_back(func_info(-5,4)); return 0;};
    if (s == "sp3d2-5"){pfv->push_back(func_info(-5,5)); return 0;};
    if (s == "sp3d2-6"){pfv->push_back(func_info(-5,6)); return 0;};

    std::string msg("Unknown function specification \"");
    msg+=s;
    msg+='"';
    yyerror(NULL, msg.c_str());

    return -1;
}

void output_funclist(t_func_vector* vp){
    for (t_func_vector::const_iterator 
	     p = vp->begin(); p != vp->end(); ++p)
    {
	std::cout << " Y_" << p->l << '_' << p->m << " ";
    }
}
