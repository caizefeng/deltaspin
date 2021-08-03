#ifndef SITES_VPARSE_HPP
#define SITES_VPARSE_HPP

#include <vector>
#include <list>

struct site_info{

    site_info(float px, float py, float pz):posx(px),posy(py),posz(pz),poscar(-1){};
    site_info(int site):posx(0),posy(0),posz(0),poscar(site){};

    float posx;
    float posy;
    float posz;

    int poscar;

};

typedef std::vector<site_info> t_site_vector;

void output_sitelist(t_site_vector*);
void push_back_fromto(t_site_vector*, int from, int to);

#endif
