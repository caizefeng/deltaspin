#ifndef FUNCTIONS_VPARSE_HPP
#define FUNCTIONS_VPARSE_HPP

#include <vector>
#include <string>

struct func_info{
    int l;
    int m;

    func_info(int l_, int m_): l(l_), m(m_){};
};

typedef std::vector<func_info> t_func_vector;

int interpret_function_string(t_func_vector*, const std::string&);
void output_funclist(t_func_vector*);

#endif
