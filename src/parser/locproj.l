%{
#include <iostream>
#include <cstring>
#include "sites.hpp"
#include "functions.hpp"
#include "radial.hpp"
#include "basis.hpp"
#include "locproj.tab.h"
using namespace std;

#define YY_DECL extern "C" int yylex()
int line_num = 1;
int return_to;
%}
%x FOUND COMMENT
%%
<INITIAL>{
      (?i:LOCPROJ)   { BEGIN(FOUND); yylval.sval = strdup(yytext); return KEYWORD; };
      \n             { ++line_num; };
}
<FOUND>{
      [0-9]+\.[0-9]+ { yylval.fval = atof(yytext); return FLOAT; };
      [0-9]+\.+      { yylval.fval = atof(yytext); return FLOAT; };
   \-+[0-9]+\.[0-9]+ { yylval.fval = atof(yytext); return FLOAT; };
         \-+\.[0-9]+ { yylval.fval = atof(yytext); return FLOAT; };
   \++[0-9]+\.[0-9]+ { yylval.fval = atof(yytext); return FLOAT; };
         \++\.[0-9]+ { yylval.fval = atof(yytext); return FLOAT; };
   \-+[0-9]+         { yylval.ival = atoi(yytext); return INT; };
   \++[0-9]+         { yylval.ival = atoi(yytext); return INT; };
      [0-9]+         { yylval.ival = atoi(yytext); return INT; };
      (?i:PR)        { return PAW_PR_MOD; };
      (?i:PS)        { return PAW_PS_MOD; };
      (?i:HY)        { return RHYDRO_MOD; };
      [a-zA-Z]       { yylval.sval = strdup(yytext); return STRING; };
      [a-zA-Z][a-zA-Z0-9\-)(]+   { yylval.sval = strdup(yytext); return STRING; };
      "="            { return EQUAL; };
      ":"            { return DELIM1; };
      "-"            { return DASH; };
      "|"            { return DELIM2; };
      "("            { return OPENKL; };
      ")"            { return CLOSEKL; };
      ","            { return KOMMA; };
      \\             { return_to = FOUND; BEGIN(COMMENT); };
      ";"            { BEGIN(INITIAL); };
      \n             { ++line_num; BEGIN(INITIAL); };
}
<COMMENT>{
      \n             { ++line_num; BEGIN(return_to); };
}
<*>{
      \#             { return_to = INITIAL; BEGIN(COMMENT); };
      [ \t]           ;
      .               ;
}
%%
