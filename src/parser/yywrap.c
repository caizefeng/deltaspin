#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

extern FILE *yyin, *yyout;

extern "C" int yywrap() {
    return (feof(yyin));
}

