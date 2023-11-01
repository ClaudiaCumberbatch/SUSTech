%{
    #define YYSTYPE char *
    #include "lex.yy.c"
    #include "stdlib.h"
    int yyerror(char* s);
%}

%token X
%token DOT
%token COLON

%%
Fair: IPv4 { printf("IPv4\n");}
    | IPv6 { printf("IPv6\n");}
    ;
IPv4: FairX DOT FairX DOT FairX DOT FairX;
FairX: X { 
    if (strlen($1)>1 && $1[0] == '0') {
        yyerror("");
        return 0;
    }
    else if (atoi($1) > 255 || atoi($1) < 0) {
        yyerror("");
        return 0;
    }
};
IPv6: Fair6 COLON Fair6 COLON Fair6 COLON Fair6 COLON Fair6 COLON Fair6 COLON Fair6 COLON Fair6;
Fair6: X {
    if (strlen($1)<1 || strlen($1)>4) {
        yyerror("");
        return 0;
    }
};
%%

int yyerror(char* s) {
    fprintf(stderr, "%s\n", "Invalid");
    return 1;
}
int main() {
    yyparse();
}
