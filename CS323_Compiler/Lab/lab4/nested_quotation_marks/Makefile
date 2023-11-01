CC=gcc
FLEX=flex
nest:
	$(FLEX) lex.l
	$(CC) lex.yy.c -o nest.out
clean:
	@rm -f lex.yy.c *.out
.PHONY: nest
