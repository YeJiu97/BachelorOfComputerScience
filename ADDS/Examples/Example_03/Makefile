all: program

program: Driver.o BookOp.o
	g++ BookOp.o Driver.o -o Driver

Driver.o: Driver.cpp
	g++ -c Driver.cpp

BookOp.o: BookOp.h BookOp.cpp
	g++ -c BookOp.cpp

clean:
	rm *.o
	rm Driver
