
#include <iostream>
#include <string>

#include "BookOp.h"
using namespace std;



int main(){
	Book b1("Norton's Star Atlas", "A. P. Norton", "510.3");
	Book b2("Norton's Star Atlas", "A. P. Norton", "510.3(C2)");
	Book b3("Handbook of DeepSky Objects", "Skiff", "510.2(C1)");
	
	cout << "Are books equal? " << (b1 == b2)  <<"\n" ;
	cout << "Compare B1 and B3:  " << (b1 <  b3)  << "\n" ;
	
}


