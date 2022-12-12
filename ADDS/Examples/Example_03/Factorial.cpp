#include <iostream>
using namespace std;

int factorial(int n) {
	cout << "Factorial at n = " << n << endl;
	if (n < 0){
		cout << "ERROR!!!!111!!!! so forth\n";
		exit(1);
	}else 
		if (n == 0){
		return 1;
	}else{
		return n * factorial(n-1);
	}
}


int main (int argc, char* argv[]) {
	// default value is 10.
	int test=10;
	cout << "Number of arguments: " << argc << endl;
	cout << "My program name is " << argv[0] << endl;
	if (argc == 2) {
		// Someone has entered a command line argument
		// atoi will return 0 if it can't parse the number.
		test = atoi(argv[1]);
	} 
	cout << "Testing factorial function with test = " << test << " yields \n";
	cout << factorial(test) << endl;
	
}
