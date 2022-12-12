
#include <iostream>
using namespace std;



int main(void) {
    int* p = new int;
    *p=100;
    cout << *p << endl;
    p=p+1;
    cout << *p << endl;
    delete p;
   
}
