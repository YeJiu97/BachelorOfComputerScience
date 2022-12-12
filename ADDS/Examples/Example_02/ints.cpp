
#include <iostream>
using namespace std;



int main(void) {
    int* p;
    for (int i=0; i< 1000000; i++){
        cout << "allocate" << endl;
        p=new int[100000000];
        delete p;
    }
}
