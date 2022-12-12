
#include <iostream>
using namespace std;
int main() {
    int i=100;
    int* ip = &i;
    cout << ip << endl;
    cout << ip+1 << endl;
    
    double d=5.0;
    double*dp = &d;
    cout << dp << endl;
    cout << dp+1 << endl;
    
    return 0;
}
