
#include <iostream>
using namespace std;

int fac (int n){
    int value=0;
    if (n==1){
        value=1;
    }
    else
    {
        value = n*fac(n-1);
    }
    
    return value;
}



int main(void) {
    int n=10;
    cout << fac(n) << endl;
    return 0;

}
