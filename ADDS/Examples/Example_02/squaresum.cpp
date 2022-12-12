
#include <iostream>
using namespace std;

int square (int n){
    n*=n;
    return n;
}

int squaresum(int n){
    int result=0;
    while(n>1){
        result+=square(n);
        n--;
    }
    return result;
}

int main(void) {
    int n=3;
    cout << squaresum(n) << endl;
    return 0;

}
