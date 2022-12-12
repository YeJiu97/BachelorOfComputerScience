
#include <iostream>
using namespace std;
int main(void) {
    int x, *xp;
    double y, *yp;
    char z, *zp;
    class{
        int temp[1000];
    } c, *cp;
    
    cout << sizeof(x) << endl;
    cout << sizeof(y) << endl;
    cout << sizeof(z) << endl;
    cout << sizeof(c) << endl;
    
    cout << sizeof(xp) << endl;
    cout << sizeof(yp) << endl;
    cout << sizeof(zp) << endl;
    cout << sizeof(cp) << endl;
    
    cout << sizeof(*xp) << endl;
    cout << sizeof(*yp) << endl;
    cout << sizeof(*zp) << endl;
    cout << sizeof(*cp) << endl;
    
    
    
}
