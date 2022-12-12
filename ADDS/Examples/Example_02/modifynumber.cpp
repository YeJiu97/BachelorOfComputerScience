
#include <iostream>
using namespace std;

void modifyNumber(int number, int numbers[]){
    number=1001;
    numbers[0]=5;
    
}

int main(void) {
    int x=1;
    int y[10];
    
    modifyNumber(x,y);
    
    cout << "x is " << x << endl;
    cout << "y[0] is " << y[0] << endl;
    
    return 0;

}
