
#include <iostream>
using namespace std;

void foo(){
    cout << "recurse"<< endl;
    foo();
}

int main(void) {
    foo();

}
