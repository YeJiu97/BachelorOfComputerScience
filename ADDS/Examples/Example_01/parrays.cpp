
#include <iostream>
using namespace std;
int main(void) {
    // insert code here...
    char s1[]="abc";
    char* s2="def";
    char* p1=s1;
    
    cout << &s1 <<"\n";
    cout << s1 <<"\n";
    cout << s2 <<"\n";
    cout << *s2 <<"\n";
    cout << p1 <<"\n";
    cout << &p1 <<"\n";
    
    cout << "end \n";
    s1[0]='z';
    s2+=1;
    p1+=1;
    
    cout << s1[0] <<"\n";
    cout << s1 <<"\n";
    cout << s2 <<"\n";
    cout << p1 <<"\n";
    
}
