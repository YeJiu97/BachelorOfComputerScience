
#include <iostream>

int main(void) {
    // insert code here...
    char c='c';
    
    char *cp = &c;
    int i = 100;
    int *ip=&i;
    int j=*ip;
    
    *ip=99;
    std::cout<< i << std::endl;
    
    *cp=*cp+1;
    std::cout << c << std::endl;
    
    std::cout << &ip << std::endl;
    
    std::cout << &i << std::endl;
    std::cout << j << std::endl;
    std::cout << ip << std::endl;
    
    char cc = c+1;
    std::cout << cc << std::endl;
    
}
