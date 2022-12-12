#include<iostream>
using namespace std;


void hanoi(int n,char from, char help,char to)
{
    if(n==1)
    {
        cout<<"Move Disk "<<n<<" from "<<from<<" to "<<to<<endl;
        return;
    }
    
    hanoi(n-1,from,to,help);
    cout<<"Move Disk "<<n<<" from "<<from<<" to "<<to<<endl;
    hanoi(n-1,help,from,to);
}

//main program
int main()
{
    int n;
    
    n=3;
    hanoi(n,'F','H','T');
    
    return 0;
}
