
#include<iostream>
using namespace std;

const int n = 20;
const int numcoins = 6;


void printTable(int** table){
    for (int i=0; i<=n; i++){
    for (int j=0; j<numcoins; j++){
            
                cout << table[i][j]<< " ";
        }
            cout << endl;
    }
    cout << endl;
}


int main(void) {
    
    int facevalue[] ={5,10,20,50,100,200};
    
    int** dptable = new int*[n+1];
    for(int i = 0; i <= n; i++){
        dptable[i] = new int[numcoins];
    }
    
    cout << "Print table" << endl;
    printTable(dptable);
    for (int i=0; i< numcoins; i++){
        dptable[0][i]=1;
    }
    
    cout << "Print table"<<endl;
    printTable(dptable);
    int j=0;
    for(int i=0; i<=n; i++){
        if (i>=facevalue[j]){
        dptable[i][j] = dptable[i-facevalue[j]][j];
    }
    }
        
      cout << "Print table after coin with facevalue: " << facevalue[j]<<endl;
        printTable(dptable);
    
    for (int j=1; j< numcoins; j++){
        
        for(int i=0; i<=n; i++){
            if (i>=facevalue[j]){
                dptable[i][j] = dptable[i][j-1] + dptable[i-facevalue[j]][j];
            }
            else{
                dptable[i][j] = dptable[i][j-1];
            }
            
        }
        cout << "Print table after considering in addition coin with facevalue: " << facevalue[j]<<endl;
        printTable(dptable);
    }
    
    
}


