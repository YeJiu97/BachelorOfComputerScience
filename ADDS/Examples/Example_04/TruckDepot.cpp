#include "TruckDepot.h"
#include <cmath>
#include <iostream>

int TruckDepot::numTrucks(int pileSize, int truckCapacity){
    int result =-1;
    int pile1;
    int pile2;
 //   std::cout << "PileSize: " << pileSize << std::endl;
    
    if (pileSize<= truckCapacity){
        result =1;
    }
    else{
        if (pileSize %2 ==0){
            pile1=pileSize/2;
            pile2 =pileSize/2;
        }else{
            pile1=pileSize/2+1;
            pile2=pileSize/2;
        }
        
        result = numTrucks(pile1,truckCapacity) + numTrucks(pile2,truckCapacity);
    }
    std::cout<<"Number of Trucks for pileSize " << pileSize <<": " << result <<std::endl;
    return result;
}
