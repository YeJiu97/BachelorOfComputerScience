#include <iostream>
using namespace std;
class CircularArray {
private:
int size;
int* array;
int arrayIndex;
public:
CircularArray(int newSize);
void addElement(int element);
int  getElementAt(int index);
void printElements();
~CircularArray();
};

CircularArray::~CircularArray() {
delete array;
}

CircularArray::CircularArray(int newSize) {
size = newSize;
array = new int[size];
arrayIndex = 0;
for (int i =0; i < size; i++) {
array[i] =0;
}
}

void CircularArray::addElement(int element) {
array[arrayIndex++] = element;
if (arrayIndex >= size){
arrayIndex= arrayIndex % size;
}
}

int CircularArray::getElementAt(int index) {
return array[index % size];
}

void CircularArray::printElements() {
for (int i =0; i < size; i++) {
cout << array[i] << " " ;
}
cout << "\n";
}

int main () {
    CircularArray C(3);
    C.addElement(5);
    C.addElement(7);
    C.addElement(9);
    cout << C.getElementAt(0) << endl;
    C.printElements();
    C.addElement(3);
    C.printElements();
    cout << C.getElementAt(0) << endl;
    
    
}
