#include<iostream>
#include<stdexcept>

using namespace std;
// see also: std::stack

template<class T>
class Stack {
    public:
        Stack():top(0) {}

        void push(const T&);
        T pop();

        bool empty() {return top == 0;}

        ~Stack() {while (!empty()) pop();}

    private:
        struct Node {
            T data;
            Node* link;
        }* top;
};

template<class T>
void Stack<T>::push(const T& obj) {
    Node* tmp = new Node;
    tmp->data = obj;
    tmp->link = top;
    top = tmp;
}

template<class T>
T Stack<T>::pop() {
    if (empty()) 
        throw runtime_error("Stack empty, cannot pop");
    
    Node* tmp = top;
    top = top->link;
    T data = tmp->data;
    delete tmp;
    return data;
}

main() {
    Stack<double> mystack;
    mystack.push(3.3);
    mystack.push(2.1);
    mystack.push(0.7);
    cout << mystack.pop() << ' ';
    cout << mystack.pop() << ' ';
    cout << mystack.pop() << ' ';
    cout << mystack.pop() << ' ';
}
