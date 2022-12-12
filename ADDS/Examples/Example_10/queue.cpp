#include<iostream>
#include<stdexcept>

using namespace std;
// see also: std::queue

template<class T>
class Queue {
    public:
        Queue():top(0),tail(0) {}

        void push(const T&);
        T pop();

        bool empty() {return top == 0;}

        ~Queue() {while (!empty()) pop();}

    private:
        struct Node {
            T data;
            Node* link;
        } *top, *tail;
};

template<class T>
void Queue<T>::push(const T& obj) {
    Node* tmp = new Node;
    tmp->data = obj;
    tmp->link = 0;

    if (tail == 0) 
        top = tail = tmp;
    else {
        tail->link = tmp;
        tail = tmp;
    }
}

template<class T>
T Queue<T>::pop() {
    if (empty()) 
        throw runtime_error("Queue empty, cannot pop");
    
    Node* tmp = top;
    top = top->link;
    T data = tmp->data;
    delete tmp;

    if (top == 0)
        tail = 0;
    return data;
}

main() {
    Queue<double> myqueue;
    myqueue.push(3.3);
    myqueue.push(2.1);
    myqueue.push(0.7);
    cout << myqueue.pop() << ' ';
    cout << myqueue.pop() << ' ';
    cout << myqueue.pop() << ' ';
    cout << myqueue.pop() << ' ';
}
