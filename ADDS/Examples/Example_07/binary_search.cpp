#include<iostream>
#include<vector>

using namespace std;
// see also binary_search, bsearch

template<class T>
bool adds_binary_search(vector<T> data, T obj) {

    int start = 0, end = data.size() - 1;

    while (start + 1 < end) {
        int mid = (start + end) / 2;
        if (data[mid] == obj)
            return true;
        else if (data[mid] > obj) 
            end = mid - 1;
        else 
            start = mid + 1;
    }

    if ((data[start] == obj) || (data[end] == obj))
        return true;

    return false;
}

int main() {
    vector<int> data_vector;

    for (int i = 1; i <= 10; i++)
        data_vector.push_back(i);

    cout << adds_binary_search(data_vector, 0) << endl;
    cout << adds_binary_search(data_vector, 1) << endl;
    cout << adds_binary_search(data_vector, 2) << endl;
    cout << adds_binary_search(data_vector, 3) << endl;
    cout << adds_binary_search(data_vector, 4) << endl;
    cout << adds_binary_search(data_vector, 5) << endl;
    cout << adds_binary_search(data_vector, 6) << endl;
    cout << adds_binary_search(data_vector, 7) << endl;
    cout << adds_binary_search(data_vector, 8) << endl;
    cout << adds_binary_search(data_vector, 9) << endl;
    cout << adds_binary_search(data_vector, 10) << endl;
    cout << adds_binary_search(data_vector, 11) << endl;
}
