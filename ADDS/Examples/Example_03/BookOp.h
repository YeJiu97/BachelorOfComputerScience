#ifndef BOOKOP_H
#define BOOKOP_H
using namespace std;
#include <string>

class Book {

	
friend  bool operator==(Book, Book);
friend int operator < (Book, Book);

public:
	Book();
	Book(string t, string a, string cn);
	
	bool borrow();
	string getCallNo();
	string getTitle();
	string getAuthor();
	
private:
	string call_no;
	string title;
	string author;
	bool onLoan;
};
#endif
