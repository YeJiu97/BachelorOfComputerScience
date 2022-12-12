
#include "BookOp.h"


#include <iostream>
#include <string>

using namespace std;

Book::Book() {};
Book::Book(string t, string a, string cn) : title(t), author(a), call_no(cn), onLoan(false) {}
	
bool Book::borrow() {
	if (onLoan) {
		cout << "Book already on load!!";
	} else {
		onLoan = true;
		cout << "Book borrowed";
	}
}

string Book::getCallNo() {
	return call_no;
}

string Book::getTitle() {
	return title;
}

string Book::getAuthor() {
	return author;
}


bool operator==(Book b1, Book b2) {
	return ( b1.title == b2.title && b1.author == b2.author);
}

int operator < (Book b1, Book b2) {
	return (b1.title).compare(b2.title);
}

