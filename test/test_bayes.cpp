#include <iostream>
#include "../bayes/bayes.h"
using namespace std;
using namespace ML;
void test_text()
{
	Bayes b;
	b.load("./data/text2");
	b.calculatePrior();
	b.show();
	string sample1 = "love my dalmation";
	cout<<sample1<<endl;
	string label = b.classify(sample1);
	cout<<"label: "<<label<<endl;
	string sample2 = "stupid worthless garbage";
	cout<<sample2<<endl;
	label = b.classify(sample2);
	cout<<"label: "<<label<<endl;
    return;
}
int main()
{
    test_text();
	return 0;
}
