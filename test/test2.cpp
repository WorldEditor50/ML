#include <iostream>
#include "bayes.h"
using namespace std;
using namespace ML;

int main()
{
	Bayes b;
	b.load("data_set2.txt");
	b.calculatePrior();
	//b.show();
	string sample1 = "hell is real";
	cout<<sample1<<endl;
	string label = b.classify(sample1);
	cout<<label<<endl;
	return 0;
}
