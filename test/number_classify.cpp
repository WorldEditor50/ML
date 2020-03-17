#include <iostream>
#include <fstream>
#include <vector>
#include "../bpnn/bpnn.h"
using namespace std;
using namespace ML;

void show(vector<double>& x)
{
    for (int i = 0; i < x.size(); i++) {
        cout<<x[i]<<" ";
        if ((i + 1) % 7 == 0) {
            cout<<endl;
        }
    }
    cout<<endl;
    return;
}
int decode(vector<double>& y)
{
    int num = -1;
    for (int i = 0; i < y.size(); i++) {
        cout<<y[i]<<" ";
        if (y[i] >= 0.9) {
            num = i;
        }
    }
    cout<<endl;
    cout<<"num = "<<num<<endl;
    return num;
}
void number_classify01()
{
    BpNet bp;
    bp.createNet(10, 56, 64, 3, 0.01);
    bp.loadFeature("./data/number.txt", 10, 56);
    bp.loadTarget("./data/number_code.txt", 10, 10);
    bp.train(20000);
    bp.saveParameter("./data/number_classify_weights.txt");
    vector<vector<double> >& x = bp.features;
    for (int j = 0; j < 10; j++) {
        vector<double>& y = bp.feedForward(x[j]);
        decode(y);
    }
    return;
}
void number_classify02()
{
    BpNet bp;
    bp.createNet(10, 56, 64, 3, 0.01);
    bp.loadFeature("./data/number.txt", 10, 56);
    bp.loadTarget("./data/number_code.txt", 10, 10);
    bp.loadParameter("./data/number_classify_weights.txt");
    vector<vector<double> >& x = bp.features;
    show(x[9]);
    vector<double>& y = bp.feedForward(x[9]);
    decode(y);
    show(x[4]);
    bp.feedForward(x[4]);
    decode(y);
    return;
}
int main()
{
    srand((unsigned int)time(NULL));
    number_classify01();
    return 0;
}
