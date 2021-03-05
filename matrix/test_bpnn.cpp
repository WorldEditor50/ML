#include <iostream>
#include "bpnn.hpp"
using namespace std;

int main()
{
    srand((unsigned int)time(nullptr));
    BPNet<double> bpNet(2, 4, 2, 1, ACTIVATE_SIGMOID);
    vector<Mat<double> > x;
    Mat<double> x1(2, 1);
    x1[0][0] = 0;
    x1[1][0] = 0;
    x.push_back(x1);
    Mat<double> x2(2, 1);
    x2[0][0] = 1;
    x2[1][0] = 0;
    x.push_back(x2);
    Mat<double> x3(2, 1);
    x3[0][0] = 0;
    x3[1][0] = 1;
    x.push_back(x3);
    Mat<double> x4(2, 1);
    x4[0][0] = 1;
    x4[1][0] = 1;
    x.push_back(x4);
    vector<Mat<double> > y;
    Mat<double> y1(1, 1);
    y1[0][0] = 0;
    y.push_back(y1);
    Mat<double> y2(1, 1);
    y2[0][0] = 1;
    y.push_back(y2);
    Mat<double> y3(1, 1);
    y3[0][0] = 1;
    y.push_back(y3);
    Mat<double> y4(1, 1);
    y4[0][0] = 0;
    y.push_back(y4);
    bpNet.train(x, y, OPT_RMSPROP, 4, 0.01, 10000);
    bpNet.feedForward(x1);
    bpNet.show();
    bpNet.feedForward(x2);
    bpNet.show();
    bpNet.feedForward(x3);
    bpNet.show();
    bpNet.feedForward(x4);
    bpNet.show();
	return 0;
}
