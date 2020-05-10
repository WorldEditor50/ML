#include <iostream>
#include "bpnn.h"
using namespace std;

int main()
{
    srand((unsigned int)time(nullptr));
    BPNet bpNet;
    bpNet.createNet(2, 4, 2, 1, ACTIVATE_SIGMOID);
    vector<Mat<float> > x;
    Mat<float> x1(2, 1);
    x1.mat[0][0] = 0;
    x1.mat[1][0] = 0;
    x.push_back(x1);
    Mat<float> x2(2, 1);
    x2.mat[0][0] = 1;
    x2.mat[1][0] = 0;
    x.push_back(x2);
    Mat<float> x3(2, 1);
    x3.mat[0][0] = 0;
    x3.mat[1][0] = 1;
    x.push_back(x3);
    Mat<float> x4(2, 1);
    x4.mat[0][0] = 1;
    x4.mat[1][0] = 1;
    x.push_back(x4);
    vector<Mat<float> > y;
    Mat<float> y1(1, 1);
    y1.mat[0][0] = 0;
    y.push_back(y1);
    Mat<float> y2(1, 1);
    y2.mat[0][0] = 1;
    y.push_back(y2);
    Mat<float> y3(1, 1);
    y3.mat[0][0] = 1;
    y.push_back(y3);
    Mat<float> y4(1, 1);
    y4.mat[0][0] = 0;
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
