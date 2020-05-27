#include <iostream>
#include "../bayes/bayes.h"
#include "../bpnn/bpnn.h"
#include "../logistics/logistics.h"
#include "../perceptron/perceptron.h"
#include "../svm/svm.h"
using namespace std;
using namespace ML;
typedef void (*pftest)(vector<vector<double> >& x);

void test_bpnn(vector<vector<double> >& x)
{
    cout<<"bpnn:"<<endl;
    BPNet bp(2, 4, 1, 1, ACTIVATE_SIGMOID, true);
    vector<vector<double> > y(4);
    for (int i = 0; i < 4; i++) {
        y[i].resize(1);
    }
    y[0][0] = 0;
    y[1][0] = 1;
    y[2][0] = 1;
    y[3][0] = 0;
    bp.Train(x, y, OPT_RMSPROP, 4, 0.01, 10000);
    vector<double>& yi = bp.GetOutput();
    for (int i = 0; i < 4; i++) {
        bp.FeedForward(x[i]);
        cout<<x[i][0]<<" "<<x[i][1]<<" "<<yi[0]<<endl;
    }
    return;
}

void test_logistics(vector<vector<double> >& x)
{
    /* not xor  */
    cout<<"logistics:"<<endl;
    Logistics l;
    l.create(2, 0.5);
    l.loadDataSet("./data/and", 4, 2);
    l.train(10000);
    l.show();
    /* classify */
    for (int i = 0; i < 4; i++) {
        double y = l.calculateOutput(x[i]);
        cout<<x[i][0]<<" "<<x[i][1]<<" "<<y<<endl;
    }
    return;
}

void test_perceptron(vector<vector<double> >& x)
{
    cout<<"perceptron:"<<endl;
    Perceptron p;
    p.create("./data/xor2", 2, 4);
    p.setKernel(KERNEL_RBF);
    p.train(10000);
    p.show();
    /* classify */
    for (int i = 0; i < 4; i++) {
        double y = p.classify(x[i]);
        cout<<x[i][0]<<" "<<x[i][1]<<" "<<y<<endl;
    }
    return;
}

void test_svm(vector<vector<double> >& x)
{
    cout<<"svm:"<<endl;
    SVM s;
    s.loadDataSet("./data/xor2", 2, 4);
    s.setParameter(KERNEL_RBF, 1, 0.0001);
    s.SMO(10000);
    s.show();
    /* classify */
    for (int i = 0; i < 4; i++) {
        double y = s.classify(x[i]);
        cout<<x[i][0]<<" "<<x[i][1]<<" "<<y<<endl;
    }
    return;
}

int main()
{
    pftest test[4] = {
        test_bpnn,
        test_logistics,
        test_perceptron,
        test_svm
    };
    vector<vector<double> > x(4);
    for (int i = 0; i < 4; i++) {
        x[i].resize(2);
    }
    x[0][0] = 1;
    x[0][1] = 1;
    x[1][0] = 1;
    x[1][1] = 0;
    x[2][0] = 0;
    x[2][1] = 1;
    x[3][0] = 0;
    x[3][1] = 0;
    srand((unsigned int)time(NULL));
    for (int i = 0; i < 4; i++) {
        test[i](x);
    }
    return 0;
}
