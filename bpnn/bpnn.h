#ifndef MLP_H
#define MLP_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
namespace ML {
/* activate method */
enum ActiveType {
    SIGMOID = 0,
    TANH,
    RELU,
    LINEAR
};
/* Optimize method */
enum OptType {
    NONE = 0,
    OPT_SGD,
    OPT_RMSPROP,
    OPT_ADAM
};
/* loss type */
enum LossType {
    MSE = 0,
    CROSS_ENTROPY
};
using Mat = std::vector<std::vector<double> >;
using Vec = std::vector<double>;
class Layer
{
public:
    enum LayerType {
      INPUT = 0,
      HIDDEN,
      OUTPUT
    };
public:
    Layer():inputDim(0), layerDim(0){}
    virtual ~Layer(){}
    explicit Layer(int inputDim, int layerDim, LayerType layerType, ActiveType activeType, LossType lossType, bool tarinFlag);
    void feedForward(Vec& x);
    void loss(Vec& yo, Vec& yt);
    void loss(Vec& l);
    void error(Vec& nextE, Mat& nextW);
    void gradient(Vec& x);
    void gradient(Vec& x, double threshold);
    void softmaxGradient(Vec& x, Vec& yo, Vec& yt);
    void SGD(double learningRate);
    void RMSProp(double rho, double learningRate);
    void Adam(double alpha1, double alpha2, double learningRate);
    void RMSPropWithClip(double rho, double learningRate, double threshold);
public:
    /* output */
    Mat W;
    Vec B;
    Vec O;
    Vec E;
    /* paramter */
    int inputDim;
    int layerDim;
    ActiveType activeType;
    LossType lossType;
    LayerType layerType;
    std::string name;
protected:
    double activate(double x);
    double dActivate(double y);
    void softmax(Vec& x, Vec& y);
    Vec softmax(Vec &x);
    double max(Vec& x);
    int argmax(Vec& x);
    double dotProduct(Vec& x1, Vec& x2);
    /* buffer for optimization */
    Mat dW;
    Mat Sw;
    Mat Vw;
    Vec dB;
    Vec Sb;
    Vec Vb;
    double alpha1_t;
    double alpha2_t;
};

class BPNN
{
private:
    int outputIndex;
    std::vector<Layer> layers;
public:
    BPNN(){}
    virtual ~BPNN(){}
    explicit BPNN(int inputDim, int hiddenDim, int hiddenLayerNum, int outputDim, bool trainFlag = false,
            ActiveType activeType = SIGMOID, LossType lossType = MSE);
    void copyTo(BPNN& dstNet);
    void softUpdateTo(BPNN& dstNet, double alpha);
    Vec& getOutput();
    int feedForward(Vec& x);
    void backPropagate(Vec& yo, Vec& yt);
    void grad(Vec &x, Vec &y, Vec &loss);
    void gradient(Vec &x, Vec &y);
    void SGD(double learningRate = 0.001);
    void RMSProp(double rho = 0.9, double learningRate = 0.001);
    void Adam(double alpha1 = 0.9, double alpha2 = 0.99, double learningRate = 0.001);
    void RMSPropWithClip(double rho = 0.9, double learningRate = 0.001, double threshold = 1);
    void optimize(OptType optType = OPT_RMSPROP, double learningRate = 0.001);
    void train(Mat& x,
            Mat& y,
            OptType optType,
            int batchSize,
            double learningRate,
            int iterateNum);
    int argmax();
    void show();
    void load(const std::string& fileName);
    void save(const std::string& fileName);
};
}
#endif // MLP_H
