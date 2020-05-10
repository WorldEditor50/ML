#ifndef BPNN_H
#define BPNN_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "matrix.hpp"
using namespace ML;
/* activate method */
#define ACTIVATE_SIGMOID 0
#define ACTIVATE_TANH    1
#define ACTIVATE_RELU    2
#define ACTIVATE_LINEAR  3
/* optimize method */
#define OPT_SGD     0
#define OPT_RMSPROP 1
#define OPT_ADAM    2
/* loss type */
#define LOSS_MSE            0
#define LOSS_CROSS_ENTROPY  1
Mat<float> LOG(Mat<float> X);
Mat<float> EXP(Mat<float> X);
Mat<float> SQRT(Mat<float> X);
Mat<float> SOFTMAX(Mat<float>& X);
class Layer {
    public:
        Layer(){}
        ~Layer(){}
        void createLayer(int inputDim, int layerDim, int activateType, int lossTye = LOSS_MSE);
        void SGD(float learningRate);
        void RMSProp(float rho, float learningRate);
        void Adam(float alpha1, float alpha2, float learningRate);
        Mat<float> Activate(Mat<float> X);
        Mat<float> dActivate(Mat<float>& Y);
        int lossType;
        int activateType;
        Mat<float> W;
        Mat<float> B;
        Mat<float> O;
        Mat<float> E;
        /* buffer for optimization */
        Mat<float> dW;
        Mat<float> Sw;
        Mat<float> Vw;
        Mat<float> dB;
        Mat<float> Sb;
        Mat<float> Vb;
        float alpha1_t;
        float alpha2_t;
        float delta;
        float decay;
};

class BPNet {
    public:
        BPNet(){}
        ~BPNet(){}
        void createNet(int inputDim, int hiddenDim, int hiddenLayerNum, int outputDim,
                int activateType = ACTIVATE_SIGMOID, int lossType = LOSS_MSE);
        void copyTo(BPNet& dstNet);
        void softUpdateTo(BPNet& net, float alpha);
        void feedForward(Mat<float>& x);
        void backPropagate(Mat<float>& Yo, Mat<float>& Yt);
        void gradient(Mat<float> &x, Mat<float> &y);
        void SGD(float learningRate = 0.001);
        void RMSProp(float rho = 0.9, float learningRate = 0.001);
        void Adam(float alpha1 = 0.9, float alpha2 = 0.99, float learningRate = 0.001);
        void optimize(int optType = OPT_RMSPROP, float learningRate = 0.001);
        void train(std::vector<Mat<float> >& x,
                std::vector<Mat<float> >& y,
                int optType,
                int batchSize,
                float learningRate,
                int iterateNum);
        int argmax();
        Mat<float>& getOutput();
        void show();
        void load(const std::string& fileName);
        void save(const std::string& fileName);
        int outputIndex;
        std::vector<Layer> layers;
};
#endif // BPNN_H
