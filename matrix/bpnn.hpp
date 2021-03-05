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
double sigmoid(double x)
{
    return exp(x) / (exp(x) + 1);
}
double relu(double x)
{
    return x > 0 ? x : 0;
}
double linear(double x)
{
    return x;
}
double dsigmoid(double y)
{
    return y * (1 - y);
}
double drelu(double y)
{
    return y > 0 ? 1 : 0;
}
double dtanh(double y)
{
    return 1 - y * y;
}
template<typename T>
Mat<T> LOG(Mat<T> x)
{
    return for_each(x, log);
}

template<typename T>
Mat<T> EXP(Mat<T> x)
{
    return for_each(x, exp);
}

template<typename T>
Mat<T> SQRT(Mat<T> x)
{
    return for_each(x, sqrt);
}

template<typename T>
Mat<T> SOFTMAX(Mat<T>& x)
{
    /* softmax works in multi-classify */
    T maxValue = max(x);
    Mat<T> delta = EXP(x - maxValue);
    T s = sum(delta);
    return delta / (s + 1e-9);
}

template<typename T>
class Layer
{
public:
    int lossType;
    int activateType;
    Mat<T> W;
    Mat<T> B;
    Mat<T> O;
    Mat<T> E;
    /* buffer for optimization */
    Mat<T> dW;
    Mat<T> Sw;
    Mat<T> Vw;
    Mat<T> dB;
    Mat<T> Sb;
    Mat<T> Vb;
    double alpha1_t;
    double alpha2_t;
public:
    Layer(){}
    ~Layer(){}
    Mat<T> Activate(const Mat<T> &x)
    {
        Mat<T> y;
        switch (activateType) {
            case ACTIVATE_SIGMOID:
                y = for_each(x, sigmoid);
                break;
            case ACTIVATE_RELU:
                y = for_each(x, relu);
                break;
            case ACTIVATE_TANH:
                y = for_each(x, tanh);
                break;
            case ACTIVATE_LINEAR:
                y = x;
                break;
            default:
                y = for_each(x, sigmoid);
                break;
        }
        return y;
    }

    Mat<T> dActivate(const Mat<T>& y)
    {
        Mat<T> dy;
        switch (activateType) {
            case ACTIVATE_SIGMOID:
                dy = for_each(y, dsigmoid);
                break;
            case ACTIVATE_RELU:
                dy = for_each(y, drelu);
                break;
            case ACTIVATE_TANH:
                dy = for_each(y, dtanh);
                break;
            case ACTIVATE_LINEAR:
                dy = y;
                dy.assign(1);
                break;
            default:
                dy = for_each(y, dsigmoid);
                break;
        }
        return dy;
    }

    Layer(int inputDim, int layerDim, int activateType, int lossType = LOSS_MSE)
    {
        this->lossType = lossType;
        this->activateType = activateType;
        W = Mat<T>(layerDim, inputDim, UNIFORM_RAND);
        B = Mat<T>(layerDim, 1, UNIFORM_RAND);
        O = Mat<T>(layerDim, 1, ZERO);
        E = Mat<T>(layerDim, 1, ZERO);
        /* buffer for optimization */
        dW = Mat<T>(layerDim, inputDim, ZERO);
        Sw = Mat<T>(layerDim, inputDim, ZERO);
        Vw = Mat<T>(layerDim, inputDim, ZERO);
        dB = Mat<T>(layerDim, 1, ZERO);
        Sb = Mat<T>(layerDim, 1, ZERO);
        Vb = Mat<T>(layerDim, 1, ZERO);
        this->alpha1_t = 1;
        this->alpha2_t = 1;
    }

    void SGD(T learningRate)
    {
        W -= dW * learningRate;
        B -= dB * learningRate;
        dW.zero();
        dB.zero();
        return;
    }

    void RMSProp(T rho, T learningRate)
    {
        Sw = Sw * rho + (dW % dW) * (1 - rho);
        Sb = Sb * rho + (dB % dB) * (1 - rho);
        W -= dW / (SQRT(Sw) + 1e-9) * learningRate;
        B -= dB / (SQRT(Sb) + 1e-9) * learningRate;
        dW.zero();
        dB.zero();
        return;
    }

    void Adam(T alpha1, T alpha2, T learningRate)
    {
        alpha1_t *= alpha1;
        alpha2_t *= alpha2;
        Vw = Vw * alpha1 + dW * (1 - alpha1);
        Vb = Vb * alpha1 + dB * (1 - alpha1);
        Sw = Sw * alpha2 + (dW % dW) * (1 - alpha2);
        Sb = Sb * alpha2 + (dB % dB) * (1 - alpha2);
        Mat<T> Vwt = Vw / (1 - alpha1_t);
        Mat<T> Vbt = Vb / (1 - alpha1_t);
        Mat<T> Swt = Sw / (1 - alpha2_t);
        Mat<T> Sbt = Sb / (1 - alpha2_t);
        W -= Vwt / (SQRT(Swt) + 1e-9) * learningRate;
        B -= Vbt / (SQRT(Sbt) + 1e-9) * learningRate;
        dW.zero();
        dB.zero();
        return;
    }

};

template<typename T>
class BPNet
{
public:
    using DataType = T;
    int outputIndex;
    std::vector<Layer<DataType> > layers;
public:
    BPNet(){}
    ~BPNet(){}
    BPNet(int inputDim, int hiddenDim, int hiddenLayerNum, int outputDim,
            int activateType = ACTIVATE_SIGMOID, int lossType = LOSS_MSE)
    {
        layers.push_back(Layer<DataType>(inputDim, hiddenDim, activateType));
        for (int i = 1; i < hiddenLayerNum; i++) {
            layers.push_back(Layer<DataType>(hiddenDim, hiddenDim, activateType));
        }
        if (lossType == LOSS_MSE) {
            layers.push_back(Layer<DataType>(hiddenDim, outputDim, activateType));
        } else if (lossType == LOSS_CROSS_ENTROPY) {
            layers.push_back(Layer<DataType>(hiddenDim, outputDim, ACTIVATE_LINEAR, LOSS_CROSS_ENTROPY));
        }
        this->outputIndex = layers.size() - 1;
        return;
    }

    void copyTo(BPNet& dstNet)
    {
        if (layers.size() != dstNet.layers.size()) {
            return;
        }
        for (int i = 0; i < layers.size(); i++) {
            dstNet.layers[i].W = layers[i].W;
            dstNet.layers[i].B = layers[i].B;
        }
        return;
    }

    void softUpdateTo(BPNet &dstNet, DataType alpha)
    {
        if (layers.size() != dstNet.layers.size()) {
            return;
        }
        for (int i = 0; i < layers.size(); i++) {
            dstNet.layers[i].W = dstNet.layers[i].W * (1 - alpha) + layers[i].W * alpha;
            dstNet.layers[i].B = dstNet.layers[i].B * (1 - alpha) + layers[i].B * alpha;
        }
        return;
    }

    void feedForward(const Mat<DataType>& x)
    {
        layers[0].O = layers[0].Activate(layers[0].W * x + layers[0].B);
        for (int i = 1; i < layers.size(); i++) {
            layers[i].O = layers[i].Activate(layers[i].W * layers[i - 1].O + layers[i].B);
            if (layers[i].lossType == LOSS_CROSS_ENTROPY) {
                layers[i].O = SOFTMAX(layers[i].O);
            }
        }
        return;
    }

    Mat<DataType>& getOutput()
    {
        Mat<DataType>& outputs = layers[outputIndex].O;
        return outputs;
    }

    void gradient(Mat<DataType> &x, Mat<DataType> &y)
    {
        /* loss */
        if (layers[outputIndex].lossType == LOSS_CROSS_ENTROPY) {
            layers[outputIndex].E = (y % LOG(layers[outputIndex].O)) * (-1);
        } else if (layers[outputIndex].lossType == LOSS_MSE){
            layers[outputIndex].E = layers[outputIndex].O - y;
        }
        /* error backpropagate */
        for (int i = outputIndex - 1; i >= 0; i--) {
            layers[i].E = layers[i + 1].W.Tr() * layers[i + 1].E;
        }
        /* calculate  gradient */
        for (int i = 0; i < layers.size(); i++) {
            if (layers[i].lossType == LOSS_CROSS_ENTROPY) {
                Mat<DataType> dy = layers[outputIndex].O - y;
                layers[i].dW += dy * layers[i - 1].O.Tr();
                layers[i].dB += dy;
            } else {
                Mat<DataType> dy = layers[i].E % layers[i].dActivate(layers[i].O);
                if (i == 0) {
                    layers[i].dW += dy * x.Tr();
                } else {
                    layers[i].dW += dy * layers[i - 1].O.Tr();
                }
                layers[i].dB += dy;
            }
        }
        return;
    }

    void SGD(DataType learningRate)
    {
        /* gradient descent */
        for (int i = 0; i < layers.size(); i++) {
            layers[i].SGD(learningRate);
        }
        return;
    }

    void RMSProp(DataType rho, DataType learningRate)
    {
        for (int i = 0; i < layers.size(); i++) {
            layers[i].RMSProp(rho, learningRate);
        }
        return;
    }

    void Adam(DataType alpha1, DataType alpha2, DataType learningRate)
    {
        for (int i = 0; i < layers.size(); i++) {
            layers[i].Adam(alpha1, alpha2, learningRate);
        }
        return;
    }

    void optimize(int optType, DataType learningRate)
    {
        switch (optType) {
            case OPT_SGD:
                SGD(learningRate);
                break;
            case OPT_RMSPROP:
                RMSProp(0.9, learningRate);
                break;
            case OPT_ADAM:
                Adam(0.9, 0.99, learningRate);
                break;
            default:
                RMSProp(0.9, learningRate);
                break;
        }
        return;
    }

    void train(std::vector<Mat<DataType> >& x,
            std::vector<Mat<DataType> >& y,
            int optType,
            int batchSize,
            float learningRate,
            int iterateNum)
    {
        int len = x.size();
        for (int i = 0; i < iterateNum; i++) {
            for (int j = 0; j < batchSize; j++) {
                int k = rand() % len;
                feedForward(x[k]);
                gradient(x[k], y[k]);
            }
            optimize(optType, learningRate);
        }
        return;
    }

    void show()
    {
        layers[outputIndex].O.show();
        return;
    }

    void load(const std::string& fileName)
    {
        std::ifstream file;
        file.open(fileName);
        for (int i = 0; i < layers.size(); i++) {
            layers[i].W.load(fileName);
            layers[i].B.load(fileName);
        }
        return;
    }

    void save(const std::string& fileName)
    {
        std::ofstream file;
        file.open(fileName);
        for (int i = 0; i < layers.size(); i++) {
            layers[i].W.save(fileName);
            layers[i].B.save(fileName);
        }
        return;
    }
};
#endif // BPNN_H
