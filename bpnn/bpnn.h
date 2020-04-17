#ifndef BPNN_H
#define BPNN_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
namespace ML {
#define SIGMOID 0
#define RELU    1
    class Layer {
        public:
            Layer(){}
            ~Layer(){}
            void createLayer(int inputDim, int layerDim, int activateMethod);
            double activate(double x);
            double derivativeActivate(double y);
            void calculateOutputs(std::vector<double>& x);
            void calculateErrors(std::vector<double>& nextErrors, std::vector<std::vector<double> >& nextWeights);
            void stochasticGradientDescent(std::vector<double>& x, double learningRate);
            void calculateBatchGradient(std::vector<double>& x);
            void batchGradientDescent(double learningRate);
            double sigmoid(double x);
            double dsigmoid(double y);
            double relu(double x);
            double drelu(double x);
            double dotProduct(std::vector<double>& x1, std::vector<double>& x2);
            int activateMethod;
            std::vector<double> outputs;
            std::vector<double> errors;
            std::vector<std::vector<double> > weights;
            std::vector<double> bias;
            std::vector<std::vector<double> > batchGradientX;
            std::vector<double> batchGradient;
    };

    class BPNet {
        public:
            BPNet(){}
            ~BPNet(){}
            void createNet(int inputDim, int hiddenDim, int hiddenLayerNum, int outputDim,
                    int activateMethod, double learningRate);
            void copyTo(BPNet& dstNet);
            std::vector<double>& getOutput();
            void feedForward(std::vector<double>& xi);
            void backPropagate(std::vector<double>& yo,
                    std::vector<double>& yt);
            void stochasticGradientDescent(std::vector<double> &x,
                    std::vector<double> &yo,
                    std::vector<double> &yt);
            void calculateBatchGradient(std::vector<double> &x,
                    std::vector<double> &yo,
                    std::vector<double> &yt);
            void updateWithBatchGradient();
            void batchGradientDescent(std::vector<std::vector<double> >& x,
                    std::vector<std::vector<double> >& yo,
                    std::vector<std::vector<double> >& yt);
            void batchGradientDescent(std::vector<std::vector<double> >& x,
                    std::vector<std::vector<double> >& y);
            void train(std::vector<std::vector<double> >& x,
                    std::vector<std::vector<double> >& y,
                    int iterateNum);
            void show();
            void loadParameter(const std::string& fileName);
            void saveParameter(const std::string& fileName);
            double learningRate;
            int outputIndex;
            std::vector<Layer> layers;
    };
}
#endif // BPNN_H
