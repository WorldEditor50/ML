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
    struct BpNetConfig {
        int inputDim;
        int hiddenDim;
        int outputDim;
        int hiddenLayerNum;
        double learningRate;
    };

    class Layer {
        public:
            Layer(){}
            ~Layer(){}
            void createLayer(int inputDim, int layerDim);
            double activate(double x);
            double derivativeActivate(double y);
            void calculateOutputs(std::vector<double>& x);
            void calculateErrors(std::vector<double>& nextErrors, std::vector<std::vector<double> >& nextWeights);
            void adjustWeight(std::vector<double>& x, double learningRate);
            double sigmoid(double x);
            double DSigmoid(double y);
            double RELU(double x);
            double DRELU(double x);
            std::vector<double> outputs;
            std::vector<double> errors;
            std::vector<std::vector<double> > weights;
            std::vector<double> bias;
    };

    class BpNet {
        public:
            BpNet(){}
            ~BpNet(){}
            void createNet(int inputDim, int hiddenDim, int outputDim, int hiddenLayerNum, double learningRate);
            void copyTo(BpNet& dstNet);
            void createNetWithConfig(BpNetConfig& config);
            void feedForward(std::vector<double>& xi);
            void backPropagate(std::vector<double>& yo, std::vector<double>& yt);
            void updateWeight(std::vector<double>& xi);
            std::vector<double>& getOutput();
            void train(int iterateNum);
            void train(std::vector<double> &x, std::vector<double> &yo, std::vector<double> &yt);
            void train(std::vector<std::vector<double> >& x, std::vector<std::vector<double> >& y, int iterateNum);
            void show();
            void loadDataSet(const std::string& fileName, int rowNum, int colNum, int featureNum);
            void loadFeature(const std::string& fileName, int rowNum, int colNum);
            void loadTarget(const std::string& fileName, int rowNum, int colNum);
            void loadParameter(const std::string& fileName);
            void saveParameter(const std::string& fileName);
            std::vector<std::vector<double> > features;
            std::vector<std::vector<double> > targets;
            double learningRate;
            int outputIndex;
            std::vector<Layer> layers;
    };
}
#endif // BPNN_H
