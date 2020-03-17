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
        int outputNeuronNum;
        int inputNeuronNum;
        int hiddenNeuronNum;
        int hiddenLayerNum;
        double learningRate;
    };

    class Layer {
        public:
            Layer(){}
            ~Layer(){}
            void createLayer(int neuronNum, int inputNum);
            double activate(double x);
            double derivativeActivate(double y);
            void calculateOutputs(std::vector<double>& x);
            void calculateErrors(std::vector<double>& nextErrors, std::vector<std::vector<double> >& nextWeights);
            void adjustWeight(std::vector<double>& x, double learningRate);
        public:
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
            void createNet(int outputNeuronNum, int inputNum, int hiddenNeuronNum, int hiddenLayerNum, double learningRate);
            void createNetWithConfig(BpNetConfig& config);
            void loadDataSet(const std::string& fileName, int rowNum, int colNum, int featureNum);
            void loadFeature(const std::string& fileName, int rowNum, int colNum);
            void loadTarget(const std::string& fileName, int rowNum, int colNum);
            void loadParameter(const std::string& fileName);
            void saveParameter(const std::string& fileName);
            void train(int iterateNum);
            void train(std::vector<std::vector<double> >& x, std::vector<std::vector<double> >& y, int iterateNum);
            std::vector<double>& feedForward(std::vector<double>& xi);
            void show();
            std::vector<std::vector<double> > features;
            std::vector<std::vector<double> > targets;
        private:
            double learningRate;
            int outputIndex;
            std::vector<Layer> layers;
            void backPropagate(std::vector<double>& yi);
            void updateWeight(std::vector<double>& xi);
    };
}
#endif // BPNN_H
