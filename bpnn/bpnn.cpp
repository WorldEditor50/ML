#include "bpnn.h"
namespace ML {
    double Layer::sigmoid(double x)
    {
        return exp(x) / (exp(x) + 1);
    }

    double Layer::DSigmoid(double y)
    {
        return y * (1 - y);
    }

    double Layer::RELU(double x)
    {
        return x > 0 ? x : 0;
    }

    double Layer::DRELU(double x)
    {
        return x > 0 ? 1 : 0;
    }

    double Layer::activate(double x)
    {
        return sigmoid(x);
    }

    double Layer::derivativeActivate(double y)
    {
        return DSigmoid(y);
    }

    void Layer::createLayer(int neuronNum, int inputNum)
    {
        if (neuronNum < 1 || inputNum < 1) {
            return;
        }
        outputs.resize(neuronNum);
        errors.resize(neuronNum);
        bias.resize(neuronNum);
        weights.resize(neuronNum);
        for (int i = 0; i < weights.size(); i++) {
            weights[i].resize(inputNum);
            /* init weights */
            for (int j = 0; j < weights[0].size(); j++) {
                weights[i][j] = double(rand() % 1000 - rand() % 1000) / 1000;
            }
            /* init bias */
            bias[i] = double(rand() % 1000 - rand() % 1000) / 1000;
            errors[i] = 0;
        }
        return;
    }

    void Layer::calculateOutputs(std::vector<double>& x)
    {
        if (x.size() != weights[0].size()) {
            std::cout<<"not same size"<<std::endl;
            return;
        }
        double y = 0;
        for (int i = 0; i < weights.size(); i++) {
            y = 0;
            for (int j = 0; j < weights[0].size(); j++) {
                y += weights[i][j] * x[j]; 
            }
            y += bias[i];
            outputs[i] = activate(y);
        }
        return;
    }

    void Layer::calculateErrors(std::vector<double>& nextErrors, std::vector<std::vector<double> >& nextWeights)
    {
        if (errors.size() != nextWeights[0].size()) {
            std::cout<<"size is not matching"<<std::endl;;
        }
        for (int i = 0; i < nextWeights[0].size(); i++) {
            for (int j = 0; j < nextWeights.size(); j++) {
                errors[i] += nextErrors[j] * nextWeights[j][i];   
            }
        }
        return;
    }

    void Layer::adjustWeight(std::vector<double>& x, double learningRate)
    {
        /*
         * e = (Activate(wx + b) - T)^2/2
         * de/dw = (Activate(wx +b) - T)*DActivate(wx + b) * x
         * de/db = (Activate(wx +b) - T)*DActivate(wx + b)
         * */
        double dOutput = 1;
        for (int i = 0; i < weights.size(); i++) {
            dOutput = derivativeActivate(outputs[i]);
            for (int j = 0; j < weights[0].size(); j++) {
                weights[i][j] -= learningRate * errors[i] * dOutput * x[j];
            }
            bias[i] -= learningRate * errors[i] * dOutput; 
            errors[i] = 0;
        }
        return;
    }

    void BpNet::createNet(int outputNeuronNum, int inputNum, int hiddenNeuronNum, int hiddenLayerNum, double learningRate)
    {
        Layer layer1; 
        layer1.createLayer(hiddenNeuronNum, inputNum);
        layers.push_back(layer1);
        for (int i = 1; i < hiddenLayerNum; i++) {
            Layer layer;
            layer.createLayer(hiddenNeuronNum, hiddenNeuronNum);
            layers.push_back(layer);
        }
        Layer outputLayer; 
        outputLayer.createLayer(outputNeuronNum, hiddenNeuronNum);
        layers.push_back(outputLayer);
        this->learningRate = learningRate;
        this->outputIndex = layers.size() - 1;
        return;
    }

    void BpNet::createNetWithConfig(BpNetConfig& config)
    {
        createNet(config.outputNeuronNum,
                config.inputNeuronNum,
                config.hiddenNeuronNum,
                config.hiddenLayerNum,
                config.learningRate);
        return;
    }

    void BpNet::loadDataSet(const std::string& fileName, int rowNum, int featureNum, int targetNum)
    {
        std::ifstream file;
        file.open(fileName);
        for (int i = 0; i < rowNum; i++) {
            std::vector<double> feature(featureNum);
            std::vector<double> target(targetNum);
            for (int j = 0; j < featureNum; j++) {
                file>>feature[j];
            }
            for (int k = 0; k < targetNum; k++) {
                file>>target[k];
            }
            features.push_back(feature);
            targets.push_back(target);
        }
        return;
    }

    std::vector<double>& BpNet::feedForward(std::vector<double>& xi)
    {
        layers[0].calculateOutputs(xi);
        for (int i = 1; i < layers.size(); i++) {
            layers[i].calculateOutputs(layers[i - 1].outputs);
        }
        std::vector<double>& outputs = layers[outputIndex].outputs;
        return outputs;
    }

    void BpNet::backPropagate(std::vector<double>& yi)
    {
        /* calculate final error */
        for (int i = 0; i < yi.size(); i++) {
            layers[outputIndex].errors[i] = layers[outputIndex].outputs[i] - yi[i]; 
        }
        /* error backpropagate */
        for (int i = outputIndex - 1; i >= 0; i--) {
            layers[i].calculateErrors(layers[i + 1].errors, layers[i + 1].weights);
        }
        return;
    }

    void BpNet::updateWeight(std::vector<double>& xi)
    {
        layers[0].adjustWeight(xi, learningRate);
        for (int i = 1; i < layers.size(); i++) {
            layers[i].adjustWeight(layers[i - 1].outputs, learningRate);
        }
        return;
    }

    void BpNet::train(int iterateNum)
    {
        train(features, targets, iterateNum);
        return;
    }

    void BpNet::train(std::vector<std::vector<double> >& x, std::vector<std::vector<double> >& y, int iterateNum)
    {
        if (x.empty() || y.empty()) {
            std::cout<<"x or y is empty"<<std::endl;
            return;
        }
        if (x.size() != y.size()) {
            std::cout<<"x != y"<<std::endl;
            return;
        }
        if (x[0].size() != layers[0].weights[0].size()) {
            std::cout<<"x != w"<<std::endl;
            return;
        }
        if (y[0].size() != layers[outputIndex].outputs.size()) {
            std::cout<<"y != output"<<std::endl;
            return;
        }
        for (int i = 0; i < iterateNum; i++) {
            int k = rand() % y.size();
            feedForward(x[k]);
            backPropagate(y[k]);
            updateWeight(x[k]);
        }
        return;
    }

    void BpNet::show()
    {
#if 0
        std::cout<<"features:"<<std::endl;;
        for (int i = 0; i < features.size(); i++) {
            for (int j = 0; j < features[0].size(); j++) {
                std::cout<<features[i][j]<<" ";
            }
            std::cout<<std::endl;;
        }
        std::cout<<"targets:"<<std::endl;;
        for (int i = 0; i < targets.size(); i++) {
            for (int j = 0; j < targets[0].size(); j++) {
                std::cout<<targets[i][j]<<" ";
            }
            std::cout<<std::endl;;
        }
#endif
        std::cout<<"outputs:"<<std::endl;;
        for (int i = 0; i < layers[outputIndex].outputs.size(); i++) {
            std::cout<<layers[outputIndex].outputs[i]<<" ";
        }
        std::cout<<std::endl;;
        return;
    }
    void BpNet::loadFeature(const std::string& fileName, int rowNum, int colNum)
    {
        if (rowNum < 1 || colNum < 1) {
            return;
        }
        std::ifstream file;
        file.open(fileName);
        for (int i = 0; i < rowNum; i++) {
            std::vector<double> feature(colNum);
            for (int j = 0; j < colNum; j++) {
                file >> feature[j];
            }
            features.push_back(feature);
        }
        return;
    }
    void BpNet::loadTarget(const std::string& fileName, int rowNum, int colNum)
    {
        if (rowNum < 1 || colNum < 1) {
            return;
        }
        std::ifstream file;
        file.open(fileName);
        for (int i = 0; i < rowNum; i++) {
            std::vector<double> target(colNum);
            for (int j = 0; j < colNum; j++) {
                file >> target[j];
            }
            targets.push_back(target);
        }
        return;
    }
    void BpNet::loadParameter(const std::string& fileName)
    {
        std::ifstream file;
        file.open(fileName);
        for (int i = 0; i < layers.size(); i++) {
            for (int j = 0; j < layers[i].weights.size(); j++) {
                for (int k = 0; k < layers[i].weights[j].size(); k++) {
                    file >> layers[i].weights[j][k];
                }
                file >> layers[i].bias[j];
            }
        }
        return;
    }
    void BpNet::saveParameter(const std::string& fileName)
    {
        std::ofstream file;
        file.open(fileName);
        for (int i = 0; i < layers.size(); i++) {
            for (int j = 0; j < layers[i].weights.size(); j++) {
                for (int k = 0; k < layers[i].weights[j].size(); k++) {
                    file << layers[i].weights[j][k];
                }
                file << layers[i].bias[j];
                file << std::endl;;
            }
        }
        return;
    }

}
