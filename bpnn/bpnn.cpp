#include "bpnn.h"
namespace ML {
    double Layer::sigmoid(double x)
    {
        return exp(x) / (exp(x) + 1);
    }

    double Layer::dsigmoid(double y)
    {
        return y * (1 - y);
    }

    double Layer::relu(double x)
    {
        return x > 0 ? x : 0;
    }

    double Layer::drelu(double x)
    {
        return x > 0 ? 1 : 0;
    }

    double Layer::activate(double x)
    {
        double y = 0;
        switch (activateMethod) {
            case SIGMOID:
                y = sigmoid(x);
                break;
            case RELU:
                y = relu(x);
                break;
            default:
                y = sigmoid(x);
                break;
        }
        return y;
    }

    double Layer::derivativeActivate(double y)
    {
        double dy = 0;
        switch (activateMethod) {
            case SIGMOID:
                dy = dsigmoid(y);
                break;
            case RELU:
                dy = drelu(y);
                break;
            default:
                dy = dsigmoid(y);
                break;
        }
        return dy;
    }

    double Layer::dotProduct(std::vector<double>& x1, std::vector<double>& x2)
    {
        double sum = 0;
        for (int i = 0; i < x1.size(); i++) {
            sum += x1[i] * x2[i];
        }
        return sum;
    }

    void Layer::createLayer(int inputDim, int layerDim, int activateMethod)
    {
        if (layerDim < 1 || inputDim < 1) {
            return;
        }
        this->activateMethod = activateMethod;
        outputs.resize(layerDim);
        errors.resize(layerDim);
        bias.resize(layerDim);
        weights.resize(layerDim);
        batchGradientX.resize(layerDim);
        batchGradient.resize(layerDim);
        for (int i = 0; i < weights.size(); i++) {
            weights[i].resize(inputDim);
            batchGradientX[i].resize(inputDim);
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
            y = dotProduct(weights[i], x) + bias[i];
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

    void Layer::stochasticGradientDescent(std::vector<double>& x, double learningRate)
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

    void Layer::calculateBatchGradient(std::vector<double>& x)
    {
        double dOutput = 1;
        for (int i = 0; i < batchGradientX.size(); i++) {
            dOutput = derivativeActivate(outputs[i]);
            for (int j = 0; j < batchGradientX[0].size(); j++) {
                batchGradientX[i][j] += errors[i] * dOutput * x[j]; 
            }
            batchGradient[i] += errors[i] * dOutput; 
            errors[i] = 0;
        }
        return;
    }

    void Layer::batchGradientDescent(double learningRate)
    {
        for (int i = 0; i < weights.size(); i++) {
            for (int j = 0; j < weights[0].size(); j++) {
                weights[i][j] -= learningRate * batchGradientX[i][j];
                batchGradientX[i][j] = 0;
            }
            bias[i] -= learningRate * batchGradient[i];
            batchGradient[i] = 0;
        }
        return;
    }

    void BPNet::createNet(int inputDim, int hiddenDim, int hiddenLayerNum, int outputDim,
            int activateMethod, double learningRate)
    {
        Layer layer1; 
        layer1.createLayer(inputDim, hiddenDim, activateMethod);
        layers.push_back(layer1);
        for (int i = 1; i < hiddenLayerNum; i++) {
            Layer layer;
            layer.createLayer(hiddenDim, hiddenDim, activateMethod);
            layers.push_back(layer);
        }
        Layer outputLayer; 
        outputLayer.createLayer(hiddenDim, outputDim, activateMethod);
        layers.push_back(outputLayer);
        this->learningRate = learningRate;
        this->outputIndex = layers.size() - 1;
        return;
    }

    void BPNet::copyTo(BPNet& dstNet)
    {
        if (layers.size() != dstNet.layers.size()) {
            return;
        }
        for (int i = 0; i < layers.size(); i++) {
            for (int j = 0; j < layers[i].weights.size(); j++) {
                for (int k = 0; k < layers[i].weights[j].size(); k++) {
                    layers[i].weights[j][k] = dstNet.layers[i].weights[j][k];
                }
                layers[i].bias[j] = dstNet.layers[i].bias[j];
            }
        }
        return;
    }

    void BPNet::feedForward(std::vector<double>& xi)
    {
        layers[0].calculateOutputs(xi);
        for (int i = 1; i < layers.size(); i++) {
            layers[i].calculateOutputs(layers[i - 1].outputs);
        }
        return;
    }

    std::vector<double>& BPNet::getOutput()
    {
        std::vector<double>& outputs = layers[outputIndex].outputs;
        return outputs;
    }

    void BPNet::backPropagate(std::vector<double>& yo, std::vector<double>& yt)
    {
        /* calculate final error */
        for (int i = 0; i < yo.size(); i++) {
            layers[outputIndex].errors[i] = yo[i] - yt[i]; 
        }
        /* error backpropagate */
        for (int i = outputIndex - 1; i >= 0; i--) {
            layers[i].calculateErrors(layers[i + 1].errors, layers[i + 1].weights);
        }
        return;
    }

    void BPNet::calculateBatchGradient(std::vector<double> &x, std::vector<double> &yo, std::vector<double> &yt)
    {
            feedForward(x);
            backPropagate(yo, yt);
            /* calculate batch gradient */
            for (int j = 0; j < layers.size(); j++) {
                if (j == 0) {
                    layers[j].calculateBatchGradient(x);
                } else {
                    layers[j].calculateBatchGradient(layers[j - 1].outputs);
                }
            }

        return;
    }

    void BPNet::updateWithBatchGradient()
    {
        /* gradient descent */
        for (int i = 0; i < layers.size(); i++) {
            layers[i].batchGradientDescent(learningRate);
        }
        return;
    }

    void BPNet::batchGradientDescent(std::vector<std::vector<double> >& x,
            std::vector<std::vector<double> >& yo,
            std::vector<std::vector<double> >& yt)
    {
        for (int i = 0; i < x.size(); i++) {
            calculateBatchGradient(x[i], yo[i], yt[i]);
        }
        updateWithBatchGradient();
        return;
    }
    void BPNet::batchGradientDescent(std::vector<std::vector<double> >& x,
            std::vector<std::vector<double> >& y)
    {
        for (int i = 0; i < x.size(); i++) {
            feedForward(x[i]);
            backPropagate(layers[outputIndex].outputs, y[i]);
            /* calculate batch gradient */
            for (int j = 0; j < layers.size(); j++) {
                if (j == 0) {
                    layers[j].calculateBatchGradient(x[i]);
                } else {
                    layers[j].calculateBatchGradient(layers[j - 1].outputs);
                }
            }
        }
        /* gradient descent */
        updateWithBatchGradient();
        return;
    }

    void BPNet::stochasticGradientDescent(std::vector<double> &x, std::vector<double> &yo, std::vector<double> &yt)
    {
        if (yo.size() != yt.size()) {
            return;
        }
        feedForward(x);
        /* calculate final error */
        backPropagate(yo, yt);
        /* gradient descent */
        for (int j = 0; j < layers.size(); j++) {
            if (j == 0) {
                layers[j].stochasticGradientDescent(x, learningRate);
            } else {
                layers[j].stochasticGradientDescent(layers[j - 1].outputs, learningRate);
            }
        }
        return;
    }


    void BPNet::train(std::vector<std::vector<double> >& x,
            std::vector<std::vector<double> >& y,
            int iterateNum)
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
            stochasticGradientDescent(x[k], layers[outputIndex].outputs, y[k]);
        }
        return;
    }

    void BPNet::show()
    {
        std::cout<<"outputs:"<<std::endl;;
        for (int i = 0; i < layers[outputIndex].outputs.size(); i++) {
            std::cout<<layers[outputIndex].outputs[i]<<" ";
        }
        std::cout<<std::endl;;
        return;
    }
    void BPNet::loadParameter(const std::string& fileName)
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
    void BPNet::saveParameter(const std::string& fileName)
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
