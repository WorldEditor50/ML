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

    double Layer::dtanh(double y)
    {
        return 1 - y * y;
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
            case ACTIVATE_SIGMOID:
                y = sigmoid(x);
                break;
            case ACTIVATE_RELU:
                y = relu(x);
                break;
            case ACTIVATE_TANH:
                y = tanh(x);
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
            case ACTIVATE_SIGMOID:
                dy = dsigmoid(y);
                break;
            case ACTIVATE_RELU:
                dy = drelu(y);
                break;
            case ACTIVATE_TANH:
                y = dtanh(y);
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
        /* buffer for optimization */
        batchGradientX.resize(layerDim);
        batchGradient.resize(layerDim);
        sx.resize(layerDim);
        sb.resize(layerDim, 0);
        vx.resize(layerDim);
        vb.resize(layerDim, 0);
        this->alpha1_t = 1;
        this->alpha2_t = 1;
        this->delta = pow(10, -8);
        /* init weights */
        for (int i = 0; i < weights.size(); i++) {
            weights[i].resize(inputDim);
            batchGradientX[i].resize(inputDim);
            sx[i].resize(inputDim, 0);
            vx[i].resize(inputDim, 0);
            /* init weights */
            for (int j = 0; j < weights[0].size(); j++) {
                weights[i][j] = double(rand() % 10000 - rand() % 10000) / 10000;
            }
            /* init bias */
            bias[i] = double(rand() % 10000 - rand() % 10000) / 10000;
            errors[i] = 0;
        }
        return;
    }

    void Layer::createSoftmaxLayer(int inputDim, int layerDim, int activateMethod)
    {

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

    void Layer::SGD(std::vector<double>& x, double learningRate)
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

    void Layer::BGD(double learningRate)
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

    void Layer::RMSProp(double rho, double learningRate)
    {
        for (int i = 0; i < weights.size(); i++) {
            for (int j = 0; j < weights[0].size(); j++) {
                double gx = batchGradientX[i][j];
                sx[i][j] = rho * sx[i][j] + (1 - rho) * gx * gx;
                weights[i][j] -= learningRate * gx / (sqrt(sx[i][j]) + delta);
                batchGradientX[i][j] = 0;
            }
            double gb = batchGradient[i];
            sb[i] = rho * sb[i] + (1 - rho) * gb * gb;
            bias[i] -= learningRate * gb / (sqrt(sb[i]) + delta);
            batchGradient[i] = 0;
        }
        return;
    }

    void Layer::Adam(double alpha1, double alpha2, double learningRate)
    {
        double v;
        double s;
        for (int i = 0; i < weights.size(); i++) {
            alpha1_t *= alpha1;
            alpha2_t *= alpha2;
            for (int j = 0; j < weights[0].size(); j++) {
                double gx = batchGradientX[i][j];
                /* momentum */
                vx[i][j] = alpha1 * vx[i][j] + (1 - alpha1) * gx;
                /* delcay factor */
                sx[i][j] = alpha2 * sx[i][j] + (1 - alpha2) * gx * gx;
                v = vx[i][j] / (1 - alpha1_t);
                s = sx[i][j] / (1 - alpha2_t);
                weights[i][j] -= learningRate * v / (sqrt(s) + delta);
                batchGradientX[i][j] = 0;
            }
            double gb = batchGradient[i];
            vb[i] = alpha1 * vb[i] + (1 - alpha1) * gb;
            sb[i] = alpha2 * sb[i] + (1 - alpha2) * gb * gb;
            v = vb[i] / (1 - alpha1_t);
            s = sb[i] / (1 - alpha2_t);
            bias[i] -= learningRate * v / (sqrt(s) + delta);
            batchGradient[i] = 0;
        }
        return;
    }

    double Layer::softmaxSum(std::vector<double> &z)
    {
        double s = 0;
        for (int i = 0; i < z.size(); i++) {
            s += exp(z[i]);
        }
        return s;
    }

    double Layer::softmax(double s, double z)
    {
        return exp(z) / s;
    }

    void BPNet::createNet(int inputDim, int hiddenDim, int hiddenLayerNum, int outputDim, int activateMethod)
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
                    dstNet.layers[i].weights[j][k] = layers[i].weights[j][k];
                }
                dstNet.layers[i].bias[j] = layers[i].bias[j];
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

    void BPNet::calculateBatchGradient(std::vector<double> &x, std::vector<double> &y)
    {
        feedForward(x);
        backPropagate(layers[outputIndex].outputs, y);
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

    void BPNet::BGD(double learningRate)
    {
        /* gradient descent */
        for (int i = 0; i < layers.size(); i++) {
            layers[i].BGD(learningRate);
        }
        return;
    }

    void BPNet::RMSProp(double rho, double learningRate)
    {
        for (int i = 0; i < layers.size(); i++) {
            layers[i].RMSProp(rho, learningRate);
        }
        return;
    }

    void BPNet::Adam(double alpha1, double alpha2, double learningRate)
    {
        for (int i = 0; i < layers.size(); i++) {
            layers[i].Adam(alpha1, alpha2, learningRate);
        }
        return;
    }

    void BPNet::SGD(std::vector<double> &x, std::vector<double> &y, double learningRate)
    {
        feedForward(x);
        /* calculate final error */
        backPropagate(layers[outputIndex].outputs, y);
        /* gradient descent */
        for (int j = 0; j < layers.size(); j++) {
            if (j == 0) {
                layers[j].SGD(x, learningRate);
            } else {
                layers[j].SGD(layers[j - 1].outputs, learningRate);
            }
        }
        return;
    }

    void BPNet::train(std::vector<std::vector<double> >& x,
            std::vector<std::vector<double> >& y,
            int optimizeMethod,
            int batchSize,
            double learningRate,
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
        int len = x.size();
        for (int i = 0; i < iterateNum; i++) {
            for (int j = 0; j < batchSize; j++) {
                int k = rand() % len;
                calculateBatchGradient(x[k], y[k]);
            }
            switch (optimizeMethod) {
                case OPT_BGD:
                    BGD(learningRate);
                    break;
                case OPT_RMSPROP:
                    RMSProp(0.9, learningRate);
                    break;
                case OPT_ADAM:
                    Adam(0.9, 0.99, learningRate);
                    break;
                default:
                    Adam(0.9, 0.99, learningRate);
                    break;
            }
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
