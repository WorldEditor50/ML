#include "bpnn.h"
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

Mat<float> LOG(Mat<float> X)
{
    return mappingTo(X, log);
}

Mat<float> EXP(Mat<float> X)
{
    return mappingTo(X, exp);
}

Mat<float> SQRT(Mat<float> X)
{
    return mappingTo(X, sqrt);
}

Mat<float> SOFTMAX(Mat<float>& X)
{
    /* softmax works in multi-classify */
    float maxValue = max(X);
    Mat<float> delta = EXP(X - maxValue);
    float s = sum(delta);
    if (s != 0) {
        X = delta / s;
    }
    return X;
}

Mat<float> Layer::Activate(Mat<float> X)
{
    Mat<float> Y;
    switch (activateType) {
        case ACTIVATE_SIGMOID:
            Y = mappingTo(X, sigmoid);
            break;
        case ACTIVATE_RELU:
            Y = mappingTo(X, relu);
            break;
        case ACTIVATE_TANH:
            Y = mappingTo(X, tanh);
            break;
        case ACTIVATE_LINEAR:
            Y = X;
            break;
        default:
            Y = mappingTo(X, sigmoid);
            break;
    }
    return Y;
}

Mat<float> Layer::dActivate(Mat<float>& Y)
{
    Mat<float> dY;
    switch (activateType) {
        case ACTIVATE_SIGMOID:
            dY = mappingTo(Y, dsigmoid);
            break;
        case ACTIVATE_RELU:
            dY = mappingTo(Y, drelu);
            break;
        case ACTIVATE_TANH:
            dY = mappingTo(Y, dtanh);
            break;
        case ACTIVATE_LINEAR:
            dY = Y;
            dY.full(1);
            break;
        default:
            dY = mappingTo(Y, dsigmoid);
            break;
    }
    return dY;
}
void Layer::createLayer(int inputDim, int layerDim, int activateType, int lossType)
{
    if (layerDim < 1 || inputDim < 1) {
        return;
    }
    this->lossType = lossType;
    this->activateType = activateType;
    W.createMat(layerDim, inputDim);
    B.createMat(layerDim, 1);
    O.createMat(layerDim, 1);
    E.createMat(layerDim, 1);
    /* buffer for optimization */
    dW.createMat(layerDim, inputDim);
    Sw.createMat(layerDim, inputDim);
    Vw.createMat(layerDim, inputDim);
    dB.createMat(layerDim, 1);
    Sb.createMat(layerDim, 1);
    Vb.createMat(layerDim, 1);
    this->alpha1_t = 1;
    this->alpha2_t = 1;
    this->delta = pow(10, -8);
    this->decay = 0.01;
    /* init */
    W.uniformRandom();
    B.uniformRandom();
    return;
}

void Layer::SGD(float learningRate)
{
    /*
     * e = (Activate(wx + b) - T)^2/2
     * de/dw = (Activate(wx +b) - T)*DActivate(wx + b) * x
     * de/db = (Activate(wx +b) - T)*DActivate(wx + b)
     * */
    W -= dW * learningRate;
    B -= dB * learningRate;
    dW.zero();
    dB.zero();
    return;
}

void Layer::RMSProp(float rho, float learningRate)
{
    Sw = Sw * rho + (dW % dW) * (1 - rho);
    Sb = Sb * rho + (dB % dB) * (1 - rho);
    W -= dW / (SQRT(Sw) + delta) * learningRate;
    B -= dB / (SQRT(Sb) + delta) * learningRate;
    dW.zero();
    dB.zero();
    return;
}

void Layer::Adam(float alpha1, float alpha2, float learningRate)
{
    alpha1_t *= alpha1;
    alpha2_t *= alpha2;
    Vw = Vw * alpha1 + dW * (1 - alpha1);
    Vb = Vb * alpha1 + dB * (1 - alpha1);
    Sw = Sw * alpha2 + (dW % dW) * (1 - alpha2);
    Sb = Sb * alpha2 + (dB % dB) * (1 - alpha2);
    Mat<float> Vwt = Vw / (1 - alpha1_t);
    Mat<float> Vbt = Vb / (1 - alpha1_t);
    Mat<float> Swt = Sw / (1 - alpha2_t);
    Mat<float> Sbt = Sb / (1 - alpha2_t);
    W -= Vwt / (SQRT(Swt) + delta) * learningRate;
    B -= Vbt / (SQRT(Sbt) + delta) * learningRate;
    dW.zero();
    dB.zero();
    return;
}

void BPNet::createNet(int inputDim, int hiddenDim, int hiddenLayerNum, int outputDim, int activateType, int lossType)
{
    Layer inputLayer;
    inputLayer.createLayer(inputDim, hiddenDim, activateType);
    layers.push_back(inputLayer);
    for (int i = 1; i < hiddenLayerNum; i++) {
        Layer hiddenLayer;
        hiddenLayer.createLayer(hiddenDim, hiddenDim, activateType);
        layers.push_back(hiddenLayer);
    }
    if (lossType == LOSS_MSE) {
        Layer outputLayer;
        outputLayer.createLayer(hiddenDim, outputDim, activateType);
        layers.push_back(outputLayer);
    }
    if (lossType == LOSS_CROSS_ENTROPY) {
        Layer softmaxLayer;
        softmaxLayer.createLayer(hiddenDim, outputDim, ACTIVATE_LINEAR, LOSS_CROSS_ENTROPY);
        layers.push_back(softmaxLayer);
    }
    this->outputIndex = layers.size() - 1;
    return;
}

void BPNet::copyTo(BPNet& dstNet)
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

void BPNet::softUpdateTo(BPNet &dstNet, float alpha)
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

void BPNet::feedForward(Mat<float>& x)
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

Mat<float>& BPNet::getOutput()
{
    Mat<float>& outputs = layers[outputIndex].O;
    return outputs;
}

void BPNet::gradient(Mat<float> &x, Mat<float> &y)
{
    /* calculate loss */
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
            Mat<float> dy = layers[outputIndex].O - y;
            layers[i].dW += dy * layers[i - 1].O.Tr();
            layers[i].dB += dy;
        } else {
            Mat<float> dy = layers[i].dActivate(layers[i].O);
            if (i == 0) {
                layers[i].dW += (layers[i].E % dy) * x.Tr();
            } else {
                layers[i].dW += (layers[i].E % dy) * layers[i - 1].O.Tr();
            }
            layers[i].dB += layers[i].E % dy;
        }
    }
    return;
}

void BPNet::SGD(float learningRate)
{
    /* gradient descent */
    for (int i = 0; i < layers.size(); i++) {
        layers[i].SGD(learningRate);
    }
    return;
}

void BPNet::RMSProp(float rho, float learningRate)
{
    for (int i = 0; i < layers.size(); i++) {
        layers[i].RMSProp(rho, learningRate);
    }
    return;
}

void BPNet::Adam(float alpha1, float alpha2, float learningRate)
{
    for (int i = 0; i < layers.size(); i++) {
        layers[i].Adam(alpha1, alpha2, learningRate);
    }
    return;
}

void BPNet::optimize(int optType, float learningRate)
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

void BPNet::train(std::vector<Mat<float> >& x,
        std::vector<Mat<float> >& y,
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

void BPNet::show()
{
    layers[outputIndex].O.show();
    return;
}

void BPNet::load(const std::string& fileName)
{
    std::ifstream file;
    file.open(fileName);
    for (int i = 0; i < layers.size(); i++) {
        layers[i].W.load(fileName);
        layers[i].B.load(fileName);
    }
    return;
}

void BPNet::save(const std::string& fileName)
{
    std::ofstream file;
    file.open(fileName);
    for (int i = 0; i < layers.size(); i++) {
        layers[i].W.save(fileName);
        layers[i].B.save(fileName);
    }
    return;
}
