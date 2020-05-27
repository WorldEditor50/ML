#include "policyGradient.h"
namespace ML {
    void DPGNet::CreateNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
                          double learningRate)
    {
        if (stateDim < 1 || hiddenDim < 1 || hiddenLayerNum < 1 || actionDim < 1) {
            return;
        }
        this->gamma = 0.9;
        this->exploringRate = 1;
        this->stateDim = stateDim;
        this->actionDim = actionDim;
        this->learningRate = learningRate;
        this->policyNet.CreateNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, ACTIVATE_SIGMOID, true, LOSS_CROSS_ENTROPY);
        return;
    }

    int DPGNet::eGreedyAction(std::vector<double> &state)
    {
        if (state.size() != stateDim) {
            return -1;
        }
        double p = double(rand() % 10000) / 10000;
        int index = 0;
        if (p < exploringRate) {
            index = RandomAction();
        } else {
            index = policyNet.FeedForward(state);
        }
        return index;
    }

    int DPGNet::RandomAction()
    {
        std::vector<double>& policyNetOutput = policyNet.GetOutput();
        policyNetOutput.assign(actionDim, 0);
        int index = rand() % actionDim;
        policyNetOutput[index] = 1;
        return index;
    }

    int DPGNet::Action(std::vector<double> &state)
    {
        return policyNet.FeedForward(state);
    }

    int DPGNet::MaxAction(std::vector<double>& value)
    {
        int index = 0;
        double maxValue = value[0];
        for (int i = 0; i < value.size(); i++) {
            if (maxValue < value[i]) {
                maxValue = value[i];
                index = i;
            }
        }
        return index;
    }

    void DPGNet::StdScore(std::vector<double> &x)
    {
        double u = 0;
        double n = 0;
        double sigma = 0;
        for (int i = 0 ; i < x.size(); i++) {
            u += x[i];
            n++;
        }
        u = u / n;
        for (int i = 0 ; i < x.size(); i++) {
            x[i] -= u;
            sigma += x[i] * x[i];
        }
        sigma = sqrt(sigma / n);
        for (int i = 0 ; i < x.size(); i++) {
            x[i] = x[i] / sigma;
        }
        return;
    }

    void DPGNet::Reinforce(std::vector<Step>& x)
    {
        double r = 0;
        std::vector<double> discountedReward(x.size());
        for (int i = x.size() - 1; i >= 0; i--) {
            r = gamma * r + x[i].reward;
            discountedReward[i] = r;
        }
        StdScore(discountedReward);
        for (int i = 0; i < x.size(); i++) { 
            int k = MaxAction(x[i].action);
            x[i].action[k] *= discountedReward[i];
            policyNet.Gradient(x[i].state, x[i].action);
        }
        policyNet.RMSProp(0.9, learningRate);
        exploringRate *= 0.99;
        exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
        return;
    }

    void DPGNet::Save(const std::string &fileName)
    {
        policyNet.Save(fileName);
        return;
    }

    void DPGNet::Load(const std::string &fileName)
    {
        policyNet.Load(fileName);
        return;
    }
}
