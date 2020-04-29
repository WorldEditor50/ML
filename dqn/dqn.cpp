#include "dqn.h"
namespace ML {
    void DQNet::createNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
            int maxMemorySize, int replaceTargetIter, int batchSize)
    {
        if (stateDim < 1 || hiddenDim < 1 || hiddenLayerNum < 1 || actionDim < 1 ||
                maxMemorySize < 1 || replaceTargetIter < 1 || batchSize < 1) {
            return;
        }
        this->gamma = 0.9;
        this->exploringRate = 1;
        this->stateDim = stateDim;
        this->actionDim = actionDim;
        this->maxMemorySize = maxMemorySize;
        this->replaceTargetIter = replaceTargetIter;
        this->batchSize = batchSize;
        this->QMainNet.createNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, ACTIVATE_RELU);
        this->QTargetNet.createNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, ACTIVATE_RELU);
        this->QMainNet.copyTo(QTargetNet);
        return;
    }

    void DQNet::perceive(std::vector<double>& state,
            std::vector<double>& action,
            std::vector<double>& nextState,
            double reward,
            bool done)
    {
        if (state.size() != stateDim || action.size() != actionDim
                || nextState.size() != stateDim) {
            return;
        }
        Transition transition;
        transition.state = state;
        transition.action = action;
        transition.nextState = nextState;
        transition.reward = reward;
        transition.done = done;
        memories.push_back(transition);
        return;
    }

    void DQNet::forget()
    {
        int k = memories.size() - 1;
        for (int i = 0; i < k / 3; i++) {
            memories.pop_front();
        }
        return;
    }

    int DQNet::eGreedyAction(std::vector<double> &state)
    {
        if (state.size() != stateDim) {
            return -1;
        }
        double p = double(rand() % 10000) / 10000;
        int index = 0;
        if (p <= exploringRate) {
            std::vector<double>& QMainNetOutput = QMainNet.getOutput();
            for (int i = 0; i < QMainNetOutput.size(); i++) {
                QMainNetOutput[i] = double(rand() % 10000) / 10000;
            }
            index = maxQ(QMainNetOutput);
        } else {
            index = action(state);
        }
        return index;
    }

    int DQNet::action(std::vector<double> &state)
    {
        int index = 0;
        QMainNet.feedForward(state);
        std::vector<double>& action = QMainNet.getOutput();
        index = maxQ(action);
        return index;
    }

    int DQNet::maxQ(std::vector<double>& q_value)
    {
        int index = 0;
        double maxValue = q_value[0];
        for (int i = 0; i < q_value.size(); i++) {
            if (maxValue < q_value[i]) {
                maxValue = q_value[i];
                index = i;
            }
        }
        return index;
    }

    void DQNet::experienceReplay(Transition& x)
    {
        std::vector<double> qTarget(actionDim);
        std::vector<double>& QTargetNetOutput = QTargetNet.getOutput();
        std::vector<double>& QMainNetOutput = QMainNet.getOutput();
        /* estimate q-target: Q-Regression */
        qTarget = x.action;
        QMainNet.feedForward(x.nextState);
        int index = maxQ(QMainNetOutput);
        if (x.done == true) {
            qTarget[index] = x.reward;
        } else {
            QTargetNet.feedForward(x.nextState);
            qTarget[index] = x.reward + gamma * QTargetNetOutput[index];
        }
        /* train QMainNet */
        QMainNet.calculateBatchGradient(x.state, x.action, qTarget);
        return;
    }

    void DQNet::learn(double learningRate, double minExploringRate)
    {
        if (memories.size() < batchSize) {
            return;
        }
        if (learningSteps % replaceTargetIter == 0) {
            std::cout<<"update target net"<<std::endl;
            /* update tagetNet */
            QMainNet.copyTo(QTargetNet);
            learningSteps = 0;
        }
        /* experience replay */
        for (int i = 0; i < batchSize; i++) {
            int k = rand() % memories.size();
            experienceReplay(memories[k]);
        }
        QMainNet.Adam(0.9, 0.99, learningRate);
        /* reduce memory */
        if (memories.size() > maxMemorySize) {
            forget();
        }
        /* update step */
        if (exploringRate > minExploringRate) {
            exploringRate *= 0.99;
        } else {
            exploringRate = minExploringRate;
        }
        learningSteps++;
        return;
    }

    void DQNet::onlineLearning(std::vector<Transition>& x, double learningRate, double minExploringRate)
    {
        if (learningSteps % replaceTargetIter == 0) {
            std::cout<<"update target net"<<std::endl;
            /* update tagetNet */
            QMainNet.copyTo(QTargetNet);
            learningSteps = 0;
        }
        for (int i = 0; i < x.size(); i++) {
            int k = rand() % x.size();
            experienceReplay(x[k]);
        }
        QMainNet.Adam(0.9, 0.99, learningRate);
        /* update step */
        if (exploringRate > minExploringRate) {
            exploringRate *= 0.99;
        } else {
            exploringRate = minExploringRate;
        }
        learningSteps++;
        return;
    }

    void DQNet::save(const std::string &fileName)
    {
        QMainNet.saveParameter(fileName);
        return;
    }

    void DQNet::load(const std::string &fileName)
    {
        QMainNet.loadParameter(fileName);
        QMainNet.copyTo(QTargetNet);
        return;
    }
}
