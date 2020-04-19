#include "dqn.h"
namespace ML {
    void DQNet::createNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
            int maxMemorySize, int replaceTargetIter, int batchSize, double learningRate)
    {
        if (stateDim < 1 || hiddenDim < 1 || hiddenLayerNum < 1 || actionDim < 1 ||
                maxMemorySize < 1 || replaceTargetIter < 1 || batchSize < 1 || learningRate < 0) {
            return;
        }
        this->gamma = 0.9;
        this->epsilonMax = 0.9;
        this->epsilon = 0;
        this->stateDim = stateDim;
        this->actionDim = actionDim;
        this->learningRate = learningRate;
        this->maxMemorySize = maxMemorySize;
        this->replaceTargetIter = replaceTargetIter;
        this->batchSize = batchSize;
        this->QMainNet.createNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, RELU, learningRate);
        this->QTargetNet.createNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, RELU, learningRate);
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
        if (p <= epsilon) {
            index = action(state);
        } else {
            std::vector<double>& QMainNetOutput = QMainNet.getOutput();
            for (int i = 0; i < QMainNetOutput.size(); i++) {
                QMainNetOutput[i] = double(rand() % 10000) / 10000;
            }
            index = maxQ(QMainNetOutput);
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
        QTargetNet.feedForward(x.nextState);
        QMainNet.feedForward(x.nextState);
        qTarget = x.action;
        int index = maxQ(QMainNetOutput);
        if (x.done == true) {
            qTarget[index] = x.reward;
        } else {
            qTarget[index] = x.reward + gamma * QTargetNetOutput[index];
        }
        /* train QMainNet */
        QMainNet.calculateBatchGradient(x.state, x.action, qTarget);
        return;
    }

    void DQNet::learn()
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
        QMainNet.updateWithBatchGradient();
        /* update step */
        if (epsilon < epsilonMax) {
            epsilon += 0.0001;
        }
        /* reduce memory */
        if (memories.size() > maxMemorySize) {
            forget();
        }
        learningSteps++;
        return;
    }

    void DQNet::onlineLearning(std::vector<Transition>& x)
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
        QMainNet.updateWithBatchGradient();
        /* update step */
        if (epsilon < epsilonMax) {
            epsilon += 0.0001;
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
        return;
    }
}
