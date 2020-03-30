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
        this->states.resize(batchSize);
        this->rewards.resize(batchSize);
        this->isEnds.resize(batchSize);
        this->q_main.resize(batchSize);
        this->q_target.resize(batchSize);
        this->q_main_next.resize(batchSize);
        this->q_target_next.resize(batchSize);
        for (int i = 0; i < batchSize; i++) {
            states[i].resize(stateDim, 0);
            q_main[i].resize(actionDim, 0);
            q_target[i].resize(actionDim, 0);
            q_main_next[i].resize(actionDim, 0);
            q_target_next[i].resize(actionDim, 0);
        }
        this->QMainNet.createNet(stateDim, hiddenDim, actionDim, hiddenLayerNum, learningRate);
        this->QTargetNet.createNet(stateDim, hiddenDim, actionDim, hiddenLayerNum, learningRate);
        this->QMainNet.copyTo(QTargetNet);
        return;
    }

    void DQNet::perceive(std::vector<double>& state,
            std::vector<double>& action,
            std::vector<double>& nextState,
            double reward,
            bool isEnd)
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
        transition.isEnd = isEnd;
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
        double p = double(rand() % 1000) / 1000;
        int index = 0;
        if (p <= epsilon) {
            index = action(state);
        } else {
            index = rand() % actionDim;
        }
        return index;
    }

    int DQNet::action(std::vector<double> &state)
    {
        int index = 0;
        QMainNet.feedForward(state);
        std::vector<double>& action = QMainNet.getOutput();
        double maxValue = action[0];
        for (int i = 0; i < action.size(); i++) {
            if (maxValue < action[i]) {
                maxValue = action[i];
                index = i;
            }
        }
        std::cout<<action[0]<<" "<<action[1]<<" "<<action[2]<<" "<<action[3]<<std::endl;
        return index;
    }

    int DQNet::maxQ(std::vector<double>& qnext)
    {
        double maxValue = qnext[0];
        int index = 0;
        for (int i = 0; i < qnext.size(); i++) {
            if (maxValue < qnext[i]) {
                maxValue = qnext[i];
                index = i;
            }
        }
        return index;
    }

    void DQNet::experienceReplay()
    {
        if (memories.size() < batchSize) {
            return;
        }
        /* sampling */
        for (int i = 0; i < batchSize; i++) {
            int index = rand() % memories.size();
            states[i] = memories[index].state;
            q_main[i] = memories[index].action;
            q_target[i] = memories[index].action;
            rewards[i] = memories[index].reward;
            isEnds[i] = memories[index].isEnd;
            /* estimate q-target: DDQN Method */
            QTargetNet.feedForward(memories[index].nextState);
            q_target_next[i] = QTargetNet.getOutput();
            QMainNet.feedForward(memories[index].nextState);
            q_main_next[i] = QMainNet.getOutput();
        }
        /* estimate q-target: DDQN Method */
        for (int i = 0; i < q_target.size(); i++) {
            int index = maxQ(q_main_next[i]);
            if (isEnds[i] == true) {
                q_target[i][index] = rewards[i];
            } else {
                q_target[i][index] = rewards[i] + gamma * q_target_next[i][index];
            }
        }
        /* train QMainNet */
        for (int i = 0; i < batchSize; i++) {
            QMainNet.batchGradientDescent(states, q_main, q_target);
        }
        return;
    }

    void DQNet::learn(int iterateNum)
    {
        if (iterateNum < 1) {
            return;
        }
        for (int i = 0; i < iterateNum; i++) {
            if (learningSteps % replaceTargetIter == 0) {
                std::cout<<"update target net"<<std::endl;
                /* update tagetNet */
                QMainNet.copyTo(QTargetNet);
            }
            /* experience replay */
            experienceReplay();
            /* update step */
            if (epsilon < epsilonMax) {
                epsilon += 0.0001;
            }
            learningSteps++;
        }
        /* reduce memory */
        if (memories.size() > maxMemorySize) {
            forget();
        }
        return;
    }

    void DQNet::onlineLearning(std::vector<Transition>& x)
    {
        std::vector<double> qTarget(actionDim);
        std::vector<double>& QTargetNetOutput = QTargetNet.getOutput();
        std::vector<double>& QMainNetOutput = QMainNet.getOutput();
        for (int i = 0; i < x.size(); i++) {
            /* estimate q-target: DDQN Method */
            QTargetNet.feedForward(x[i].nextState);
            QMainNet.feedForward(x[i].nextState);
            qTarget = QMainNetOutput;
            int index = maxQ(QMainNetOutput);
            if (x[i].isEnd == true) {
                qTarget[index] = x[i].reward;
            } else {
                qTarget[index] = x[i].reward + gamma * QTargetNetOutput[index];
            }
            /* train QMainNet */
            QMainNet.stochasticGradientDescent(x[i].state, x[i].action, qTarget);
            /* add to memories */
            this->memories.push_back(x[i]);
        }
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
