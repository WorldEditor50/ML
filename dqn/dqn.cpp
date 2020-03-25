#include "dqn.h"
namespace ML {
    void DQNet::createNet(int stateDim, int actionDim, int maxMemorySize, int replaceTargetIter, int batchSize)
    {

        this->learningRate = 0.000001;
        this->gamma = 0.9;
        this->epsilonMax = 0.9;
        this->epsilon = 0;
        this->maxMemorySize = maxMemorySize;
        this->batchSize = batchSize;
        this->states.resize(batchSize);
        this->q_eval.resize(batchSize);
        this->q_next.resize(batchSize);
        this->q_target.resize(batchSize);
        for (int i = 0; i < batchSize; i++) {
            states[i].resize(stateDim, 0);
            q_eval[i].resize(actionDim, 0);
            q_next[i].resize(actionDim, 0);
            q_target[i].resize(actionDim, 0);
        }
        int hiddenDim = 32;
        int hiddenLayerNum = 2;
        this->evalNet.createNet(stateDim, hiddenDim, actionDim, hiddenLayerNum, learningRate);
        this->targetNet.createNet(stateDim, hiddenDim, actionDim, hiddenLayerNum, learningRate);
        this->evalNet.copyTo(targetNet);
        return;
    }

    void DQNet::perceive(std::vector<double>& state, std::vector<double>& action, std::vector<double>& nextState, double reward);
    {
        Transition transition;
        transition.state = state;
        transition.action = action;
        transition.nextState = nextState;
        transition.reward = reward;
        memories.push_back(transition);
        return;
    }

    void DQNet::forget()
    {
        if (memories.size() < 1024) {
            return;
        }
        int k = memories.size() - 1;
        for (int i = 0; i < k / 3; i++) {
            memories.pop_front();
        }
        return;
    }

    int DQNet::chooseAction(std::vector<double> &state)
    {
        double p = double(rand() % 1000) / 1000;
        int index = 0;
        if (p <= epsilon) {
            index = action(state);
        } else {
            index = rand() % action.size();
        }
        return index;
    }

    int DQNet::action(std::vector<double> &state)
    {
        int index = 0;
        evalNet.feedForward(state);
        std::vector<double>& action = evalNet.getOutput();
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

    void DQNet::experienceReplay()
    {
        std::vector<double>& targetNetOutput = targetNet.getOutput();
        for (int i = 0; i < batchSize; i++) {
            int index = rand() % memories.size();
            /* copy statw */
            states[i] = memories[index].state;
            /* copy action */
            q_eval[i] = memories[index].action;
            q_target[i] = memories[index].action;
            /* copy reward */
            rewards[i] = memories[index].reward;
            /* calculate q-target */
            targetNet.feedForward(memories[index].nextState);
            q_next[i].assign(targetNetOutput.begin(), targetNetOutput.end());
        }
        return;
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

    void DQNet::learn()
    {
        if (memories.size() < batchSize) {
            return;
        }
        if (learningSteps % replaceTargetIter == 0) {
            std::cout<<"update target net"<<std::endl;
            /* update tagetNet */
            evalNet.copyTo(targetNet);
            /* reduce memory */
            forget();
        }
        /* experience replay */
        experienceReplay();
        /* estimate q-target */
        for (int i = 0; i < q_target.size(); i++) {
            int index = maxQ(q_next[i]);
            q_target[i][index] = rewards[i] + gamma * q_next[i][index];
        }
        /* train evalNet */
        for (int i = 0; i < batchSize; i++) {
            evalNet.train(states[i], q_eval[i], q_target[i]);
        }
        /* update step */
        if (epsilon < epsilonMax) {
            epsilon += 0.0001;
        }
        learningSteps++;
        return;
    }

    void DQNet::save(const std::string &fileName)
    {
        evalNet.saveParameter(fileName);
        return;
    }

    void DQNet::load(const std::string &fileName)
    {
        evalNet.loadParameter(fileName);
        return;
    }
}
