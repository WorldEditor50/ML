#ifndef DQNN_H
#define DQNN_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <deque>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "../bpnn/bpnn.h"
#include "rl_basic.h"

namespace ML {

class DQN
{
public:
    int stateDim;
    int actionDim;
    double gamma;
    double exploringRate;
    int learningSteps;
    BPNN QMainNet;
    BPNN QTargetNet;
    std::deque<Transition> memories;
public:
    DQN(){}
    explicit DQN(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim);
    ~DQN(){}
    void perceive(std::vector<double>& state,
                  std::vector<double>& action,
                  std::vector<double>& nextState,
                  double reward,
                  bool done);
    std::vector<double>& greedyAction(std::vector<double>& state);
    int randomAction();
    int action(std::vector<double>& state);
    int maxQ(std::vector<double>& q_value);
    void experienceReplay(Transition& x);
    void learn(OptType optType = OPT_RMSPROP,
               int maxMemorySize = 4096,
               int replaceTargetIter = 256,
               int batchSize = 32,
               double learningRate = 0.001);
    void save(const std::string& fileName);
    void load(const std::string& fileName);
};
}
#endif // DQNN_H
