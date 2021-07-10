#ifndef PPO_H
#define PPO_H
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

class PPO
{
private:
    int stateDim;
    int actionDim;
    double gamma;
    double beta;
    double delta;
    double epsilon;
    double exploringRate;
    int learningSteps;
    BPNN actorP;
    BPNN actorQ;
    BPNN critic;
public:
    PPO(){}
    explicit PPO(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim);
    ~PPO(){}
    int greedyAction(std::vector<double>& state);
    int action(std::vector<double>& state);
    double KLmean(std::vector<double>& p, std::vector<double>& q);
    double getValue(std::vector<double> &s);
    int maxAction(std::vector<double>& value);
    double clip(double x, double sup, double inf);
    void learnWithKLpenalty(OptType optType, double learningRate, std::vector<Transition>& x);
    void learnWithClipObject(OptType optType, double learningRate, std::vector<Transition>& x);
    void save(const std::string &actorPara, const std::string &criticPara);
    void load(const std::string &actorPara, const std::string &criticPara);
};
}
#endif // PPO_H
