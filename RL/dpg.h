#ifndef POLICY_GRADIENT_H
#define POLICY_GRADIENT_H
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

class DPG
{
public:
    int stateDim;
    int actionDim;
    double gamma;
    double exploringRate;
    double learningRate;
    BPNN policyNet;
public:
    DPG(){}
    explicit DPG(int stateDim,
                 int hiddenDim,
                 int hiddenLayerNum,
                 int actionDim);
    ~DPG(){}
    int greedyAction(std::vector<double>& state);
    int randomAction();
    int action(std::vector<double>& state);
    int maxAction(std::vector<double>& value);
    void zscore(std::vector<double>& x);
    void reinforce(OptType optType, double learningRate, std::vector<Step>& x);
    void save(const std::string& fileName);
    void load(const std::string& fileName);
};
}
#endif // POLICY_GRADIENT_H
