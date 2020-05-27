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
namespace ML {
    struct Step {
        std::vector<double> state;
        std::vector<double> action;
        double reward;
    };
    class DPGNet {
        public:
            DPGNet(){}
            ~DPGNet(){}
            void CreateNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
                           double learningRate = 0.001);
            int eGreedyAction(std::vector<double>& state);
            int RandomAction();
            int Action(std::vector<double>& state);
            int MaxAction(std::vector<double>& value);
            void StdScore(std::vector<double>& x);
            void Reinforce(std::vector<Step>& steps);
            void Save(const std::string& fileName);
            void Load(const std::string& fileName);
            int stateDim;
            int actionDim;
            double gamma;
            double exploringRate;
            double learningRate;
            BPNet policyNet;
    };
}
#endif // POLICY_GRADIENT_H
