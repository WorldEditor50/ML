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
namespace ML {
    struct Transition {
        std::vector<double> state;
        std::vector<double> action;
        std::vector<double> nextState;
        double reward;
    };
    class DQNet {
        public:
            DQNet(){}
            ~DQNet(){}
            void createNet(int stateDim,
                    int actionDim,
                    int hiddenDim,
                    int hiddenLayerNum,
                    int maxMemorySize,
                    int replaceTargetIter,
                    int batchSize,
                    double learningRate);
            void perceive(std::vector<double>& state,
                    std::vector<double>& action,
                    std::vector<double>& nextState,
                    double reward);
            void forget();
            int eGreedyAction(std::vector<double>& state);
            int action(std::vector<double>& state);
            void experienceReplay();
            int maxQ(std::vector<double>& qnext);
            void learn(int iterateNum);
            void save(const std::string& fileName);
            void load(const std::string& fileName);
            double gamma;
            double epsilonMax;
            double epsilon;
            int maxMemorySize;
            int batchSize;
            double learningRate;
            int learningSteps;
            int replaceTargetIter;
            BpNet evalNet;
            BpNet targetNet;
            std::deque<Transition> memories;
            std::vector<std::vector<double> > states;
            std::vector<double> rewards;
            std::vector<std::vector<double> > q_eval;
            std::vector<std::vector<double> > q_next;
            std::vector<std::vector<double> > q_target;
    };
}
#endif // DQNN_H
