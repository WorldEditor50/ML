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
        bool isEnd;
    };
    class DQNet {
        public:
            DQNet(){}
            ~DQNet(){}
            void createNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
                    int maxMemorySize, int replaceTargetIter, int batchSize, double learningRate);
            void perceive(std::vector<double>& state,
                    std::vector<double>& action,
                    std::vector<double>& nextState,
                    double reward,
                    bool isEnd);
            void forget();
            int eGreedyAction(std::vector<double>& state);
            int action(std::vector<double>& state);
            int maxQ(std::vector<double>& q_value);
            void experienceReplay(Transition& x);
            void learn();
            void onlineLearning(std::vector<Transition>& x);
            void save(const std::string& fileName);
            void load(const std::string& fileName);
            int stateDim;
            int actionDim;
            double gamma;
            double epsilonMax;
            double epsilon;
            int maxMemorySize;
            int batchSize;
            double learningRate;
            int learningSteps;
            int replaceTargetIter;
            BPNet QMainNet;
            BPNet QTargetNet;
            std::deque<Transition> memories;
    };
}
#endif // DQNN_H
