#ifndef RL_BASIC_H
#define RL_BASIC_H
#include <vector>

namespace ML {

struct Step
{
    std::vector<double> state;
    std::vector<double> action;
    double reward;
    Step(){}
    Step(std::vector<double>& s, std::vector<double>& a, double r)
        :state(s), action(a), reward(r) {}
};

struct Transition
{
    std::vector<double> state;
    std::vector<double> action;
    std::vector<double> nextState;
    double reward;
    bool done;
    Transition(){}
    explicit Transition(std::vector<double>& s, std::vector<double>& a,
               std::vector<double>& s_, double r, bool d)
    {
        state = s;
        action = a;
        nextState = s_;
        reward = r;
        done = d;
    }
};

}
#endif // RL_BASIC_H
