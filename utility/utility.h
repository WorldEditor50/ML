#ifndef UTILITY_H
#define UTILITY_H
#include <vector>
#include <string>
#include <cmath>
namespace ML {


template<typename T>
class Sigmoid {
public:
    static T _(T x) {return exp(x) / (1 + exp(x));}
    static T d(T y) {return y * (1 - y);}
};

template<typename T>
class Relu {
public:
    static T _(T x) {return x < 0 ? 0 : x;}
    static T d(T x) {return x < 0 ? 0 : 1;}
};

template<typename T>
class Tanh {
public:
    static T _(T x) {return tanh(x);}
    static T d(T y) {return 1 - y * y;}
};

template<typename T>
class Linear {
public:
    static T _(T x) {return x;}
    static T d(T y) {return 1 - y * y;}
};

template<typename T>
T dotProduct(const std::vector<T> &x1, const std::vector<T> &x2)
{
    T s = 0;
    for (int i = 0; i < x1.size(); i++) {
        s += x1[i] * x2[i];
    }
    return s;
}

}
#endif // UTILITY_H
