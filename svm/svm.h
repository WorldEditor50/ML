#ifndef SVM_H
#define SVM_H
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include "../utility/utility.h"
namespace ML {


namespace Kernel {

template<typename T>
class RBF {
public:
    static T _(std::vector<T>& x1, std::vector<T>& x2)
    {
        T sigma = 1;
        T xL2 = 0.0;
        xL2 = dotProduct(x1, x1) + dotProduct(x2, x2) - 2 * dotProduct(x1, x2);
        xL2 = xL2 / (-2 * sigma * sigma);
        return exp(xL2);
    }
};
template<typename T>
class Laplace {
public:
    static T _(std::vector<T>& x1, std::vector<T>& x2)
    {
        T sigma = 1;
        T xL2 = 0.0;
        xL2 = dotProduct(x1, x1) + dotProduct(x2, x2) - 2 * dotProduct(x1, x2);
        xL2 = sqrt(xL2) / sigma * (-1);
        return exp(xL2);
    }
};

template<typename T>
class Sigmoid {
public:
    static T _(std::vector<T>& x1, std::vector<T>& x2)
    {
        T beta1 = 1;
        T theta = -1;
        return tanh(beta1 * dotProduct(x1, x2) + theta);
    }
};

template<typename T>
class Polynomial {
public:
    static T _(std::vector<T>& x1, std::vector<T>& x2)
    {
        T d = 1.0;
        T p = 100;
        return pow(dotProduct(x1, x2) + d, p);
    }
};

template<typename T>
class Linear {
public:
    static T _(std::vector<T>& x1, std::vector<T>& x2)
    {
        return dotProduct(x1, x2);
    }
};

}

template<template<typename> class KernelF, typename T = double>
class SVM
{
public:
    struct SupportVector
    {
        T alpha;
        T y;
        std::vector<T> x;
    };
    std::vector<std::vector<T> > x;
    std::vector<T> y;
    std::vector<T> alpha;
    std::vector<SupportVector> sv;
    T b;
    T C;
    T tolerance;
    T learningRate;
public:
    explicit SVM(T C, T tolerance)
    {
        this->C = C;
        this->tolerance = tolerance;
    }

    T classify(std::vector<T>& xi)
    {
        T label = 0.0;
        T sum = 0.0;
        for (int j = 0; j < sv.size(); j++) {
            sum += sv[j].alpha * sv[j].y * KernelF<T>::_(sv[j].x, xi);
        }
        sum += b;
        /* f(x) = sign(sum(alpha_j * yj * K(xj, x)))
         * */
        if (sum >= 0) {
            label = 1.0;
        } else {
            label = -1.0;
        }
        return label;
    }

    void train(int n)
    {
        /* perceptron training */
        for (int k = 0; k < n; k++) {
            int i = 0;
            T sum = 0.0;
            i = rand() % x.size();
            for (int j = 0; j < x.size(); j++) {
                sum += alpha[j] * y[j] * KernelF<T>::_(x[j], x[i]);
            }
            sum += b;
            /* if yi * sum(alpha_j * yj * K(xj, xi)) <= 0,
             * then update alpha and b
             * */
            if (y[i] * sum <= 0) {
                alpha[i] += learningRate;
                b += learningRate * y[i];
            }
        }
        return;
    }

    void SMO(int n)
    {
        int k = 0;
        while (k < n) {
            bool alphaOptimized = false;
            for (int i = 0; i < x.size(); i++) {
                T Ei = g(x[i]) - y[i];
                T alpha_i = alpha[i];
                if (KKT(y[i], Ei, alpha[i])) {
                    int j = random(i);
                    T Ej = g(x[j]) - y[j];
                    T alpha_j = alpha[j];
                    /* optimize alpha[j] */
                    T L = 0;
                    T H = 0;
                    if (y[i] != y[j]) {
                        L = std::max(0, alpha[j] - alpha[i]);
                        H = std::min(C, C + alpha[j] - alpha[i]);
                    } else {
                        L = std::max(0, alpha[j] + alpha[i] - C);
                        H = std::min(C, alpha[j] + alpha[i]);
                    }
                    if (L == H) {
                        continue;
                    }
                    T Kii = KernelF<T>::_(x[i], x[i]);
                    T Kjj = KernelF<T>::_(x[j], x[j]);
                    T Kij = KernelF<T>::_(x[i], x[j]);
                    T eta = Kii + Kjj - 2 * Kij;
                    if (eta <= 0) {
                        continue;
                    }
                    alpha[j] += y[j] * (Ei - Ej) / eta;
                    if (alpha[j] > H) {
                        alpha[j] = H;
                    } else if (alpha[j] < L) {
                        alpha[j] = L;
                    }
                    if (abs(alpha[j] - alpha_j) < tolerance) {
                        continue;
                    }
                    /* optimize alpha[i] */
                    alpha[i] += y[i] * y[j] * (alpha_j - alpha[j]);
                    /* update b */
                    T b1 = b - Ei - y[i] * Kii * (alpha[i] - alpha_i)
                        - y[j] * Kij * (alpha[j] - alpha_j);
                    T b2 = b - Ej - y[i] * Kij * (alpha[i] - alpha_i)
                        - y[j] * Kjj * (alpha[j] - alpha_j);
                    if (alpha[i] > 0 && alpha[i] < C) {
                        b = b1;
                    } else if (alpha[j] > 0 && alpha[j] < C) {
                        b = b2;
                    } else {
                        b = (b1 + b2) / 2;
                    }
                    alphaOptimized = true;
                }
            }
            if (alphaOptimized == false) {
                k++;
            } else {
                k = 0;
            }
        }
        /* save support vectors */
        for (int i = 0; i < alpha.size(); i++) {
            if (alpha[i] > 0) {
                SupportVector v;
                v.alpha = alpha[i];
                v.y = y[i];
                v.x.assign(x[i].begin(), x[i].end());
                sv.push_back(v);
            }
        }
        return;
    }
private:
    bool KKT(T yi, T Ei, T alpha_i)
    {
        return ((yi * Ei < -1 * tolerance) && (alpha_i < C) ||
                (yi * Ei > tolerance) && (alpha_i > 0));
    }

    T g(std::vector<T>& xi)
    {
        T sum = 0.0;
        for (int j = 0; j < x.size(); j++) {
            sum += alpha[j] * y[j] * KernelF<T>::_(x[j], xi);
        }
        sum += b;
        return sum;
    }
    int random(int i)
    {
        int j = 0;
        j = rand() % x.size();
        while (j == i) {
            j = rand() % x.size();
        }
        return j;
    }

    void loadData(const std::string& fileName, int cols, int rows)
    {
        std::ifstream file;
        file.open(fileName);
        /* create feature space */
        y.resize(rows);
        x.resize(rows);
        alpha.resize(rows);
        for (int i = 0; i < rows; i++) {
            x[i].resize(cols);
        }
        /* load data */
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                file >> x[i][j];
            }
            file >> y[i];
        }
        file.close();
        /* init parameter */
        for (int i = 0; i < alpha.size(); i++) {
            alpha[i] = 0.0;
        }
        b = 0.0;
        C = 1.0;
        learningRate = 0.1;
        tolerance = 0.00001;
        return;
    }

    void show()
    {
        std::cout<<"alpha = ";
        for (int i = 0; i < alpha.size(); i++) {
            std::cout<< alpha[i] <<" ";
        }
        std::cout<<std::endl;
        return;
    }

};
}
#endif // SVM_H
