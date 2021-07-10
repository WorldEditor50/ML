#ifndef LINEAR_MODEL_H
#define LINEAR_MODEL_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "../utility/utility.h"
namespace ML {

template <template<typename> class ActivateF = Sigmoid, typename T = double>
class LinearModel
{
public:
    std::vector<T> w;
    T b;
public:
    explicit LinearModel(int dim)
    {
        w = std::vector<T>(dim);
        for (int i = 0; i < w.size(); i++) {
            w[i] = T(rand() % 10000 - rand() % 10000) / 10000;
        }
        b = T(rand() % 10000 - rand() % 10000) / 10000;
    }
    ~LinearModel(){}

    void loadDataSet(const std::string& fileName, int rowNum, int colNum)
    {
#if 0
        std::ifstream file;
        file.open(fileName);
        x.resize(rowNum);
        y.resize(rowNum);
        for (int i = 0; i < rowNum; i++) {
            x[i].resize(colNum);
            for (int j = 0; j < colNum; j++) {
                file>>x[i][j];
            }
            file>>y[i];
        }
#endif
        return;
    }

    T output(std::vector<T>& xi)
    {
        return ActivateF<T>::_(dotProduct(w, xi) + b);
    }

    void update(T y, T yi, std::vector<T>& xi, T learningRate)
    {
        /*
           E = (Activate(sum(w*x) + b) - r)^2/2
           dE/dwi = (Activate(sum(w*x) + b) - r) * dActivate(sum(w*x) + b) * xi
           y = Activate(sum(w*x) + b)
           dE/dwi = (y - r) * dy * xi;
           dE/db = (y - r) * dy
           */
        for (int i = 0; i < w.size(); i++) {
            w[i] -= learningRate * (y - yi) * ActivateF<T>::d(y) * xi[i];
        }
        b -= learningRate * (y - yi) * ActivateF<T>::d(y);
        return;
    }

    void train(std::vector<std::vector<T> > &x,
               std::vector<T> &y,
               double learningRate,
               int batchSize,
               int iterateNum)
    {
        if (x.empty()) {
            return;
        }
        int k = 0;
        for (int i = 0; i < iterateNum; i++) {
            for (int j = 0; j < batchSize; j++) {
                k = rand() % x.size();
                update(output(x[k]), y[k], x[k], learningRate);
            }
        }
        return;
    }

    void show()
    {
        std::cout<<"w = ";
        for (int i = 0; i < w.size(); i++) {
            std::cout<<w[i]<<" ";
        }
        std::cout<<std::endl;
        return;
    }

};
}
#endif // LINEAR_MODEL_H
