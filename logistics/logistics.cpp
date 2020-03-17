#include "logistics.h"
namespace ML {

    double Logistics::sigmoid(double x)
    {
        return exp(x) / (1 + exp(x));
    }

    double Logistics::derivative_sigmoid(double y)
    {
        return y * ( 1 - y);
    }

    void Logistics::loadDataSet(const std::string& fileName, int rowNum, int colNum)
    {
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
        return;
    }

    void Logistics::create(int featureNum, double learningRate)
    {
        w.resize(featureNum);
        for (int i = 0; i < w.size(); i++) {
            w[i] = double(rand() % 10000 - rand() % 10000) / 10000;
        }
        b = double(rand() % 10000 - rand() % 10000) / 10000;
        this->learningRate = learningRate;
        return;
    }

    double Logistics::calculateOutput(std::vector<double>& xi)
    {
        double expect = 0;
        for (int i = 0; i < xi.size(); i++) {
            expect += w[i] * xi[i];
        }
        expect += b;
        return sigmoid(expect);
    }

    void Logistics::adjustWeight(double y, double yi, std::vector<double>& xi)
    {
        /* 
           E = (Activate(sum(w*x) + b) - r)^2/2
           dE/dwi = (Activate(sum(w*x) + b) - r) * dActivate(sum(w*x) + b) * xi 
           y = Activate(sum(w*x) + b)
           dE/dwi = (y - r) * dy * xi;
           dE/db = (y - r) * dy
           */
        for (int i = 0; i < w.size(); i++) {
            w[i] -= learningRate * (y - yi) * derivative_sigmoid(y) * xi[i];
        }
        b -= learningRate * (y - yi) * derivative_sigmoid(y);
        return;
    }

    void Logistics::train(int iterateNum)
    {
        if (x.empty()) {
            return;
        }
        double yi = 0;
        int k = 0;
        for (int i = 0; i < iterateNum; i++) {
            k = rand() % x.size();
            yi = calculateOutput(x[k]);
            adjustWeight(yi, y[k], x[k]);
        }
        return;
    }

    void Logistics::show()
    {
        std::cout<<"w = ";
        for (int i = 0; i < w.size(); i++) {
            std::cout<<w[i]<<" ";
        }
        std::cout<<std::endl;
        return;
    }
}
