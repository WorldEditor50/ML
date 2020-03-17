#ifndef LOGISTICS_H
#define LOGISTICS_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
namespace ML {
    class Logistics {
        public:
            Logistics(){}
            ~Logistics(){}
            void create(int featureNum, double learningRate);
            void loadDataSet(const std::string& fileName, int rowNum, int colNum);
            void loadParameter(const std::string& fileName);
            void saveParameter(const std::string& fileName);
            double calculateOutput(std::vector<double>& xi);
            void train(int iterateNum);
            void show();
        private:
            double sigmoid(double x);
            double derivative_sigmoid(double y);
            void adjustWeight(double y, double r, std::vector<double>& x);
            std::vector<std::vector<double> > x;
            std::vector<double> y;
            std::vector<double> w;
            double b;
            double learningRate;
    };
}
#endif // LOGISTICS_H
