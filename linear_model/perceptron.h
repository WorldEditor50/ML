#ifndef PERCEPTRON_H
#define PERCEPTRON_H
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <fstream>
namespace ML {
#define KERNEL_RBF        0
#define KERNEL_POLYNOMIAL 1
#define KERNEL_LINEAR     2
class Perceptron
{
public:
    Perceptron(){}
    ~Perceptron(){}
    double classify(std::vector<double>& xi);
    void train(int n);
    void create(const std::string& fileName, int cols, int rows);
    void setKernel(int kernelType);
    void show();
private:
    double kernel_rbf(std::vector<double>& x1, std::vector<double>& x2);
    double kernel_polynomial(std::vector<double>& x1, std::vector<double>& x2);
    double kernel(std::vector<double>& x1, std::vector<double>& x2, int kernelType);
    double dotProduct(std::vector<double>& x1, std::vector<double>& x2);
    std::vector<std::vector<double> > x;
    std::vector<double> y;
    std::vector<double> alpha;
    double b;
    double learningRate;
    int kernelType;
};
}
#endif // PERCEPTRON_H
