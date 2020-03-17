#ifndef SVM_H
#define SVM_H
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <fstream>
namespace ML {
#ifndef KERNEL_RBF
#define KERNEL_RBF        0
#endif
#ifndef KERNEL_LAPLACE 
#define KERNEL_LAPLACE    1
#endif
#ifndef KERNEL_SIGMOID
#define KERNEL_SIGMOID    2
#endif
#ifndef KERNEL_POLYNOMIAL
#define KERNEL_POLYNOMIAL 3
#endif
#ifndef KERNEL_LINEAR
#define KERNEL_LINEAR     4
#endif
    struct SupportVector {
        double alpha;
        double y;
        std::vector<double> x;
    };
    class SVM {
        public:
            SVM(){}
            ~SVM(){}
            double classify(std::vector<double>& xi);
            void train(int n);
            void SMO(int n);
            void loadDataSet(const std::string& fileName, int cols, int rows);
            void setParameter(int kernelType, double C, double tolerance);
            void loadParameter(const std::string& fileName);
            void saveParameter(const std::string& fileName);
            void show();
        private:
            bool KKT(double yi, double Ei, double alpha_i);
            double g(std::vector<double>& xi);
            double kernel_rbf(std::vector<double> x1, std::vector<double> x2);
            double kernel_laplace(std::vector<double> x1, std::vector<double> x2);
            double kernel_sigmoid(std::vector<double> x1, std::vector<double> x2);
            double kernel_polynomial(std::vector<double> x1, std::vector<double> x2);
            double kernel_linear(std::vector<double> x1, std::vector<double> x2);
            double kernel(std::vector<double> x1, std::vector<double> x2, int kernelType);
            double dotProduct(std::vector<double> x1, std::vector<double> x2);
            int random(int i);
            double max(double x1, double x2);
            double min(double x1, double x2);
            std::vector<std::vector<double> > x;
            std::vector<double> y;
            std::vector<double> alpha;
            std::vector<SupportVector> sv;
            double b;
            double C;
            double tolerance;
            double learningRate;
            int kernelType;
    };
}
#endif // SVM_H
