#include "perceptron.h"
namespace ML {
    double Perceptron::classify(std::vector<double>& xi)
    {
        double label = 0.0;
        double sum = 0.0;
        for (int j = 0; j < x.size(); j++) {
            sum += alpha[j] * y[j] * kernel(x[j], xi, kernelType);
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

    void Perceptron::train(int n)
    {
        for (int k = 0; k < n; k++) {
            int i = 0;
            double sum = 0.0;
            i = rand() % x.size();
            for (int j = 0; j < x.size(); j++) {
                sum += alpha[j] * y[j] * kernel(x[j], x[i], kernelType);
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

    void Perceptron::create(const std::string& fileName, int cols, int rows)
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
        learningRate = 0.1;
        kernelType = KERNEL_LINEAR;
        return;
    }

    void Perceptron::show()
    {
        std::cout<<"alpha = ";
        for (int i = 0; i < alpha.size(); i++) {
            std::cout<< alpha[i] <<" ";
        }
        std::cout<<std::endl;
        return;
    }

    void Perceptron::setKernel(int kernelType)
    {
        this->kernelType = kernelType;
        return;
    }

    double Perceptron::kernel_rbf(std::vector<double>& x1, std::vector<double>& x2)
    {
        /* rbf */
        double sigma = 1;
        double xL2 = 0.0;
        xL2 = dotProduct(x1, x1) + dotProduct(x2, x2) - 2 * dotProduct(x1, x2);
        xL2 = xL2 / (-2 * sigma *sigma);
        return exp(xL2);
    }

    double Perceptron::kernel_polynomial(std::vector<double>& x1, std::vector<double>& x2)
    {
        return pow(dotProduct(x1, x2) + 1, 100);	
    }

    double Perceptron::kernel(std::vector<double>& x1, std::vector<double>& x2, int kernelType)
    {
        double innerProduct = 0;
        switch (kernelType) {
            case KERNEL_RBF:
                innerProduct = kernel_rbf(x1, x2);
                break;
            case KERNEL_POLYNOMIAL:
                innerProduct = kernel_polynomial(x1, x2);	
                break;
            case KERNEL_LINEAR:
                innerProduct = dotProduct(x1, x2);	
                break;
            default:
                innerProduct = dotProduct(x1, x2);	
                break;
        }
        return innerProduct;
    }

    double Perceptron::dotProduct(std::vector<double>& x1, std::vector<double>& x2)
    {
        double sum = 0.0;
        for (int i = 0; i < x1.size(); i++) {
            sum += x1[i] * x2[i];
        }
        return sum;
    }
}
