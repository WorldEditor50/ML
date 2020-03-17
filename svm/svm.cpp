#include "svm.h"
namespace ML {

    double SVM::classify(std::vector<double>& xi)
    {
        double label = 0.0;
        double sum = 0.0;
        for (int j = 0; j < sv.size(); j++) {
            sum += sv[j].alpha * sv[j].y * kernel(sv[j].x, xi, kernelType);
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

    void SVM::train(int n)
    {
        /* perceptron training */
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

    void SVM::SMO(int n)
    {
        int k = 0;
        while (k < n) {
            bool alphaOptimized = false;
            for (int i = 0; i < x.size(); i++) {
                double Ei = g(x[i]) - y[i];
                double alpha_i = alpha[i];
                if (KKT(y[i], Ei, alpha[i])) {
                    int j = random(i);
                    double Ej = g(x[j]) - y[j];
                    double alpha_j = alpha[j];
                    /* optimize alpha[j] */
                    double L = 0;
                    double H = 0;
                    if (y[i] != y[j]) {
                        L = max(0, alpha[j] - alpha[i]);
                        H = min(C, C + alpha[j] - alpha[i]);
                    } else {
                        L = max(0, alpha[j] + alpha[i] - C);
                        H = min(C, alpha[j] + alpha[i]);
                    }
                    if (L == H) {
                        continue;
                    }
                    double Kii = kernel(x[i], x[i], kernelType);
                    double Kjj = kernel(x[j], x[j], kernelType);
                    double Kij = kernel(x[i], x[j], kernelType);
                    double eta = Kii + Kjj - 2 * Kij;
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
                    double b1 = b - Ei - y[i] * Kii * (alpha[i] - alpha_i)
                        - y[j] * Kij * (alpha[j] - alpha_j);
                    double b2 = b - Ej - y[i] * Kij * (alpha[i] - alpha_i)
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

    bool SVM::KKT(double yi, double Ei, double alpha_i)
    {
        return ((yi * Ei < -1 * tolerance) && (alpha_i < C) ||
                (yi * Ei > tolerance) && (alpha_i > 0));
    }

    double SVM::g(std::vector<double>& xi)
    {
        double sum = 0.0;
        for (int j = 0; j < x.size(); j++) {
            sum += alpha[j] * y[j] * kernel(x[j], xi, kernelType);
        }
        sum += b;
        return sum;
    }
    int SVM::random(int i)
    {
        int j = 0;
        j = rand() % x.size();
        while (j == i) {
            j = rand() % x.size();
        }
        return j;
    }
    double SVM::max(double x1, double x2)
    {
        double x3 = 0;
        if (x1 > x2) {
            x3 = x1;
        } else {
            x3 = x2;
        }
        return x3;
    }
    double SVM::min(double x1, double x2)
    {
        double x3 = 0;
        if (x1 < x2) {
            x3 = x1;
        } else {
            x3 = x2;
        }
        return x3;
    }
    void SVM::loadDataSet(const std::string& fileName, int cols, int rows)
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
        kernelType = KERNEL_LINEAR;
        return;
    }

    void SVM::loadParameter(const std::string& fileName)
    {

        return;
    }
    void SVM::saveParameter(const std::string& fileName)
    {

        return;
    }

    void SVM::show()
    {
        std::cout<<"alpha = ";
        for (int i = 0; i < alpha.size(); i++) {
            std::cout<< alpha[i] <<" ";
        }
        std::cout<<std::endl;
        return;
    }

    void SVM::setParameter(int kernelType, double C, double tolerance)
    {
        this->kernelType = kernelType;
        this->C = C;
        this->tolerance = tolerance;
        return;
    }

    double SVM::kernel_rbf(std::vector<double> x1, std::vector<double> x2)
    {
        double sigma = 1;
        double xL2 = 0.0;
        xL2 = dotProduct(x1, x1) + dotProduct(x2, x2) - 2 * dotProduct(x1, x2);
        xL2 = xL2 / (-2 * sigma *sigma);
        return exp(xL2);
    }

    double SVM::kernel_laplace(std::vector<double> x1, std::vector<double> x2)
    {
        double sigma = 1;
        double xL2 = 0.0;
        xL2 = dotProduct(x1, x1) + dotProduct(x2, x2) - 2 * dotProduct(x1, x2);
        xL2 = sqrt(xL2) / sigma * (-1);
        return exp(xL2);
    }

    double SVM::kernel_sigmoid(std::vector<double> x1, std::vector<double> x2)
    {
        double beta1 = 1;
        double theta = -1;
        return tanh(beta1 * dotProduct(x1, x2) + theta);
    }

    double SVM::kernel_polynomial(std::vector<double> x1, std::vector<double> x2)
    {
        double d = 1.0;
        double p = 100;
        return pow(dotProduct(x1, x2) + d, p);	
    }

    double SVM::kernel(std::vector<double> x1, std::vector<double> x2, int kernelType)
    {
        double innerProduct = 0;
        switch (kernelType) {
            case KERNEL_RBF:
                innerProduct = kernel_rbf(x1, x2);
                break;
            case KERNEL_LAPLACE:
                innerProduct = kernel_laplace(x1, x2);
                break;
            case KERNEL_SIGMOID:
                innerProduct = kernel_sigmoid(x1, x2);
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

    double SVM::dotProduct(std::vector<double> x1, std::vector<double> x2)
    {
        double sum = 0.0;
        for (int i = 0; i < x1.size(); i++) {
            sum += x1[i] * x2[i];
        }
        return sum;
    }
}
