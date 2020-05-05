#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <memory>
namespace ML {
    template<typename T>
        class Mat {
            public:
                Mat():rows(0), cols(0){}
                ~Mat(){}
                /* create */
                Mat(int rows, int cols);
                Mat(const Mat<T>& X);
                void createMat(int rows, int cols);
                /* init */
                void zero();
                void full(T x);
                void identity();
                void random(int n);
                void uniformRandom();
                void copy(const Mat<T>& X);
                void copy(std::vector<T>& x);
                /* get element */
                T at(int row, int col);
                std::vector<T> column(int col);
                std::vector<T> convertToVector();
                /* basic operation */
                Mat<T> operator = (const Mat<T>& X);
                Mat<T> operator + (const Mat<T>& X);
                Mat<T> operator - (const Mat<T>& X);
                Mat<T> operator * (const Mat<T>& X);
                Mat<T> operator / (const Mat<T>& X);
                Mat<T> operator % (const Mat<T>& X);
                Mat<T>& operator += (const Mat<T>& X);
                Mat<T>& operator -= (const Mat<T>& X);
                Mat<T>& operator /= (const Mat<T>& X);
                Mat<T>& operator %= (const Mat<T>& X);
                Mat<T> operator + (T x);
                Mat<T> operator - (T x);
                Mat<T> operator * (T x);
                Mat<T> operator / (T x);
                Mat<T>& operator += (T x);
                Mat<T>& operator -= (T x);
                Mat<T>& operator *= (T x);
                Mat<T>& operator /= (T x);
                T max(std::vector<T>& x);
                T min(std::vector<T>& x);
                T dotProduct(std::vector<T>& x1, std::vector<T>& x2);
                Mat<T>& mappingTo(double (*func)(double x));
                /* tranformation */
                Mat<T> Tr();
                Mat<T> inverse();
                Mat<T> diagonalize();
                Mat<T> othogonalize();
                /* show */
                void show();
                /* read and write */
                void save(const std::string& fileName);
                void load(const std::string& fileName);
                int rows;
                int cols;
                std::vector<std::vector<T> > mat;
        };

    template<typename T>
        Mat<T>::Mat(int rows, int cols)
        {
            createMat(rows, cols);
        }

    template<typename T>
        Mat<T>::Mat(const Mat<T>& X)
        {
            createMat(X.rows, X.cols);
            copy(X);
        }

    template<typename T>
        void Mat<T>::createMat(int rows, int cols)
        {
            this->rows = rows;
            this->cols = cols;
            this->mat = std::vector<std::vector<T> >(this->rows);
            for (int i = 0; i < rows; i++) {
                mat[i] = std::vector<T>(this->cols, 0);
            }
            return;
        }

    /* init */
    template<typename T>
        void Mat<T>::zero()
        {
            full(0);
            return;
        }

    template<typename T>
        void Mat<T>::full(T x)
        {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mat[i][j] = x;
                }
            }
            return;
        }

    template<typename T>
        void Mat<T>::identity()
        {
            if (rows != cols) {
                return;
            }
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (i == j) {
                        mat[i][j] = 1;
                    } else {
                        mat[i][j] = 0;
                    }
                }
            }
            return;
        }

    template<typename T>
        void Mat<T>::random(int n)
        {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mat[i][j] = T(rand() % n);
                }
            }
            return;
        }

    template<typename T>
        void Mat<T>::uniformRandom()
        {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mat[i][j] = T(rand() % 10000 - rand() % 10000) / 10000;
                }
            }
            return;
        }

    template<typename T>
        void Mat<T>::copy(const Mat<T>& X)
        {
            if (X.rows != rows || X.cols != cols) {
                return;
            }
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mat[i][j] = X.mat[i][j];
                }
            }
            return;
        }

    template<typename T>
        void Mat<T>::copy(std::vector<T>& x)
        {
            if (x.size() < rows * cols) {
                return;
            }
            int k = 0;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mat[i][j] = x[k];
                    k++;
                }
            }
            return;
        }

    /* get element */
    template<typename T>
        T Mat<T>::at(int row, int col)
        {
            if (row >= rows || col >= cols || row < 0 || cols < 0) {
                return -10000;
            }
            return mat[row][col];
        }

    template<typename T>
        std::vector<T> Mat<T>::column(int col)
        {
            std::vector<T> columnT;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (j == col) {
                        columnT.push_back(mat[i][j]);
                    }
                }
            }
            return columnT;
        }

    template<typename T>
        std::vector<T> Mat<T>::convertToVector()
        {
            std::vector<T> x;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    x.push_back(mat[i][j]);
                }
            }
            return x;
        }

    /* show */
    template<typename T>
        void Mat<T>::show()
        {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    std::cout<<mat[i][j]<<" ";
                }
                std::cout<<std::endl;
            }
            return;
        }

    /* read and write */
    template<typename T>
        void Mat<T>::save(const std::string& fileName)
        {
            std::ofstream file;
            file.open(fileName);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    file << mat[i][j]<<" ";
                }
                file << std::endl;
            }
            return;
        }

    template<typename T>
        void Mat<T>::load(const std::string& fileName)
        {
            std::ifstream file;
            file.open(fileName);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    file >> mat[i][j];
                }
            }
            return;
        }

    /* basic operation */
    template<typename T>
        Mat<T> Mat<T>::operator = (const Mat<T>& X)
        {
            /* make sure the copy constructor is completed */
            if (this == &X) {
                return *this;
            }
            if (rows == 0 || cols == 0) {
                this->createMat(X.rows, X.cols);
            }
            if (X.rows != rows || X.cols != cols) {
                std::cout<<"size not equal"<<std::endl;
                return *this;
            }
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mat[i][j] = X.mat[i][j];
                }
            }
            return *this;
        }

    template<typename T>
        Mat<T> Mat<T>::operator + (const Mat<T>& X)
        {
            if ((rows != X.rows) || (cols != X.cols)) {
                std::cout<<"size not equal"<<std::endl;
                return *this;
            }
            Mat<T> Y(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Y.mat[i][j] = mat[i][j] + X.mat[i][j];
                }
            }
            return Y;
        }

    template<typename T>
        Mat<T> Mat<T>::operator - (const Mat<T>& X)
        {
            if (rows != X.rows || (cols != X.cols)) {
                return *this;
            }
            Mat<T> Y(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Y.mat[i][j] = mat[i][j] - X.mat[i][j];
                }
            }
            return Y;
        }

    template<typename T>
        Mat<T> Mat<T>::operator * (const Mat<T>& X)
        {
            if (cols != X.rows) {
                return *this;
            }
            /* (m, p) x (p, n) = (m, n) */
            int m = rows;
            int p = cols;
            int n = X.cols;
            Mat<T> Y(m, n);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < p; k++) {
                        Y.mat[i][j] += mat[i][k] * X.mat[k][j];
                    }
                }
            }
            return Y;
        }

    template<typename T>
        Mat<T> Mat<T>::operator / (const Mat<T>& X)
        {
            if (rows != X.rows || (cols != X.cols)) {
                return *this;
            }
            Mat<T> Y(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (X.mat[i][j] == 0) {
                        return *this;
                    } else {
                        Y.mat[i][j] = mat[i][j] / X.mat[i][j];
                    }
                }
            }
            return Y;
        }

    template<typename T>
        Mat<T> Mat<T>::operator % (const Mat<T>& X)
        {
            if (rows != X.rows || (cols != X.cols)) {
                Mat<T> nullMat;
                return nullMat;
            }
            Mat<T> Y(rows, cols);
            for (int i = 0; i < Y.rows; i++) {
                for (int j = 0; j < Y.cols; j++) {
                    Y.mat[i][j] = mat[i][j] * X.mat[i][j];
                }
            }
            return Y;
        }

    template<typename T>
        Mat<T>& Mat<T>::operator += (const Mat<T>& X)
        {
            if (rows != X.rows || (cols != X.cols)) {
                return *this;
            }
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mat[i][j] += X.mat[i][j];
                }
            }
            return *this;
        }

    template<typename T>
        Mat<T>& Mat<T>::operator -= (const Mat<T>& X)
        {
            if (rows != X.rows || (cols != X.cols)) {
                return *this;
            }
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mat[i][j] -= X.mat[i][j];
                }
            }
            return *this;
        }

    template<typename T>
        Mat<T>& Mat<T>::operator /= (const Mat<T>& X)
        {
            if (rows != X.rows || (cols != X.cols)) {
                return *this;
            }
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mat[i][j] /= X.mat[i][j];
                }
            }
            return *this;
        }

    template<typename T>
        Mat<T>& Mat<T>::operator %= (const Mat<T>& X)
        {
            if (rows != X.rows || (cols != X.cols)) {
                return *this;
            }
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mat[i][j] *= X.mat[i][j];
                }
            }
            return *this;
        }

    template<typename T>
        Mat<T> Mat<T>::operator + (T x)
        {
            Mat<T> Y(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Y.mat[i][j] = mat[i][j] + x;
                }
            }
            return Y;
        }

    template<typename T>
        Mat<T> Mat<T>::operator - (T x)
        {
            Mat<T> Y(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Y.mat[i][j] = mat[i][j] - x;
                }
            }
            return Y;
        }

    template<typename T>
        Mat<T> Mat<T>::operator * (T x)
        {
            Mat<T> Y(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Y.mat[i][j] = mat[i][j] * x;
                }
            }
            return Y;
        }

    template<typename T>
        Mat<T> Mat<T>::operator / (T x)
        {
            Mat<T> Y(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Y.mat[i][j] = mat[i][j] / x;
                }
            }
            return Y;
        }

    template<typename T>
        Mat<T>& Mat<T>::operator += (T x)
        {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mat[i][j] += x;
                }
            }
            return *this;
        }

    template<typename T>
        Mat<T>& Mat<T>::operator -= (T x)
        {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mat[i][j] -= x;
                }
            }
            return *this;
        }

    template<typename T>
        Mat<T>& Mat<T>::operator *= (T x)
        {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mat[i][j] *= x;
                }
            }
            return *this;
        }

    template<typename T>
        Mat<T>& Mat<T>::operator /= (T x)
        {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mat[i][j] /= x;
                }
            }
            return *this;
        }

    template<typename T>
        T Mat<T>::max(std::vector<T>& x)
        {
            T maxT = x[0];
            for (int i = 0; i < x.size(); i++) {
                if (maxT < x[i]) {
                    maxT = x[i];
                }
            }
            return maxT;
        }

    template<typename T>
        T Mat<T>::min(std::vector<T>& x)
        {
            T minT = x[0];
            for (int i = 0; i < x.size(); i++) {
                if (minT > x[i]) {
                    minT = x[i];
                }
            }
            return minT;
        }

    template<typename T>
        T Mat<T>::dotProduct(std::vector<T>& x1, std::vector<T>& x2)
        {
            T s = 0;
            for (int i = 0; i < x1.size(); i++) {
                s += x1[i] * x2[i];
            }
            return s;
        }

    /* tranformat<T>ion */
    template<typename T>
        Mat<T> Mat<T>::Tr()
        {
            Mat<T> Y;
            Y.createMat(cols,rows);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Y.mat[j][i] = mat[i][j];
                }
            }
            return Y;
        }

    template<typename T>
        Mat<T> Mat<T>::inverse()
        {
            Mat<T> M;
            return M;
        }

    template<typename T>
        Mat<T> Mat<T>::diagonalize()
        {
            Mat<T> M;
            return M;
        }

    template<typename T>
        Mat<T> Mat<T>::othogonalize()
        {
            Mat<T> M;
            return M;
        }

    template<typename T>
        Mat<T>& Mat<T>::mappingTo(double (*func)(double x))
        {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mat[i][j] = func(mat[i][j]);
                }
            }
            return *this;
        }
}
#endif // MATRIX_H
