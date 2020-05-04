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
                Mat(const Mat<T>& M);
                void createMat(int rows, int cols);
                /* init */
                void zero();
                void full(T x);
                void identity();
                void random(int n);
                void uniformRand();
                void copy(const Mat<T>& M);
                void copy(std::vector<T>& x);
                /* get element */
                T at(int row, int col);
                std::vector<T> column(int col);
                std::vector<T> convertToVector();
                /* row and column */
                void addRow(std::vector<T>& x);
                void addColumn(std::vector<T>& x);
                void delRow();
                void delColumn();
                /* basic operation */
                Mat<T> operator = (const Mat<T>& M);
                Mat<T> operator + (const Mat<T>& M);
                Mat<T> operator - (const Mat<T>& M);
                Mat<T> operator * (const Mat<T>& M);
                static Mat<T> multiply(const Mat<T>& M1, const Mat<T>& M2);
                Mat<T> operator + (T x);
                Mat<T> operator - (T x);
                Mat<T> operator * (T x);
                Mat<T> operator / (T x);
                T max(std::vector<T>& x);
                T min(std::vector<T>& x);
                T dotProduct(std::vector<T>& x1, std::vector<T>& x2);
                /* tranformation */
                Mat<T> transpose();
                Mat<T> inverse();
                Mat<T> diagonalize();
                Mat<T> othogonalize();
                /* attribute */
                bool semiPostive();
                int rank();
                bool independent(std::vector<T>& x1, std::vector<T>& x2);
                /* show */
                void show();
                /* read and write */
                void save(const std::string& fileName);
                void load(const std::string& fileName);
                int rows;
                int cols;
                std::vector<std::vector<T> > Mt;
        };

    template<typename T>
        Mat<T>::Mat(int rows, int cols)
        {
            this->rows = rows;
            this->cols = cols;
            this->Mt = std::vector<std::vector<T> >(this->rows);
            for (int i = 0; i < rows; i++) {
                Mt[i] = std::vector<T>(this->cols, 0);
            }
        }

    template<typename T>
        Mat<T>::Mat(const Mat<T>& M)
        {
            this->rows = M.rows;
            this->cols = M.cols;
            this->Mt = std::vector<std::vector<T> >(this->rows);
            for (int i = 0; i < rows; i++) {
                Mt[i] = std::vector<T>(this->cols, 0);
            }
            copy(M);
        }

    template<typename T>
        void Mat<T>::createMat(int rows, int cols)
        {
            this->rows = rows;
            this->cols = cols;
            this->Mt = std::vector<std::vector<T> >(this->rows);
            for (int i = 0; i < rows; i++) {
                Mt[i] = std::vector<T>(this->cols, 0);
            }
            return;
        }

    /* init */
    template<typename T>
        void Mat<T>::zero()
        {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Mt[i][j] = 0;
                }
            }
            return;
        }

    template<typename T>
        void Mat<T>::full(T x)
        {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Mt[i][j] = x;
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
                        Mt[i][j] = 1;
                    } else {
                        Mt[i][j] = 0;
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
                    Mt[i][j] = T(rand() % n);
                }
            }
            return;
        }

    template<typename T>
        void Mat<T>::uniformRand()
        {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Mt[i][j] = T(rand() % 10000 - rand() % 10000) / 10000;
                }
            }
            return;
        }

    template<typename T>
        void Mat<T>::copy(const Mat<T>& M)
        {
            if (M.rows != rows || M.cols != cols) {
                return;
            }
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Mt[i][j] = M.Mt[i][j];
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
                    Mt[i][j] = x[k];
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
            return Mt[row][col];
        }

    template<typename T>
        std::vector<T> Mat<T>::column(int col)
        {
            std::vector<T> columnT;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (j == col) {
                        columnT.push_back(Mt[i][j]);
                    }
                }
            }
            return columnT;
        }

    template<typename T>
        std::vector<T> Mat<T>::convertToVector()
        {
            std::vector<T> vt;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    vt.push_back(Mt[i][j]);
                }
            }
            return vt;
        }

    /* row and column */
    template<typename T>
        void Mat<T>::addRow(std::vector<T>& x)
        {
            if (x.size() != cols) {
                return;
            }
            Mt.push_back(x);
            return;
        }

    template<typename T>
        void Mat<T>::addColumn(std::vector<T>& x)
        {
            if (x.size() != rows) {
                return;
            }
            for (int i = 0; i < rows; i++) {
                Mt[i].push_back(x[i]);
            }
            return;
        }

    template<typename T>
        void Mat<T>::delRow()
        {
            Mt.pop_back();
            return;
        }

    template<typename T>
        void Mat<T>::delColumn()
        {
            for (int i = 0; i < rows; i++) {
                Mt[i].pop_back();
            }
            return;
        }

    /* show */
    template<typename T>
        void Mat<T>::show()
        {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    std::cout<<Mt[i][j]<<" ";
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
                    file << Mt[i][j]<<" ";
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
                    file >> Mt[i][j];
                }
            }
            return;
        }

    /* basic operation */
    template<typename T>
        Mat<T> Mat<T>::operator = (const Mat<T>& M)
        {
            /* make sure the copy constructor is completed */
            if (this == &M) {
                return *this;
            }
            if (rows == 0 || cols == 0) {
                this->createMat(M.rows, M.cols);
            }
            if (M.rows != rows || M.cols != cols) {
                std::cout<<"size not equal"<<std::endl;
                return *this;
            }
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Mt[i][j] = M.Mt[i][j];
                }
            }
            return *this;
        }

    template<typename T>
        Mat<T> Mat<T>::operator + (const Mat<T>& M)
        {
            if ((rows != M.rows) || (cols != M.cols)) {
                std::cout<<"size not equal"<<std::endl;
                return *this;
            }
            Mat<T> Mp(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Mp.Mt[i][j] = Mt[i][j] + M.Mt[i][j];
                }
            }
            return Mp;
        }

    template<typename T>
        Mat<T> Mat<T>::operator - (const Mat<T>& M)
        {
            if (rows != M.rows || (cols != M.cols)) {
                return *this;
            }
            Mat<T> Mp(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Mp.Mt[i][j] = Mt[i][j] - M.Mt[i][j];
                }
            }
            return Mp;
        }

    template<typename T>
        Mat<T> Mat<T>::operator * (const Mat<T>& M)
        {
            if (cols != M.rows) {
                return *this;
            }
            /* (m, p) x (p, n) = (m, n) */
            int m = rows;
            int p = cols;
            int n = M.cols;
            Mat<T> Mp(m, n);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < p; k++) {
                        Mp.Mt[i][j] += Mt[i][k] * M.Mt[k][j];
                    }
                }
            }
            return Mp;
        }

    template<typename T>
        Mat<T> Mat<T>::multiply(const Mat<T>& M1, const Mat<T>& M2)
        {
            if (M1.rows != M2.rows || (M1.cols != M2.cols)) {
                Mat<T> nullMat;
                return nullMat;
            }
            Mat<T> Mp(M1.rows, M1.cols);
            for (int i = 0; i < Mp.rows; i++) {
                for (int j = 0; j < Mp.cols; j++) {
                    Mp.Mt[i][j] = M1.Mt[i][j] * M2.Mt[i][j];
                }
            }
            return Mp;
        }

    template<typename T>
        Mat<T> Mat<T>::operator + (T x)
        {
            Mat<T> Mp(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Mp.Mt[i][j] = Mt[i][j] + x;
                }
            }
            return Mp;
        }

    template<typename T>
        Mat<T> Mat<T>::operator - (T x)
        {
            Mat<T> Mp(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Mp.Mt[i][j] = Mt[i][j] - x;
                }
            }
            return Mp;
        }

    template<typename T>
        Mat<T> Mat<T>::operator * (T x)
        {
            Mat<T> Mp(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Mp.Mt[i][j] = Mt[i][j] * x;
                }
            }
            return Mp;
        }

    template<typename T>
        Mat<T> Mat<T>::operator / (T x)
        {
            Mat<T> Mp(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Mp.Mt[i][j] = Mt[i][j] / x;
                }
            }
            return Mp;
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
        Mat<T> Mat<T>::transpose()
        {
            Mat<T> M;
            if (rows == cols) {
                M.createMat(rows,cols);
            } else {
                M.createMat(cols,rows);
            }
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    M.Mt[j][i] = Mt[i][j];
                }
            }
            return M;
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

    /* attribute */
    template<typename T>
        bool Mat<T>::semiPostive()
        {
            return true;
        }

    template<typename T>
        int Mat<T>::rank()
        {
            return 0;
        }

    template<typename T>
        bool Mat<T>::independent(std::vector<T>& x1, std::vector<T>& x2)
        {
            return true;
        }
}
#endif // MATRIX_H
