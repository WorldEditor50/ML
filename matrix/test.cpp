#include <iostream>
#include <memory>
#include "matrix.hpp"
using namespace std;
using namespace ML;

void test_save()
{
    Mat<int> M(3, 4);
    M.random(10);
    M.show();
    M.save("./m1");
    return;
}
void test_load()
{
    Mat<int> M;
    M.createMat(3, 4);
    M.load("./m1");
    M.show();
    return;
}
void test_zero()
{
    Mat<int> M(4, 4);
    M.identity();
    M.show();
    M.zero();
    M.show();
    return;
}
void test_copy()
{
    Mat<int> M1;
    M1.createMat(3, 4);
    M1.load("./m1");
    Mat<int> M2(M1);
    M2.copy(M1);
    M2.show();
    cout<<M2.at(0, 0)<<endl;
    vector<int> col = M2.column(3);
    for (int i = 0; i < col.size(); i++) {
        cout<<col[i]<<" ";
    }
    cout<<endl;
    return;
}
void test_assign()
{
    Mat<int> M1(3, 4);
    M1.load("./m1");
    Mat<int> M2(3, 4);
    M2 = M1;
    M2.show();
    return;
}
void test_plusConst()
{
    Mat<int> M(3, 3);
    M.random(3);
    M.show();
    M = M + 3;
    M.show();
    Mat<int> M1(3, 3);
    M1 = M + 1;
    M1.show();
    M.show();
    Mat<int> M2(3, 3);
    M2 = M1 + 1 + 1;
    M2.show();
    return;
}
void test_minusConst()
{
    Mat<int> M(3, 3);
    M.random(3);
    M.show();
    M = M - 3;
    M.show();
    Mat<int> M1(3, 3);
    M1 = M - 1 - 1;
    M1.show();
    return;
}
void test_multiplyConst()
{
    Mat<int> M(3, 3);
    M.random(3);
    M.show();
    M = M * 2;
    M.show();
    Mat<int> M1(3, 3);
    M1 = M * 2 *2;
    M1.show();
    return;
}
void test_divideConst()
{
    Mat<float> M(3, 3);
    M.random(3);
    M.show();
    M = M / 2;
    M.show();
    Mat<float> M1(3, 3);
    M1 = M / 2 / 2;
    M1.show();
    return;
}
void test_plusMat()
{
    Mat<int> M1(3, 3);
    M1.random(3);
    M1.show();
    Mat<int> M2(3, 3);
    M2.random(3);
    M2.show();
    Mat<int> M3;
    M3 = M2 + M1;
    M3.show();
    Mat<int> I(3, 3);
    I.identity();
    Mat<int> M4 = M3 + I + I * 10;
    M4.show();
    return;
}
void test_minusMat()
{
    Mat<int> M1(3, 3);
    M1.random(3);
    M1.show();
    Mat<int> M2(3, 3);
    M2.random(3);
    M2.show();
    Mat<int> M3;
    M3 = M2 - M1;
    M3.show();
    Mat<int> I(3, 3);
    I.identity();
    Mat<int> M4 = M3 * 2 - I - I;
    M4.show();
    return;
}
void test_multiplyMat()
{
    Mat<int> M1(3, 3);
    M1.random(3);
    M1.show();
    Mat<int> M2(3, 3);
    M2.random(3);
    M2.show();
    Mat<int> M3;
    M3 = M2 * M1;
    M3.show();
    Mat<int> I(3, 3);
    I.identity();
    Mat<int> M4 = M3 * I * I * 2;
    M4.show();
    return;
}
void test_mixOperation()
{
    Mat<float> M1(3, 3);
    M1.random(3);
    M1.show();
    Mat<float> M2(3, 3);
    M2.random(3);
    M2.show();
    Mat<float> M3 = M2 * M1 + M2 * 2 - M1 / 3;
    M3.show();
    return;
}
void test_transpose()
{
    Mat<int> M1(3, 4);
    M1.random(9);
    M1.show();
    Mat<int> M2;
    M2 = M1.Tr();
    M2.show();
    return;
}
int main()
{
    srand((unsigned int)time(nullptr));
    test_mixOperation();
	return 0;
}
