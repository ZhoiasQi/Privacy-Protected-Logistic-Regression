#include "util.hpp"

using namespace std;
using namespace Eigen;
using namespace emp;
using Eigen::Matrix;

void vector2d_to_RowMatrixXd(vector<vector<double>>& x, RowMatrixXd& X){
    for(int i = 0; (unsigned int)i < x.size(); i++){
        X.row(i) << Map<RowVectorXd>(x[i].data(), x[i].size());
        // 将二维向量转换为行主序的 MatrixXd 对象的第 i 行
    }
    return;
}

void vector_to_ColVectorXd(vector<double>& x, ColVectorXd& X){
    X << Map<ColVectorXd>(x.data(), x.size());
    // 将一维向量转换为列主序的 VectorXd 对象
    return;
}

void vector_to_RowVectorXi64(vector<uint64_t>& x, RowVectorXi64& X){
    X << Map<RowVectorXi64>(x.data(), x.size());
    // 将一维无符号整数向量转换为行主序的 VectorXi64 对象
    return;
}

void vector2d_to_RowMatrixXi64(std::vector<std::vector<uint64_t>>& x, RowMatrixXi64& X){
    for(int i = 0; (unsigned int)i < x.size(); i++){
        X.row(i) << Map<RowVectorXi64>(x[i].data(), x[i].size());
        // 将二维无符号整数向量转换为行主序的 MatrixXi64 对象的第 i 行
    }
    return;
}

void vector2d_to_ColMatrixXi64(vector<vector<uint64_t>>& x, ColMatrixXi64& X){
    for(int i = 0; (unsigned int)i < x.size(); i++){
        X.col(i) << Map<ColVectorXi64>(x[i].data(), x[i].size());
        // 将二维无符号整数向量转换为列主序的 MatrixXi64 对象的第 i 列
    }
    return;
}

void vector_to_ColVectorXi64(vector<uint64_t>& x, ColVectorXi64& X){
    X << Map<ColVectorXi64>(x.data(), x.size());
    // 将一维无符号整数向量转换为列主序的 VectorXi64 对象
}

void RowMatrixXi64_to_vector2d(RowMatrixXi64 X, vector<vector<uint64_t>>& x){
    for(int i = 0; i < X.rows(); i++){
        for(int j = 0; j < X.cols(); j++){
            x[i][j] = X(i, j);
            // 将行主序的 MatrixXi64 对象转换回二维无符号整数向量
        }
    }
    return;
}

vector<uint64_t> ColVectorXi64_to_vector(ColVectorXi64 X){
    vector<uint64_t> x(X.data(), X.data() + X.rows());
    // 将列主序的 VectorXi64 对象转换为一维无符号整数向量
    return x;
}

void print128_num(emp::block var){
    uint64_t *v64val = (uint64_t*) &var;
    // 2024/1/18将两个llx改为lx
    printf("%016lX %016lX", v64val[1], v64val[0]);
    // 打印 emp::block 类型的变量的值
}

void print_binary(uint64_t int_) {
    for (int i = 0; i < 64; i++){
        cout << (int_ & 1);
        int_ >>= 1;
        // 打印十进制整数的二进制表示
    }
}

void int_to_bool(bool* bool_, uint64_t int_) {
    for (int i = 0; i < 64; i++){
        bool_[i] = (int_ & 1);
        int_ >>= 1;
        // 将十进制整数转换为布尔数组
    }
}

uint64_t extract_lo64(__m128i x){ // 提取 __m128i 类型变量的低64位整数
    return (uint64_t) _mm_cvtsi128_si64(x);
}

uint64_t extract_hi64(__m128i x){ // 提取 __m128i 类型变量的高64位整数
    uint64_t *v64val = (uint64_t*) &x;
    return (uint64_t) v64val[1];
}
