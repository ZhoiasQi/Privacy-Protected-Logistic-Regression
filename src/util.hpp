#ifndef UTIL_HPP
#define UTIL_HPP

#include "defines.hpp" // 包含自定义的定义文件
#include <iostream> // 输入输出流
#include <vector> // 向量容器
#include <emp-tool/emp-tool.h> // 引入 emp-tool 库

void vector2d_to_RowMatrixXd(std::vector<std::vector<double>>& x, RowMatrixXd& X); // 将二维向量转换为行矩阵
void vector_to_ColVectorXd(std::vector<double>& x, ColVectorXd& X); // 将一维向量转换为列向量
void vector_to_RowVectorXi64(std::vector<uint64_t>& x, RowVectorXi64& X); // 将一维无符号整数向量转换为行整数向量
void vector2d_to_RowMatrixXi64(std::vector<std::vector<uint64_t>>& x, RowMatrixXi64& X); // 将二维无符号整数向量转换为行整数矩阵
void vector2d_to_ColMatrixXi64(std::vector<std::vector<uint64_t>>& x, ColMatrixXi64& X); // 将二维无符号整数向量转换为列整数矩阵
void vector_to_ColVectorXi64(std::vector<uint64_t>& x, ColVectorXi64& X); // 将一维无符号整数向量转换为列整数向量
void RowMatrixXi64_to_vector2d(RowMatrixXi64 X, std::vector<std::vector<uint64_t>>& x); // 将行整数矩阵转换回二维无符号整数向量
std::vector<uint64_t> ColVectorXi64_to_vector(ColVectorXi64 X); // 将列整数向量转换为一维无符号整数向量

void print128_num(emp::block var); // 打印 emp::block 类型的变量
void print_binary(uint64_t int_); // 打印十进制整数的二进制表示
void int_to_bool(bool* bool_, uint64_t int_); // 将十进制整数转换为布尔数组

uint64_t extract_lo64(__m128i x); // 提取 __m128i 类型变量的低64位整数
uint64_t extract_hi64(__m128i x); // 提取 __m128i 类型变量的高64位整数

template<class Derived>
void send(emp::NetIO* io, Eigen::PlainObjectBase<Derived>& X) {
    io->send_data(X.data(), X.rows() * X.cols() * sizeof(uint64_t)); // 使用 emp::NetIO 发送行主序的 Eigen 对象
    return;
}

template<class Derived>
void recv(emp::NetIO* io, Eigen::PlainObjectBase<Derived>& X) {
    io->recv_data(X.data(), X.rows() * X.cols() * sizeof(uint64_t)); // 使用 emp::NetIO 接收行主序的 Eigen 对象
    return;
}

#endif // UTIL_HPP
