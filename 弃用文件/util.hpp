#ifndef UTIL_HPP
#define UTIL_HPP

#include "defines.hpp" // �����Զ���Ķ����ļ�
#include <iostream> // ���������
#include <vector> // ��������
#include <emp-tool/emp-tool.h> // ���� emp-tool ��

void vector2d_to_RowMatrixXd(std::vector<std::vector<double>>& x, RowMatrixXd& X); // ����ά����ת��Ϊ�о���
void vector_to_ColVectorXd(std::vector<double>& x, ColVectorXd& X); // ��һά����ת��Ϊ������
void vector_to_RowVectorXi64(std::vector<uint64_t>& x, RowVectorXi64& X); // ��һά�޷�����������ת��Ϊ����������
void vector2d_to_RowMatrixXi64(std::vector<std::vector<uint64_t>>& x, RowMatrixXi64& X); // ����ά�޷�����������ת��Ϊ����������
void vector2d_to_ColMatrixXi64(std::vector<std::vector<uint64_t>>& x, ColMatrixXi64& X); // ����ά�޷�����������ת��Ϊ����������
void vector_to_ColVectorXi64(std::vector<uint64_t>& x, ColVectorXi64& X); // ��һά�޷�����������ת��Ϊ����������
void RowMatrixXi64_to_vector2d(RowMatrixXi64 X, std::vector<std::vector<uint64_t>>& x); // ������������ת���ض�ά�޷�����������
std::vector<uint64_t> ColVectorXi64_to_vector(ColVectorXi64 X); // ������������ת��Ϊһά�޷�����������

void print128_num(emp::block var); // ��ӡ emp::block ���͵ı���
void print_binary(uint64_t int_); // ��ӡʮ���������Ķ����Ʊ�ʾ
void int_to_bool(bool* bool_, uint64_t int_); // ��ʮ��������ת��Ϊ��������

uint64_t extract_lo64(__m128i x); // ��ȡ __m128i ���ͱ����ĵ�64λ����
uint64_t extract_hi64(__m128i x); // ��ȡ __m128i ���ͱ����ĸ�64λ����

template<class Derived>
void send(emp::NetIO* io, Eigen::PlainObjectBase<Derived>& X) {
    io->send_data(X.data(), X.rows() * X.cols() * sizeof(uint64_t)); // ʹ�� emp::NetIO ����������� Eigen ����
    return;
}

template<class Derived>
void recv(emp::NetIO* io, Eigen::PlainObjectBase<Derived>& X) {
    io->recv_data(X.data(), X.rows() * X.cols() * sizeof(uint64_t)); // ʹ�� emp::NetIO ����������� Eigen ����
    return;
}

#endif // UTIL_HPP
