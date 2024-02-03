#ifndef DEFINES_HPP
#define DEFINES_HPP

#include <Eigen/Dense>  // ���� Eigen ���ͷ�ļ�

#define BATCH_SIZE 128  // �����������СΪ 128
#define BITLEN 64  // ����λ����Ϊ 64
#define LEARNING_RATE_INV 128  // ����ѧϰ�ʵĵ�����1/LEARNING_RATE��
#define DEBUG 1  // �������ģʽ���أ�1 Ϊ������0 Ϊ�رգ�

#define SCALING_FACTOR 8192  // �������ӣ��������ݵ����ţ�����Ϊ 13 λ��

extern int PARTY;  // ����һ��ȫ�ֱ��� PARTY����ʾ�ɶԱ�ţ�

typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXi64;  // ����������� uint64_t ���͵ľ���
typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ColMatrixXi64;  // ����������� uint64_t ���͵ľ���
typedef Eigen::Matrix<uint64_t, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorXi64;  // ����������� uint64_t ���͵�����
typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, 1, Eigen::ColMajor> ColVectorXi64;  // ����������� uint64_t ���͵�����
typedef Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> SignedRowMatrixXi64;  // ����������� int64_t ���͵ľ���
typedef Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> SignedColMatrixXi64;  // ����������� int64_t ���͵ľ���
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXd;  // ����������� double ���͵ľ���
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ColMatrixXd;  // ����������� double ���͵ľ���
typedef Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorXd;  // ����������� double ���͵�����
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor> ColVectorXd;  // ����������� double ���͵�����

struct SetupTriples{
    RowMatrixXi64 Ai;  // ������� Ai ����
    ColMatrixXi64 Bi;  // ������� Bi ����
    ColMatrixXi64 Ci;  // ������� Ci ����
    ColMatrixXi64 Bi_;  // ������� Bi' ����
    ColMatrixXi64 Ci_;  // ������� Ci' ����
};

#endif  // DEFINES_HPP
