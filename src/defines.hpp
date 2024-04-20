#ifndef DEFINES_HPP
#define DEFINES_HPP

#include <Eigen/Dense>  // ���� Eigen ���ͷ�ļ�

#define BATCH_SIZE 5 // �����������СΪ 128,��ʱ�ĳ�16����
#define BITLEN 64  // ����λ����Ϊ 64
#define LEARNING_RATE_INV 256  // ����ѧϰ�ʵĵ�����1/LEARNING_RATE��
#define DEBUG 1  // �������ģʽ���أ�1 Ϊ������0 Ϊ�رգ�

#define CAROL 2 //�ڲ��Խ׶���carol��Ϊ�ն�2������

#define SCALING_FACTOR 8192  // �������ӣ��������ݵ����ţ�����Ϊ 13 λ��
#define HALF_OF_ONE 4096

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
    RowMatrixXi64 Ai;  // ������� Ai ����Ai �Ĵ�СΪn*d�ľ��󣬼�U
    ColMatrixXi64 Bi;  // ������� Bi ����Bi �Ĵ�СΪd*t�ľ�����Ϊ����ֵ��������Ϊ������������ѵ��,��V
    ColMatrixXi64 Ci;  // ������� Ci ����Ci �Ĵ�С��������ab�ĳ˻���Ϊ��������
    ColMatrixXi64 Bi_;  // ������� Bi' ����Bi_ �Ĵ�С����V'
    ColMatrixXi64 Ci_;  // ������� Ci' ����Ci_ �Ĵ�С��������UbT��V'�ĳ˻���Ϊ��������
};

#endif  // DEFINES_HPP
