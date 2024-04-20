#ifndef DEFINES_HPP
#define DEFINES_HPP

#include <Eigen/Dense>  // 包含 Eigen 库的头文件

#define BATCH_SIZE 5 // 定义批处理大小为 128,暂时改成16试试
#define BITLEN 64  // 定义位长度为 64
#define LEARNING_RATE_INV 256  // 定义学习率的倒数（1/LEARNING_RATE）
#define DEBUG 1  // 定义调试模式开关（1 为开启，0 为关闭）

#define CAROL 2 //在测试阶段让carol作为终端2的主体

#define SCALING_FACTOR 8192  // 缩放因子，用于数据的缩放（精度为 13 位）
#define HALF_OF_ONE 4096

extern int PARTY;  // 声明一个全局变量 PARTY（表示派对编号）

typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXi64;  // 定义行主序的 uint64_t 类型的矩阵
typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ColMatrixXi64;  // 定义列主序的 uint64_t 类型的矩阵
typedef Eigen::Matrix<uint64_t, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorXi64;  // 定义行主序的 uint64_t 类型的向量
typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, 1, Eigen::ColMajor> ColVectorXi64;  // 定义列主序的 uint64_t 类型的向量
typedef Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> SignedRowMatrixXi64;  // 定义行主序的 int64_t 类型的矩阵
typedef Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> SignedColMatrixXi64;  // 定义列主序的 int64_t 类型的矩阵
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXd;  // 定义行主序的 double 类型的矩阵
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ColMatrixXd;  // 定义列主序的 double 类型的矩阵
typedef Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorXd;  // 定义行主序的 double 类型的向量
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor> ColVectorXd;  // 定义列主序的 double 类型的向量

struct SetupTriples{
    RowMatrixXi64 Ai;  // 行主序的 Ai 矩阵，Ai 的大小为n*d的矩阵，即U
    ColMatrixXi64 Bi;  // 列主序的 Bi 矩阵，Bi 的大小为d*t的矩阵，行为特征值个数，列为迭代次数用来训练,即V
    ColMatrixXi64 Ci;  // 列主序的 Ci 矩阵，Ci 的大小，用来存ab的乘积，为辅助矩阵
    ColMatrixXi64 Bi_;  // 列主序的 Bi' 矩阵，Bi_ 的大小，即V'
    ColMatrixXi64 Ci_;  // 列主序的 Ci' 矩阵，Ci_ 的大小，用来存UbT和V'的乘积，为辅助矩阵
};

#endif  // DEFINES_HPP
