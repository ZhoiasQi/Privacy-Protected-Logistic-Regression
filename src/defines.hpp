#ifndef DEFINES_HPP
#define DEFINES_HPP

#include <Eigen/Dense>  

#define BITLEN 64 

#define BATCH_SIZE 16
#define LEARNING_RATE_INV 32

#define BATCH_SIZE_I 64
#define LEARNING_RATE_INV_I 1024

#define DEBUG 0

#define CAROL 2 

#define B_TO_MB 1048576

#define SCALING_FACTOR 8192
#define HALF_OF_ONE 4096

#define A_L 81
#define B_H 85

extern int PARTY;  // 声明一个全局变量 PARTY（表示派对编号）

typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXi64;  
typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ColMatrixXi64;  
typedef Eigen::Matrix<uint64_t, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorXi64;  
typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, 1, Eigen::ColMajor> ColVectorXi64; 
typedef Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> SignedRowMatrixXi64;  
typedef Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> SignedColMatrixXi64;  
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXd;  
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ColMatrixXd;  
typedef Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorXd; 
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor> ColVectorXd; 

struct SetupTriples{
    RowMatrixXi64 Ai; 
    ColMatrixXi64 Bi;  
    ColMatrixXi64 Ci;  
    ColMatrixXi64 Bi_;  
    ColMatrixXi64 Ci_;  
};

struct Traffic{
    double offline;
    double online;
};

#endif  
