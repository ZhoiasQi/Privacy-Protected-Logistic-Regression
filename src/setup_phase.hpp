#ifndef SETUP_HPP
#define SETUP_HPP

#include "util.hpp"
#include <emp-ot/emp-ot.h>

/**
 * SetupPhase 类用于执行设置阶段的操作。
 */
class SetupPhase{
public:
    int n, d, t;
    int party;
    RowMatrixXi64 Ai; // 矩阵 Ai
    ColMatrixXi64 Bi; // 矩阵 Bi
    ColMatrixXi64 Ci; // 矩阵 Ci
    ColMatrixXi64 Bi_; // 矩阵 Bi_
    ColMatrixXi64 Ci_; // 矩阵 Ci_

    emp::NetIO* io; // 网络 I/O 对象
    emp::SHOTExtension<emp::NetIO>* send_ot; // OT 发送方对象
    emp::SHOTExtension<emp::NetIO>* recv_ot; // OT 接收方对象
    emp::PRG prg; // 伪随机生成器对象

    /**
     * SetupPhase 构造函数用于初始化对象。
     * @param n 数据行数
     * @param d 数据列数
     * @param t 矩阵 Bi、Ci 的列数
     * @param io 网络 I/O 对象指针
     */
    SetupPhase(int n, int d, int t, emp::NetIO* io){
        this->n = n;
        this->d = d;
        this->t = t;
        this->io = io;
        this->send_ot = new emp::SHOTExtension<emp::NetIO>(io); // 初始化 OT 发送方对象
        this->recv_ot = new emp::SHOTExtension<emp::NetIO>(io); // 初始化 OT 接收方对象
        this->party = PARTY; // 设置当前参与方

        Ai.resize(n, d); // 调整矩阵 Ai 大小
        Bi.resize(d, t); // 调整矩阵 Bi 大小
        Ci.resize(BATCH_SIZE, t); // 调整矩阵 Ci 大小
        Bi_.resize(BATCH_SIZE, t); // 调整矩阵 Bi_ 大小
        Ci_.resize(d, t); // 调整矩阵 Ci_ 大小

        initialize_matrices(); // 初始化矩阵
        std::cout << "Matrices Initialized" << std::endl;
    }

    void initialize_matrices(); // 初始化矩阵
    void generateMTs(); // 生成 MTs
    void secure_mult(int N, int D, vector<vector<uint64_t>>& a,
                     vector<uint64_t>& b, vector<uint64_t> &c); // 安全乘法
    void getMTs(SetupTriples* triples); // 获取 MTs
    void verify(); // 验证函数
};

#endif // SETUP_HPP
