#ifndef SETUP_HPP
#define SETUP_HPP

#include "util.hpp"  // 包含常用的工具函数
#include <emp-ot/emp-ot.h>  // 包含emp-ot库

/**
 * SetupPhase 类用于执行协议的设置阶段操作。
 */
class SetupPhase{
public:
    int n, d, t;  // 数据行数，数据列数，以及 Bi 和 Ci 的矩阵维度
    int party;  // 当前参与方（ALICE或BOB）
    RowMatrixXi64 Ai;  // 矩阵 Ai
    ColMatrixXi64 Bi;  // 矩阵 Bi
    ColMatrixXi64 Ci;  // 矩阵 Ci
    ColMatrixXi64 Bi_;  // 矩阵 Bi_
    ColMatrixXi64 Ci_;  // 矩阵 Ci_

    emp::NetIO* io;  // 网络 I/O 对象指针
    emp::SHOTExtension<emp::NetIO>* send_ot;  // OT 发送端对象指针
    emp::SHOTExtension<emp::NetIO>* recv_ot;  // OT 接收端对象指针
    emp::PRG prg;  // 伪随机数生成器对象

    /**
     * 构造函数 SetupPhase 用于初始化对象。
     * @param n 数据行数
     * @param d 数据列数
     * @param t 矩阵 Bi 和 Ci 的维度
     * @param io 网络 I/O 对象指针
     */
    SetupPhase(int n, int d, int t, emp::NetIO* io){
        this->n = n;
        this->d = d;
        this->t = t;
        this->io = io;
        this->send_ot = new emp::SHOTExtension<emp::NetIO>(io);  // 初始化 OT 发送端对象
        this->recv_ot = new emp::SHOTExtension<emp::NetIO>(io);  // 初始化 OT 接收端对象
        this->party = PARTY;  // 设置当前参与方

        Ai.resize(n, d);  // 调整矩阵 Ai 的大小
        Bi.resize(d, t);  // 调整矩阵 Bi 的大小
        Ci.resize(BATCH_SIZE, t);  // 调整矩阵 Ci 的大小
        Bi_.resize(BATCH_SIZE, t);  // 调整矩阵 Bi_ 的大小
        Ci_.resize(d, t);  // 调整矩阵 Ci_ 的大小

        initialize_matrices();  // 初始化矩阵
        std::cout << "Matrices Initialized" << std::endl;  // 输出消息：矩阵已初始化
    }

    void initialize_matrices();  // 初始化矩阵的方法
    void generateMTs();  // 生成 MTs 的方法
    void secure_mult(int N, int D, vector<vector<uint64_t>>& a,
                     vector<uint64_t>& b, vector<uint64_t> &c);  // 安全乘法的方法
    void getMTs(SetupTriples* triples);  // 获取 MTs 的方法
    void verify();  // 验证函数
};

#endif // SETUP_HPP
