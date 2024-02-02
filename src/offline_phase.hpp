#ifndef OFFLINE_PHASE_H
#define OFFLINE_PHASE_H

#include <emp-ot/emp-ot.h>
#include "util.hpp"

using namespace emp;
using namespace std;

class OfflineSetUp{
public:
    int n, d, t;
    int party;
    RowMatrixXi64 Ai;  // 矩阵 Ai
    ColMatrixXi64 Bi;  // 矩阵 Bi
    ColMatrixXi64 Ci;  // 矩阵 Ci
    ColMatrixXi64 Bi_;  // 矩阵 Bi_
    ColMatrixXi64 Ci_;  // 矩阵 Ci_

    NetIO* io;  // 网络 I/O 对象指针
    SHOTExtension<emp::NetIO>* send_ot;  // OT 发送端对象指针
    SHOTExtension<emp::NetIO>* recv_ot;  // OT 接收端对象指针
    PRG prg;  // 伪随机数生成器对象

    OfflineSetUp(int n, int d, int t, NetIO* io){
        this->n = n;
        this->d = d;
        this->t = t;
        this->party = PARTY;
        this->io = io;
        this->send_ot = new SHOTExtension<NetIO>(io);
        this->recv_ot = new SHOTExtension<NetIO>(io);

        Ai.resize(n, d);  // 调整矩阵 Ai 的大小为n*d的矩阵，即和训练集一样
        Bi.resize(d, t);  // 调整矩阵 Bi 的大小为d*t的矩阵，行为特征值个数，列为迭代次数用来训练
        Ci.resize(BATCH_SIZE, t);  // 调整矩阵 Ci 的大小
        Bi_.resize(BATCH_SIZE, t);  // 调整矩阵 Bi_ 的大小
        Ci_.resize(d, t);  // 调整矩阵 Ci_ 的大小

        initialize_matrices();  // 初始化矩阵
        cout << "Matrices Initialized" << endl;  
    }

    void initialize_matrices();
    void generateMTs();
    void secure_mult(int N, int D, vector<vector<uint64_t>>& a, vector<uint64_t>& b, vector<uint64_t> &c);
    void getMTs(SetupTriples *triples);
    void verify();
};

#endif