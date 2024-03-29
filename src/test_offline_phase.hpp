#ifndef TEST_OFFLINE_PHASE_HPP
#define TEST_OFFLINE_PHASE_HPP

#include <emp-ot/emp-ot.h>
#include "util.hpp"

using namespace emp;
using namespace std;

class TestSetUp{
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

    TestSetUp(int n, int d, int t, NetIO* io){
        this->n = n;
        this->d = d;
        this->t = 1;
        this->party = PARTY;
        this->io = io;
        this->send_ot = new SHOTExtension<NetIO>(io);
        this->recv_ot = new SHOTExtension<NetIO>(io);

        Ai.resize(n, d);  // 调整矩阵 Ai 的大小为n*d的矩阵，即U
        Bi.resize(d, t);  // 调整矩阵 Bi 的大小为d*t的矩阵，行为特征值个数，列为迭代次数用来训练,即V
        Ci.resize(n, t);  // 调整矩阵 Ci 的大小，用来存ab的乘积，为辅助矩阵
        Bi_.resize(n, t);  // 调整矩阵 Bi_ 的大小，即V'
        Ci_.resize(d, t);  // 调整矩阵 Ci_ 的大小，用来存UbT和V'的乘积，为辅助矩阵

        initialize_matrices();  // 初始化矩阵
        cout << "Matrices Initialized" << endl;  
    }

    void initialize_matrices();
    void generateMTs();
    void secure_mult(int N, int D, vector<vector<uint64_t>>& a, vector<uint64_t>& b, vector<uint64_t> &c);
    void getMTs(SetupTriples *triples);
};

#endif