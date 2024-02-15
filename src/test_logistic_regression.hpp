#ifndef T_LOGISTIC_REGRESSION_H
#define T_LOGISTIC_REGRESSION_H

#include "test_offline_phase.hpp"
#include "test_online_phase.hpp"
#include <iostream>

using namespace std;

class TestLogisticRegression{
public:
    emp::NetIO* io;  // 网络输入输出对象指针
    int party;  // 当前参与方（ALICE或BOB）
    int n, d, t;  // 训练数据大小、特征数、迭代次数
    RowMatrixXi64 X;  // 训练数据矩阵
    ColVectorXi64 Y;  // 训练标签向量
    ColVectorXi64 w;  // 权重向量（整数类型）
    ColVectorXi64 wi;
    ColVectorXi64 prediction;
    TestSetUp* setup;
    TestOnlinePhase* online;  // 在线阶段对象指针

    //测试阶段构造函数
    TestLogisticRegression(RowMatrixXi64& testing_data, ColVectorXi64& testing_labels, TestingParams params, emp::NetIO* io) {
        this->n = params.n;
        this->d = params.d;
        this->t = 1;
        this->X = testing_data;
        this->Y = testing_labels;
        this->io = io;
        this->party = PARTY;
        this->w.resize(d);
        this->wi.resize(d);
        this->prediction.resize(d);
        
        this->setup = new TestSetUp(n, d, t, io);
        setup->generateMTs();

        SetupTriples triples;
        setup->getMTs(&triples);

        RowMatrixXi64 Xi(X.rows(), X.cols());  // 初始化存储加密数据的矩阵
        ColVectorXi64 Yi(Y.rows(), Y.cols());  // 初始化存储加密标签的向量

        if (party == CAROL) {  // 如果当前参与方是CAROL
            emp::PRG prg;  // 伪随机数生成器对象
            RowMatrixXi64 rX(X.rows(), X.cols());  // 随机数据矩阵
            ColVectorXi64 rY(Y.rows(), Y.cols());  // 随机标签向量
            prg.random_data(rX.data(), X.rows() * X.cols() * sizeof(uint64_t));  // 生成随机数据 rX
            prg.random_data(rY.data(), Y.rows() * Y.cols() * sizeof(uint64_t));  // 生成随机数据 rY
            Xi = X + rX;  // 加密后的训练数据
            Yi = Y + rY;  // 加密后的训练标签
            rX *= -1;  // 对 rX 中的元素取反
            rY *= -1;  // 对 rY 中的元素取反
            send<RowMatrixXi64>(io, rX);  // 发送随机数据 rX
            send<ColVectorXi64>(io, rY);  // 发送随机数据 rY

            cout << "Carol has secretly sent the data to Alice" << endl;
        } else {  // 如果当前参与方是Alice
            recv<RowMatrixXi64>(io, Xi);  // 从对方接收加密后的训练数据
            recv<ColVectorXi64>(io, Yi);  // 从对方接收加密后的训练标签

            cout << "Alice has received the secret data from Carol" << endl;

        }
    }

    void getW(ColVectorXi64 w);

    void secret_share_w();

    void test_model();
};

#endif

