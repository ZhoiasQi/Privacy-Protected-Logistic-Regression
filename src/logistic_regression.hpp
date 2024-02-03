#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "offline_phase.hpp"
#include "online_phase.hpp"
#include <iostream>

using namespace std;

class LogisticRegression{
public:
    emp::NetIO* io;  // 网络输入输出对象指针
    int party;  // 当前参与方（ALICE或BOB）
    int n, d, t;  // 训练数据大小、特征数、迭代次数
    RowMatrixXi64 X;  // 训练数据矩阵
    ColVectorXi64 Y;  // 训练标签向量
    ColVectorXi64 w;  // 权重向量（整数类型）
    ColVectorXd w_d;  // 权重向量（双精度浮点数类型）
    OfflineSetUp* setup;  // 设置阶段对象指针

    //OnlinePhase* online;  // 在线阶段对象指针

    LogisticRegression(RowMatrixXi64& training_data, ColVectorXi64& training_labels, TrainingParams params, emp::NetIO* io) {
        this->n = params.n;  // 设置训练数据大小
        this->d = params.d;  // 设置特征数
        this->t = (params.n) / BATCH_SIZE;  // 计算迭代次数
        this->X = training_data;  // 拷贝训练数据
        this->Y = training_labels;  // 拷贝训练标签
        this->io = io;  // 设置网络输入输出对象指针
        this->party = PARTY;  // 设置当前参与方（ALICE或BOB）
        this->w.resize(d);  // 调整权重向量大小
        this->w_d.resize(d);  // 调整权重向量大小

        this->setup = new OfflineSetUp(n, d, t, io);
        setup->generateMTs();  // 生成随机三元组

        SetupTriples triples;
        setup->getMTs(&triples);  // 获取随机三元组


        // RowMatrixXi64 Xi(X.rows(), X.cols());  // 初始化存储加密数据的矩阵
        // ColVectorXi64 Yi(Y.rows(), Y.cols());  // 初始化存储加密标签的向量
        
        // if (party == emp::ALICE) {  // 如果当前参与方是ALICE
        //     emp::PRG prg;  // 伪随机数生成器对象
        //     RowMatrixXi64 rX(X.rows(), X.cols());  // 随机数据矩阵
        //     ColVectorXi64 rY(Y.rows(), Y.cols());  // 随机标签向量
        //     prg.random_data(rX.data(), X.rows() * X.cols() * sizeof(uint64_t));  // 生成随机数据 rX
        //     prg.random_data(rY.data(), Y.rows() * Y.cols() * sizeof(uint64_t));  // 生成随机数据 rY
        //     Xi = X + rX;  // 加密后的训练数据
        //     Yi = Y + rY;  // 加密后的训练标签
        //     rX *= -1;  // 对 rX 中的元素取反
        //     rY *= -1;  // 对 rY 中的元素取反
        //     send<RowMatrixXi64>(io, rX);  // 发送随机数据 rX
        //     send<ColVectorXi64>(io, rY);  // 发送随机数据 rY
        // } else {  // 如果当前参与方是BOB
        //     recv<RowMatrixXi64>(io, Xi);  // 从对方接收加密后的训练数据
        //     recv<ColVectorXi64>(io, Yi);  // 从对方接收加密后的训练标签
        // }

        // this->online = new OnlinePhase(params, io, &triples);  // 创建在线阶段对象
        // online->initialize(Xi, Yi);  // 在在线阶段初始化

        // train_model();  // 训练模型
    }

    // // 训练模型的方法
    // void train_model();

    // // 测试模型的方法
    // void test_model(RowMatrixXd& testing_data, ColVectorXd& testing_labels);

};

#endif