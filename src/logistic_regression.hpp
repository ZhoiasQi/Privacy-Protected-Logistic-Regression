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

    
    // OnlinePhase* online;  // 在线阶段对象指针



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

        this->setup = new OfflineSetUp();
    }

    void train_model();

};

#endif