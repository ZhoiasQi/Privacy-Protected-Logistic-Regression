#include "online_phase.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

// 初始化在线阶段
void OnlinePhase::initialize(RowMatrixXi64& Xi, ColVectorXi64& Yi){
    this->Xi = Xi; // 初始化输入数据矩阵 Xi
    this->Yi = Yi; // 初始化输入标签向量 Yi

    for (int i = 0; i < d; i++){
        wi(i) = 0; // 初始化 wi 向量，所有元素置为 0
    }

    Ui = triples->Ai; // 初始化 Ui 矩阵

    Ei = Xi - Ui; // 初始化 Ei 矩阵

    Vi = triples->Bi; // 初始化 Vi 矩阵
    Vi_ = triples->Bi_; // 初始化 Vi_ 矩阵
    Zi = triples->Ci; // 初始化 Zi 矩阵
    Zi_ = triples->Ci_; // 初始化 Zi_ 矩阵

    if (party == ALICE)
        send<RowMatrixXi64>(io, Ei); // 如果当前参与方为 ALICE，则发送 Ei 矩阵
    else
        recv<RowMatrixXi64>(io, E); // 如果当前参与方为 BOB，则接收 E 矩阵
    if (party == BOB)
        send<RowMatrixXi64>(io, Ei); // 如果当前参与方为 BOB，则发送 Ei 矩阵
    else
        recv<RowMatrixXi64>(io, E); // 如果当前参与方为 ALICE，则接收 E 矩阵

    E += Ei; // 更新 E 矩阵
}

// 在线阶段训练每个批次的数据
void OnlinePhase::train_batch(int iter, int indexLo){
    RowMatrixXi64 X = Xi.block(indexLo, 0, BATCH_SIZE, d); // 提取当前批次的输入数据
    ColVectorXi64 Y = Yi.segment(indexLo, BATCH_SIZE); // 提取当前批次的标签向量
    RowMatrixXi64 Eb = E.block(indexLo, 0, BATCH_SIZE, d); // 提取当前批次的加噪矩阵
    ColVectorXi64 V = Vi.col(iter); // 提取 Vi 矩阵的第 iter 列
    ColVectorXi64 V_ = Vi_.col(iter); // 提取 Vi_ 矩阵的第 iter 列
    ColVectorXi64 Z = Zi.col(iter); // 提取 Zi 矩阵的第 iter 列
    ColVectorXi64 Z_ = Zi_.col(iter); // 提取 Zi_ 矩阵的第 iter 列

    Fi = wi - V; // 计算 Fi 向量

    ColVectorXi64 D(BATCH_SIZE);
    ColVectorXi64 Y_(BATCH_SIZE);
    ColVectorXi64 Fi_(BATCH_SIZE);
    ColVectorXi64 F_(BATCH_SIZE);
    ColVectorXi64 delta(d);

    if (party == ALICE)
        send<ColVectorXi64>(io, Fi); // 如果当前参与方为 ALICE，则发送 Fi 向量
    else
        recv<ColVectorXi64>(io, F); // 如果当前参与方为 BOB，则接收 F 向量

    if (party == BOB)
        send<ColVectorXi64>(io, Fi); // 如果当前参与方为 BOB，则发送 Fi 向量
    else
        recv<ColVectorXi64>(io, F); // 如果当前参与方为 ALICE，则接收 F 向量

    F += Fi; // 更新 F 向量

    Y_ = -i * (Eb * F)  + X * F + Eb * wi + Z; // 计算 Y_ 向量

    truncate<ColVectorXi64>(i, SCALING_FACTOR, Y_); // 对 Y_ 向量进行截断操作

    D = Y_ - Y; // 计算 D 向量

    Fi_ = D - V_; // 计算 Fi_ 向量

    if (party == ALICE)
        send<ColVectorXi64>(io, Fi_); // 如果当前参与方为 ALICE，则发送 Fi_ 向量
    else
        recv<ColVectorXi64>(io, F_); // 如果当前参与方为 BOB，则接收 F_ 向量

    if (party == BOB)
        send<ColVectorXi64>(io, Fi_); // 如果当前参与方为 BOB，则发送 Fi_ 向量
    else
        recv<ColVectorXi64>(io, F_); // 如果当前参与方为 ALICE，则接收 F_ 向量

    F_ += Fi_; // 更新 F_ 向量

    RowMatrixXi64 Et = Eb.transpose(); // 计算 Eb 的转置矩阵
    RowMatrixXi64 Xt = X.transpose(); // 计算 X 的转置矩阵

    delta = -i * (Et * F_) + Xt * F_ + Et * D + Z_; // 计算 delta 向量

    truncate<ColVectorXi64>(i, SCALING_FACTOR, delta); // 对 delta 向量进行截断操作
    truncate<ColVectorXi64>(i, alpha_inv * BATCH_SIZE, delta); // 对 delta 向量进行截断操作

    wi -= delta; // 更新 wi 向量
}
