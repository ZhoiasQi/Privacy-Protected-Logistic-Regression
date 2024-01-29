#ifndef ONLINE_HPP
#define ONLINE_HPP

#include <math.h>
#include "util.hpp"

// 将输入矩阵 X 缩放，并存储到 x 中
template<class Derived, class OtherDerived>
void scale(Eigen::PlainObjectBase<Derived>& X, Eigen::PlainObjectBase<OtherDerived>& x) {
    Derived scaled_X = X * SCALING_FACTOR; // 缩放矩阵 X
    x = scaled_X.template cast<uint64_t>(); // 将缩放后的矩阵转换为 uint64_t 类型，并存储到 x
    return;
}

// 将输入矩阵 X 还原缩放，并存储到 x 中
template<class Derived, class OtherDerived>
void descale(Eigen::PlainObjectBase<Derived>& X, Eigen::PlainObjectBase<OtherDerived>& x) {
    Derived signed_X = X * SCALING_FACTOR; // 还原缩放矩阵 X
    x = (X.template cast<int64_t>()).template cast<double>(); // 将还原缩放后的矩阵转换为 double 类型，并存储到 x
    x /= SCALING_FACTOR; // 还原缩放因子
    return;
}

// 对输入矩阵 X 进行截断操作
template<class Derived>
void truncate(int i, uint64_t scaling_factor, Eigen::PlainObjectBase<Derived>& X) {
    if (i == 1)
        X = -1 * X; // 如果 i 为 1，则将矩阵 X 中的元素取负
    X /= scaling_factor; // 按照缩放因子缩放矩阵 X
    if (i == 1)
        X = -1 * X; // 如果 i 为 1，则将矩阵 X 中的元素再取负
    return;
}

struct TrainingParams {
    int n;  // 训练样本数量
    int d;  // 特征维度
    int alpha_inv = LEARNING_RATE_INV;  // 学习率的倒数
};

class OnlinePhase {
public:
    int party, port;
    int n, d, t, i, alpha_inv;
    SetupTriples* triples;
    emp::NetIO* io;
    emp::PRG prg;
    RowMatrixXi64 Xi, Ui, E, Ei;
    ColVectorXi64 Yi, F, Fi, wi;
    ColMatrixXi64 Vi, Zi, Vi_, Zi_;

    OnlinePhase(TrainingParams params, emp::NetIO* io, SetupTriples* triples) {
        this->n = params.n; // 样本数
        this->d = params.d; // 特征数
        this->t = (params.n) / BATCH_SIZE; // 迭代次数
        this->party = PARTY; // 当前参与方（ALICE或BOB）
        this->alpha_inv = params.alpha_inv; // 学习率的倒数
        this->io = io; // 网络输入输出对象指针
        this->triples = triples; // 伪随机数对象指针

        if (party == emp::ALICE)
            i = 0; // 当前参与方为ALICE时，i为0
        else
            i = 1; // 当前参与方为BOB时，i为1

        Xi.resize(n, d); // 初始化 Xi 矩阵
        Ui.resize(n, d); // 初始化 Ui 矩阵
        E.resize(n, d); // 初始化 E 矩阵
        Ei.resize(n, d); // 初始化 Ei 矩阵
        Yi.resize(n); // 初始化 Yi 向量
        Fi.resize(d); // 初始化 Fi 向量
        F.resize(d); // 初始化 F 向量
        wi.resize(d); // 初始化 wi 向量
        Vi.resize(d, t); // 初始化 Vi 矩阵
        Zi.resize(BATCH_SIZE, t); // 初始化 Zi 矩阵
        Vi_.resize(BATCH_SIZE, t); // 初始化 Vi_ 矩阵
        Zi_.resize(d, t); // 初始化 Zi_ 矩阵
    }

    void initialize(RowMatrixXi64& Xi, ColVectorXi64& Yi); // 初始化在线阶段
    void train_batch(int iter, int indexLo); // 在线阶段训练每个批次的数据
};

#endif // ONLINE_HPP
