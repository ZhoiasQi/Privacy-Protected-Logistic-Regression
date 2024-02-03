#ifndef ONLINE_HPP
#define ONLINE_HPP

#include <math.h>
#include "util.hpp"

template<class Derived, class OtherDerived>
void scale(Eigen::PlainObjectBase<Derived>& X, Eigen::PlainObjectBase<OtherDerived>& x) {
    Derived scaled_X = X * SCALING_FACTOR; // 矩阵X进行缩放
    x = scaled_X.template cast<uint64_t>(); // 将缩放后的矩阵X转换为uint64_t类型并存储在变量x中
    return;
}

template<class Derived, class OtherDerived>
void descale(Eigen::PlainObjectBase<Derived>& X, Eigen::PlainObjectBase<OtherDerived>& x) {
    Derived signed_X = X * SCALING_FACTOR; // 矩阵X进行反缩放
    x = (X.template cast<int64_t>()).template cast<double>(); // 将反缩放后的矩阵X转换为double类型并存储在变量x中
    x /= SCALING_FACTOR; // 对变量x进行进一步的缩放
    return;
}

template<class Derived>
void truncate(int i, uint64_t scaling_factor, Eigen::PlainObjectBase<Derived>& X) {
    if (i == 1)
        X = -1 * X; // 如果i为1，则将矩阵X的元素取相反数
    X /= scaling_factor; // 对矩阵X进行缩放，除以缩放因子scaling_factor
    if (i == 1)
        X = -1 * X; // 如果i为1，则再次将矩阵X的元素取相反数，恢复原始值
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
        this->n = params.n; // 设置训练样本数量
        this->d = params.d; // 设置特征维度
        this->t = (params.n) / BATCH_SIZE; // 计算迭代次数
        this->party = PARTY; // 设置当前参与方，ALICE或BOB
        this->alpha_inv = params.alpha_inv; // 设置学习率的倒数
        this->io = io; // 设置网络I/O对象
        this->triples = triples; // 设置三元组对象

        if (party == emp::ALICE)
            i = 0; // 如果当前参与方为ALICE，设置i为0
        else
            i = 1; // 如果当前参与方为BOB，设置i为1

        Xi.resize(n, d); // 初始化矩阵Xi的大小为n×d
        Ui.resize(n, d); // 初始化矩阵Ui的大小为n×d
        E.resize(n, d); // 初始化矩阵E的大小为n×d
        Ei.resize(n, d); // 初始化矩阵Ei的大小为n×d
        Yi.resize(n); // 初始化向量Yi的大小为n
        Fi.resize(d); // 初始化向量Fi的大小为d
        F.resize(d); // 初始化向量F的大小为d
        wi.resize(d); // 初始化向量wi的大小为d
        Vi.resize(d, t); // 初始化矩阵Vi的大小为d×t
        Zi.resize(BATCH_SIZE, t); // 初始化矩阵Zi的大小为BATCH_SIZE×t
        Vi_.resize(BATCH_SIZE, t); // 初始化矩阵Vi'的大小为BATCH_SIZE×t
        Zi_.resize(d, t); // 初始化矩阵Zi'的大小为d×
    }

    void initialize(RowMatrixXi64& Xi, ColVectorXi64& Yi); // 鍒濆鍖栧湪绾块樁娈�
    void train_batch(int iter, int indexLo); // 鍦ㄧ嚎闃舵璁粌姣忎釜鎵规鐨勬暟鎹�
};

#endif // ONLINE_HPP
