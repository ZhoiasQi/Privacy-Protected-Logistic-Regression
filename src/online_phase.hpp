#ifndef ONLINE_HPP
#define ONLINE_HPP

#include <math.h>
#include "util.hpp"

using namespace emp;

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

class OnlinePhase{
public:
    int party, port;
    int n, d, t;
    int i, alpha_inv;
    SetupTriples * triples;
    emp::NetIO* io;
    emp::PRG prg;
    RowMatrixXi64 Xi, Ui, E, Ei;
    ColVectorXi64 Yi, F, Fi, wi;
    ColMatrixXi64 Vi, Zi, Vi_, Zi_;

    OnlinePhase(TrainingParams params, emp::NetIO* io, SetupTriples* triples) {
        this->party = PARTY;
        this->n = params.n;
        this->d = params.d;
        this->t = (params.n) / BATCH_SIZE;
        this->alpha_inv = params.alpha_inv;
        this->io = io;
        this->triples = triples;
        this->i = party == ALICE ? 0 : 1;

        Xi.resize(n, d);
        Ui.resize(n, d);
        Vi.resize(d, t);
        Vi_.resize(BATCH_SIZE, t);
        


    }
};

#endif 
