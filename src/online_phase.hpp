#ifndef ONLINE_HPP
#define ONLINE_HPP

#include <math.h>
#include "util.hpp"
#include "sigmoid.hpp"

using namespace emp;

template<class Derived, class OtherDerived>
void scale(Eigen::PlainObjectBase<Derived>& X, Eigen::PlainObjectBase<OtherDerived>& x) {
    Derived scaled_X = X * SCALING_FACTOR; // 矩阵X进行缩放
    x = scaled_X.template cast<uint64_t>(); // 将缩放后的矩阵X转换为uint64_t类型并存储在变量x中
    return;
}

template<class Derived, class OtherDerived>
void descale(Eigen::PlainObjectBase<Derived>& X, Eigen::PlainObjectBase<OtherDerived>& x) {
    Derived signed_X = X * SCALING_FACTOR; 
    x = (X.template cast<int64_t>()).template cast<double>(); 
    x /= SCALING_FACTOR;
    return;
}

template<class Derived>
void truncate(int i, uint64_t scaling_factor, Eigen::PlainObjectBase<Derived>& X) {
    if (i == 1)
        X = -1 * X;
    X /= scaling_factor; 
    if (i == 1)
        X = -1 * X; 
    return;
}

struct TrainingParams {
    int n; 
    int d; 
    int alpha_inv = LEARNING_RATE_INV;  
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
        E.resize(n, d);
        Ei.resize(n, d);
        Yi.resize(n);
        Fi.resize(d);
        F.resize(d);
        wi.resize(d);
        Vi.resize(d, t);
        Zi.resize(BATCH_SIZE, t);
        Vi_.resize(BATCH_SIZE, t);
        Zi_.resize(d, t);
    }

    void initialize(RowMatrixXi64& Xi, ColVectorXi64& Yi);

    void train_batch(int iter, int indexLo);

};

uint64_t Mypow(uint64_t a, int b);

#endif 
