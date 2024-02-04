#ifndef ONLINE_HPP
#define ONLINE_HPP

#include <math.h>
#include "util.hpp"

using namespace emp;

template<class Derived, class OtherDerived>
void scale(Eigen::PlainObjectBase<Derived>& X, Eigen::PlainObjectBase<OtherDerived>& x) {
    Derived scaled_X = X * SCALING_FACTOR; // ����X��������
    x = scaled_X.template cast<uint64_t>(); // �����ź�ľ���Xת��Ϊuint64_t���Ͳ��洢�ڱ���x��
    return;
}

template<class Derived, class OtherDerived>
void descale(Eigen::PlainObjectBase<Derived>& X, Eigen::PlainObjectBase<OtherDerived>& x) {
    Derived signed_X = X * SCALING_FACTOR; // ����X���з�����
    x = (X.template cast<int64_t>()).template cast<double>(); // �������ź�ľ���Xת��Ϊdouble���Ͳ��洢�ڱ���x��
    x /= SCALING_FACTOR; // �Ա���x���н�һ��������
    return;
}

template<class Derived>
void truncate(int i, uint64_t scaling_factor, Eigen::PlainObjectBase<Derived>& X) {
    if (i == 1)
        X = -1 * X; // ���iΪ1���򽫾���X��Ԫ��ȡ�෴��
    X /= scaling_factor; // �Ծ���X�������ţ�������������scaling_factor
    if (i == 1)
        X = -1 * X; // ���iΪ1�����ٴν�����X��Ԫ��ȡ�෴�����ָ�ԭʼֵ
    return;
}

struct TrainingParams {
    int n;  // ѵ����������
    int d;  // ����ά��
    int alpha_inv = LEARNING_RATE_INV;  // ѧϰ�ʵĵ���
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
