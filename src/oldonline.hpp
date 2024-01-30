#ifndef ONLINE_HPP
#define ONLINE_HPP

#include <math.h>
#include "util.hpp"

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
        this->n = params.n; // ����ѵ����������
        this->d = params.d; // ��������ά��
        this->t = (params.n) / BATCH_SIZE; // �����������
        this->party = PARTY; // ���õ�ǰ���뷽��ALICE��BOB
        this->alpha_inv = params.alpha_inv; // ����ѧϰ�ʵĵ���
        this->io = io; // ��������I/O����
        this->triples = triples; // ������Ԫ�����

        if (party == emp::ALICE)
            i = 0; // �����ǰ���뷽ΪALICE������iΪ0
        else
            i = 1; // �����ǰ���뷽ΪBOB������iΪ1

        Xi.resize(n, d); // ��ʼ������Xi�Ĵ�СΪn��d
        Ui.resize(n, d); // ��ʼ������Ui�Ĵ�СΪn��d
        E.resize(n, d); // ��ʼ������E�Ĵ�СΪn��d
        Ei.resize(n, d); // ��ʼ������Ei�Ĵ�СΪn��d
        Yi.resize(n); // ��ʼ������Yi�Ĵ�СΪn
        Fi.resize(d); // ��ʼ������Fi�Ĵ�СΪd
        F.resize(d); // ��ʼ������F�Ĵ�СΪd
        wi.resize(d); // ��ʼ������wi�Ĵ�СΪd
        Vi.resize(d, t); // ��ʼ������Vi�Ĵ�СΪd��t
        Zi.resize(BATCH_SIZE, t); // ��ʼ������Zi�Ĵ�СΪBATCH_SIZE��t
        Vi_.resize(BATCH_SIZE, t); // ��ʼ������Vi'�Ĵ�СΪBATCH_SIZE��t
        Zi_.resize(d, t); // ��ʼ������Zi'�Ĵ�СΪd��
    }

    void initialize(RowMatrixXi64& Xi, ColVectorXi64& Yi); // 初始化在线阶殄1�7
    void train_batch(int iter, int indexLo); // 在线阶段训练每个批次的数捄1�7
};

#endif // ONLINE_HPP
