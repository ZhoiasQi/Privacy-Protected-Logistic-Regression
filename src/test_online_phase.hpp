#ifndef T_ONLINE_HPP
#define T_ONLINE_HPP

#include <math.h>
#include "util.hpp"
#include "sigmoid.hpp"

using namespace emp;

template<class Derived>
void truncateTest(int i, uint64_t scaling_factor, Eigen::PlainObjectBase<Derived>& X) {
    if (i == 1)
        X = -1 * X; // 如果i为1，则将矩阵X的元素取相反数
    X /= scaling_factor; // 对矩阵X进行缩放，除以缩放因子scaling_factor
    if (i == 1)
        X = -1 * X; // 如果i为1，则再次将矩阵X的元素取相反数，恢复原始值
    return;
}

struct TestingParams {
    int n;
    int d;
};

class TestOnlinePhase{
public:
    int party, port;
    int n, d, t;
    int i;
    SetupTriples * triples;
    emp::NetIO* io;
    emp::PRG prg;
    RowMatrixXi64 Xi, Ui, E, Ei;
    ColVectorXi64 Yi, F, Fi, wi;
    ColMatrixXi64 Vi, Zi, Vi_, Zi_;

    TestOnlinePhase(TestingParams params, emp::NetIO* io, SetupTriples* triples) {
        this->party = PARTY;
        this->n = params.n;
        this->d = params.d;
        this->t = 1;
        this->io = io;
        this->triples = triples;
        this->i = party == CAROL ? 0 : 1;

        Xi.resize(n, d);
        Ui.resize(n, d);
        E.resize(n, d);
        Ei.resize(n, d);
        Yi.resize(n);
        Fi.resize(d);
        F.resize(d);
        wi.resize(d);
        Vi.resize(d, t);
        Zi.resize(n, t);
        Vi_.resize(n, t);
        Zi_.resize(d, t);
    }

    void initialize(RowMatrixXi64& Xi, ColVectorXi64& Yi);

    void test_model(int iter, int indexLo);

};

#endif 
