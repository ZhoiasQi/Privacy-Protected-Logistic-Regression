#ifndef T_ONLINE_HPP
#define T_ONLINE_HPP

#include <math.h>
#include "util.hpp"
#include "sigmoid.hpp"

using namespace emp;

template<class Derived, class OtherDerived>
void descaleTest(Eigen::PlainObjectBase<Derived>& X, Eigen::PlainObjectBase<OtherDerived>& x) {
    Derived signed_X = X * SCALING_FACTOR; 
    x = (X.template cast<int64_t>()).template cast<double>(); 
    x /= SCALING_FACTOR; 
    return;
}

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
    ColVectorXi64 wi, F, Fi, prediction_i;
    ColMatrixXi64 Vi, Zi, Vi_, Zi_;

    TestOnlinePhase(TestingParams params, emp::NetIO* io, SetupTriples* triples) {
        this->party = PARTY;
        this->n = params.n;
        this->d = params.d;
        this->t = 1;
        this->io = io;
        this->triples = triples;
        
        if(party == CAROL){
            i = 0;
        }
        else{
            i = 1;
        }

        Xi.resize(n, d);
        Ui.resize(n, d);
        E.resize(n, d);
        Ei.resize(n, d);
        wi.resize(d);
        Fi.resize(d);
        F.resize(d);
        prediction_i.resize(n);
        Vi.resize(d, t);
        Zi.resize(n, t);
    }

    void initialize(RowMatrixXi64& Xi, ColVectorXi64& wi);

    void test_model(int iter, int indexLo);

};

#endif 
