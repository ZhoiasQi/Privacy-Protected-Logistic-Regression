#ifndef ONLINE_HPP
#define ONLINE_HPP

#include <math.h>
#include "util.hpp"

// 灏嗚緭鍏ョ煩闃� X 缂╂斁锛屽苟瀛樺偍鍒� x 涓�
template<class Derived, class OtherDerived>
void scale(Eigen::PlainObjectBase<Derived>& X, Eigen::PlainObjectBase<OtherDerived>& x) {
    Derived scaled_X = X * SCALING_FACTOR; // 缂╂斁鐭╅樀 X
    x = scaled_X.template cast<uint64_t>(); // 灏嗙缉鏀惧悗鐨勭煩闃佃浆鎹负 uint64_t 绫诲瀷锛屽苟瀛樺偍鍒� x
    return;
}

// 灏嗚緭鍏ョ煩闃� X 杩樺師缂╂斁锛屽苟瀛樺偍鍒� x 涓�
template<class Derived, class OtherDerived>
void descale(Eigen::PlainObjectBase<Derived>& X, Eigen::PlainObjectBase<OtherDerived>& x) {
    Derived signed_X = X * SCALING_FACTOR; // 杩樺師缂╂斁鐭╅樀 X
    x = (X.template cast<int64_t>()).template cast<double>(); // 灏嗚繕鍘熺缉鏀惧悗鐨勭煩闃佃浆鎹负 double 绫诲瀷锛屽苟瀛樺偍鍒� x
    x /= SCALING_FACTOR; // 杩樺師缂╂斁鍥犲瓙
    return;
}

// 瀵硅緭鍏ョ煩闃� X 杩涜鎴柇鎿嶄綔
template<class Derived>
void truncate(int i, uint64_t scaling_factor, Eigen::PlainObjectBase<Derived>& X) {
    if (i == 1)
        X = -1 * X; // 濡傛灉 i 涓� 1锛屽垯灏嗙煩闃� X 涓殑鍏冪礌鍙栬礋
    X /= scaling_factor; // 鎸夌収缂╂斁鍥犲瓙缂╂斁鐭╅樀 X
    if (i == 1)
        X = -1 * X; // 濡傛灉 i 涓� 1锛屽垯灏嗙煩闃� X 涓殑鍏冪礌鍐嶅彇璐�
    return;
}

struct TrainingParams {
    int n;  // 璁粌鏍锋湰鏁伴噺
    int d;  // 鐗瑰緛缁村害
    int alpha_inv = LEARNING_RATE_INV;  // 瀛︿範鐜囩殑鍊掓暟
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
        this->n = params.n; // 鏍锋湰鏁�
        this->d = params.d; // 鐗瑰緛鏁�
        this->t = (params.n) / BATCH_SIZE; // 杩唬娆℃暟
        this->party = PARTY; // 褰撳墠鍙備笌鏂癸紙ALICE鎴朆OB锛�
        this->alpha_inv = params.alpha_inv; // 瀛︿範鐜囩殑鍊掓暟
        this->io = io; // 缃戠粶杈撳叆杈撳嚭瀵硅薄鎸囬拡
        this->triples = triples; // 浼殢鏈烘暟瀵硅薄鎸囬拡

        if (party == emp::ALICE)
            i = 0; // 褰撳墠鍙備笌鏂逛负ALICE鏃讹紝i涓�0
        else
            i = 1; // 褰撳墠鍙備笌鏂逛负BOB鏃讹紝i涓�1

        Xi.resize(n, d); // 鍒濆鍖� Xi 鐭╅樀
        Ui.resize(n, d); // 鍒濆鍖� Ui 鐭╅樀
        E.resize(n, d); // 鍒濆鍖� E 鐭╅樀
        Ei.resize(n, d); // 鍒濆鍖� Ei 鐭╅樀
        Yi.resize(n); // 鍒濆鍖� Yi 鍚戦噺
        Fi.resize(d); // 鍒濆鍖� Fi 鍚戦噺
        F.resize(d); // 鍒濆鍖� F 鍚戦噺
        wi.resize(d); // 鍒濆鍖� wi 鍚戦噺
        Vi.resize(d, t); // 鍒濆鍖� Vi 鐭╅樀
        Zi.resize(BATCH_SIZE, t); // 鍒濆鍖� Zi 鐭╅樀
        Vi_.resize(BATCH_SIZE, t); // 鍒濆鍖� Vi_ 鐭╅樀
        Zi_.resize(d, t); // 鍒濆鍖� Zi_ 鐭╅樀
    }

    void initialize(RowMatrixXi64& Xi, ColVectorXi64& Yi); // 鍒濆鍖栧湪绾块樁娈�
    void train_batch(int iter, int indexLo); // 鍦ㄧ嚎闃舵璁粌姣忎釜鎵规鐨勬暟鎹�
};

#endif // ONLINE_HPP
