#include "online_phase.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

// 鍒濆鍖栧湪绾块樁娈�
void OnlinePhase::initialize(RowMatrixXi64& Xi, ColVectorXi64& Yi){
    this->Xi = Xi; // 鍒濆鍖栬緭鍏ユ暟鎹煩闃� Xi
    this->Yi = Yi; // 鍒濆鍖栬緭鍏ユ爣绛惧悜閲� Yi

    for (int i = 0; i < d; i++){
        wi(i) = 0; // 鍒濆鍖� wi 鍚戦噺锛屾墍鏈夊厓绱犵疆涓� 0
    }

    Ui = triples->Ai; // 鍒濆鍖� Ui 鐭╅樀

    Ei = Xi - Ui; // 鍒濆鍖� Ei 鐭╅樀

    Vi = triples->Bi; // 鍒濆鍖� Vi 鐭╅樀
    Vi_ = triples->Bi_; // 鍒濆鍖� Vi_ 鐭╅樀
    Zi = triples->Ci; // 鍒濆鍖� Zi 鐭╅樀
    Zi_ = triples->Ci_; // 鍒濆鍖� Zi_ 鐭╅樀

    if (party == ALICE)
        send<RowMatrixXi64>(io, Ei); // 濡傛灉褰撳墠鍙備笌鏂逛负 ALICE锛屽垯鍙戦€� Ei 鐭╅樀
    else
        recv<RowMatrixXi64>(io, E); // 濡傛灉褰撳墠鍙備笌鏂逛负 BOB锛屽垯鎺ユ敹 E 鐭╅樀
    if (party == BOB)
        send<RowMatrixXi64>(io, Ei); // 濡傛灉褰撳墠鍙備笌鏂逛负 BOB锛屽垯鍙戦€� Ei 鐭╅樀
    else
        recv<RowMatrixXi64>(io, E); // 濡傛灉褰撳墠鍙備笌鏂逛负 ALICE锛屽垯鎺ユ敹 E 鐭╅樀

    E += Ei; // 鏇存柊 E 鐭╅樀
}

// 鍦ㄧ嚎闃舵璁粌姣忎釜鎵规鐨勬暟鎹�
void OnlinePhase::train_batch(int iter, int indexLo){
    RowMatrixXi64 X = Xi.block(indexLo, 0, BATCH_SIZE, d); // 鎻愬彇褰撳墠鎵规鐨勮緭鍏ユ暟鎹�
    ColVectorXi64 Y = Yi.segment(indexLo, BATCH_SIZE); // 鎻愬彇褰撳墠鎵规鐨勬爣绛惧悜閲�
    RowMatrixXi64 Eb = E.block(indexLo, 0, BATCH_SIZE, d); // 鎻愬彇褰撳墠鎵规鐨勫姞鍣煩闃�
    ColVectorXi64 V = Vi.col(iter); // 鎻愬彇 Vi 鐭╅樀鐨勭 iter 鍒�
    ColVectorXi64 V_ = Vi_.col(iter); // 鎻愬彇 Vi_ 鐭╅樀鐨勭 iter 鍒�
    ColVectorXi64 Z = Zi.col(iter); // 鎻愬彇 Zi 鐭╅樀鐨勭 iter 鍒�
    ColVectorXi64 Z_ = Zi_.col(iter); // 鎻愬彇 Zi_ 鐭╅樀鐨勭 iter 鍒�

    Fi = wi - V; // 璁＄畻 Fi 鍚戦噺

    ColVectorXi64 D(BATCH_SIZE);
    ColVectorXi64 Y_(BATCH_SIZE);
    ColVectorXi64 Fi_(BATCH_SIZE);
    ColVectorXi64 F_(BATCH_SIZE);
    ColVectorXi64 delta(d);

    if (party == ALICE)
        send<ColVectorXi64>(io, Fi); // 濡傛灉褰撳墠鍙備笌鏂逛负 ALICE锛屽垯鍙戦€� Fi 鍚戦噺
    else
        recv<ColVectorXi64>(io, F); // 濡傛灉褰撳墠鍙備笌鏂逛负 BOB锛屽垯鎺ユ敹 F 鍚戦噺

    if (party == BOB)
        send<ColVectorXi64>(io, Fi); // 濡傛灉褰撳墠鍙備笌鏂逛负 BOB锛屽垯鍙戦€� Fi 鍚戦噺
    else
        recv<ColVectorXi64>(io, F); // 濡傛灉褰撳墠鍙備笌鏂逛负 ALICE锛屽垯鎺ユ敹 F 鍚戦噺

    F += Fi; // 鏇存柊 F 鍚戦噺

    Y_ = -i * (Eb * F)  + X * F + Eb * wi + Z; // 璁＄畻 Y_ 鍚戦噺

    truncate<ColVectorXi64>(i, SCALING_FACTOR, Y_); // 瀵� Y_ 鍚戦噺杩涜鎴柇鎿嶄綔

    D = Y_ - Y; // 璁＄畻 D 鍚戦噺

    Fi_ = D - V_; // 璁＄畻 Fi_ 鍚戦噺

    if (party == ALICE)
        send<ColVectorXi64>(io, Fi_); // 濡傛灉褰撳墠鍙備笌鏂逛负 ALICE锛屽垯鍙戦€� Fi_ 鍚戦噺
    else
        recv<ColVectorXi64>(io, F_); // 濡傛灉褰撳墠鍙備笌鏂逛负 BOB锛屽垯鎺ユ敹 F_ 鍚戦噺

    if (party == BOB)
        send<ColVectorXi64>(io, Fi_); // 濡傛灉褰撳墠鍙備笌鏂逛负 BOB锛屽垯鍙戦€� Fi_ 鍚戦噺
    else
        recv<ColVectorXi64>(io, F_); // 濡傛灉褰撳墠鍙備笌鏂逛负 ALICE锛屽垯鎺ユ敹 F_ 鍚戦噺

    F_ += Fi_; // 鏇存柊 F_ 鍚戦噺

    RowMatrixXi64 Et = Eb.transpose(); // 璁＄畻 Eb 鐨勮浆缃煩闃�
    RowMatrixXi64 Xt = X.transpose(); // 璁＄畻 X 鐨勮浆缃煩闃�

    delta = -i * (Et * F_) + Xt * F_ + Et * D + Z_; // 璁＄畻 delta 鍚戦噺

    truncate<ColVectorXi64>(i, SCALING_FACTOR, delta); // 瀵� delta 鍚戦噺杩涜鎴柇鎿嶄綔
    truncate<ColVectorXi64>(i, alpha_inv * BATCH_SIZE, delta); // 瀵� delta 鍚戦噺杩涜鎴柇鎿嶄綔

    wi -= delta; // 鏇存柊 wi 鍚戦噺
}
