#include "online_phase.hpp" // 包含自定义的头文件 online_phase.hpp

using namespace Eigen; // 使用 Eigen 命名空间
using Eigen::Matrix; // 导入 Eigen 中的 Matrix 类型
using namespace emp; // 使用 emp 命名空间
using namespace std; // 使用标准命名空间

void OnlinePhase::initialize(RowMatrixXi64& Xi, ColVectorXi64& Yi){
    this->Xi = Xi; // 将 Xi 赋值给类成员变量 Xi
    this->Yi = Yi; // 将 Yi 赋值给类成员变量 Yi

    for (int i = 0; i < d; i++){ // 循环初始化 wi 数组中的元素为 0
        wi(i) = 0;
    }

    Ui = triples->Ai; // 将 triples->Ai 赋值给类成员变量 Ui

    Ei = Xi - Ui; // 计算 Ei，Xi 减去 Ui

    Vi = triples->Bi; // 将 triples->Bi 赋值给类成员变量 Vi
    Vi_ = triples->Bi_; // 将 triples->Bi_ 赋值给类成员变量 Vi_
    Zi = triples->Ci; // 将 triples->Ci 赋值给类成员变量 Zi
    Zi_ = triples->Ci_; // 将 triples->Ci_ 赋值给类成员变量 Zi_

    if (party == ALICE) // 判断当前 party 是否为 ALICE
        send<RowMatrixXi64>(io, Ei); // 若是 ALICE 发送 Ei
    else
        recv<RowMatrixXi64>(io, E); // 若不是 ALICE 接收 E

    if (party == BOB) // 判断当前 party 是否为 BOB
        send<RowMatrixXi64>(io, Ei); // 若是 BOB 发送 Ei
    else
        recv<RowMatrixXi64>(io, E); // 若不是 BOB 接收 E

    E += Ei; // 将 E 和 Ei 相加
}

void OnlinePhase::train_batch(int iter, int indexLo){
    RowMatrixXi64 X = Xi.block(indexLo, 0, BATCH_SIZE, d); // 根据 indexLo 和 BATCH_SIZE 从 Xi 中截取子矩阵 X
    ColVectorXi64 Y = Yi.segment(indexLo, BATCH_SIZE); // 根据 indexLo 和 BATCH_SIZE 从 Yi 中截取子向量 Y
    RowMatrixXi64 Eb = E.block(indexLo, 0, BATCH_SIZE, d); // 根据 indexLo 和 BATCH_SIZE 从 E 中截取子矩阵 Eb
    ColVectorXi64 V = Vi.col(iter); // 取 Vi 中第 iter 列作为向量 V
    ColVectorXi64 V_ = Vi_.col(iter); // 取 Vi_ 中第 iter 列作为向量 V_
    ColVectorXi64 Z = Zi.col(iter); // 取 Zi 中第 iter 列作为向量 Z
    ColVectorXi64 Z_ = Zi_.col(iter); // 取 Zi_ 中第 iter 列作为向量 Z_

    Fi = wi - V; // 计算 Fi，wi 减去 V

    ColVectorXi64 D(BATCH_SIZE); // 初始化长度为 BATCH_SIZE 的向量 D
    ColVectorXi64 Y_(BATCH_SIZE); // 初始化长度为 BATCH_SIZE 的向量 Y_
    ColVectorXi64 Fi_(BATCH_SIZE); // 初始化长度为 BATCH_SIZE 的向量 Fi_
    ColVectorXi64 F_(BATCH_SIZE); // 初始化长度为 BATCH_SIZE 的向量 F_
    ColVectorXi64 delta(d); // 初始化长度为 d 的向量 delta

    if (party == ALICE) // 判断当前 party 是否为 ALICE
        send<ColVectorXi64>(io, Fi); // 若是 ALICE 发送 Fi
    else
        recv<ColVectorXi64>(io, F); // 若不是 ALICE 接收 F

    if (party == BOB) // 判断当前 party 是否为 BOB
        send<ColVectorXi64>(io, Fi); // 若是 BOB 发送 Fi
    else
        recv<ColVectorXi64>(io, F); // 若不是 BOB 接收 F

    F += Fi; // 将 F 和 Fi 相加

    Y_ = -i * (Eb * F)  + X * F + Eb * wi + Z; // 计算 Y_

    truncate<ColVectorXi64>(i, SCALING_FACTOR, Y_); // 截断 Y_

    D = Y_ - Y; // 计算 D

    Fi_ = D - V_; // 计算 Fi_

    if (party == ALICE) // 判断当前 party 是否为 ALICE
        send<ColVectorXi64>(io, Fi_); // 若是 ALICE 发送 Fi_
    else
        recv<ColVectorXi64>(io, F_); // 若不是 ALICE 接收 F_

    if (party == BOB) // 判断当前 party 是否为 BOB
        send<ColVectorXi64>(io, Fi_); // 若是 BOB 发送 Fi_
    else
        recv<ColVectorXi64>(io, F_); // 若不是 BOB 接收 F_

    F_ += Fi_; // 将 F_ 和 Fi_ 相加

    RowMatrixXi64 Et = Eb.transpose(); // Eb 的转置矩阵
    RowMatrixXi64 Xt = X.transpose(); // X 的转置矩阵

    delta = -i * (Et * F_) + Xt * F_ + Et * D + Z_; // 计算 delta

    truncate<ColVectorXi64>(i, SCALING_FACTOR, delta); // 截断 delta
    truncate<ColVectorXi64>(i, alpha_inv * BATCH_SIZE, delta); // 截断 delta

    wi -= delta; // 更新 wi
}
