#include "test_online_phase.hpp"

void TestOnlinePhase::initialize(RowMatrixXi64& Xi, ColVectorXi64& Yi){
    this->Xi = Xi;
    this->Yi = Yi;
    
    for(int i = 0; i < d; i++){
        prediction_i(i) = 0;
    }

    Ui = triples->Ai;
    Ei = Xi - Ui;
    Vi = triples->Bi;
    Vi_ = triples->Bi_;
    Zi = triples->Ci;
    Zi_ = triples->Ci_;

    if(party == CAROL){
        send<RowMatrixXi64>(io, Ei);
    }
    else{
        recv<RowMatrixXi64>(io, E);
    }

    if(party == ALICE){
        send<RowMatrixXi64>(io, Ei);
    }
    else{
        recv<RowMatrixXi64>(io, E);
    }

    E = E + Ei;
}

void TestOnlinePhase::test_model(int iter, int indexLo){
    
    //block参数分别为起始行、起始列、行数、列数
    //segment参数分别为起始行、行数
    RowMatrixXi64 X = Xi.block(indexLo, 0, n, d); // 根据 indexLo 和 BATCH_SIZE 从 Xi 中截取子矩阵 X
    ColVectorXi64 Y = Yi.segment(indexLo, n); // 根据 indexLo 和 BATCH_SIZE 从 Yi 中截取子向量 Y
    RowMatrixXi64 Eb = E.block(indexLo, 0, n, d); // 根据 indexLo 和 BATCH_SIZE 从 E 中截取子矩阵 Eb
    ColVectorXi64 V = Vi.col(iter); // 取 Vi 中第 iter 列作为向量 V
    ColVectorXi64 V_ = Vi_.col(iter); // 取 Vi_ 中第 iter 列作为向量 V_
    ColVectorXi64 Z = Zi.col(iter); // 取 Zi 中第 iter 列作为向量 Z
    ColVectorXi64 Z_ = Zi_.col(iter); // 取 Zi_ 中第 iter 列作为向量 Z_、

    Fi = prediction_i - V;

    //一些中间结果辅助向量
    // ColVectorXi64 D(n);
    ColVectorXi64 Y_(n);
    // ColVectorXi64 Sig(n);
    // ColVectorXi64 Fi_(n);
    // ColVectorXi64 F_(n);
    // ColVectorXi64 delta(d);

    if(party == ALICE){
        send<ColVectorXi64>(io, Fi);
    }
    else{
        recv<ColVectorXi64>(io, F);
    }

    if(party == BOB){
        send<ColVectorXi64>(io, Fi);
    }
    else{
        recv<ColVectorXi64>(io, F);
    }

    F = F + Fi;

    Y_ = -i * (Eb * F) + X * F + Eb * prediction_i + Z;

    truncateTest<ColVectorXi64>(i, SCALING_FACTOR, Y_);

    prediction_i = Y_;

    // //TODO: 逻辑回归还没写
    // Sig = sigmoid(Y_);

    // D = Sig - Y;

    // Fi_ = D - V_;

    // if (party == ALICE){
    //     send<ColVectorXi64>(io, Fi_);
    // }
    // else{
    //     recv<ColVectorXi64>(io, F_);
    // }

    // if (party == BOB){
    //     send<ColVectorXi64>(io, Fi_);
    // }
    // else{
    //     recv<ColVectorXi64>(io, F_);
    // }

    // F_ = F_ + Fi_;

    // RowMatrixXi64 Xt = X.transpose();
    // RowMatrixXi64 Ebt = Eb.transpose();

    // delta = -i * (Ebt * F_) + Xt * F_ + Ebt * D + Z_;

    // truncateTest<ColVectorXi64>(i, SCALING_FACTOR, delta);
    // //乘学习率，除batch_size, 即除（学习率的倒数*BATCH_SIZE）
    // truncateTest<ColVectorXi64>(i, alpha_inv * BATCH_SIZE, delta);

    // prediction_i = prediction_i - delta;
    
}