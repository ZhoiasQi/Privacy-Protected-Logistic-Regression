#include "online_phase.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

extern Traffic traffic;

void OnlinePhase::initialize(RowMatrixXi64& Xi, ColVectorXi64& Yi){
    this->Xi = Xi;
    this->Yi = Yi;

    for(int i = 0; i < d; i++){
        wi(i) = 0;
    }

    Ui = triples->Ai;
    Ei = Xi - Ui;
    Vi = triples->Bi;
    Vi_ = triples->Bi_;
    Zi = triples->Ci;
    Zi_ = triples->Ci_;

    if(party == ALICE){
        send<RowMatrixXi64>(io, Ei);
    }
    else{
        recv<RowMatrixXi64>(io, E);
    }

    if(party == BOB){
        send<RowMatrixXi64>(io, Ei);
    }
    else{
        recv<RowMatrixXi64>(io, E);
    }

    traffic.online += (sizeof(Ei) + sizeof(E))  / (double)B_TO_MB; 

    E = E + Ei;
}

void OnlinePhase::train_batch(int iter, int indexLo){
    
    //block参数分别为起始行、起始列、行数、列数
    //segment参数分别为起始行、行数
    RowMatrixXi64 X = Xi.block(indexLo, 0, BATCH_SIZE, d); // 根据 indexLo 和 BATCH_SIZE 从 Xi 中截取子矩阵 X
    ColVectorXi64 Y = Yi.segment(indexLo, BATCH_SIZE); // 根据 indexLo 和 BATCH_SIZE 从 Yi 中截取子向量 Y
    RowMatrixXi64 Eb = E.block(indexLo, 0, BATCH_SIZE, d); // 根据 indexLo 和 BATCH_SIZE 从 E 中截取子矩阵 Eb
    ColVectorXi64 V = Vi.col(iter); // 取 Vi 中第 iter 列作为向量 V
    ColVectorXi64 V_ = Vi_.col(iter); // 取 Vi_ 中第 iter 列作为向量 V_
    ColVectorXi64 Z = Zi.col(iter); // 取 Zi 中第 iter 列作为向量 Z
    ColVectorXi64 Z_ = Zi_.col(iter); // 取 Zi_ 中第 iter 列作为向量 Z_

    Fi = wi - V;

    //一些中间结果辅助向量
    ColVectorXi64 D(BATCH_SIZE);
    ColVectorXi64 Y_(BATCH_SIZE);
    ColVectorXi64 Sig(BATCH_SIZE);
    ColVectorXi64 Fi_(BATCH_SIZE);
    ColVectorXi64 F_(BATCH_SIZE);
    ColVectorXi64 delta(d);

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

    traffic.online += (sizeof(Fi) + sizeof(F))  / (double)B_TO_MB; 

    F = F + Fi;

    Y_ = -i * (Eb * F) + X * F + Eb * wi + Z;

    truncate<ColVectorXi64>(i, SCALING_FACTOR, Y_);

    Sig = sigmoid(Y_, i, io);

    D = Sig - Y;

    Fi_ = D - V_;

    if (party == ALICE){
        send<ColVectorXi64>(io, Fi_);
    }
    else{
        recv<ColVectorXi64>(io, F_);
    }

    if (party == BOB){
        send<ColVectorXi64>(io, Fi_);
    }
    else{
        recv<ColVectorXi64>(io, F_);
    }

    traffic.online += (sizeof(Fi_) + sizeof(F_))  / (double)B_TO_MB; 

    F_ = F_ + Fi_;

    RowMatrixXi64 Xt = X.transpose();
    RowMatrixXi64 Ebt = Eb.transpose();

    delta = -i * (Ebt * F_) + Xt * F_ + Ebt * D + Z_;

    truncate<ColVectorXi64>(i, SCALING_FACTOR, delta);

    int min = std::min(iter, 15);
    //cout << min;
    auto k = alpha_inv;

    if(iter < 5){
        k = alpha_inv;
    }
    else if(iter < 30){
        int e = (iter - 5) / 3;
        k = Mypow(k, e);
    }
    else{
        k = Mypow(k, 10);
    }

    truncate<ColVectorXi64>(i, k * BATCH_SIZE, delta);

    wi = wi - delta;

}

uint64_t Mypow(uint64_t a, int b){
    for(int i = 0; i < b; i++){
        a *= 2;
    }
    return a;
}