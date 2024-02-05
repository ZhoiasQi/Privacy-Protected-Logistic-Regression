#include "online_phase.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

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

    

}