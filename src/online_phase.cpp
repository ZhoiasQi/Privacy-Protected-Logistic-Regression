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
    
    //block�����ֱ�Ϊ��ʼ�С���ʼ�С�����������
    //segment�����ֱ�Ϊ��ʼ�С�����
    RowMatrixXi64 X = Xi.block(indexLo, 0, BATCH_SIZE, d); // ���� indexLo �� BATCH_SIZE �� Xi �н�ȡ�Ӿ��� X
    ColVectorXi64 Y = Yi.segment(indexLo, BATCH_SIZE); // ���� indexLo �� BATCH_SIZE �� Yi �н�ȡ������ Y
    RowMatrixXi64 Eb = E.block(indexLo, 0, BATCH_SIZE, d); // ���� indexLo �� BATCH_SIZE �� E �н�ȡ�Ӿ��� Eb
    ColVectorXi64 V = Vi.col(iter); // ȡ Vi �е� iter ����Ϊ���� V
    ColVectorXi64 V_ = Vi_.col(iter); // ȡ Vi_ �е� iter ����Ϊ���� V_
    ColVectorXi64 Z = Zi.col(iter); // ȡ Zi �е� iter ����Ϊ���� Z
    ColVectorXi64 Z_ = Zi_.col(iter); // ȡ Zi_ �е� iter ����Ϊ���� Z_

    Fi = wi - V;

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

    //һЩ�м�����������
    ColVectorXi64 D(BATCH_SIZE);
    ColVectorXi64 Y_(BATCH_SIZE);
    ColVectorXi64 Sig(BATCH_SIZE);
    ColVectorXi64 Fi_(BATCH_SIZE);
    ColVectorXi64 F_(BATCH_SIZE);
    ColVectorXi64 delta(d);
    
    Y_ = -i * (Eb * F) + X * F + Eb * wi + Z;

    truncate(i, SCALING_FACTOR, Y_);

    //TODO: �߼��ع黹ûд
    Sig = sigmoid(Y_);

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

    F_ = F_ + Fi_;

    RowMatrixXi64 Xt = X.transpose();
    RowMatrixXi64 Ebt = Eb.transpose();

    delta = -i * (Ebt * F_) + Xt * F + Ebt * D + Z;

    truncate<ColVectorXi64>(i, SCALING_FACTOR, delta);
    //��ѧϰ�ʣ���batch_size, ������ѧϰ�ʵĵ���*BATCH_SIZE��
    truncate<ColVectorXi64>(i, alpha_inv * BATCH_SIZE, delta);

    wi = wi - delta;

}