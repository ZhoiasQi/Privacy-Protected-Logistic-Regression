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

    

}