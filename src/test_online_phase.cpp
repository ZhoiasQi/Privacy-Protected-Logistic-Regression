#include "test_online_phase.hpp"

using namespace std;

void TestOnlinePhase::initialize(RowMatrixXi64& Xi, ColVectorXi64& Wi){
    this->Xi = Xi;
    this->wi = Wi;

    Ui = triples->Ai;

    Ei = Xi - Ui;

    Vi = triples->Bi;
    Zi = triples->Ci;

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
    
    //block�����ֱ�Ϊ��ʼ�С���ʼ�С�����������
    //segment�����ֱ�Ϊ��ʼ�С�����
    RowMatrixXi64 X = Xi.block(indexLo, 0, n, d); // ���� indexLo �� BATCH_SIZE �� Xi �н�ȡ�Ӿ��� X
    RowMatrixXi64 Eb = E.block(indexLo, 0, n, d); // ���� indexLo �� BATCH_SIZE �� E �н�ȡ�Ӿ��� Eb
    ColVectorXi64 V = Vi.col(iter); // ȡ Vi �е� iter ����Ϊ���� V
    ColVectorXi64 Z = Zi.col(iter); // ȡ Zi �е� iter ����Ϊ���� Z

    Fi = wi - V;

    ColVectorXi64 Y_(n);

    if(party == CAROL){
        send<ColVectorXi64>(io, Fi);
    }
    else{
        recv<ColVectorXi64>(io, F);
    }

    if(party == ALICE){
        send<ColVectorXi64>(io, Fi);
    }
    else{
        recv<ColVectorXi64>(io, F);
    }

    F = F + Fi;

    Y_ = -i * (Eb * F) + X * F + Eb * wi + Z;

    truncateTest<ColVectorXi64>(i, SCALING_FACTOR, Y_);

    prediction_i = Y_;
    
}