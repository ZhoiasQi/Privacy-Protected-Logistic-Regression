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
    
    //block�����ֱ�Ϊ��ʼ�С���ʼ�С�����������
    //segment�����ֱ�Ϊ��ʼ�С�����
    RowMatrixXi64 X = Xi.block(indexLo, 0, n, d); // ���� indexLo �� BATCH_SIZE �� Xi �н�ȡ�Ӿ��� X
    ColVectorXi64 Y = Yi.segment(indexLo, n); // ���� indexLo �� BATCH_SIZE �� Yi �н�ȡ������ Y
    RowMatrixXi64 Eb = E.block(indexLo, 0, n, d); // ���� indexLo �� BATCH_SIZE �� E �н�ȡ�Ӿ��� Eb
    ColVectorXi64 V = Vi.col(iter); // ȡ Vi �е� iter ����Ϊ���� V
    ColVectorXi64 V_ = Vi_.col(iter); // ȡ Vi_ �е� iter ����Ϊ���� V_
    ColVectorXi64 Z = Zi.col(iter); // ȡ Zi �е� iter ����Ϊ���� Z
    ColVectorXi64 Z_ = Zi_.col(iter); // ȡ Zi_ �е� iter ����Ϊ���� Z_��

    Fi = prediction_i - V;

    //һЩ�м�����������
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

    // //TODO: �߼��ع黹ûд
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
    // //��ѧϰ�ʣ���batch_size, ������ѧϰ�ʵĵ���*BATCH_SIZE��
    // truncateTest<ColVectorXi64>(i, alpha_inv * BATCH_SIZE, delta);

    // prediction_i = prediction_i - delta;
    
}