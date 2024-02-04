#include "online_phase.hpp" // �����Զ����ͷ�ļ� online_phase.hpp

using namespace Eigen; // ʹ�� Eigen �����ռ�
using Eigen::Matrix; // ���� Eigen �е� Matrix ����
using namespace emp; // ʹ�� emp �����ռ�
using namespace std; // ʹ�ñ�׼�����ռ�

void OnlinePhase::initialize(RowMatrixXi64& Xi, ColVectorXi64& Yi){
    this->Xi = Xi; // �� Xi ��ֵ�����Ա���� Xi
    this->Yi = Yi; // �� Yi ��ֵ�����Ա���� Yi

    for (int i = 0; i < d; i++){ // ѭ����ʼ�� wi �����е�Ԫ��Ϊ 0
        wi(i) = 0;
    }

    Ui = triples->Ai; // �� triples->Ai ��ֵ�����Ա���� Ui

    Ei = Xi - Ui; // ���� Ei��Xi ��ȥ Ui

    Vi = triples->Bi; // �� triples->Bi ��ֵ�����Ա���� Vi
    Vi_ = triples->Bi_; // �� triples->Bi_ ��ֵ�����Ա���� Vi_
    Zi = triples->Ci; // �� triples->Ci ��ֵ�����Ա���� Zi
    Zi_ = triples->Ci_; // �� triples->Ci_ ��ֵ�����Ա���� Zi_

    if (party == ALICE) // �жϵ�ǰ party �Ƿ�Ϊ ALICE
        send<RowMatrixXi64>(io, Ei); // ���� ALICE ���� Ei
    else
        recv<RowMatrixXi64>(io, E); // ������ ALICE ���� E

    if (party == BOB) // �жϵ�ǰ party �Ƿ�Ϊ BOB
        send<RowMatrixXi64>(io, Ei); // ���� BOB ���� Ei
    else
        recv<RowMatrixXi64>(io, E); // ������ BOB ���� E

    E += Ei; // �� E �� Ei ���
}

void OnlinePhase::train_batch(int iter, int indexLo){
    RowMatrixXi64 X = Xi.block(indexLo, 0, BATCH_SIZE, d); // ���� indexLo �� BATCH_SIZE �� Xi �н�ȡ�Ӿ��� X
    ColVectorXi64 Y = Yi.segment(indexLo, BATCH_SIZE); // ���� indexLo �� BATCH_SIZE �� Yi �н�ȡ������ Y
    RowMatrixXi64 Eb = E.block(indexLo, 0, BATCH_SIZE, d); // ���� indexLo �� BATCH_SIZE �� E �н�ȡ�Ӿ��� Eb
    ColVectorXi64 V = Vi.col(iter); // ȡ Vi �е� iter ����Ϊ���� V
    ColVectorXi64 V_ = Vi_.col(iter); // ȡ Vi_ �е� iter ����Ϊ���� V_
    ColVectorXi64 Z = Zi.col(iter); // ȡ Zi �е� iter ����Ϊ���� Z
    ColVectorXi64 Z_ = Zi_.col(iter); // ȡ Zi_ �е� iter ����Ϊ���� Z_

    Fi = wi - V; // ���� Fi��wi ��ȥ V

    ColVectorXi64 D(BATCH_SIZE); // ��ʼ������Ϊ BATCH_SIZE ������ D
    ColVectorXi64 Y_(BATCH_SIZE); // ��ʼ������Ϊ BATCH_SIZE ������ Y_
    ColVectorXi64 Fi_(BATCH_SIZE); // ��ʼ������Ϊ BATCH_SIZE ������ Fi_
    ColVectorXi64 F_(BATCH_SIZE); // ��ʼ������Ϊ BATCH_SIZE ������ F_
    ColVectorXi64 delta(d); // ��ʼ������Ϊ d ������ delta

    if (party == ALICE) // �жϵ�ǰ party �Ƿ�Ϊ ALICE
        send<ColVectorXi64>(io, Fi); // ���� ALICE ���� Fi
    else
        recv<ColVectorXi64>(io, F); // ������ ALICE ���� F

    if (party == BOB) // �жϵ�ǰ party �Ƿ�Ϊ BOB
        send<ColVectorXi64>(io, Fi); // ���� BOB ���� Fi
    else
        recv<ColVectorXi64>(io, F); // ������ BOB ���� F

    F += Fi; // �� F �� Fi ���

    Y_ = -i * (Eb * F)  + X * F + Eb * wi + Z; // ���� Y_

    truncate<ColVectorXi64>(i, SCALING_FACTOR, Y_); // �ض� Y_

    D = Y_ - Y; // ���� D

    Fi_ = D - V_; // ���� Fi_

    if (party == ALICE) // �жϵ�ǰ party �Ƿ�Ϊ ALICE
        send<ColVectorXi64>(io, Fi_); // ���� ALICE ���� Fi_
    else
        recv<ColVectorXi64>(io, F_); // ������ ALICE ���� F_

    if (party == BOB) // �жϵ�ǰ party �Ƿ�Ϊ BOB
        send<ColVectorXi64>(io, Fi_); // ���� BOB ���� Fi_
    else
        recv<ColVectorXi64>(io, F_); // ������ BOB ���� F_

    F_ += Fi_; // �� F_ �� Fi_ ���

    RowMatrixXi64 Et = Eb.transpose(); // Eb ��ת�þ���
    RowMatrixXi64 Xt = X.transpose(); // X ��ת�þ���

    delta = -i * (Et * F_) + Xt * F_ + Et * D + Z_; // ���� delta

    truncate<ColVectorXi64>(i, SCALING_FACTOR, delta); // �ض� delta
    truncate<ColVectorXi64>(i, alpha_inv * BATCH_SIZE, delta); // �ض� delta

    wi -= delta; // ���� wi
}
