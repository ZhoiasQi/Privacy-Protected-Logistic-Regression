#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "offline_phase.hpp"
#include "online_phase.hpp"
#include <iostream>

using namespace std;

class LogisticRegression{
public:
    emp::NetIO* io;  // ���������������ָ��
    int party;  // ��ǰ���뷽��ALICE��BOB��
    int n, d, t;  // ѵ�����ݴ�С������������������
    RowMatrixXi64 X;  // ѵ�����ݾ���
    ColVectorXi64 Y;  // ѵ����ǩ����
    ColVectorXi64 w;  // Ȩ���������������ͣ�
    ColVectorXd w_d;  // Ȩ��������˫���ȸ��������ͣ�
    OfflineSetUp* setup;  // ���ý׶ζ���ָ��

    
    // OnlinePhase* online;  // ���߽׶ζ���ָ��



    LogisticRegression(RowMatrixXi64& training_data, ColVectorXi64& training_labels, TrainingParams params, emp::NetIO* io) {
        this->n = params.n;  // ����ѵ�����ݴ�С
        this->d = params.d;  // ����������
        this->t = (params.n) / BATCH_SIZE;  // �����������
        this->X = training_data;  // ����ѵ������
        this->Y = training_labels;  // ����ѵ����ǩ
        this->io = io;  // �������������������ָ��
        this->party = PARTY;  // ���õ�ǰ���뷽��ALICE��BOB��
        this->w.resize(d);  // ����Ȩ��������С
        this->w_d.resize(d);  // ����Ȩ��������С

        this->setup = new OfflineSetUp();
    }

    void train_model();

};

#endif