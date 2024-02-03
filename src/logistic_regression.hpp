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

    //OnlinePhase* online;  // ���߽׶ζ���ָ��

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

        this->setup = new OfflineSetUp(n, d, t, io);
        setup->generateMTs();  // ���������Ԫ��

        SetupTriples triples;
        setup->getMTs(&triples);  // ��ȡ�����Ԫ��


        // RowMatrixXi64 Xi(X.rows(), X.cols());  // ��ʼ���洢�������ݵľ���
        // ColVectorXi64 Yi(Y.rows(), Y.cols());  // ��ʼ���洢���ܱ�ǩ������
        
        // if (party == emp::ALICE) {  // �����ǰ���뷽��ALICE
        //     emp::PRG prg;  // α���������������
        //     RowMatrixXi64 rX(X.rows(), X.cols());  // ������ݾ���
        //     ColVectorXi64 rY(Y.rows(), Y.cols());  // �����ǩ����
        //     prg.random_data(rX.data(), X.rows() * X.cols() * sizeof(uint64_t));  // ����������� rX
        //     prg.random_data(rY.data(), Y.rows() * Y.cols() * sizeof(uint64_t));  // ����������� rY
        //     Xi = X + rX;  // ���ܺ��ѵ������
        //     Yi = Y + rY;  // ���ܺ��ѵ����ǩ
        //     rX *= -1;  // �� rX �е�Ԫ��ȡ��
        //     rY *= -1;  // �� rY �е�Ԫ��ȡ��
        //     send<RowMatrixXi64>(io, rX);  // ����������� rX
        //     send<ColVectorXi64>(io, rY);  // ����������� rY
        // } else {  // �����ǰ���뷽��BOB
        //     recv<RowMatrixXi64>(io, Xi);  // �ӶԷ����ռ��ܺ��ѵ������
        //     recv<ColVectorXi64>(io, Yi);  // �ӶԷ����ռ��ܺ��ѵ����ǩ
        // }

        // this->online = new OnlinePhase(params, io, &triples);  // �������߽׶ζ���
        // online->initialize(Xi, Yi);  // �����߽׶γ�ʼ��

        // train_model();  // ѵ��ģ��
    }

    // // ѵ��ģ�͵ķ���
    // void train_model();

    // // ����ģ�͵ķ���
    // void test_model(RowMatrixXd& testing_data, ColVectorXd& testing_labels);

};

#endif