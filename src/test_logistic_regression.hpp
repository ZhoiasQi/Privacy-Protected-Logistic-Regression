#ifndef T_LOGISTIC_REGRESSION_H
#define T_LOGISTIC_REGRESSION_H

#include "test_offline_phase.hpp"
#include "test_online_phase.hpp"
#include <iostream>

using namespace std;

class TestLogisticRegression{
public:
    emp::NetIO* io;  // ���������������ָ��
    int party;  // ��ǰ���뷽��ALICE��BOB��
    int n, d, t;  // ѵ�����ݴ�С������������������
    RowMatrixXi64 X;  // ѵ�����ݾ���
    ColVectorXi64 Y;  // ѵ����ǩ����
    ColVectorXi64 w;  // Ȩ���������������ͣ�
    ColVectorXi64 wi;
    ColVectorXi64 prediction;
    TestSetUp* setup;
    TestOnlinePhase* online;  // ���߽׶ζ���ָ��

    //���Խ׶ι��캯��
    TestLogisticRegression(RowMatrixXi64& testing_data, ColVectorXi64& testing_labels, TestingParams params, emp::NetIO* io) {
        this->n = params.n;
        this->d = params.d;
        this->t = 1;
        this->X = testing_data;
        this->Y = testing_labels;
        this->io = io;
        this->party = PARTY;
        this->w.resize(d);
        this->wi.resize(d);
        this->prediction.resize(d);
        
        this->setup = new TestSetUp(n, d, t, io);
        setup->generateMTs();

        SetupTriples triples;
        setup->getMTs(&triples);

        RowMatrixXi64 Xi(X.rows(), X.cols());  // ��ʼ���洢�������ݵľ���
        ColVectorXi64 Yi(Y.rows(), Y.cols());  // ��ʼ���洢���ܱ�ǩ������

        if (party == CAROL) {  // �����ǰ���뷽��CAROL
            emp::PRG prg;  // α���������������
            RowMatrixXi64 rX(X.rows(), X.cols());  // ������ݾ���
            ColVectorXi64 rY(Y.rows(), Y.cols());  // �����ǩ����
            prg.random_data(rX.data(), X.rows() * X.cols() * sizeof(uint64_t));  // ����������� rX
            prg.random_data(rY.data(), Y.rows() * Y.cols() * sizeof(uint64_t));  // ����������� rY
            Xi = X + rX;  // ���ܺ��ѵ������
            Yi = Y + rY;  // ���ܺ��ѵ����ǩ
            rX *= -1;  // �� rX �е�Ԫ��ȡ��
            rY *= -1;  // �� rY �е�Ԫ��ȡ��
            send<RowMatrixXi64>(io, rX);  // ����������� rX
            send<ColVectorXi64>(io, rY);  // ����������� rY

            cout << "Carol has secretly sent the data to Alice" << endl;
        } else {  // �����ǰ���뷽��Alice
            recv<RowMatrixXi64>(io, Xi);  // �ӶԷ����ռ��ܺ��ѵ������
            recv<ColVectorXi64>(io, Yi);  // �ӶԷ����ռ��ܺ��ѵ����ǩ

            cout << "Alice has received the secret data from Carol" << endl;

        }
    }

    void getW(ColVectorXi64 w);

    void secret_share_w();

    void test_model();
};

#endif

